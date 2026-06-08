#!/usr/bin/env python3
"""
실제 생성 텍스트 샘플러 — 체크포인트를 로드해 GSM8K dev 질문에 대한
모델 출력을 파일로 저장한다.

Usage
-----
    python scripts/eval/sample_generate.py \
        --exp_id G_A1_direct \
        --out_dir /mnt/outputs \
        --n 20 \
        --use_gpu

    # 또는 체크포인트 경로 직접 지정
    python scripts/eval/sample_generate.py \
        --checkpoint /mnt/outputs/G_B1_skel_best.pt \
        --out_dir /mnt/outputs \
        --n 20 --use_gpu

Outputs
-------
    {out_dir}/{exp_id}_samples.md   -- 마크다운 (사람이 읽기 좋은 형태)
    {out_dir}/{exp_id}_samples.json -- JSON (분석용)
"""

import argparse
import json
import os
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "scripts"))
sys.path.insert(0, str(_ROOT / "scripts" / "eval"))

import torch
import gsm8k_eval
from run_ablation import ReasoningGPT, _load_dev, _make_gpt2_args, load_model


def get_args():
    p = argparse.ArgumentParser()
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--exp_id", help="실험 ID (예: G_A1_direct). out_dir/{exp_id}_best.pt 로드")
    g.add_argument("--checkpoint", help="체크포인트 파일 경로 직접 지정")
    p.add_argument("--out_dir", default="/mnt/outputs")
    p.add_argument("--arith_init_path", default="/mnt/cot_large_integer_arithmetic_pretrain.pt")
    p.add_argument("--n", type=int, default=20, help="샘플링할 dev 예시 수")
    p.add_argument("--rung", default="gsm8k", choices=["gsm8k", "multiarith"])
    p.add_argument("--max_new_tokens", type=int, default=256)
    p.add_argument("--use_gpu", action="store_true")
    p.add_argument("--seed", type=int, default=11711)
    return p.parse_args()


def main():
    args = get_args()
    device = torch.device("cuda" if args.use_gpu and torch.cuda.is_available() else "cpu")

    # ── 체크포인트 로드 ────────────────────────────────────────────────────
    if args.exp_id:
        ckpt_path = os.path.join(args.out_dir, f"{args.exp_id}_best.pt")
        exp_label = args.exp_id
    else:
        ckpt_path = args.checkpoint
        exp_label = Path(ckpt_path).stem

    if not os.path.exists(ckpt_path):
        print(f"[ERROR] 체크포인트 없음: {ckpt_path}")
        sys.exit(1)

    print(f"Loading: {ckpt_path}")
    gpt2_args = _make_gpt2_args()
    model = ReasoningGPT(gpt2_args)
    saved = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model.load_state_dict(saved.get("model", saved), strict=True)
    model = model.to(device)
    model.eval()
    print(f"Loaded (epoch {saved.get('epoch', '?')})")

    # ── dev 로드 ───────────────────────────────────────────────────────────
    dev = _load_dev(args.rung)
    samples = dev[:args.n]
    print(f"Generating {len(samples)} samples on {args.rung} dev ...")

    # ── 생성 ──────────────────────────────────────────────────────────────
    results = []
    correct = 0

    for i, rec in enumerate(samples):
        prompt = f"Question: {rec['question']}\n\nReasoning:\n"
        enc = model.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
        with torch.no_grad():
            full = model.generate(enc["input_ids"].to(device),
                                  max_new_tokens=args.max_new_tokens)
        generation = full[len(prompt):] if full.startswith(prompt) else full

        pred, source = gsm8k_eval.extract_pred_answer(generation)
        gold = float(rec["gold_answer"]) if rec["gold_answer"] is not None else None
        is_correct = (pred is not None and gold is not None
                      and abs(pred - gold) <= 1e-6)
        fmt_ok = gsm8k_eval.is_format_valid(generation)
        has_rep = gsm8k_eval.has_repetition(generation)
        if is_correct:
            correct += 1

        results.append({
            "idx": i,
            "question": rec["question"],
            "gold": gold,
            "generation": generation,
            "pred": pred,
            "pred_source": source,
            "correct": is_correct,
            "format_valid": fmt_ok,
            "has_repetition": has_rep,
        })

        status = "✓" if is_correct else "✗"
        print(f"  [{i+1:3d}/{len(samples)}] {status}  pred={pred}  gold={gold}  rep={has_rep}")

    acc = correct / len(samples)
    print(f"\nacc={acc:.3f}  ({correct}/{len(samples)})")

    # ── JSON 저장 ──────────────────────────────────────────────────────────
    json_path = os.path.join(args.out_dir, f"{exp_label}_samples.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump({"exp_id": exp_label, "n": len(samples), "acc": acc,
                   "samples": results}, f, ensure_ascii=False, indent=2)
    print(f"JSON → {json_path}")

    # ── 마크다운 저장 ──────────────────────────────────────────────────────
    md_path = os.path.join(args.out_dir, f"{exp_label}_samples.md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(f"# {exp_label} — 생성 샘플 ({len(samples)}개)\n\n")
        f.write(f"**acc={acc:.3f}** ({correct}/{len(samples)})\n\n")
        f.write("---\n\n")
        for r in results:
            status = "✓ 정답" if r["correct"] else "✗ 오답"
            rep_flag = " ⚠️ 루핑" if r["has_repetition"] else ""
            fmt_flag = "" if r["format_valid"] else " ⚠️ 형식오류"
            f.write(f"## [{r['idx']+1}] {status}{rep_flag}{fmt_flag}\n\n")
            f.write(f"**Question:** {r['question']}\n\n")
            f.write(f"**Gold:** {r['gold']}  |  **Pred:** {r['pred']} ({r['pred_source']})\n\n")
            f.write("**Generation:**\n```\n")
            # 루핑 시 최대 40줄만 표시
            lines = r["generation"].splitlines()
            if len(lines) > 40:
                f.write("\n".join(lines[:40]))
                f.write(f"\n... ({len(lines)-40}줄 생략)\n")
            else:
                f.write(r["generation"])
            f.write("\n```\n\n---\n\n")
    print(f"MD  → {md_path}")


if __name__ == "__main__":
    main()
