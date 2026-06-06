#!/usr/bin/env python3
import sys
from pathlib import Path
_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "scripts" / "train"))
sys.path.insert(0, str(_ROOT / "scripts" / "eval"))

"""
Generate DPO (chosen, rejected) pairs using a trained SFT checkpoint.

chosen  = gold_reasoning from gsm8k_dpo_source.jsonl
rejected = SFT model's wrong generation (correct answers are skipped)

Usage:
    python generate_rejected.py --checkpoint 9_10-1e-05-reasoning.pt --use_gpu

Output:
    data/dpo_pairs.jsonl   {prompt, chosen, rejected}
"""

import argparse
import json
import os

import torch
from tqdm import tqdm

from reasoning_generation import ReasoningGPT, seed_everything
import gsm8k_eval

DPO_SOURCE = os.path.join("data", "gsm8k_dpo_source.jsonl")
OUT_PATH = os.path.join("data", "dpo_pairs.jsonl")


def main():
    args = get_args()
    seed_everything(args.seed)
    device = torch.device("cuda") if args.use_gpu else torch.device("cpu")

    saved = torch.load(args.checkpoint, weights_only=False)
    model = ReasoningGPT(saved["args"]).to(device)
    model.load_state_dict(saved["model"])
    model.eval()

    with open(DPO_SOURCE, "r", encoding="utf-8") as f:
        sources = [json.loads(line) for line in f]

    if args.limit:
        sources = sources[: args.limit]

    pairs = []
    skipped_correct = 0
    skipped_no_answer = 0

    for rec in tqdm(sources, desc="generating rejected"):
        prompt = f"Question: {rec['question']}\n\nReasoning:\n"
        enc = model.tokenizer(
            prompt, return_tensors="pt", truncation=True, max_length=512
        ).to(device)

        _, generated = model.generate(
            enc["input_ids"],
            temperature=args.temperature,
            top_p=args.top_p,
            max_length=args.max_new_tokens,
        )

        continuation = generated[len(prompt):] if generated.startswith(prompt) else generated

        # Skip if SFT got it right — can't use as rejected
        if gsm8k_eval.is_correct(continuation, rec["gold_answer"]):
            skipped_correct += 1
            continue

        # Skip if no answer extracted at all (degenerate output)
        pred, _ = gsm8k_eval.extract_pred_answer(continuation)
        if pred is None:
            skipped_no_answer += 1
            continue

        pairs.append({
            "prompt": prompt,
            "chosen": rec["gold_reasoning"],
            "rejected": continuation,
        })

    os.makedirs(os.path.dirname(OUT_PATH) if os.path.dirname(OUT_PATH) else ".", exist_ok=True)
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        for pair in pairs:
            f.write(json.dumps(pair, ensure_ascii=False) + "\n")

    total = len(sources)
    print(f"\n=== DPO pair generation complete ===")
    print(f"Total source examples : {total}")
    print(f"Skipped (SFT correct) : {skipped_correct} ({skipped_correct/total*100:.1f}%)")
    print(f"Skipped (no answer)   : {skipped_no_answer} ({skipped_no_answer/total*100:.1f}%)")
    print(f"Valid DPO pairs saved : {len(pairs)} -> {OUT_PATH}")


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to SFT .pt checkpoint")
    parser.add_argument("--use_gpu", action="store_true")
    parser.add_argument("--seed", type=int, default=11711)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument("--limit", type=int, default=0,
                        help="If >0, only process first N examples (smoke test).")
    return parser.parse_args()


if __name__ == "__main__":
    main()
