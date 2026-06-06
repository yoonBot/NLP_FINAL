#!/usr/bin/env python3
import sys
from pathlib import Path
_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "scripts" / "train"))
sys.path.insert(0, str(_ROOT / "scripts" / "eval"))

"""
Generate DPO (chosen, rejected) pairs using a trained SFT checkpoint for e-SNLI.

chosen  = gold explanation + label (from esnli_dpo_source.jsonl)
rejected = SFT model's wrong explanation + label (correct predictions are skipped)

Usage:
    python generate_rejected_esnli.py --checkpoint 2_1e-05-esnli.pt --use_gpu
"""

import argparse
import json
import os
import torch
from tqdm import tqdm

from reasoning_generation_esnli import ESNLIGPT, seed_everything, add_arguments
import eval_esnli

DPO_SOURCE = os.path.join("data", "esnli_dpo_source.jsonl")
OUT_PATH = os.path.join("data", "esnli_dpo_pairs.jsonl")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to SFT .pt checkpoint")
    parser.add_argument("--use_gpu", action="store_true")
    parser.add_argument("--seed", type=int, default=11711)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--max_new_tokens", type=int, default=128)
    parser.add_argument("--limit", type=int, default=0, help="Smoke test limit")
    args = parser.parse_args()

    seed_everything(args.seed)
    device = torch.device("cuda") if args.use_gpu else torch.device("cpu")

    # Load SFT model
    saved = torch.load(args.checkpoint, weights_only=False)
    model_args = saved["args"]
    model_args = add_arguments(model_args)
    model = ESNLIGPT(model_args).to(device)
    model.load_state_dict(saved["model"])
    model.eval()

    with open(DPO_SOURCE, "r", encoding="utf-8") as f:
        sources = [json.loads(line) for line in f]

    if args.limit > 0:
        sources = sources[:args.limit]

    pairs = []
    skipped_correct = 0
    skipped_no_answer = 0

    print(f"Generating DPO rejected outputs using checkpoint {args.checkpoint}...")

    for rec in tqdm(sources, desc="generating rejected"):
        prompt = f"Premise: {rec['premise']}\nHypothesis: {rec['hypothesis']}\n\nExplanation:\n"
        enc = model.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512).to(device)

        _, generated = model.generate(
            enc["input_ids"],
            temperature=args.temperature,
            top_p=args.top_p,
            max_length=args.max_new_tokens,
        )

        continuation = generated[len(prompt):] if generated.startswith(prompt) else generated
        pred_label = eval_esnli.extract_pred_label(continuation)

        gold_label = rec["gold_label"].lower()
        gold_explanation = rec["gold_explanation"]
        
        # SFT got it right - skip
        if pred_label == gold_label:
            skipped_correct += 1
            continue

        # Degenerate prediction - skip
        if pred_label == "none":
            skipped_no_answer += 1
            continue

        # Setup Chosen and Rejected reasoning
        chosen = f"{gold_explanation}\nTherefore, the relationship is {rec['gold_label']}.\n"
        rejected = continuation

        pairs.append({
            "prompt": prompt,
            "chosen": chosen,
            "rejected": rejected,
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

if __name__ == "__main__":
    main()
