#!/usr/bin/env python3
import sys
from pathlib import Path
_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "scripts" / "train"))

"""
Base GPT-2 baseline on the GSM8K dev split (RESEARCH_DESIGN.md priority #1).

Runs the *un-fine-tuned* GPT-2 on data/gsm8k_dev_prompts.txt with greedy decoding
(do_sample=False, per design doc 6.2), parses answers with the shared gsm8k_eval
parser, and reports the primary + secondary metrics. This produces the missing
base-accuracy number that the H1 (Base vs CoT-SFT) comparison needs.

Run (inside the cs224n conda env, GPU recommended):
    python baseline_gsm8k.py --use_gpu --model_size gpt2

Outputs:
    outputs/baseline_<model_size>_generations.txt   full generations per dev item
    outputs/baseline_<model_size>_metrics.json       aggregate metrics
"""

import argparse
import json
import os
import re

import torch

from reasoning_generation import ReasoningGPT, add_arguments, seed_everything
import gsm8k_eval

DEV_PROMPTS = os.path.join("data", "gsm8k_dev_prompts.txt")
DEV_JSONL = os.path.join("data", "gsm8k_dev.jsonl")


def load_dev_prompts(path):
    """Return list of (id, prompt_text) parsed from the held-out-format file."""
    with open(path, "r", encoding="utf-8") as f:
        text = f.read()
    blocks = re.split(r"\n\s*\d+\s*\n", text)
    blocks = [b.strip() for b in blocks if b.strip()]
    return list(enumerate(blocks))


def load_gold(path):
    gold = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            gold[rec["id"]] = rec["gold_answer"]
    return gold


def main():
    args = get_args()
    args = add_arguments(args)
    seed_everything(args.seed)

    device = torch.device("cuda") if args.use_gpu else torch.device("cpu")
    model = ReasoningGPT(args).to(device)
    model.eval()

    prompts = load_dev_prompts(DEV_PROMPTS)
    gold = load_gold(DEV_JSONL)
    if args.limit:
        prompts = prompts[: args.limit]

    records = []
    gen_lines = []
    for idx, prompt in prompts:
        encoding = model.tokenizer(
            prompt, return_tensors="pt", padding=False, truncation=True, max_length=900
        ).to(device)
        _, generated = model.generate(
            encoding["input_ids"],
            temperature=args.temperature,
            top_p=args.top_p,
            max_length=args.max_new_tokens,
        )
        # Score only the continuation (drop the echoed prompt) for answer extraction.
        continuation = generated[len(prompt):] if generated.startswith(prompt) else generated
        records.append({"generation": continuation, "gold": gold.get(idx)})
        gen_lines.append(f"\n=== dev {idx} (gold={gold.get(idx)}) ===\n{generated}\n")

    metrics = gsm8k_eval.evaluate(records)
    metrics["model_size"] = args.model_size
    metrics["decoding"] = "greedy(argmax)"

    os.makedirs("outputs", exist_ok=True)
    gen_path = os.path.join("outputs", f"baseline_{args.model_size}_generations.txt")
    met_path = os.path.join("outputs", f"baseline_{args.model_size}_metrics.json")
    with open(gen_path, "w", encoding="utf-8") as f:
        f.writelines(gen_lines)
    with open(met_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print(json.dumps(metrics, indent=2))
    print(f"Generations -> {gen_path}")
    print(f"Metrics     -> {met_path}")


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--use_gpu", action="store_true")
    parser.add_argument("--seed", type=int, default=11711)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument("--limit", type=int, default=0,
                        help="If >0, only evaluate the first N dev items (quick smoke test).")
    parser.add_argument("--model_size", type=str,
                        choices=["gpt2", "gpt2-medium", "gpt2-large"], default="gpt2")
    return parser.parse_args()


if __name__ == "__main__":
    main()
