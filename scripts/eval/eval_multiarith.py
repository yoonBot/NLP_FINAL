#!/usr/bin/env python3
"""
Evaluate a trained ReasoningGPT checkpoint on the MultiArith dev set.

Usage:
    # SFT checkpoint (.pt)
    python eval_multiarith.py --checkpoint 9_10-1e-05-reasoning.pt --use_gpu

    # DPO / HuggingFace model directory
    python eval_multiarith.py --checkpoint outputs/dpo_gpt2_model --use_gpu

Outputs:
    outputs/<name>_multiarith_metrics.json    aggregate metrics
    outputs/<name>_multiarith_generations.txt full generations per dev item
"""

import sys
from pathlib import Path
_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "scripts" / "train"))

import argparse
import json
import os

import torch

from reasoning_generation import ReasoningGPT, add_arguments, seed_everything
import gsm8k_eval

DEV_JSONL = os.path.join("data", "multiarith_dev.jsonl")


def load_dev(path: str) -> list[dict]:
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            records.append(json.loads(line))
    return records


def main():
    args = get_args()
    seed_everything(args.seed)
    device = torch.device("cuda") if args.use_gpu else torch.device("cpu")

    if os.path.isdir(args.checkpoint):
        from transformers import GPT2LMHeadModel, GPT2Tokenizer
        model = GPT2LMHeadModel.from_pretrained(args.checkpoint).to(device)
        tokenizer = GPT2Tokenizer.from_pretrained(args.checkpoint)
        tokenizer.pad_token = tokenizer.eos_token
        use_hf = True
    else:
        saved = torch.load(args.checkpoint, weights_only=False)
        model_args = saved["args"]
        model = ReasoningGPT(model_args).to(device)
        model.load_state_dict(saved["model"])
        model.eval()
        use_hf = False

    dev = load_dev(DEV_JSONL)
    if args.limit:
        dev = dev[: args.limit]
    print(f"Evaluating on {len(dev)} MultiArith dev examples...")

    eval_records = []
    gen_lines = []

    for rec in dev:
        prompt = f"Question: {rec['question']}\n\nReasoning:\n"

        if use_hf:
            enc = tokenizer(
                prompt, return_tensors="pt", truncation=True, max_length=512
            ).to(device)
            with torch.no_grad():
                out_ids = model.generate(
                    **enc,
                    max_new_tokens=args.max_new_tokens,
                    do_sample=False,
                    pad_token_id=tokenizer.eos_token_id,
                )
            generated = tokenizer.decode(out_ids[0], skip_special_tokens=True)
        else:
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
        eval_records.append({"generation": continuation, "gold": rec["gold_answer"]})
        gen_lines.append(
            f"\n=== dev {rec['id']} (gold={rec['gold_answer']}) ===\n{generated}\n"
        )

    metrics = gsm8k_eval.evaluate(eval_records)
    metrics["checkpoint"] = args.checkpoint
    metrics["dataset"] = "multiarith"

    name = os.path.splitext(os.path.basename(args.checkpoint.rstrip("/\\")))[0]
    os.makedirs("outputs", exist_ok=True)
    met_path = os.path.join("outputs", f"{name}_multiarith_metrics.json")
    gen_path = os.path.join("outputs", f"{name}_multiarith_generations.txt")

    with open(met_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    with open(gen_path, "w", encoding="utf-8") as f:
        f.writelines(gen_lines)

    print(json.dumps(metrics, indent=2))
    print(f"Metrics     -> {met_path}")
    print(f"Generations -> {gen_path}")


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True,
                        help=".pt file or HuggingFace model directory")
    parser.add_argument("--use_gpu", action="store_true")
    parser.add_argument("--seed", type=int, default=11711)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument("--limit", type=int, default=0,
                        help="If >0, only evaluate first N dev items (smoke test).")
    return parser.parse_args()


if __name__ == "__main__":
    main()
