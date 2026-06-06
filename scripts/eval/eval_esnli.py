#!/usr/bin/env python3
import sys
from pathlib import Path
_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "scripts" / "train"))

"""
Evaluate a trained ESNLIGPT checkpoint on the e-SNLI dev set.
"""

import argparse
import json
import os
import re
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from reasoning_generation_esnli import ESNLIGPT, add_arguments, seed_everything

DEV_JSONL = os.path.join("data", "esnli_dev.jsonl")

def load_dev(path):
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            records.append(json.loads(line))
    return records

def extract_pred_label(generation):
    # Match the standard format: "Therefore, the relationship is [label]"
    match = re.search(r"relationship is\s*(entailment|neutral|contradiction)", generation, re.IGNORECASE)
    if match:
        return match.group(1).lower()
    
    # Fallback to search for any label words in the entire generation
    matches = re.findall(r"\b(entailment|neutral|contradiction)\b", generation, re.IGNORECASE)
    if matches:
        return matches[-1].lower()
    
    return "none"

def is_format_valid(generation):
    return bool(re.search(r"Therefore,\s+the\s+relationship\s+is\s+(entailment|neutral|contradiction)", generation, re.IGNORECASE))

def has_repetition(generation, min_line_len=8, threshold=2):
    counts = {}
    for line in generation.splitlines():
        key = line.strip()
        if len(key) < min_line_len:
            continue
        counts[key] = counts.get(key, 0) + 1
        if counts[key] >= threshold:
            return True
    return False

def evaluate(records):
    n = len(records)
    if n == 0:
        return {"n": 0}

    correct = 0
    no_answer = 0
    format_valid = 0
    repetition = 0

    for r in records:
        gen = r["generation"]
        gold = r["gold_label"].lower()

        pred = extract_pred_label(gen)
        if pred == "none":
            no_answer += 1
        if is_format_valid(gen):
            format_valid += 1
        if has_repetition(gen):
            repetition += 1
        if pred == gold:
            correct += 1

    return {
        "n": n,
        "exact_accuracy": correct / n,
        "no_answer_rate": no_answer / n,
        "format_valid_rate": format_valid / n,
        "repetition_rate": repetition / n,
    }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True, help=".pt file or HF directory")
    parser.add_argument("--use_gpu", action="store_true")
    parser.add_argument("--seed", type=int, default=11711)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--max_new_tokens", type=int, default=128)
    parser.add_argument("--limit", type=int, default=0, help="If >0, only evaluate first N items")
    args = parser.parse_args()

    seed_everything(args.seed)
    device = torch.device("cuda") if args.use_gpu else torch.device("cpu")

    # Load model
    if os.path.isdir(args.checkpoint):
        hf_model = GPT2LMHeadModel.from_pretrained(args.checkpoint).to(device)
        tokenizer = GPT2Tokenizer.from_pretrained(args.checkpoint)
        tokenizer.pad_token = tokenizer.eos_token
        use_hf = True
    else:
        saved = torch.load(args.checkpoint, weights_only=False)
        model_args = saved["args"]
        model_args = add_arguments(model_args)
        model = ESNLIGPT(model_args).to(device)
        model.load_state_dict(saved["model"])
        model.eval()
        tokenizer = model.tokenizer
        use_hf = False

    dev = load_dev(DEV_JSONL)
    if args.limit > 0:
        dev = dev[:args.limit]

    eval_records = []
    gen_lines = []

    print(f"Starting evaluation of {len(dev)} examples on e-SNLI...")

    for rec in dev:
        prompt = f"Premise: {rec['premise']}\nHypothesis: {rec['hypothesis']}\n\nExplanation:\n"

        if use_hf:
            enc = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512).to(device)
            with torch.no_grad():
                out_ids = hf_model.generate(
                    **enc,
                    max_new_tokens=args.max_new_tokens,
                    do_sample=True,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    pad_token_id=tokenizer.eos_token_id,
                )
            generated = tokenizer.decode(out_ids[0], skip_special_tokens=True)
        else:
            enc = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512).to(device)
            _, generated = model.generate(
                enc["input_ids"],
                temperature=args.temperature,
                top_p=args.top_p,
                max_length=args.max_new_tokens,
            )

        continuation = generated[len(prompt):] if generated.startswith(prompt) else generated
        eval_records.append({
            "generation": continuation,
            "gold_label": rec["gold_label"]
        })
        
        pred_label = extract_pred_label(continuation)
        gen_lines.append(
            f"\n=== dev {rec['id']} (gold={rec['gold_label']}, pred={pred_label}) ===\n"
            f"Premise: {rec['premise']}\n"
            f"Hypothesis: {rec['hypothesis']}\n"
            f"Generated Continuation:\n{continuation}\n"
        )

    metrics = evaluate(eval_records)
    metrics["checkpoint"] = args.checkpoint

    name = os.path.splitext(os.path.basename(args.checkpoint.rstrip("/\\")))[0]
    os.makedirs("outputs", exist_ok=True)
    met_path = os.path.join("outputs", f"{name}_esnli_metrics.json")
    gen_path = os.path.join("outputs", f"{name}_esnli_generations.txt")

    with open(met_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    with open(gen_path, "w", encoding="utf-8") as f:
        f.writelines(gen_lines)

    print("\n--- Evaluation Metrics ---")
    print(json.dumps(metrics, indent=2))
    print(f"Metrics saved to: {met_path}")
    print(f"Generations saved to: {gen_path}")

if __name__ == "__main__":
    main()
