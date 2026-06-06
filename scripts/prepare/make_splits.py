#!/usr/bin/env python3
"""
Deterministic split of the existing 5,000 GSM8K examples into SFT / DPO / dev.

Reads data/gsm8k_small_train.txt (already on disk, 5,000 examples) rather than
re-downloading, so the split is reproducible against exactly what we trained on.

Approved split (RESEARCH_DESIGN.md + planning session):
    SFT  : 3,000   (CoT-SFT training, native GSM8K format)
    DPO  : 1,500   (source for chosen/rejected pair generation)
    dev  :   500   (held-out evaluation, parser-scored)
seed = 11711 (matches seed_everything in the training code)

The official GSM8K *test* split is NOT touched here (frozen for final eval, R4).

Outputs:
    data/gsm8k_sft_train.txt     training-format blocks (Question + Reasoning + #### N + EOS)
    data/gsm8k_dpo_source.jsonl  {id, question, gold_answer, gold_reasoning}
    data/gsm8k_dev.jsonl         {id, question, gold_answer}
    data/gsm8k_dev_prompts.txt   prompt-only blocks for inference (held-out format)
"""

import sys
from pathlib import Path
_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT / "scripts" / "eval"))

import json
import os
import random
import re

SEED = 11711
N_SFT = 3000
N_DPO = 1500
N_DEV = 500

DATA_DIR = str(_ROOT / "data")
SRC = os.path.join(DATA_DIR, "gsm8k_small_train.txt")

# Import the shared parser so gold extraction is identical to evaluation.
from gsm8k_eval import extract_gold_answer

_BLOCK_QUESTION_RE = re.compile(r"Question:\s*(.*?)\s*Reasoning:\s*(.*)", re.DOTALL)


def parse_source(path):
    """Parse the native train file into a list of {question, reasoning, gold} dicts."""
    with open(path, "r", encoding="utf-8") as f:
        text = f.read()

    raw_blocks = [b.strip() for b in text.split("<|endoftext|>") if b.strip()]
    examples = []
    for block in raw_blocks:
        m = _BLOCK_QUESTION_RE.search(block)
        if not m:
            continue
        question = m.group(1).strip()
        reasoning = m.group(2).strip()
        gold = extract_gold_answer(reasoning)
        if gold is None:
            # Skip any block without a parseable gold answer rather than
            # silently keeping an unscoreable example.
            continue
        examples.append({"question": question, "reasoning": reasoning, "gold": gold})
    return examples


def write_sft(examples, path):
    """Write native training-format blocks (same shape the model already learned)."""
    with open(path, "w", encoding="utf-8") as f:
        for i, ex in enumerate(examples):
            f.write(
                f"{i}\n\nQuestion: {ex['question']}\n\n"
                f"Reasoning:\n{ex['reasoning']}\n\n<|endoftext|>\n\n"
            )


def write_jsonl(examples, path, include_reasoning):
    with open(path, "w", encoding="utf-8") as f:
        for i, ex in enumerate(examples):
            rec = {"id": i, "question": ex["question"], "gold_answer": ex["gold"]}
            if include_reasoning:
                rec["gold_reasoning"] = ex["reasoning"]
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def write_prompts(examples, path):
    """Prompt-only file for inference (mirrors the held-out file format)."""
    with open(path, "w", encoding="utf-8") as f:
        for i, ex in enumerate(examples):
            f.write(f"{i}\n\nQuestion: {ex['question']}\n\nReasoning:\n\n")


def main():
    examples = parse_source(SRC)
    print(f"Parsed {len(examples)} scoreable examples from {os.path.basename(SRC)}")

    total_needed = N_SFT + N_DPO + N_DEV
    if len(examples) < total_needed:
        raise SystemExit(
            f"Not enough examples: have {len(examples)}, need {total_needed}."
        )

    # Deterministic shuffle, then disjoint slices (no question reused across splits, R4).
    rng = random.Random(SEED)
    indices = list(range(len(examples)))
    rng.shuffle(indices)

    sft_idx = indices[:N_SFT]
    dpo_idx = indices[N_SFT:N_SFT + N_DPO]
    dev_idx = indices[N_SFT + N_DPO:N_SFT + N_DPO + N_DEV]

    # Sanity: no overlap.
    assert len(set(sft_idx) & set(dpo_idx)) == 0
    assert len(set(sft_idx) & set(dev_idx)) == 0
    assert len(set(dpo_idx) & set(dev_idx)) == 0

    sft = [examples[i] for i in sft_idx]
    dpo = [examples[i] for i in dpo_idx]
    dev = [examples[i] for i in dev_idx]

    write_sft(sft, os.path.join(DATA_DIR, "gsm8k_sft_train.txt"))
    write_jsonl(dpo, os.path.join(DATA_DIR, "gsm8k_dpo_source.jsonl"), include_reasoning=True)
    write_jsonl(dev, os.path.join(DATA_DIR, "gsm8k_dev.jsonl"), include_reasoning=False)
    write_prompts(dev, os.path.join(DATA_DIR, "gsm8k_dev_prompts.txt"))

    print(f"SFT : {len(sft)} -> data/gsm8k_sft_train.txt")
    print(f"DPO : {len(dpo)} -> data/gsm8k_dpo_source.jsonl")
    print(f"dev : {len(dev)} -> data/gsm8k_dev.jsonl + data/gsm8k_dev_prompts.txt")
    print("All splits disjoint; gold answers present for every dev example.")


if __name__ == "__main__":
    main()
