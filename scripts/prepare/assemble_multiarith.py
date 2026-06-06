#!/usr/bin/env python3
"""
Validate externally-generated CoT and assemble the augmented SFT training file.

Takes the CoT that Antigravity produced for each augmented problem, validates it
against the equation-derived gold answer (every <<expr=result>>result block must
compute correctly AND the final #### N must equal the gold answer), discards any
that fail, then writes the final GSM8K-format SFT file = original train + verified
augmented examples.

Inputs:
    data/multiarith_sft_train_base.txt   (original train CoT, from prepare_multiarith.py)
    data/multiarith_aug_cot.jsonl        (Antigravity output)
        {src_id, variant, question, gold_answer, equation, cot}
    data/multiarith_dev.jsonl            (for the leakage assertion)

Output:
    data/multiarith_sft_train_aug.txt

Usage:
    python assemble_multiarith.py
"""

import json
import os
import re

AUG_COT_PATH = os.path.join("data", "multiarith_aug_cot.jsonl")
SFT_TRAIN_BASE = os.path.join("data", "multiarith_sft_train_base.txt")
DEV_JSONL = os.path.join("data", "multiarith_dev.jsonl")
OUT_PATH = os.path.join("data", "multiarith_sft_train_aug.txt")

_BLOCK = re.compile(r"<<\s*([0-9+\-*/().\s]+?)\s*=\s*(-?\d+(?:\.\d+)?)\s*>>")
_FINAL = re.compile(r"####\s*(-?\d+(?:\.\d+)?)")


def cot_is_valid(cot: str, gold: float, tol: float = 1e-6) -> bool:
    """True iff every <<expr=res>> block computes correctly and #### == gold."""
    blocks = _BLOCK.findall(cot)
    if not blocks:
        return False
    for expr, claimed in blocks:
        try:
            computed = eval(expr, {"__builtins__": {}}, {})  # arithmetic only (regex-gated)
        except Exception:
            return False
        if abs(computed - float(claimed)) > tol:
            return False
    m = _FINAL.search(cot)
    if not m:
        return False
    return abs(float(m.group(1)) - float(gold)) <= tol


def format_sft(idx: str, question: str, cot: str) -> str:
    return (
        f"{idx}\n\n"
        f"Question: {question.strip()}\n\n"
        f"Reasoning:\n{cot.strip()}\n\n"
        f"<|endoftext|>\n\n"
    )


def main():
    if not os.path.exists(AUG_COT_PATH):
        raise SystemExit(
            f"{AUG_COT_PATH} not found. Run augment_multiarith.py and have Antigravity "
            f"fill in the 'cot' field first."
        )

    dev_ids = set()
    with open(DEV_JSONL, encoding="utf-8") as f:
        for line in f:
            dev_ids.add(json.loads(line)["id"])

    accepted = 0
    rejected = 0
    leaked = 0
    aug_blocks = []
    with open(AUG_COT_PATH, encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            # Leakage guard: an augmented example must never come from a dev source.
            if rec["src_id"] in dev_ids:
                leaked += 1
                continue
            if not rec.get("cot") or not cot_is_valid(rec["cot"], rec["gold_answer"]):
                rejected += 1
                continue
            ex_id = f"{rec['src_id']}_aug{rec['variant']}"
            aug_blocks.append(format_sft(ex_id, rec["question"], rec["cot"]))
            accepted += 1

    assert leaked == 0, f"LEAKAGE: {leaked} augmented examples derive from dev sources!"

    base_text = ""
    if os.path.exists(SFT_TRAIN_BASE):
        with open(SFT_TRAIN_BASE, encoding="utf-8") as f:
            base_text = f.read()
    base_count = base_text.count("<|endoftext|>")

    with open(OUT_PATH, "w", encoding="utf-8") as f:
        f.write(base_text)
        if base_text and not base_text.endswith("\n\n"):
            f.write("\n\n")
        f.writelines(aug_blocks)

    print(f"base train examples   : {base_count}")
    print(f"augmented accepted    : {accepted}")
    print(f"augmented rejected    : {rejected}  (failed arithmetic/format/gold check)")
    print(f"dev-source leakage    : {leaked}  (must be 0)")
    print(f"total written          : {base_count + accepted}  -> {OUT_PATH}")


if __name__ == "__main__":
    main()
