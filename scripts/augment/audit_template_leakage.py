#!/usr/bin/env python3
"""
Template-level leakage audit for MultiArith train/dev splits.

Each problem is reduced to a template key by:
  - replacing all numbers (integers and decimals) with NUM
  - normalising whitespace

If a dev problem shares the same template as any train problem,
the model may be "memorising a template" rather than generalising.

Outputs
-------
data/multiarith_dev_template_labels.jsonl
    {id, question, gold_answer, template_key, in_train_template}

Prints a summary: how many dev items are in-template vs held-out.

Usage
-----
    python audit_template_leakage.py
    python audit_template_leakage.py --train_sft data/multiarith_sft_train_aug.txt
"""

import argparse
import json
import os
import re

MULTIARITH_TRAIN_IDS = os.path.join("data", "multiarith_train_ids.json")
DEV_JSONL = os.path.join("data", "multiarith_dev.jsonl")
MULTIARITH_JSON = os.path.join("data", "MultiArith.json")
OUT_JSONL = os.path.join("data", "multiarith_dev_template_labels.jsonl")

_NUM_RE = re.compile(r"\d+(?:\.\d+)?")


def to_template(text: str) -> str:
    """Strip numbers and normalise whitespace → template key."""
    return " ".join(_NUM_RE.sub("NUM", text).split()).lower()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--train_sft",
        type=str,
        default=None,
        help="Optional SFT .txt file to extract train questions from "
             "(falls back to MultiArith.json + train_ids).",
    )
    args = parser.parse_args()

    # ── collect train question templates ──────────────────────────────────
    train_templates: set[str] = set()

    if args.train_sft and os.path.exists(args.train_sft):
        # parse GSM8K-format SFT file: "Question: ..." lines
        q_re = re.compile(r"^Question:\s*(.+)$", re.MULTILINE)
        with open(args.train_sft, encoding="utf-8") as f:
            text = f.read()
        for m in q_re.finditer(text):
            train_templates.add(to_template(m.group(1)))
        print(f"Train questions from SFT file : {len(train_templates)}")
    else:
        train_ids = set(json.load(open(MULTIARITH_TRAIN_IDS, encoding="utf-8")))
        problems = json.load(open(MULTIARITH_JSON, encoding="utf-8"))
        for p in problems:
            if p["iIndex"] in train_ids:
                train_templates.add(to_template(p["sQuestion"]))
        print(f"Train questions from MultiArith.json : {len(train_templates)}")

    # ── audit dev ─────────────────────────────────────────────────────────
    dev_records = []
    with open(DEV_JSONL, encoding="utf-8") as f:
        for line in f:
            dev_records.append(json.loads(line))

    in_tmpl = 0
    held_out = 0
    with open(OUT_JSONL, "w", encoding="utf-8") as out:
        for rec in dev_records:
            key = to_template(rec["question"])
            overlap = key in train_templates
            if overlap:
                in_tmpl += 1
            else:
                held_out += 1
            out.write(json.dumps({
                "id": rec["id"],
                "question": rec["question"],
                "gold_answer": rec["gold_answer"],
                "template_key": key,
                "in_train_template": overlap,
            }, ensure_ascii=False) + "\n")

    total = len(dev_records)
    print(f"\nDev template audit (n={total})")
    print(f"  in-template (train overlap) : {in_tmpl}  ({100*in_tmpl/total:.1f}%)")
    print(f"  held-out-template           : {held_out} ({100*held_out/total:.1f}%)")
    print(f"  -> {OUT_JSONL}")
    return {"total": total, "in_template": in_tmpl, "held_out": held_out}


if __name__ == "__main__":
    main()
