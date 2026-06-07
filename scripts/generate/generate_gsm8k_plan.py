#!/usr/bin/env python3
"""
Generate two plan-CoT variants of data/gsm8k_sft_train.txt (3000 blocks).

Outputs
-------
data/gsm8k_sft_train_plan_skeleton.txt
    Operator-only plan — no numbers, no intermediate results.
    Plan: Solve in N steps. (1) multiply; (2) add; ...; then give the final answer.

data/gsm8k_sft_train_plan_entity.txt
    Entity-abstract plan (PS+ style) — natural-language subject + operator,
    numbers stripped.  Approximates Plan-and-Solve PS+ (Wang et al. ACL 2023).
    Plan: find Kim's height, then calculate Tamara's height, then give the final answer.
"""

import os
import re
import sys

SRC  = "data/gsm8k_sft_train.txt"
SKEL = "data/gsm8k_sft_train_plan_skeleton.txt"
ENT  = "data/gsm8k_sft_train_plan_entity.txt"

_CALC_RE  = re.compile(r"<<([^>]*)>>")
_NUM_RE   = re.compile(r"-?\$?[\d,]+(?:\.\d+)?%?")


# ── operator extraction ──────────────────────────────────────────────────────

def _operator(expr: str) -> str:
    """Return the dominant arithmetic operator word for an expression."""
    expr = expr.split("=")[0].strip()
    has = lambda op: op in expr
    if has("*") and not any(has(o) for o in ["+", "-", "/"]):
        return "multiply"
    if has("/") and not any(has(o) for o in ["+", "-", "*"]):
        return "divide"
    if has("+") and not any(has(o) for o in ["-", "*", "/"]):
        return "add"
    if has("-") and not any(has(o) for o in ["+", "*", "/"]):
        return "subtract"
    if re.match(r"[\d.]+\s*=\s*[\d.]+", expr):
        return "find the value"
    return "calculate"


# ── entity description extraction ────────────────────────────────────────────

_STOP = {"let", "the", "a", "an", "is", "are", "was", "were", "be", "been",
         "of", "in", "at", "by", "for", "with", "on", "each", "per",
         "total", "how", "many", "much", "find", "calculate"}

def _entity_words(text: str) -> str:
    """Strip numbers/$/%/punctuation from text; return up to 5 meaningful words."""
    text = _CALC_RE.sub("", text)         # remove <<...>> blocks
    text = _NUM_RE.sub("", text)          # remove numbers
    text = re.sub(r"[$%*+/=\(\)\[\]\{\}]", " ", text)
    words = [w.strip(".,;:") for w in text.split()
             if w.strip(".,;:").lower() not in _STOP and len(w.strip(".,;:")) > 1]
    return " ".join(words[:5]).strip()


def _entity_desc(line: str, expr: str) -> str:
    """Build one plan step description from a reasoning line + its calc expression."""
    op = _operator(expr)
    pre = line[:line.find("<<")].strip() if "<<" in line else line

    # self-assignment like <<24=24>>: try to name the variable from LHS of '='
    raw_expr = expr.split("=")[0].strip()
    if re.match(r"^[\d.]+$", raw_expr):
        # pure number LHS — extract subject from line text
        subj = _entity_words(pre.split("=")[0] if "=" in pre else pre)
        return f"find {subj}" if subj else "find the value"

    subject = _entity_words(pre)
    if subject:
        return f"{op} {subject}"
    return op


# ── plan builders ─────────────────────────────────────────────────────────────

def _extract_calc_steps(reasoning: str):
    """Return list of (line, expr) tuples for lines containing <<...>>."""
    steps = []
    for line in reasoning.split("\n"):
        m = _CALC_RE.search(line)
        if m:
            steps.append((line, m.group(1)))
    return steps


def _skeleton_plan(steps) -> str:
    if not steps:
        return "Plan: Solve directly; then give the final answer."
    ops = [_operator(expr) for _, expr in steps]
    body = "; ".join(f"({i+1}) {op}" for i, op in enumerate(ops))
    return f"Plan: Solve in {len(ops)} steps. {body}; then give the final answer."


def _entity_plan(steps) -> str:
    if not steps:
        return "Plan: Solve directly; then give the final answer."
    descs = [_entity_desc(line, expr) for line, expr in steps]
    return "Plan: " + ", then ".join(descs) + ", then give the final answer."


# ── block processor ───────────────────────────────────────────────────────────

def _insert_plan(block: str, plan_fn) -> str:
    """Insert 'Plan: ...' between Question and Reasoning in one block."""
    m = re.search(r"(Reasoning:)", block)
    if not m:
        return block
    reasoning_start = m.start()
    reasoning_text  = block[reasoning_start:]
    steps = _extract_calc_steps(reasoning_text)
    plan  = plan_fn(steps)
    prefix = block[:reasoning_start].rstrip()
    return f"{prefix}\n\n{plan}\n\n{reasoning_text}"


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    missing = [p for p in [SKEL, ENT] if not os.path.exists(p)]
    if not missing:
        print("[skip] both plan files already exist")
        return 0

    if not os.path.exists(SRC):
        print(f"Error: {SRC} not found.")
        return 1

    with open(SRC, encoding="utf-8") as f:
        content = f.read()

    blocks = content.split("<|endoftext|>")

    skel_blocks = []
    ent_blocks  = []
    for i, block in enumerate(blocks):
        if not block.strip():
            skel_blocks.append(block)
            ent_blocks.append(block)
            continue
        skel_blocks.append(_insert_plan(block, _skeleton_plan))
        ent_blocks.append(_insert_plan(block, _entity_plan))

    for path, processed, label in [
        (SKEL, skel_blocks, "skeleton"),
        (ENT,  ent_blocks,  "entity"),
    ]:
        if os.path.exists(path):
            print(f"[skip] {path} already exists")
            continue
        out = "<|endoftext|>".join(processed)
        with open(path, "w", encoding="utf-8") as f:
            f.write(out)
        n = out.count("<|endoftext|>")
        print(f"[{label}] wrote {path}  ({n} blocks)")

    # preview first 2 blocks of each
    for path, label in [(SKEL, "skeleton"), (ENT, "entity")]:
        print(f"\n=== {label} preview (first 2 blocks) ===")
        preview_blocks = open(path, encoding="utf-8").read().split("<|endoftext|>")
        shown = 0
        for b in preview_blocks:
            if b.strip():
                print(b.strip())
                print("---")
                shown += 1
                if shown >= 2:
                    break

    return 0


if __name__ == "__main__":
    sys.exit(main())
