#!/usr/bin/env python3
"""
Plan-then-solve CoT generator for MultiArith problems.

Prepends a one-line "Plan:" header to existing CoT that names the two operations
in natural language order.  The header is derived purely from the equation tree —
no LLM needed.

Plan line examples
------------------
  ((32+42)-35)  →  Plan: (1) add 32 and 42; (2) subtract 35 from the result
  ((6*7)+8)     →  Plan: (1) multiply 6 and 7; (2) add 8 to the result
  (5*(3+4))     →  Plan: (1) add 3 and 4; (2) multiply 5 by the result

Inputs
------
  data/multiarith_cot_raw.jsonl          base CoT  {iIndex, sQuestion, answer, cot}
  data/multiarith_aug_cot.jsonl          augmented {src_id, variant, question, gold_answer, equation, cot}
  data/MultiArith.json                   for equations of base problems
  data/multiarith_train_ids.json
  data/multiarith_dev.jsonl              (to enforce dev exclusion)

Outputs
-------
  data/multiarith_sft_train_plan.txt          base train with Plan:
  data/multiarith_sft_train_plan_aug.txt      base + number-aug with Plan:

Usage
-----
    python generate_cot_plan.py
"""

import ast
import json
import os
import re

BASE_RAW       = os.path.join("data", "multiarith_cot_raw.jsonl")
AUG_COT        = os.path.join("data", "multiarith_aug_cot.jsonl")
MULTIARITH_JSON = os.path.join("data", "MultiArith.json")
TRAIN_IDS      = os.path.join("data", "multiarith_train_ids.json")
DEV_JSONL      = os.path.join("data", "multiarith_dev.jsonl")

OUT_BASE_PLAN  = os.path.join("data", "multiarith_sft_train_plan.txt")
OUT_AUG_PLAN   = os.path.join("data", "multiarith_sft_train_plan_aug.txt")

_OP_SYM  = {ast.Add: "+", ast.Sub: "-", ast.Mult: "*", ast.Div: "/"}
_OP_WORD = {"+": "add", "-": "subtract", "*": "multiply", "/": "divide"}
# Sentence template for left-deep outer op: "{op_word} {c} {prep}"
# Exception: divide uses different word order handled in make_plan_line
_OP_PREP = {
    "+": "to the result",
    "-": "from the result",
    "*": "by the result",
    "/": "by the result",   # "divide {c} by the result" → handled below
}

_BLOCK = re.compile(r"<<\s*([0-9+\-*/().\s]+?)\s*=\s*(-?\d+(?:\.\d+)?)\s*>>")
_FINAL = re.compile(r"####\s*(-?\d+(?:\.\d+)?)")


def _eval_node(node):
    if isinstance(node, ast.Constant):
        return node.value
    if isinstance(node, ast.BinOp):
        a, b = _eval_node(node.left), _eval_node(node.right)
        op = _OP_SYM[type(node.op)]
        if op == "+": return a + b
        if op == "-": return a - b
        if op == "*": return a * b
        return a // b
    raise ValueError


def make_plan_line(equation: str) -> str:
    """Return a natural-language Plan: line from the equation string."""
    rhs = equation.split("=", 1)[-1].strip()
    # handle float literals like 32.0 → cast to int where exact
    rhs = re.sub(r"\b(\d+)\.0\b", r"\1", rhs)
    tree = ast.parse(rhs, mode="eval").body

    def _fmt(v):
        iv = int(v)
        return str(iv) if iv == v else str(v)

    def _step_desc(op: str, x: str, y: str) -> str:
        """Natural language for 'compute x op y'."""
        if op == "+": return f"add {x} and {y}"
        if op == "-": return f"subtract {y} from {x}"
        if op == "*": return f"multiply {x} by {y}"
        return f"divide {x} by {y}"

    def _step2_from_result(op: str, operand_str: str) -> str:
        """Natural language for 'apply op to (inner result) with outer operand'."""
        if op == "+": return f"add {operand_str} to the result"
        if op == "-": return f"subtract {operand_str} from the result"
        if op == "*": return f"multiply the result by {operand_str}"
        return f"divide the result by {operand_str}"

    if isinstance(tree.left, ast.BinOp):
        # Left-deep: ((a op1 b) op2 c)
        inner_op = _OP_SYM[type(tree.left.op)]
        outer_op = _OP_SYM[type(tree.op)]
        a = _eval_node(tree.left.left)
        b = _eval_node(tree.left.right)
        c = _eval_node(tree.right)
        return (
            f"(1) {_step_desc(inner_op, _fmt(a), _fmt(b))}; "
            f"(2) {_step2_from_result(outer_op, _fmt(c))}"
        )
    elif isinstance(tree.right, ast.BinOp):
        # Right-deep: (a op1 (b op2 c))
        outer_op = _OP_SYM[type(tree.op)]
        inner_op = _OP_SYM[type(tree.right.op)]
        a = _eval_node(tree.left)
        b = _eval_node(tree.right.left)
        c = _eval_node(tree.right.right)
        return (
            f"(1) {_step_desc(inner_op, _fmt(b), _fmt(c))}; "
            f"(2) {_step2_from_result(outer_op, _fmt(a))}"
        )
    else:
        # Flat single-op (rare in MultiArith but handle gracefully)
        op = _OP_SYM[type(tree.op)]
        a = _eval_node(tree.left)
        b = _eval_node(tree.right)
        return f"(1) {_step_desc(op, _fmt(a), _fmt(b))}"


def prepend_plan(cot: str, plan_line: str) -> str:
    return f"Plan: {plan_line}\n{cot.strip()}"


def cot_is_valid(cot: str, gold: float, tol: float = 1e-6) -> bool:
    blocks = _BLOCK.findall(cot)
    if not blocks:
        return False
    for expr, claimed in blocks:
        try:
            computed = eval(expr, {"__builtins__": {}}, {})
        except Exception:
            return False
        if abs(computed - float(claimed)) > tol:
            return False
    m = _FINAL.search(cot)
    if not m:
        return False
    return abs(float(m.group(1)) - float(gold)) <= tol


def format_sft(idx, question: str, cot: str) -> str:
    return (
        f"{idx}\n\n"
        f"Question: {question.strip()}\n\n"
        f"Reasoning:\n{cot.strip()}\n\n"
        f"<|endoftext|>\n\n"
    )


def main():
    problems   = {p["iIndex"]: p for p in json.load(open(MULTIARITH_JSON, encoding="utf-8"))}
    train_ids  = set(json.load(open(TRAIN_IDS, encoding="utf-8")))
    dev_ids    = {json.loads(l)["id"] for l in open(DEV_JSONL, encoding="utf-8")}

    # ── base CoT with Plan: ───────────────────────────────────────────────
    base_blocks = []
    with open(BASE_RAW, encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            idx = rec["iIndex"]
            if idx not in train_ids or idx in dev_ids:
                continue
            if not (rec.get("cot") and "####" in rec["cot"]):
                continue
            prob = problems[idx]
            try:
                plan_line = make_plan_line(prob["lEquations"][0])
            except Exception:
                plan_line = None
            cot = rec["cot"].strip()
            if plan_line:
                cot = prepend_plan(cot, plan_line)
            base_blocks.append(format_sft(idx, prob["sQuestion"], cot))

    os.makedirs("data", exist_ok=True)
    with open(OUT_BASE_PLAN, "w", encoding="utf-8") as f:
        f.writelines(base_blocks)
    print(f"Base plan CoT  : {len(base_blocks)} examples  -> {OUT_BASE_PLAN}")

    # ── augmented CoT with Plan: ──────────────────────────────────────────
    if not os.path.exists(AUG_COT):
        print(f"WARNING: {AUG_COT} not found; skipping augmented plan file.")
        return

    aug_blocks = []
    err = 0
    with open(AUG_COT, encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            if rec["src_id"] in dev_ids:
                continue
            if not rec.get("cot") or not cot_is_valid(rec["cot"], rec["gold_answer"]):
                err += 1
                continue
            try:
                plan_line = make_plan_line(rec["equation"])
            except Exception:
                plan_line = None
            cot = rec["cot"].strip()
            if plan_line:
                cot = prepend_plan(cot, plan_line)
            ex_id = f"{rec['src_id']}_aug{rec['variant']}"
            aug_blocks.append(format_sft(ex_id, rec["question"], cot))

    with open(OUT_AUG_PLAN, "w", encoding="utf-8") as f:
        f.writelines(base_blocks)   # base first
        f.writelines(aug_blocks)
    print(f"Aug plan CoT   : {len(base_blocks)} base + {len(aug_blocks)} aug = "
          f"{len(base_blocks)+len(aug_blocks)} total  -> {OUT_AUG_PLAN}")
    if err:
        print(f"  (skipped {err} aug records with invalid CoT)")


if __name__ == "__main__":
    main()
