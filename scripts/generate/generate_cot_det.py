#!/usr/bin/env python3
"""
Deterministic CoT generator for augmented MultiArith problems.

Each problem in multiarith_aug_problems.jsonl has an `equation` field like
`((42 + 20) - 51)`.  All MultiArith problems are 2-step, so the tree is
always left-deep `((a op b) op c)` or right-deep `(a op (b op c))`.

Output format matches GSM8K SFT:
    <<expr=result>>result
    ...
    #### N

No LLM needed — every value is derived from the equation.

Output: data/multiarith_aug_cot.jsonl
"""

import ast
import json
import os

AUG_PROBLEMS = os.path.join("data", "multiarith_aug_problems.jsonl")
OUT_PATH = os.path.join("data", "multiarith_aug_cot.jsonl")

_OP_SYM = {ast.Add: "+", ast.Sub: "-", ast.Mult: "*", ast.Div: "/"}
_OP_WORD = {"+": "plus", "-": "minus", "*": "times", "/": "divided by"}


def _eval(node):
    if isinstance(node, ast.Constant):
        return node.value
    if isinstance(node, ast.BinOp):
        a, b = _eval(node.left), _eval(node.right)
        op = _OP_SYM[type(node.op)]
        if op == "+":
            return a + b
        if op == "-":
            return a - b
        if op == "*":
            return a * b
        return a // b  # guaranteed exact by augment_multiarith
    raise ValueError(f"Unexpected node: {ast.dump(node)}")


def _fmt_num(v):
    return str(int(v)) if isinstance(v, float) and v == int(v) else str(v)


def _step_str(a, op, b, r):
    """One CoT line: `A op B = <<A op B=R>>R`"""
    a_s, b_s, r_s = _fmt_num(a), _fmt_num(b), _fmt_num(r)
    expr = f"{a_s}{op}{b_s}"
    return f"<<{expr}={r_s}>>{r_s}"


def make_cot(equation: str, gold: int) -> str:
    """Return the 2-step GSM8K-format CoT string for this equation."""
    tree = ast.parse(equation, mode="eval").body  # BinOp at top level

    top_op = _OP_SYM[type(tree.op)]

    if isinstance(tree.left, ast.BinOp):
        # Left-deep: ((a op1 b) op2 c)
        inner = tree.left
        inner_op = _OP_SYM[type(inner.op)]
        a = _eval(inner.left)
        b = _eval(inner.right)
        inner_r = _eval(inner)
        c = _eval(tree.right)
        final_r = _eval(tree)

        step1 = _step_str(a, inner_op, b, inner_r)
        step2 = _step_str(inner_r, top_op, c, final_r)
        lines = [
            f"{_fmt_num(a)} {_OP_WORD[inner_op]} {_fmt_num(b)} = {step1}",
            f"{_fmt_num(inner_r)} {_OP_WORD[top_op]} {_fmt_num(c)} = {step2}",
        ]
    elif isinstance(tree.right, ast.BinOp):
        # Right-deep: (a op1 (b op2 c))
        inner = tree.right
        inner_op = _OP_SYM[type(inner.op)]
        b = _eval(inner.left)
        c = _eval(inner.right)
        inner_r = _eval(inner)
        a = _eval(tree.left)
        final_r = _eval(tree)

        step1 = _step_str(b, inner_op, c, inner_r)
        step2 = _step_str(a, top_op, inner_r, final_r)
        lines = [
            f"{_fmt_num(b)} {_OP_WORD[inner_op]} {_fmt_num(c)} = {step1}",
            f"{_fmt_num(a)} {_OP_WORD[top_op]} {_fmt_num(inner_r)} = {step2}",
        ]
    else:
        # Flat (should not occur for 2-step problems): a op b
        a = _eval(tree.left)
        b = _eval(tree.right)
        r = _eval(tree)
        step1 = _step_str(a, top_op, b, r)
        lines = [f"{_fmt_num(a)} {_OP_WORD[top_op]} {_fmt_num(b)} = {step1}"]

    cot = "\n".join(lines) + f"\n#### {gold}"
    return cot


def main():
    problems = []
    with open(AUG_PROBLEMS, encoding="utf-8") as f:
        for line in f:
            problems.append(json.loads(line))

    ok = err = 0
    with open(OUT_PATH, "w", encoding="utf-8") as out:
        for rec in problems:
            try:
                cot = make_cot(rec["equation"], rec["gold_answer"])
                row = dict(rec)
                row["cot"] = cot
                out.write(json.dumps(row, ensure_ascii=False) + "\n")
                ok += 1
            except Exception as e:
                err += 1
                print(f"  ERROR src={rec['src_id']} v={rec['variant']}: {e}")

    print(f"Generated : {ok}")
    print(f"Errors    : {err}")
    print(f"Output    : {OUT_PATH}")


if __name__ == "__main__":
    main()
