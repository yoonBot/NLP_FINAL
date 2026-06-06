#!/usr/bin/env python3
"""
MultiArith number-substitution augmentation (train-source only).

For each training-source problem, generates up to K number-variants by resampling
the operands in its equation under constraints that preserve MultiArith's invariants
(all intermediate values and the final answer are positive integers; division is
exact; subtraction is non-negative). The question text is edited *by character
offset* using lAlignments, so substitution is exact even when numbers repeat.

CoT is NOT produced here — it is generated externally (Antigravity) from the emitted
{question, gold_answer}. See assemble_multiarith.py for the validation + assembly step.

Inputs:
    data/MultiArith.json
    data/multiarith_train_ids.json   (from prepare_multiarith.py)

Output:
    data/multiarith_aug_problems.jsonl
        {src_id, variant, question, gold_answer, equation}

Usage:
    python augment_multiarith.py --k 5
    python augment_multiarith.py --k 8 --seed 7
"""

import argparse
import ast
import json
import os
import random
import re

MULTIARITH_PATH = os.path.join("data", "MultiArith.json")
TRAIN_IDS_PATH = os.path.join("data", "multiarith_train_ids.json")
OUT_PATH = os.path.join("data", "multiarith_aug_problems.jsonl")

_NUM_AT_OFFSET = re.compile(r"\d+(?:\.\d+)?")
_OP_SYMBOL = {ast.Add: "+", ast.Sub: "-", ast.Mult: "*", ast.Div: "/"}


class NotExact(Exception):
    """Raised when a division would not yield an integer."""


def leaf_values_in_order(node, out):
    """Collect numeric leaf values left-to-right (matches lAlignments order)."""
    if isinstance(node, ast.Constant):
        out.append(node.value)
    elif isinstance(node, ast.BinOp):
        leaf_values_in_order(node.left, out)
        leaf_values_in_order(node.right, out)
    elif isinstance(node, ast.UnaryOp):
        leaf_values_in_order(node.operand, out)


def evaluate(node, vals_iter, results):
    """Evaluate the tree using substituted leaf values; collect every BinOp result.

    Raises NotExact on non-integer division. Returns the node's value.
    """
    if isinstance(node, ast.Constant):
        return next(vals_iter)
    if isinstance(node, ast.UnaryOp):  # unary minus
        return -evaluate(node.operand, vals_iter, results)
    if isinstance(node, ast.BinOp):
        a = evaluate(node.left, vals_iter, results)
        b = evaluate(node.right, vals_iter, results)
        op = _OP_SYMBOL[type(node.op)]
        if op == "+":
            r = a + b
        elif op == "-":
            r = a - b
        elif op == "*":
            r = a * b
        else:  # "/"
            if b == 0 or a % b != 0:
                raise NotExact
            r = a // b
        results.append(r)
        return r
    raise ValueError(f"Unsupported node: {ast.dump(node)}")


def unparse(node, new_leaves):
    """Render the equation RHS with substituted integer leaves (fully parenthesized)."""
    it = iter(new_leaves)

    def render(n):
        if isinstance(n, ast.Constant):
            return str(next(it))
        if isinstance(n, ast.UnaryOp):
            return f"-{render(n.operand)}"
        if isinstance(n, ast.BinOp):
            return f"({render(n.left)} {_OP_SYMBOL[type(n.op)]} {render(n.right)})"
        raise ValueError
    return render(node)


def sample_variants(problem, k, rng, max_attempts):
    """Yield up to k valid (new_question, gold, equation_str) variants."""
    # lAlignments offsets index into the ORIGINAL (unstripped) sQuestion; splice on
    # that string and strip only at the end.
    question = problem["sQuestion"]
    alignments = problem["lAlignments"]
    rhs = problem["lEquations"][0].split("=", 1)[-1]
    tree = ast.parse(rhs, mode="eval").body

    orig_leaves = []
    leaf_values_in_order(tree, orig_leaves)

    # Safety: alignment count must match leaf count, and each offset must point at
    # the matching number — otherwise we cannot substitute the question reliably.
    if len(orig_leaves) != len(alignments):
        return
    spans = []  # (start, end, original_value)
    for off, leaf_val in zip(alignments, orig_leaves):
        m = _NUM_AT_OFFSET.match(question, off)
        if not m or float(m.group()) != float(leaf_val):
            return  # misaligned; skip this problem entirely
        spans.append((off, m.end(), leaf_val))

    seen = {tuple(int(v) for v in orig_leaves)}  # exclude the original combo
    produced = 0
    for _ in range(max_attempts):
        if produced >= k:
            break
        new_vals = [rng.randint(1, max(2 * int(v), 12)) for v in orig_leaves]
        key = tuple(new_vals)
        if key in seen:
            continue

        results = []
        try:
            final = evaluate(tree, iter(new_vals), results)
        except NotExact:
            continue
        # All intermediates, the final answer, and leaves must be positive integers.
        if any((not isinstance(r, int)) or r < 1 for r in results):
            continue
        if final < 1:
            continue

        seen.add(key)

        # Build the new question by splicing from right to left (offsets stay valid).
        new_q = question
        for (start, end, _), nv in sorted(zip(spans, new_vals), key=lambda t: t[0][0],
                                          reverse=True):
            new_q = new_q[:start] + str(nv) + new_q[end:]

        produced += 1
        yield new_q.strip(), int(final), unparse(tree, new_vals)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--k", type=int, default=5, help="Max variants per source problem.")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--max_attempts", type=int, default=400,
                        help="Sampling attempts per problem before giving up.")
    args = parser.parse_args()

    rng = random.Random(args.seed)

    problems = {p["iIndex"]: p for p in json.load(open(MULTIARITH_PATH, encoding="utf-8"))}
    train_ids = json.load(open(TRAIN_IDS_PATH, encoding="utf-8"))

    os.makedirs("data", exist_ok=True)

    total = 0
    skipped_align = 0
    under_k = 0
    with open(OUT_PATH, "w", encoding="utf-8") as out:
        for src_id in train_ids:
            prob = problems[src_id]
            variants = list(sample_variants(prob, args.k, rng, args.max_attempts))
            if not variants:
                # Either misaligned or no valid sample found.
                orig_leaves = []
                leaf_values_in_order(
                    ast.parse(prob["lEquations"][0].split("=", 1)[-1], mode="eval").body,
                    orig_leaves)
                if len(orig_leaves) != len(prob["lAlignments"]):
                    skipped_align += 1
                continue
            if len(variants) < args.k:
                under_k += 1
            for vi, (q, gold, eq) in enumerate(variants):
                out.write(json.dumps({
                    "src_id": src_id,
                    "variant": vi,
                    "question": q,
                    "gold_answer": gold,
                    "equation": eq,
                }, ensure_ascii=False) + "\n")
                total += 1

    print(f"train-source problems : {len(train_ids)}")
    print(f"skipped (misaligned)  : {skipped_align}")
    print(f"produced < k variants : {under_k}")
    print(f"augmented problems     : {total}  -> {OUT_PATH}")


if __name__ == "__main__":
    main()
