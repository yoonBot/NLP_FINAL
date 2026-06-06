#!/usr/bin/env python3
"""
Entity-substitution augmentation for MultiArith.

For each training-source problem, generates up to K entity-swapped variants
by replacing person names and common item nouns with alternatives drawn from
a curated bank.  Numbers and equation structure are kept fixed (unlike the
number-substitution augmentation in augment_multiarith.py).

Goals
-----
  - Increase surface diversity so the model must attend to relational language,
    not just memorise name-number associations.
  - All substitutions are deterministic given a seed.
  - Output is the same {question, gold_answer, equation} format as
    augment_multiarith.py so generate_cot_det.py can consume it directly.

Inputs
------
  data/MultiArith.json
  data/multiarith_train_ids.json

Output
------
  data/multiarith_entity_aug_problems.jsonl
    {src_id, variant, question, gold_answer, equation}

Usage
-----
    python augment_entities.py --k 3
"""

import argparse
import json
import os
import random
import re

MULTIARITH_PATH = os.path.join("data", "MultiArith.json")
TRAIN_IDS_PATH  = os.path.join("data", "multiarith_train_ids.json")
OUT_PATH        = os.path.join("data", "multiarith_entity_aug_problems.jsonl")

# ── Entity bank ───────────────────────────────────────────────────────────────
# Person names (gender-neutral or mixed; keep realistic)
_PERSON_BANK = [
    "Alex", "Jordan", "Sam", "Casey", "Riley", "Morgan", "Taylor", "Avery",
    "Quinn", "Blake", "Drew", "Reese", "Parker", "Logan", "Skyler",
    "Brooke", "Cameron", "Dana", "Emery", "Finley", "Hayden", "Jamie",
    "Kendall", "Lane", "Macy", "Nolan", "Oakley", "Paige", "Remy", "Sage",
    "Shawn", "Tatum", "Uma", "Val", "Wren", "Xan", "Yara", "Zion",
    "Aiden", "Bella", "Charlie", "Diana", "Eli", "Fiona", "Gabe", "Hana",
    "Ivan", "Jess", "Kim", "Liam", "Mia", "Nina", "Owen", "Pia",
]
# Common item nouns (countable, easy to swap; chosen to fit arithmetic word problems)
_ITEM_BANK = [
    "stickers", "stamps", "marbles", "coins", "cards", "books", "toys",
    "crayons", "pencils", "erasers", "buttons", "beads", "shells",
    "ribbons", "clips", "pins", "jars", "boxes", "bags", "cups",
    "apples", "oranges", "bananas", "grapes", "berries", "pears",
    "cookies", "cupcakes", "muffins", "donuts", "candies", "chocolates",
    "balls", "blocks", "puzzles", "magnets", "patches", "tickets",
]

# Regex to find capitalised word sequences (likely proper nouns / names)
_PROPER_RE = re.compile(r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\b")

# Common nouns in the item bank (lowercase) for matching
_ITEM_SET   = set(w.lower() for w in _ITEM_BANK)
# Also match plurals by stripping trailing 's' for lookup
_ITEM_RE    = re.compile(
    r"\b(" + "|".join(re.escape(w) for w in sorted(_ITEM_BANK, key=len, reverse=True)) + r")\b",
    re.IGNORECASE,
)


def _swap_persons(text: str, rng: random.Random, k: int) -> list[str]:
    """Generate up to k variants by swapping each distinct person name."""
    names_found = []
    seen = set()
    for m in _PROPER_RE.finditer(text):
        w = m.group(1)
        # Skip if it looks like a unit (e.g. "Monday") or is just one common word
        if w.lower() in {"halloween", "monday", "tuesday", "wednesday",
                         "thursday", "friday", "saturday", "sunday",
                         "january", "february", "march", "april", "may",
                         "june", "july", "august", "september", "october",
                         "november", "december", "mr", "mrs", "ms", "dr"}:
            continue
        if w not in seen:
            seen.add(w)
            names_found.append(w)

    if not names_found:
        return []

    available = [p for p in _PERSON_BANK if p not in seen]
    rng.shuffle(available)

    variants = []
    for _ in range(k):
        if not available:
            break
        mapping = {}
        for name in names_found:
            if available:
                mapping[name] = available.pop(0)
        if not mapping:
            break
        new_text = text
        for old, new in mapping.items():
            new_text = re.sub(r"\b" + re.escape(old) + r"\b", new, new_text)
        if new_text != text:
            variants.append(new_text)
    return variants


def _swap_items(text: str, rng: random.Random, k: int) -> list[str]:
    """Generate up to k variants by swapping item nouns."""
    items_found = list({m.group(1).lower() for m in _ITEM_RE.finditer(text)})
    if not items_found:
        return []

    available = [it for it in _ITEM_BANK if it.lower() not in set(items_found)]
    rng.shuffle(available)

    variants = []
    for _ in range(k):
        if not available:
            break
        mapping = {}
        for item in items_found:
            if available:
                mapping[item] = available.pop(0)
        if not mapping:
            break

        def replace_item(m):
            key = m.group(1).lower()
            if key in mapping:
                repl = mapping[key]
                # Preserve capitalisation of the first char
                if m.group(1)[0].isupper():
                    return repl[0].upper() + repl[1:]
                return repl
            return m.group(1)

        new_text = _ITEM_RE.sub(replace_item, text)
        if new_text != text:
            variants.append(new_text)
    return variants


def sample_entity_variants(problem: dict, k: int, rng: random.Random) -> list[tuple]:
    """Return up to k (new_question, gold_answer, equation) triples."""
    q        = problem["sQuestion"].strip()
    gold     = problem["lSolutions"][0]
    equation = problem["lEquations"][0]

    candidates = []
    # First try swapping names (usually highest signal)
    candidates.extend(_swap_persons(q, rng, k))
    # Then items, deduplicated
    seen_qs = set(candidates) | {q}
    for v in _swap_items(q, rng, k):
        if v not in seen_qs:
            candidates.append(v)
            seen_qs.add(v)

    results = []
    for i, new_q in enumerate(candidates[:k]):
        results.append((new_q, gold, equation))
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--k",    type=int, default=3, help="Max variants per problem.")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    rng = random.Random(args.seed)

    problems  = {p["iIndex"]: p for p in json.load(open(MULTIARITH_PATH, encoding="utf-8"))}
    train_ids = json.load(open(TRAIN_IDS_PATH, encoding="utf-8"))

    os.makedirs("data", exist_ok=True)
    total = 0
    with open(OUT_PATH, "w", encoding="utf-8") as out:
        for src_id in train_ids:
            prob = problems[src_id]
            variants = sample_entity_variants(prob, args.k, rng)
            for vi, (q, gold, eq) in enumerate(variants):
                out.write(json.dumps({
                    "src_id":      src_id,
                    "variant":     vi,
                    "question":    q,
                    "gold_answer": int(gold) if float(gold) == int(gold) else gold,
                    "equation":    eq,
                }, ensure_ascii=False) + "\n")
                total += 1

    print(f"train-source problems  : {len(train_ids)}")
    print(f"entity-aug problems    : {total}  -> {OUT_PATH}")


if __name__ == "__main__":
    main()
