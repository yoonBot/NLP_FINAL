#!/usr/bin/env python3
"""
GSM8K answer parser + evaluator.

Shared across base / CoT-SFT / DPO evaluation so every model is scored with the
*same* parser and the *same* metric definitions (a requirement in
RESEARCH_DESIGN.md section 10).

Primary metric:
  - exact_accuracy: extracted final answer == gold final answer (numeric match)

Secondary metrics (RESEARCH_DESIGN.md 10.2):
  - no_answer_rate:   fraction of generations with no extractable number
  - format_valid_rate: fraction that follow the trained answer format (#### N or Answer: N)
  - repetition_rate:  fraction flagged as containing a repeated reasoning line
                      (this is partly a *decoding* artifact, not model incapacity;
                       see reasoning_generation.generate() greedy decoding)

The parser keys on `#### N` first (GSM8K-native gold/target format) and falls
back to `Answer: N`, then to the last number in the text.
"""

import re

# A number such as: 1,234.56  $1200  -3  72%  (we normalize all of these)
_NUMBER = r"-?\$?\d[\d,]*(?:\.\d+)?%?"

_GOLD_RE = re.compile(r"####\s*(" + _NUMBER + r")")
_ANSWER_RE = re.compile(r"Answer:\s*(" + _NUMBER + r")", re.IGNORECASE)
_HASH_RE = re.compile(r"####\s*(" + _NUMBER + r")")
_ANY_NUMBER_RE = re.compile(_NUMBER)


def normalize_number(raw):
    """Normalize a raw numeric string to a comparable float, or None on failure.

    Strips $, commas, and a trailing %. Returns a float so that '72' and '72.0'
    compare equal. Returns None if nothing parseable.
    """
    if raw is None:
        return None
    s = raw.strip().replace("$", "").replace(",", "").rstrip("%").strip()
    if s in ("", "-", "."):
        return None
    try:
        return float(s)
    except ValueError:
        return None


def extract_gold_answer(reasoning_text):
    """Extract the gold final answer from a GSM8K answer field (the `#### N` line)."""
    m = _GOLD_RE.search(reasoning_text)
    if m:
        return normalize_number(m.group(1))
    return None


def extract_pred_answer(generation_text):
    """Extract the model's predicted final answer from a generation.

    Priority: `#### N`  ->  `Answer: N`  ->  last number in the text.
    Returns (value_or_None, source) where source is one of
    {"hash", "answer", "last_number", "none"}.
    """
    m = _HASH_RE.search(generation_text)
    if m:
        return normalize_number(m.group(1)), "hash"

    m = _ANSWER_RE.search(generation_text)
    if m:
        return normalize_number(m.group(1)), "answer"

    matches = _ANY_NUMBER_RE.findall(generation_text)
    if matches:
        return normalize_number(matches[-1]), "last_number"

    return None, "none"


def is_format_valid(generation_text):
    """True if the generation follows a trained answer format (#### N or Answer: N)."""
    return bool(_HASH_RE.search(generation_text) or _ANSWER_RE.search(generation_text))


def has_repetition(generation_text, min_line_len=8, threshold=2):
    """Flag exact-line repetition in the reasoning (a greedy-decoding artifact).

    Returns True if any non-trivial line (>= min_line_len chars after strip)
    appears at least `threshold` times.
    """
    counts = {}
    for line in generation_text.splitlines():
        key = line.strip()
        if len(key) < min_line_len:
            continue
        counts[key] = counts.get(key, 0) + 1
        if counts[key] >= threshold:
            return True
    return False


def is_correct(generation_text, gold_value, tol=1e-6):
    """True if the predicted answer numerically matches the gold answer."""
    pred, _ = extract_pred_answer(generation_text)
    if pred is None or gold_value is None:
        return False
    return abs(pred - gold_value) <= tol


def evaluate(records):
    """Aggregate metrics over a list of records.

    Each record is a dict with keys:
      - "generation": the model's full generated text
      - "gold": gold final answer value (float) OR "gold_reasoning" text to parse
    """
    n = len(records)
    if n == 0:
        return {"n": 0}

    correct = 0
    no_answer = 0
    format_valid = 0
    repetition = 0

    for r in records:
        gen = r["generation"]
        gold = r.get("gold")
        if gold is None and "gold_reasoning" in r:
            gold = extract_gold_answer(r["gold_reasoning"])

        pred, source = extract_pred_answer(gen)
        if pred is None:
            no_answer += 1
        if is_format_valid(gen):
            format_valid += 1
        if has_repetition(gen):
            repetition += 1
        if pred is not None and gold is not None and abs(pred - gold) <= 1e-6:
            correct += 1

    return {
        "n": n,
        "exact_accuracy": correct / n,
        "no_answer_rate": no_answer / n,
        "format_valid_rate": format_valid / n,
        "repetition_rate": repetition / n,
    }


def _selftest():
    # Gold extraction
    assert extract_gold_answer("blah\n#### 72") == 72.0
    assert extract_gold_answer("no marker here") is None
    # normalization variants
    assert normalize_number("$1,200") == 1200.0
    assert normalize_number("72%") == 72.0
    assert normalize_number("3.5") == 3.5
    assert normalize_number("abc") is None
    # pred extraction priority
    assert extract_pred_answer("steps\n#### 256")[0] == 256.0
    assert extract_pred_answer("steps\n#### 256")[1] == "hash"
    assert extract_pred_answer("Final.\nAnswer: 10")[0] == 10.0
    assert extract_pred_answer("Final.\nAnswer: 10")[1] == "answer"
    assert extract_pred_answer("he had 5 then 8 apples")[0] == 8.0
    assert extract_pred_answer("he had 5 then 8 apples")[1] == "last_number"
    assert extract_pred_answer("no numbers at all")[0] is None
    # format / repetition / correctness
    assert is_format_valid("#### 5")
    assert not is_format_valid("just words 5")
    assert has_repetition("She uses 20 students.\nShe uses 20 students.")
    assert not has_repetition("step one done\nstep two done")
    assert is_correct("#### 72", 72.0)
    assert not is_correct("#### 71", 72.0)
    # aggregate
    recs = [
        {"generation": "#### 72", "gold": 72.0},
        {"generation": "#### 71", "gold": 72.0},
        {"generation": "no answer here words", "gold_reasoning": "x\n#### 5"},
    ]
    m = evaluate(recs)
    assert m["n"] == 3
    assert abs(m["exact_accuracy"] - 1 / 3) < 1e-9
    assert abs(m["no_answer_rate"] - 1 / 3) < 1e-9
    print("gsm8k_eval self-test: all assertions passed.")


if __name__ == "__main__":
    _selftest()
