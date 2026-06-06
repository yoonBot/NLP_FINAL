#!/usr/bin/env python3
"""
MultiArith CoT distillation via Gemini.

Usage:
    python generate_cot_multiarith.py
    python generate_cot_multiarith.py --limit 10  # smoke test

Output:
    data/multiarith_cot_train.txt   SFT-ready (GSM8K format, <|endoftext|> separated)
    data/multiarith_cot_raw.jsonl   raw per-example records (resumable)
"""

import argparse
import json
import os
import time

from dotenv import load_dotenv
from google import genai
from google.genai import types

MULTIARITH_PATH = "data/MultiArith.json"
RAW_OUT = "data/multiarith_cot_raw.jsonl"
TXT_OUT = "data/multiarith_cot_train.txt"
MODEL = "gemini-3.1-flash-lite"

SYSTEM_PROMPT = """\
You are a math tutoring assistant. Given a word problem, write a step-by-step \
reasoning chain a student would use to solve it.

Rules:
1. Each step on its own line.
2. Show every arithmetic calculation using: <<expression=result>>result
   Example: 32 + 42 = <<32+42=74>>74
3. The final answer must be on the last line in the format: #### {number}
4. Keep each step one sentence. Maximum 4 steps.
5. Use only the numbers given in the problem. Do not invent numbers.

Output ONLY the reasoning steps and the final #### line. No preamble."""


def build_prompt(body: str) -> str:
    return f"Problem: {body.strip()}"


def already_done(raw_path: str) -> set:
    """Only count examples that actually have a valid CoT response."""
    done = set()
    if not os.path.exists(raw_path):
        return done
    with open(raw_path, "r", encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            if rec.get("cot") and "####" in rec["cot"]:
                done.add(rec["iIndex"])
    return done


def format_sft(index: int, body: str, cot: str) -> str:
    return (
        f"{index}\n\n"
        f"Question: {body.strip()}\n\n"
        f"Reasoning:\n{cot.strip()}\n\n"
        f"<|endoftext|>\n\n"
    )


def main():
    load_dotenv()

    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=0,
                        help="If >0, only process first N examples (smoke test).")
    parser.add_argument("--delay", type=float, default=4.5,
                        help="Seconds between API calls. Free tier limit: 15 req/min → use 4.5s.")
    args = parser.parse_args()

    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY not found in .env")

    client = genai.Client(api_key=api_key)

    with open(MULTIARITH_PATH, "r", encoding="utf-8") as f:
        problems = json.load(f)

    if args.limit:
        problems = problems[: args.limit]

    done = already_done(RAW_OUT)
    print(f"Total problems: {len(problems)}, already done: {len(done)}")

    os.makedirs("data", exist_ok=True)

    with open(RAW_OUT, "a", encoding="utf-8") as raw_f:
        for prob in problems:
            idx = prob["iIndex"]
            if idx in done:
                continue

            body = prob["sQuestion"]
            answer = prob["lSolutions"][0]

            try:
                response = client.models.generate_content(
                    model=MODEL,
                    contents=build_prompt(body),
                    config=types.GenerateContentConfig(
                        system_instruction=SYSTEM_PROMPT,
                        temperature=0.0,
                    ),
                )
                cot = (response.text or "").strip()
            except Exception as e:
                print(f"[{idx}] ERROR: {e}")
                cot = ""

            rec = {
                "iIndex": idx,
                "sQuestion": body,
                "answer": answer,
                "cot": cot,
            }
            raw_f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            raw_f.flush()

            ok = "####" in cot
            print(f"[{idx}] {'OK' if ok else 'MISSING####'} | {cot[:80].replace(chr(10), ' ')}")

            time.sleep(args.delay)

    # Assemble SFT txt from raw jsonl
    print(f"\nAssembling {TXT_OUT} ...")
    records = {}
    with open(RAW_OUT, "r", encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            records[rec["iIndex"]] = rec

    skipped = 0
    with open(TXT_OUT, "w", encoding="utf-8") as f:
        for idx in sorted(records):
            rec = records[idx]
            if not rec["cot"] or "####" not in rec["cot"]:
                print(f"  SKIP [{idx}] — missing #### or empty")
                skipped += 1
                continue
            f.write(format_sft(idx, rec["sQuestion"], rec["cot"]))

    total = len(records) - skipped
    print(f"Done. {total}/{len(records)} examples written to {TXT_OUT}")


if __name__ == "__main__":
    main()
