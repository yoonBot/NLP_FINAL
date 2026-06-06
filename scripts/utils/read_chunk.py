#!/usr/bin/env python3
import argparse
import json
import os

PROBLEMS_PATH = "data/multiarith_aug_problems.jsonl"

def get_done_set(part: int) -> set:
    done = set()
    part_path = f"data/_aug_cot_part{part:02d}.jsonl"
    if os.path.exists(part_path):
        with open(part_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                    done.add((rec["src_id"], rec["variant"]))
                except Exception:
                    pass
    return done

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--part", type=int, required=True, help="Partition index (0-9)")
    parser.add_argument("--limit", type=int, default=25, help="Number of items to fetch")
    args = parser.parse_args()

    part = args.part
    limit = args.limit

    if not os.path.exists(PROBLEMS_PATH):
        print(f"Error: {PROBLEMS_PATH} not found.")
        return

    # Load all problems
    problems = []
    with open(PROBLEMS_PATH, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                problems.append(json.loads(line))

    # Divide 2550 problems into 10 partitions of 255 items each
    total_items = len(problems)
    items_per_part = 255
    start_idx = part * items_per_part
    end_idx = min((part + 1) * items_per_part, total_items)

    part_problems = problems[start_idx:end_idx]
    done_set = get_done_set(part)

    # Filter out completed items
    todo_problems = []
    for prob in part_problems:
        key = (prob["src_id"], prob["variant"])
        if key not in done_set:
            todo_problems.append(prob)

    if not todo_problems:
        print("ALL_DONE")
        return

    # Output the next batch
    batch = todo_problems[:limit]
    print(json.dumps(batch, indent=2, ensure_ascii=False))

if __name__ == "__main__":
    main()
