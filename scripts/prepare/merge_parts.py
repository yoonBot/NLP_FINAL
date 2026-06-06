#!/usr/bin/env python3
import json
import os
import re

PROBLEMS_PATH = "data/multiarith_aug_problems.jsonl"
FINAL_OUT_PATH = "data/multiarith_aug_cot.jsonl"

def parse_cot_expressions(cot_text):
    pattern = r"<<([^>=]+)=([^>]+)>>"
    matches = re.findall(pattern, cot_text)
    return [(expr.strip(), res_str.strip()) for expr, res_str in matches]

def verify_cot_ok(cot_text, gold_answer):
    # Verify inline equations
    expr_blocks = parse_cot_expressions(cot_text)
    if not expr_blocks:
        # If no expressions are found, let's treat it as failure if the problem had arithmetic
        pass
    
    for expr, res_str in expr_blocks:
        if not re.match(r"^[0-9\+\-\*\/\(\)\s\.]+$", expr):
            return False
        try:
            calculated = eval(expr, {"__builtins__": None}, {})
            target = float(res_str)
            if abs(calculated - target) > 1e-5:
                return False
        except Exception:
            return False
            
    # Verify final line
    lines = [line.strip() for line in cot_text.strip().split("\n") if line.strip()]
    if not lines:
        return False

    last_line = lines[-1]
    match = re.match(r"^####\s*(-?\d+(\.\d+)?)$", last_line)
    if not match:
        return False
    
    ans_val = float(match.group(1))
    gold_val = float(gold_answer)
    if abs(ans_val - gold_val) > 1e-5:
        return False
        
    return True

def main():
    records = []
    part_files = []
    
    # Identify all part files
    for i in range(10):
        part_file = f"data/_aug_cot_part{i:02d}.jsonl"
        if os.path.exists(part_file):
            part_files.append(part_file)
            with open(part_file, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        records.append(json.loads(line))

    if not records:
        print("Error: No part files or records found to merge.")
        return

    print(f"Read {len(records)} records from {len(part_files)} part files.")

    # Sort records by src_id and variant
    records.sort(key=lambda r: (r["src_id"], r["variant"]))

    # Perform validation checks on merged records
    pass_count = 0
    fail_count = 0
    
    for rec in records:
        cot = rec.get("cot", "")
        gold_answer = rec["gold_answer"]
        if verify_cot_ok(cot, gold_answer):
            pass_count += 1
        else:
            fail_count += 1

    # Write merged records
    with open(FINAL_OUT_PATH, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(f"\nMerged file written to: {FINAL_OUT_PATH}")
    print(f"Total lines: {len(records)}")
    print(f"Self-validation PASS: {pass_count}")
    print(f"Self-validation FAIL: {fail_count}")

    # Remove partition files upon successful merge
    removed_count = 0
    for part_file in part_files:
        try:
            os.remove(part_file)
            removed_count += 1
        except Exception as e:
            print(f"Warning: Failed to delete {part_file}: {e}")
            
    print(f"Cleaned up {removed_count} partition files.")

if __name__ == "__main__":
    main()
