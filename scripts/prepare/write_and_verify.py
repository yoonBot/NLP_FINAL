#!/usr/bin/env python3
import argparse
import json
import os
import re

PROBLEMS_PATH = "data/multiarith_aug_problems.jsonl"

def parse_cot_expressions(cot_text):
    # Find all <<expr=result>> blocks
    pattern = r"<<([^>=]+)=([^>]+)>>"
    matches = re.findall(pattern, cot_text)
    
    results = []
    for expr, res_str in matches:
        expr = expr.strip()
        res_str = res_str.strip()
        results.append((expr, res_str))
    return results

def verify_cot(cot_text, gold_answer):
    errors = []
    
    # 1. Verify all inline expressions
    expr_blocks = parse_cot_expressions(cot_text)
    for expr, res_str in expr_blocks:
        # Safety check for characters allowed in eval
        if not re.match(r"^[0-9\+\-\*\/\(\)\s\.]+$", expr):
            errors.append(f"Invalid characters in expression: <<{expr}={res_str}>>")
            continue
        
        try:
            # Safe eval with restricted globals/locals
            calculated = eval(expr, {"__builtins__": None}, {})
            target = float(res_str)
            if abs(calculated - target) > 1e-5:
                errors.append(f"Arithmetic mismatch: <<{expr}={res_str}>> calculated to {calculated}, not {target}")
        except Exception as e:
            errors.append(f"Failed to evaluate expression <<{expr}={res_str}>>: {e}")
            
    # 2. Verify the final line format and match with gold answer
    lines = [line.strip() for line in cot_text.strip().split("\n") if line.strip()]
    if not lines:
        errors.append("CoT is empty")
        return errors

    last_line = lines[-1]
    match = re.match(r"^####\s*(-?\d+(\.\d+)?)$", last_line)
    if not match:
        errors.append(f"Last line must be exactly '#### <answer>', found: '{last_line}'")
    else:
        ans_val = float(match.group(1))
        gold_val = float(gold_answer)
        if abs(ans_val - gold_val) > 1e-5:
            errors.append(f"Final answer mismatch: got #### {match.group(1)}, expected gold_answer {gold_answer}")
            
    return errors

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--part", type=int, required=True, help="Partition index (0-9)")
    parser.add_argument("--force", action="store_true", help="Force write even if validation fails (best-effort)")
    args = parser.parse_args()

    part = args.part
    force = args.force

    temp_path = f"data/temp_cot_part{part:02d}.json"
    part_path = f"data/_aug_cot_part{part:02d}.jsonl"

    if not os.path.exists(temp_path):
        print(f"Error: Temporary file {temp_path} not found.")
        exit(1)

    # Load temp cot data
    with open(temp_path, "r", encoding="utf-8") as f:
        cot_data = json.load(f)

    # Load all original problems for lookup
    problems = {}
    with open(PROBLEMS_PATH, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rec = json.loads(line)
                problems[(rec["src_id"], rec["variant"])] = rec

    validation_failed = False
    failed_reports = []
    records_to_write = []

    for item in cot_data:
        src_id = item["src_id"]
        variant = item["variant"]
        cot = item["cot"]

        key = (src_id, variant)
        if key not in problems:
            print(f"Error: Problem with src_id {src_id}, variant {variant} not found in source dataset.")
            exit(1)

        original_prob = problems[key]
        gold_answer = original_prob["gold_answer"]

        # Run verification
        errors = verify_cot(cot, gold_answer)
        
        # Build output record (keep original fields and append/overwrite cot)
        rec = original_prob.copy()
        rec["cot"] = cot

        if errors:
            validation_failed = True
            failed_reports.append({
                "src_id": src_id,
                "variant": variant,
                "question": original_prob["question"],
                "gold_answer": gold_answer,
                "cot": cot,
                "errors": errors
            })
        
        records_to_write.append(rec)

    if validation_failed and not force:
        print("VALIDATION_FAILED")
        print(json.dumps(failed_reports, indent=2, ensure_ascii=False))
        exit(1)

    # Write records to part file
    os.makedirs("data", exist_ok=True)
    with open(part_path, "a", encoding="utf-8") as f:
        for rec in records_to_write:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    # Clean up temp file
    try:
        os.remove(temp_path)
    except Exception:
        pass

    print("SUCCESS")
    if validation_failed and force:
        print(f"Warning: Wrote {len(records_to_write)} records with validation failures under --force.")
    else:
        print(f"Successfully validated and wrote {len(records_to_write)} records.")

if __name__ == "__main__":
    main()
