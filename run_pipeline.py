#!/usr/bin/env python3
"""
Server pipeline — equivalent to colab_pipeline_math.ipynb.

Run from NLP_FINAL root:
    python run_pipeline.py --use_gpu --arith_init_path /mnt/cot_large_integer_arithmetic_pretrain.pt

Checkpoints and metrics go to --out_dir (default: /mnt/outputs).
"""

import argparse
import gc
import json
import os
import sys
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts" / "eval"))
sys.path.insert(0, str(ROOT / "scripts" / "generate"))

import torch

# ── CLI args ────────────────────────────────────────────────────────────────

def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--arith_init_path", default="cot_large_integer_arithmetic_pretrain.pt",
                   help="Path to arithmetic pre-trained checkpoint (.pt)")
    p.add_argument("--out_dir", default="/mnt/outputs",
                   help="Output dir for checkpoints and metrics (use /mnt/ for persistence)")
    p.add_argument("--use_gpu", action="store_true")
    p.add_argument("--seed", type=int, default=11711)
    p.add_argument("--batch_size", type=int, default=32,
                   help="Training batch size (32 is safe for A100 11GB with gpt2-small)")
    p.add_argument("--skip_data_prep", action="store_true",
                   help="Skip data generation step (if already done)")
    return p.parse_args()


# ── helpers ─────────────────────────────────────────────────────────────────

def _exists(p):
    return os.path.exists(p) and os.path.getsize(p) > 0

def _log(msg, log_path):
    ts = time.strftime("%H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(line + "\n")

def _run(cmd):
    import subprocess
    result = subprocess.run([sys.executable] + cmd, check=True)
    return result


# ── STEP 1: data generation (idempotent) ────────────────────────────────────

def run_data_prep():
    print("\n=== STEP 1: DATA PREPARATION ===\n")

    # 1-a: base split
    if not _exists("data/multiarith_sft_train_base.txt"):
        print("Running prepare_multiarith.py ...")
        _run(["scripts/prepare/prepare_multiarith.py"])
    else:
        print("[skip] multiarith base split already exists")

    # 1-b: number-substitution augmentation
    if not _exists("data/multiarith_aug_problems.jsonl"):
        print("Running augment_multiarith.py --k 10 ...")
        _run(["scripts/augment/augment_multiarith.py", "--k", "10"])
    else:
        print("[skip] number-aug problems already exist")

    # 1-c: deterministic CoT for augmented problems
    if not _exists("data/multiarith_aug_cot.jsonl"):
        print("Running generate_cot_det.py ...")
        _run(["scripts/generate/generate_cot_det.py"])
    else:
        print("[skip] aug CoT already exists")

    # 1-d: assemble number-aug SFT file
    if not _exists("data/multiarith_sft_train_aug.txt"):
        print("Running assemble_multiarith.py ...")
        _run(["scripts/prepare/assemble_multiarith.py"])
    else:
        print("[skip] numaug SFT already assembled")

    # 1-e: plan-then-solve CoT
    if not _exists("data/multiarith_sft_train_plan.txt"):
        print("Running generate_cot_plan.py ...")
        _run(["scripts/generate/generate_cot_plan.py"])
    else:
        print("[skip] plan CoT already exists")

    # 1-f: entity augmentation
    if not _exists("data/multiarith_entity_aug_problems.jsonl"):
        print("Running augment_entities.py --k 3 ...")
        _run(["scripts/augment/augment_entities.py", "--k", "3"])
    else:
        print("[skip] entity-aug problems already exist")

    # 1-g: CoT for entity-aug problems
    ENTITY_COT = "data/multiarith_entity_aug_cot.jsonl"
    if not _exists(ENTITY_COT):
        print("Generating CoT for entity-aug problems ...")
        import ast, json as _json, re
        from generate_cot_det import make_cot
        from generate_cot_plan import make_plan_line, prepend_plan
        ok = err = 0
        with open("data/multiarith_entity_aug_problems.jsonl") as fin, \
             open(ENTITY_COT, "w") as fout:
            for line in fin:
                rec = _json.loads(line)
                try:
                    cot = make_cot(rec["equation"], rec["gold_answer"])
                    plan = make_plan_line(rec["equation"])
                    cot_with_plan = prepend_plan(cot, plan)
                    row = dict(rec); row["cot"] = cot_with_plan
                    fout.write(_json.dumps(row, ensure_ascii=False) + "\n")
                    ok += 1
                except Exception:
                    err += 1
        print(f"  entity CoT: {ok} ok, {err} err")
    else:
        print("[skip] entity CoT already exists")

    # 1-h: assemble entity-aug SFT file
    ENTITY_SFT = "data/multiarith_sft_train_entity_aug.txt"
    if not _exists(ENTITY_SFT):
        import json as _json, re
        _BLOCK = re.compile(r"<<\s*([0-9+\-*/().\s]+?)\s*=\s*(-?\d+(?:\.\d+)?)\s*>>")
        _FINAL = re.compile(r"####\s*(-?\d+(?:\.\d+)?)")
        def _valid(cot, gold):
            for expr, claimed in _BLOCK.findall(cot):
                try:
                    if abs(eval(expr, {"__builtins__": {}}, {}) - float(claimed)) > 1e-6:
                        return False
                except:
                    return False
            m = _FINAL.search(cot)
            return m and abs(float(m.group(1)) - float(gold)) <= 1e-6
        dev_ids = {_json.loads(l)["id"] for l in open("data/multiarith_dev.jsonl")}
        blocks = []
        plan_aug = "data/multiarith_sft_train_plan_aug.txt"
        if _exists(plan_aug):
            blocks.append(open(plan_aug).read())
        with open(ENTITY_COT) as f:
            for line in f:
                rec = _json.loads(line)
                if rec["src_id"] in dev_ids: continue
                if not _valid(rec["cot"], rec["gold_answer"]): continue
                ex_id = f"{rec['src_id']}_ent{rec['variant']}"
                blocks.append(f"{ex_id}\n\nQuestion: {rec['question'].strip()}\n\nReasoning:\n{rec['cot'].strip()}\n\n<|endoftext|>\n\n")
        with open(ENTITY_SFT, "w") as f:
            f.writelines(blocks)
        print(f"  entity_aug SFT: {len(blocks)} blocks -> {ENTITY_SFT}")
    else:
        print("[skip] entity-aug SFT already assembled")

    # 1-i: GSM8K + MultiArith combos
    GSM8K_PLUS_MA     = "data/gsm8k_plus_ma_sft_train.txt"
    GSM8K_PLUS_ENTITY = "data/gsm8k_plus_entity_sft_train.txt"
    ENTITY_SFT        = "data/multiarith_sft_train_entity_aug.txt"
    for combo_path, ma_path in [
        (GSM8K_PLUS_MA,     "data/multiarith_sft_train_plan_aug.txt"),
        (GSM8K_PLUS_ENTITY, ENTITY_SFT),
    ]:
        if not _exists(combo_path):
            print(f"Building {combo_path} ...")
            gsm_text = open("data/gsm8k_sft_train.txt").read()
            ma_text  = open(ma_path).read() if _exists(ma_path) else ""
            with open(combo_path, "w") as f:
                f.write(gsm_text)
                if ma_text:
                    f.write("\n" + ma_text)
        else:
            print(f"[skip] {combo_path} already exists")

    # 1-j: template leakage audit
    if not _exists("data/multiarith_dev_template_labels.jsonl"):
        print("Running audit_template_leakage.py ...")
        _run(["scripts/augment/audit_template_leakage.py",
              "--train_sft", "data/multiarith_sft_train_plan_aug.txt"])
    else:
        print("[skip] template labels already computed")

    ENTITY_SFT = "data/multiarith_sft_train_entity_aug.txt"
    recipe_path = {
        "base":              "data/multiarith_sft_train_base.txt",
        "numaug":            "data/multiarith_sft_train_aug.txt",
        "plan_base":         "data/multiarith_sft_train_plan.txt",
        "plan_numaug":       "data/multiarith_sft_train_plan_aug.txt",
        "entity_numaug":     ENTITY_SFT,
        "gsm8k":             "data/gsm8k_sft_train.txt",
        "gsm8k_plus_ma":     "data/gsm8k_plus_ma_sft_train.txt",
        "gsm8k_plus_entity": "data/gsm8k_plus_entity_sft_train.txt",
    }
    print("\nRecipe → file check:")
    for recipe, path in recipe_path.items():
        status = "OK" if _exists(path) else "MISSING"
        print(f"  {recipe:<25} {status}  {path}")
    print("\nData preparation complete.")
    return recipe_path


# ── STEP 2: experiment runner ────────────────────────────────────────────────

def run_experiments(args, recipe_path):
    print("\n=== STEP 2: EXPERIMENTS ===\n")
    from scripts.run_ablation import run_experiment

    os.makedirs(args.out_dir, exist_ok=True)
    log_path = os.path.join(args.out_dir, "run_log.txt")

    EXPERIMENTS = [
        {"id": "MA_plan_numaug",
         "init": "arith", "recipe": "plan_numaug", "rung": "multiarith",
         "epochs": 40, "lr": 1e-5, "patience": 8, "batch_size": args.batch_size},
        {"id": "G_A1_direct",
         "init": "arith", "recipe": "gsm8k", "rung": "gsm8k",
         "epochs": 12, "lr": 1e-5, "patience": 4, "batch_size": args.batch_size, "eval_n": 150},
        {"id": "G_A2_mixed",
         "init": "arith", "recipe": "gsm8k_plus_ma", "rung": "gsm8k",
         "epochs": 12, "lr": 1e-5, "patience": 4, "batch_size": args.batch_size, "eval_n": 150},
        {"id": "G_B1_curriculum",
         "init": "ckpt:MA_plan_numaug", "recipe": "gsm8k", "rung": "gsm8k",
         "epochs": 12, "lr": 5e-6, "patience": 4, "batch_size": args.batch_size, "eval_n": 150},
        {"id": "G_B2_curriculum_mix",
         "init": "ckpt:MA_plan_numaug", "recipe": "gsm8k_plus_ma", "rung": "gsm8k",
         "epochs": 12, "lr": 5e-6, "patience": 4, "batch_size": args.batch_size, "eval_n": 150},
    ]

    device_str = "cuda" if args.use_gpu else "cpu"
    all_results = []

    _log(f"Starting {len(EXPERIMENTS)} experiments", log_path)
    _log(f"arith_init_path: {args.arith_init_path}", log_path)
    _log(f"out_dir: {args.out_dir}", log_path)

    for exp in EXPERIMENTS:
        exp_id = exp["id"]
        train_data = recipe_path.get(exp["recipe"])

        if not train_data or not os.path.exists(train_data):
            _log(f"[SKIP-missing] {exp_id}: recipe file not found", log_path)
            continue

        cfg = {
            "id":         exp_id,
            "init":       exp["init"],
            "train_data": train_data,
            "rung":       exp["rung"],
            "epochs":     exp.get("epochs", 40),
            "lr":         exp.get("lr", 1e-5),
            "patience":   exp.get("patience", 8),
            "batch_size": exp.get("batch_size", args.batch_size),
            "eval_n":     exp.get("eval_n"),
        }

        try:
            result = run_experiment(
                cfg,
                arith_init_path=args.arith_init_path,
                out_dir=args.out_dir,
                seed=args.seed,
                device_str=device_str,
            )
            all_results.append(result)
            _log(f"[DONE] {exp_id}  acc={result['best_accuracy']:.3f}  "
                 f"elapsed={result['elapsed_s']:.0f}s", log_path)
        except Exception:
            tb = traceback.format_exc()
            _log(f"[ERROR] {exp_id}\n{tb}", log_path)

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    _log(f"Finished. {len(all_results)} results collected.", log_path)
    return all_results


# ── STEP 3: results summary ──────────────────────────────────────────────────

def print_results(out_dir):
    print("\n=== RESULTS ===\n")
    import glob
    rows = []
    for p in sorted(glob.glob(os.path.join(out_dir, "*_metrics.json"))):
        try:
            m = json.load(open(p))
            if "exp_id" in m:
                rows.append(m)
        except Exception:
            continue

    if not rows:
        print("No results found.")
        return

    hdr = ("exp_id", "init", "rung", "acc", "fmt", "ep", "sec")
    print(f"{'exp_id':<24} {'init':<14} {'rung':<11} {'acc':>7} {'fmt':>7} {'ep':>5} {'sec':>7}")
    print("-" * 80)
    for r in rows:
        print(f"{str(r.get('exp_id','?'))[:23]:<24} "
              f"{str(r.get('init','?'))[:13]:<14} "
              f"{str(r.get('rung','?')):<11} "
              f"{r.get('best_accuracy', 0):>7.3f} "
              f"{r.get('format_valid_rate', 0):>7.3f} "
              f"{str(r.get('best_epoch','?')):>5} "
              f"{str(int(r['elapsed_s'])) if 'elapsed_s' in r else '?':>7}")


# ── main ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    os.chdir(ROOT)
    args = get_args()

    if not args.skip_data_prep:
        recipe_path = run_data_prep()
    else:
        # rebuild recipe_path without running prep
        ENTITY_SFT = "data/multiarith_sft_train_entity_aug.txt"
        recipe_path = {
            "base":              "data/multiarith_sft_train_base.txt",
            "numaug":            "data/multiarith_sft_train_aug.txt",
            "plan_base":         "data/multiarith_sft_train_plan.txt",
            "plan_numaug":       "data/multiarith_sft_train_plan_aug.txt",
            "entity_numaug":     ENTITY_SFT,
            "gsm8k":             "data/gsm8k_sft_train.txt",
            "gsm8k_plus_ma":     "data/gsm8k_plus_ma_sft_train.txt",
            "gsm8k_plus_entity": "data/gsm8k_plus_entity_sft_train.txt",
        }

    if not _exists(args.arith_init_path):
        print(f"\n[ERROR] arith_init_path not found: {args.arith_init_path}")
        print("Copy cot_large_integer_arithmetic_pretrain.pt to /mnt/ first.")
        sys.exit(1)

    run_experiments(args, recipe_path)
    print_results(args.out_dir)
