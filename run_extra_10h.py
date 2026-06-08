#!/usr/bin/env python3
"""Run the 10-hour extra GSM8K experiment queue on the server.

Intended server usage:
    cd /mnt/NLP_FINAL
    python run_extra_10h.py 2>&1 | tee -a /mnt/outputs/extra_10h.log

The queue is intentionally idempotent. Existing metrics / SC result files are
reused so the command can be restarted after an SSH disconnect or container
hiccup.
"""

from __future__ import annotations

import gc
import json
import os
import subprocess
import sys
import time
import traceback
from pathlib import Path


ROOT = Path(__file__).resolve().parent
OUT_DIR = Path(os.environ.get("NLP_FINAL_OUT_DIR", "/mnt/outputs"))
SC_DIR = OUT_DIR / "sc_eval"
ARITH_INIT = Path(
    os.environ.get("NLP_FINAL_ARITH_INIT", "/mnt/cot_large_integer_arithmetic_pretrain.pt")
)

SC_K = int(os.environ.get("NLP_FINAL_SC_K", "8"))
SC_LIMIT = int(os.environ.get("NLP_FINAL_SC_LIMIT", "150"))
SC_TEMPERATURE = os.environ.get("NLP_FINAL_SC_TEMPERATURE", "0.8")
SC_TOP_P = os.environ.get("NLP_FINAL_SC_TOP_P", "0.95")
USE_GPU = os.environ.get("NLP_FINAL_USE_GPU", "1") != "0"


def log(message: str) -> None:
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{ts}] {message}", flush=True)


def require_file(path: Path, label: str) -> None:
    if not path.exists() or path.stat().st_size == 0:
        raise FileNotFoundError(f"{label} missing or empty: {path}")


def sc_output_path(checkpoint: Path) -> Path:
    return SC_DIR / f"{checkpoint.stem}_gsm8k_sc_k{SC_K}_eval.json"


def sc_samples_path(checkpoint: Path) -> Path:
    return SC_DIR / f"{checkpoint.stem}_gsm8k_sc_k{SC_K}_samples.json"


def existing_sc_is_complete(path: Path, checkpoint: Path) -> bool:
    if not path.exists() or path.stat().st_size == 0:
        return False
    samples_path = sc_samples_path(checkpoint)
    if not samples_path.exists() or samples_path.stat().st_size == 0:
        return False
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return False
    return data.get("n") == SC_LIMIT and data.get("k") == SC_K


def run_self_consistency(checkpoint: Path, label: str) -> None:
    require_file(checkpoint, f"{label} checkpoint")
    SC_DIR.mkdir(parents=True, exist_ok=True)
    out_path = sc_output_path(checkpoint)
    if existing_sc_is_complete(out_path, checkpoint):
        log(f"[SKIP] SC {label}: {out_path}")
        return

    cmd = [
        sys.executable,
        "scripts/eval/eval_self_consistency.py",
        "--checkpoint",
        str(checkpoint),
        "--rung",
        "gsm8k",
        "--k",
        str(SC_K),
        "--temperature",
        str(SC_TEMPERATURE),
        "--top_p",
        str(SC_TOP_P),
        "--limit",
        str(SC_LIMIT),
        "--out_dir",
        str(SC_DIR),
        "--save_samples",
    ]
    if USE_GPU:
        cmd.append("--use_gpu")

    log(f"[RUN] SC {label}: {' '.join(cmd)}")
    subprocess.run(cmd, cwd=ROOT, check=True)
    log(f"[DONE] SC {label}: {out_path}")


def run_training_experiment(cfg: dict) -> None:
    metrics_path = OUT_DIR / f"{cfg['id']}_metrics.json"
    if metrics_path.exists() and metrics_path.stat().st_size > 0:
        log(f"[SKIP] train {cfg['id']}: {metrics_path}")
        return

    sys.path.insert(0, str(ROOT))
    sys.path.insert(0, str(ROOT / "scripts"))
    from scripts.run_ablation import run_experiment

    log(
        "[RUN] train {id}: lr={lr} epochs={epochs} patience={patience} "
        "batch={batch_size} data={train_data}".format(**cfg)
    )
    result = run_experiment(
        cfg,
        arith_init_path=str(ARITH_INIT),
        out_dir=str(OUT_DIR),
        seed=11711,
        device_str="cuda" if USE_GPU else "cpu",
    )
    log(
        "[DONE] train {exp_id}: acc={best_accuracy:.4f} fmt={format_valid_rate:.4f} "
        "repeat={repetition_rate:.4f} best_ep={best_epoch}".format(**result)
    )

    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass
    gc.collect()


def print_json_summary(path: Path) -> None:
    if not path.exists() or path.stat().st_size == 0:
        log(f"[MISS] {path}")
        return
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        log(f"[WARN] could not parse {path}: {exc}")
        return

    keys = [
        "exp_id",
        "best_accuracy",
        "format_valid_rate",
        "repetition_rate",
        "no_answer_rate",
        "best_epoch",
        "total_epochs_run",
        "self_consistency_accuracy",
        "any_correct_rate",
        "mean_vote_agreement",
        "n",
        "k",
    ]
    compact = {k: data[k] for k in keys if k in data}
    log(f"[SUMMARY] {path.name}: {json.dumps(compact, sort_keys=True)}")


def main() -> int:
    os.chdir(ROOT)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    SC_DIR.mkdir(parents=True, exist_ok=True)

    require_file(ARITH_INIT, "arithmetic init")
    require_file(OUT_DIR / "MA_plan_numaug_best.pt", "MA curriculum checkpoint")

    base_checkpoints = [
        ("G_B2_ent", OUT_DIR / "G_B2_ent_best.pt"),
        ("G_B2_skel", OUT_DIR / "G_B2_skel_best.pt"),
    ]

    extra_experiments = [
        {
            "id": "G_B2_ent_lr1e6",
            "init": "ckpt:MA_plan_numaug",
            "train_data": "data/gsm8k_plan_entity_plus_ma.txt",
            "rung": "gsm8k",
            "epochs": 8,
            "lr": 1e-6,
            "patience": 3,
            "batch_size": 4,
            "eval_n": 150,
        },
        {
            "id": "G_B2_skel_lr1e6",
            "init": "ckpt:MA_plan_numaug",
            "train_data": "data/gsm8k_plan_skeleton_plus_ma.txt",
            "rung": "gsm8k",
            "epochs": 8,
            "lr": 1e-6,
            "patience": 3,
            "batch_size": 4,
            "eval_n": 150,
        },
    ]

    log("=== 10h extra GSM8K queue start ===")
    log(f"root={ROOT}")
    log(f"out_dir={OUT_DIR}")
    log(f"sc_dir={SC_DIR}")
    log(
        f"SC settings: k={SC_K}, limit={SC_LIMIT}, "
        f"T={SC_TEMPERATURE}, top_p={SC_TOP_P}, use_gpu={USE_GPU}"
    )

    try:
        log("=== Phase 1: SC on existing best checkpoints ===")
        for label, ckpt in base_checkpoints:
            run_self_consistency(ckpt, label)

        log("=== Phase 2: lower-LR continuation experiments ===")
        for cfg in extra_experiments:
            require_file(ROOT / cfg["train_data"], f"{cfg['id']} train data")
            run_training_experiment(cfg)

        log("=== Phase 3: SC on lower-LR checkpoints ===")
        for cfg in extra_experiments:
            run_self_consistency(OUT_DIR / f"{cfg['id']}_best.pt", cfg["id"])

        log("=== Final summary ===")
        for label, ckpt in base_checkpoints:
            print_json_summary(sc_output_path(ckpt))
            log(f"[RAW] {sc_samples_path(ckpt)}")
        for cfg in extra_experiments:
            print_json_summary(OUT_DIR / f"{cfg['id']}_metrics.json")
            ckpt = OUT_DIR / f"{cfg['id']}_best.pt"
            print_json_summary(sc_output_path(ckpt))
            log(f"[RAW] {sc_samples_path(ckpt)}")

        log("=== 10h extra GSM8K queue complete ===")
        return 0
    except Exception:
        log("[ERROR] queue failed")
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
