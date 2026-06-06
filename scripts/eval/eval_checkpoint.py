#!/usr/bin/env python3
import sys
from pathlib import Path
_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "scripts"))
sys.path.insert(0, str(_ROOT / "scripts" / "train"))

"""
Evaluate an ALREADY-TRAINED ReasoningGPT checkpoint on a dev set with GREEDY
decoding. No training is performed — this measures zero-shot transfer
(e.g. a MultiArith-trained checkpoint evaluated on GSM8K).

Reuses the model and eval logic from run_ablation so the metrics match the
training-time numbers exactly.

Usage
-----
    # MultiArith-trained checkpoint, evaluated zero-shot on GSM8K
    python eval_checkpoint.py \
        --checkpoint outputs/B1a_numaug_arith_best.pt \
        --rung gsm8k \
        --use_gpu

    # smoke test on first 5 items
    python eval_checkpoint.py \
        --checkpoint outputs/B1a_numaug_arith_best.pt \
        --rung gsm8k --limit 5 --use_gpu
"""

import argparse
import json
import os

import torch

from run_ablation import (
    ReasoningGPT,
    _make_gpt2_args,
    evaluate,
    _load_dev,
    _load_template_labels,
    seed_everything,
)

SEED = 11711


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True,
                   help=".pt file saved by run_ablation (dict with 'model' key) "
                        "or the arith-init pt.")
    p.add_argument("--rung", choices=["multiarith", "gsm8k"], default="gsm8k",
                   help="Selects the dev set via run_ablation._load_dev(rung).")
    p.add_argument("--out_dir", default="outputs")
    p.add_argument("--limit", type=int, default=0,
                   help="If >0, only evaluate first N dev items (smoke test).")
    p.add_argument("--max_new_tokens", type=int, default=256)
    p.add_argument("--use_gpu", action="store_true")
    p.add_argument("--tag", default="",
                   help="Optional suffix for the output filename so the same "
                        "checkpoint evaluated on different rungs doesn't overwrite.")
    return p.parse_args()


def main():
    args = get_args()
    seed_everything(SEED)

    device = torch.device("cuda") if (args.use_gpu and torch.cuda.is_available()) \
        else torch.device("cpu")

    # ── build model + load checkpoint weights ──────────────────────────────
    model = ReasoningGPT(_make_gpt2_args())
    saved = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    state_dict = saved["model"] if isinstance(saved, dict) and "model" in saved else saved
    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()

    # ── dev set ────────────────────────────────────────────────────────────
    dev = _load_dev(args.rung)
    if args.limit > 0:
        dev = dev[:args.limit]

    template_labels = _load_template_labels() if args.rung == "multiarith" else {}

    print(f"Evaluating checkpoint '{args.checkpoint}' on {args.rung} "
          f"dev ({len(dev)} items, greedy decoding, device={device})...")

    # ── greedy eval (reuses run_ablation.evaluate) ─────────────────────────
    metrics = evaluate(model, dev, template_labels,
                       max_new_tokens=args.max_new_tokens)

    metrics["checkpoint"] = args.checkpoint
    metrics["rung"] = args.rung
    metrics["n"] = len(dev)
    metrics["decoding"] = "greedy"

    # ── write output ───────────────────────────────────────────────────────
    name = os.path.splitext(os.path.basename(args.checkpoint.rstrip("/\\")))[0]
    suffix = f"_{args.tag}" if args.tag else ""
    os.makedirs(args.out_dir, exist_ok=True)
    out_path = os.path.join(args.out_dir, f"{name}_{args.rung}{suffix}_eval.json")

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print(json.dumps(metrics, indent=2))
    print(f"Metrics -> {out_path}")


if __name__ == "__main__":
    main()
