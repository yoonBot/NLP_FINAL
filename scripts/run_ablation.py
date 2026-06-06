#!/usr/bin/env python3
import sys
from pathlib import Path
_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "scripts" / "eval"))

"""
Two-arm ablation runner: arith-init vs vanilla-init.

Callable as a module (from the notebook) or as a CLI script.

Usage
-----
    # from Python
    from run_ablation import run_experiment
    metrics = run_experiment(cfg, arith_init_path="cot_large_integer_arithmetic_pretrain.pt")

    # from CLI
    python run_ablation.py \
        --exp_id B1a_numaug_arith \
        --init arith \
        --train_data data/multiarith_sft_train_aug.txt \
        --rung multiarith \
        --epochs 40 --lr 1e-5 --patience 8 \
        --arith_init_path cot_large_integer_arithmetic_pretrain.pt \
        --out_dir outputs \
        --use_gpu
"""

import argparse
import json
import os
import random
import time

import numpy as np
import torch
import torch.nn.functional as F
from einops import rearrange
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import GPT2Tokenizer

import gsm8k_eval
from gpt_datasets import ReasoningDataset
from models.gpt2 import GPT2Model
from optimizer import AdamW

# ── constants ──────────────────────────────────────────────────────────────
DEV_MULTIARITH = os.path.join("data", "multiarith_dev.jsonl")
DEV_GSM8K      = os.path.join("data", "gsm8k_dev.jsonl")
TEMPLATE_LABELS = os.path.join("data", "multiarith_dev_template_labels.jsonl")
TQDM_DISABLE   = True   # keep logs clean in notebook


def seed_everything(seed=11711):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


# ── model helpers ──────────────────────────────────────────────────────────

def _make_gpt2_args():
    import argparse as _ap
    a = _ap.Namespace()
    a.model_size = "gpt2"
    a.d = 768; a.l = 12; a.num_heads = 12
    return a


class ReasoningGPT(torch.nn.Module):
    def __init__(self, args):
        super().__init__()
        self.gpt = GPT2Model.from_pretrained(
            model=args.model_size, d=args.d, l=args.l, num_heads=args.num_heads
        )
        self.tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        self.tokenizer.pad_token = self.tokenizer.eos_token
        for p in self.gpt.parameters():
            p.requires_grad = True

    def forward(self, input_ids, attention_mask):
        out = self.gpt(input_ids, attention_mask)
        return self.gpt.hidden_state_to_token(out["last_hidden_state"])

    def get_device(self):
        return next(self.gpt.parameters()).device

    @torch.no_grad()
    def generate(self, input_ids, max_new_tokens=256):
        device = self.get_device()
        ids    = input_ids.to(device)
        mask   = torch.ones_like(ids)
        max_ctx = self.gpt.pos_embedding.num_embeddings
        for _ in range(max_new_tokens):
            if ids.size(1) >= max_ctx:
                break
            logits = self.forward(ids, mask)[:, -1, :]
            next_t = torch.argmax(logits, dim=-1, keepdim=True)
            if next_t.item() == self.tokenizer.eos_token_id:
                break
            ids  = torch.cat([ids,  next_t], dim=1)
            mask = torch.cat([mask, torch.ones((1, 1), dtype=torch.int64, device=device)], dim=1)
        return self.tokenizer.decode(ids[0].cpu().tolist())


def load_model(init: str, arith_init_path: str, device,
               out_dir: str = "outputs") -> ReasoningGPT:
    """Load model weights.

    init values
    -----------
    "arith"         : arithmetic-pretrained checkpoint (arith_init_path)
    "vanilla"       : HuggingFace GPT-2 pretrained
    "ckpt:<exp_id>" : checkpoint saved by a previous run_experiment call
                      (loads outputs/<exp_id>_best.pt)
    """
    args = _make_gpt2_args()
    model = ReasoningGPT(args)
    if init == "arith":
        saved = torch.load(arith_init_path, map_location="cpu", weights_only=False)
        sd = saved.get("model", saved)
        model.load_state_dict(sd, strict=True)
        print(f"  [init] arith weights from {arith_init_path}")
    elif init.startswith("ckpt:"):
        exp_id = init[len("ckpt:"):]
        ckpt_path = os.path.join(out_dir, f"{exp_id}_best.pt")
        if not os.path.exists(ckpt_path):
            raise FileNotFoundError(
                f"Curriculum init: {ckpt_path} not found. "
                f"Run experiment '{exp_id}' first."
            )
        saved = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        sd = saved.get("model", saved)
        model.load_state_dict(sd, strict=True)
        print(f"  [init] curriculum checkpoint from {ckpt_path}")
    else:
        print("  [init] vanilla GPT-2 (HuggingFace pretrained)")
    return model.to(device)


# ── per-epoch evaluation ────────────────────────────────────────────────────

def _load_dev(rung: str) -> list[dict]:
    path = DEV_MULTIARITH if rung == "multiarith" else DEV_GSM8K
    with open(path, encoding="utf-8") as f:
        return [json.loads(l) for l in f]


def _load_template_labels() -> dict[int, bool]:
    """id -> in_train_template.  Returns {} if file missing."""
    if not os.path.exists(TEMPLATE_LABELS):
        return {}
    out = {}
    with open(TEMPLATE_LABELS, encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            out[rec["id"]] = rec["in_train_template"]
    return out


def evaluate(model: ReasoningGPT, dev: list[dict], template_labels: dict,
             max_new_tokens: int = 256) -> dict:
    model.eval()
    records    = []
    in_tmpl_records  = []
    out_tmpl_records = []

    for rec in dev:
        prompt = f"Question: {rec['question']}\n\nReasoning:\n"
        enc = model.tokenizer(
            prompt, return_tensors="pt", truncation=True, max_length=512
        )
        generated = model.generate(enc["input_ids"], max_new_tokens=max_new_tokens)
        continuation = generated[len(prompt):] if generated.startswith(prompt) else generated
        entry = {"generation": continuation, "gold": rec["gold_answer"]}
        records.append(entry)
        tid = rec["id"]
        if tid in template_labels:
            if template_labels[tid]:
                in_tmpl_records.append(entry)
            else:
                out_tmpl_records.append(entry)

    metrics = gsm8k_eval.evaluate(records)
    if in_tmpl_records:
        m_in = gsm8k_eval.evaluate(in_tmpl_records)
        metrics["in_template_accuracy"]  = m_in["exact_accuracy"]
    if out_tmpl_records:
        m_out = gsm8k_eval.evaluate(out_tmpl_records)
        metrics["held_out_template_accuracy"] = m_out["exact_accuracy"]
    return metrics


# ── training + early stop ───────────────────────────────────────────────────

def run_experiment(cfg: dict, arith_init_path: str, out_dir: str = "outputs",
                   seed: int = 11711, device_str: str = "cuda") -> dict:
    """
    Train one arm and return metrics dict.

    cfg keys
    --------
    id          : str   experiment id (used for checkpoint/metrics filename)
    init        : str   "arith" | "vanilla"
    train_data  : str   path to SFT .txt file
    rung        : str   "multiarith" | "gsm8k"
    epochs      : int   max epochs
    lr          : float learning rate
    patience    : int   early-stop patience (default 8)
    batch_size  : int   (default 8)
    max_eval_tokens : int (default 256)
    """
    exp_id     = cfg["id"]
    metrics_path = os.path.join(out_dir, f"{exp_id}_metrics.json")

    # ── idempotent skip ───────────────────────────────────────────────────
    if os.path.exists(metrics_path):
        with open(metrics_path) as f:
            saved_metrics = json.load(f)
        print(f"  [SKIP] {exp_id} already done  "
              f"(acc={saved_metrics.get('best_accuracy', '?'):.3f})")
        return saved_metrics

    seed_everything(seed)
    device = torch.device(device_str if torch.cuda.is_available() else "cpu")
    os.makedirs(out_dir, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"  EXP  : {exp_id}")
    print(f"  init : {cfg['init']}  |  data : {cfg['train_data']}")
    print(f"  rung : {cfg['rung']}  |  device : {device}")
    print(f"{'='*60}")

    # ── data ──────────────────────────────────────────────────────────────
    dataset = ReasoningDataset(cfg["train_data"])
    loader  = DataLoader(
        dataset, shuffle=True,
        batch_size=cfg.get("batch_size", 8),
        collate_fn=dataset.collate_fn,
    )
    dev_records      = _load_dev(cfg["rung"])
    template_labels  = _load_template_labels() if cfg["rung"] == "multiarith" else {}

    # ── per-epoch eval subset (early-stop signal only) ─────────────────────
    eval_n   = cfg.get("eval_n")
    dev_eval = dev_records[:eval_n] if eval_n else dev_records
    print(f"  dev: {len(dev_records)} total  |  per-epoch eval on first {len(dev_eval)}")

    # ── model + optimiser ─────────────────────────────────────────────────
    model = load_model(cfg["init"], arith_init_path, device, out_dir=out_dir)
    optimizer = AdamW(model.parameters(), lr=cfg["lr"])

    # ── training loop ─────────────────────────────────────────────────────
    patience  = cfg.get("patience", 8)
    best_acc  = -1.0
    best_epoch = -1
    best_ckpt  = None
    no_improve = 0
    epoch_log  = []

    t0 = time.time()
    for epoch in range(cfg["epochs"]):
        model.train()
        total_loss = 0.0
        n_batches  = 0
        for batch in tqdm(loader, desc=f"  epoch {epoch}", disable=TQDM_DISABLE):
            b_ids  = batch["token_ids"].to(device)
            b_mask = batch["attention_mask"].to(device)
            optimizer.zero_grad()
            logits = model(b_ids, b_mask)
            logits = rearrange(logits[:, :-1].contiguous(), "b t d -> (b t) d")
            labels = b_ids[:, 1:].clone()
            labels[b_mask[:, 1:] == 0] = -100
            loss = F.cross_entropy(logits, labels.flatten(), ignore_index=-100)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            n_batches  += 1

        avg_loss = total_loss / max(n_batches, 1)

        # ── dev eval ──────────────────────────────────────────────────────
        dev_metrics = evaluate(
            model, dev_eval, template_labels,
            max_new_tokens=cfg.get("max_eval_tokens", 256),
        )
        acc = dev_metrics["exact_accuracy"]
        print(f"  epoch {epoch:3d}  loss={avg_loss:.4f}  "
              f"acc={acc:.4f}  fmt={dev_metrics['format_valid_rate']:.4f}  "
              f"elapsed={time.time()-t0:.0f}s")

        row = {"epoch": epoch, "loss": avg_loss, **dev_metrics}
        epoch_log.append(row)

        # ── checkpoint best ───────────────────────────────────────────────
        if acc > best_acc:
            best_acc   = acc
            best_epoch = epoch
            no_improve = 0
            ckpt_path  = os.path.join(out_dir, f"{exp_id}_best.pt")
            torch.save({"model": model.state_dict(), "args": _make_gpt2_args(),
                        "epoch": epoch}, ckpt_path)
            best_ckpt = ckpt_path
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"  [early stop] no improvement for {patience} epochs.")
                break

    # ── final metrics ─────────────────────────────────────────────────────
    best_dev = next(
        (r for r in epoch_log if r["epoch"] == best_epoch), epoch_log[-1]
    )

    # ── FINAL FULL EVALUATION on the complete dev set ──────────────────────
    # The per-epoch loop drove early-stop using the fast subset (dev_eval).
    # Reload the best checkpoint (in-memory weights are the LAST epoch's,
    # not the best epoch's) and evaluate on the FULL dev_records for the
    # headline numbers.
    if best_ckpt is not None:
        saved = torch.load(best_ckpt, map_location="cpu", weights_only=False)
        model.load_state_dict(saved["model"])
        full_metrics = evaluate(
            model, dev_records, template_labels,
            max_new_tokens=cfg.get("max_eval_tokens", 256),
        )
        print(f"  FINAL full-dev eval: "
              f"acc={full_metrics['exact_accuracy']:.4f} (n={len(dev_records)})")
    else:
        # No epoch improved (shouldn't happen) — fall back to last-epoch metrics.
        print("  [warn] no best checkpoint; falling back to last-epoch metrics.")
        full_metrics = best_dev

    result = {
        "exp_id":         exp_id,
        "init":           cfg["init"],
        "train_data":     cfg["train_data"],
        "rung":           cfg["rung"],
        "best_epoch":     best_epoch,
        "best_accuracy":  full_metrics.get("exact_accuracy"),
        "format_valid_rate": full_metrics.get("format_valid_rate"),
        "no_answer_rate":    full_metrics.get("no_answer_rate"),
        "repetition_rate":   full_metrics.get("repetition_rate"),
        "in_template_accuracy":       full_metrics.get("in_template_accuracy"),
        "held_out_template_accuracy": full_metrics.get("held_out_template_accuracy"),
        "subset_eval_n":      eval_n,
        "full_dev_n":         len(dev_records),
        "subset_best_accuracy": best_acc,
        "checkpoint":     best_ckpt,
        "total_epochs_run": len(epoch_log),
        "elapsed_s":      round(time.time() - t0, 1),
    }
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    print(f"  -> metrics saved to {metrics_path}")
    return result


# ── CLI entry point ─────────────────────────────────────────────────────────

def _get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--exp_id",         required=True)
    p.add_argument("--init",           choices=["arith", "vanilla"], required=True)
    p.add_argument("--train_data",     required=True)
    p.add_argument("--rung",           choices=["multiarith", "gsm8k"], required=True)
    p.add_argument("--epochs",         type=int,   default=40)
    p.add_argument("--lr",             type=float, default=1e-5)
    p.add_argument("--patience",       type=int,   default=8)
    p.add_argument("--batch_size",     type=int,   default=8)
    p.add_argument("--max_eval_tokens",type=int,   default=256)
    p.add_argument("--arith_init_path",default="cot_large_integer_arithmetic_pretrain.pt")
    p.add_argument("--out_dir",        default="outputs")
    p.add_argument("--seed",           type=int,   default=11711)
    p.add_argument("--use_gpu",        action="store_true")
    return p.parse_args()


if __name__ == "__main__":
    args = _get_args()
    cfg = {
        "id":               args.exp_id,
        "init":             args.init,
        "train_data":       args.train_data,
        "rung":             args.rung,
        "epochs":           args.epochs,
        "lr":               args.lr,
        "patience":         args.patience,
        "batch_size":       args.batch_size,
        "max_eval_tokens":  args.max_eval_tokens,
    }
    device_str = "cuda" if args.use_gpu else "cpu"
    run_experiment(cfg, arith_init_path=args.arith_init_path,
                   out_dir=args.out_dir, seed=args.seed, device_str=device_str)
