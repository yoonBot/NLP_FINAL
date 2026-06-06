#!/usr/bin/env python3
import sys
from pathlib import Path
_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "scripts"))
sys.path.insert(0, str(_ROOT / "scripts" / "train"))

"""
Self-consistency evaluation (Wang et al. 2022) of an ALREADY-TRAINED checkpoint.

For each dev question we sample k chain-of-thought generations with temperature
+ nucleus (top-p) decoding, extract each final answer, and take the MAJORITY
VOTE over the extracted numeric answers. NO training is involved.

This reuses the model class and helpers from run_ablation.py and the answer
parser from gsm8k_eval.py so the scoring is identical to greedy evaluation; the
only difference is the decoding strategy (sampling + majority vote vs. argmax).

Usage
-----
    python eval_self_consistency.py \
        --checkpoint outputs/B1a_numaug_arith_best.pt \
        --rung multiarith \
        --k 8 --temperature 0.8 --top_p 0.95 \
        --out_dir outputs --use_gpu

Outputs
-------
    outputs/<checkpoint_basename>_<rung>_sc_k{k}_eval.json
"""

import argparse
import json
import os
from collections import Counter

import torch

import gsm8k_eval
from run_ablation import (
    ReasoningGPT,
    _load_dev,
    _load_template_labels,
    _make_gpt2_args,
    seed_everything,
)

# ── constants ──────────────────────────────────────────────────────────────
DEFAULT_K            = 8
DEFAULT_TEMPERATURE  = 0.8
DEFAULT_TOP_P        = 0.95
DEFAULT_MAX_NEW_TOK  = 256
DEFAULT_SEED         = 11711
DEFAULT_OUT_DIR      = "outputs"

PROMPT_MAX_LENGTH    = 512      # truncate the encoded prompt to this many tokens
VOTE_KEY_DECIMALS    = 6        # round predicted values to this many decimals for vote keying
CORRECT_TOL          = 1e-6     # numeric tolerance for "matches gold"
MIN_KEEP_TOKENS      = 1        # top-p: always keep at least this many tokens


# ── sampling generate (nucleus / top-p) ─────────────────────────────────────

@torch.no_grad()
def sample_generate(model, input_ids, max_new_tokens, temperature, top_p, device):
    """Sampling variant of ReasoningGPT.generate (run_ablation.py).

    Mirrors the greedy token loop exactly (attention-mask handling, eos stop,
    context-length guard) but replaces argmax with temperature + top-p (nucleus)
    sampling.

    Returns the decoded full text (prompt + continuation).
    """
    ids  = input_ids.to(device)
    mask = torch.ones_like(ids)
    max_ctx = model.gpt.pos_embedding.num_embeddings

    for _ in range(max_new_tokens):
        if ids.size(1) >= max_ctx:
            break

        logits = model.forward(ids, mask)[:, -1, :]      # (1, vocab)
        logits = logits / temperature
        probs = torch.softmax(logits, dim=-1)            # (1, vocab)

        # ── top-p (nucleus) filtering ──────────────────────────────────────
        sorted_probs, sorted_idx = torch.sort(probs, descending=True, dim=-1)
        cumprobs = torch.cumsum(sorted_probs, dim=-1)
        # Keep tokens up to and including the one that pushes cumprob >= top_p.
        keep = cumprobs < top_p
        # Shift right so the first token crossing the threshold is also kept.
        keep[..., 1:] = keep[..., :-1].clone()
        keep[..., :MIN_KEEP_TOKENS] = True               # always keep the top-1
        sorted_probs = sorted_probs * keep
        sorted_probs = sorted_probs / sorted_probs.sum(dim=-1, keepdim=True)

        # Sample within the sorted/filtered distribution, map back to vocab id.
        sampled_sorted_pos = torch.multinomial(sorted_probs, num_samples=1)  # (1, 1)
        next_t = torch.gather(sorted_idx, -1, sampled_sorted_pos)            # (1, 1)

        if next_t.item() == model.tokenizer.eos_token_id:
            break

        ids  = torch.cat([ids, next_t], dim=1)
        mask = torch.cat(
            [mask, torch.ones((1, 1), dtype=torch.int64, device=device)], dim=1
        )

    return model.tokenizer.decode(ids[0].cpu().tolist())


# ── per-question self-consistency ───────────────────────────────────────────

def _continuation(generated, prompt):
    return generated[len(prompt):] if generated.startswith(prompt) else generated


def self_consistency_predict(samples_text, prompt):
    """Majority-vote over the k sampled generations.

    Returns (voted_value_or_None, top_count, sample_pred_values) where
    sample_pred_values is the per-sample extracted value (may contain None).
    """
    sample_values = []
    # Map a rounded float key -> a representative actual value, plus vote counts.
    counter = Counter()
    representative = {}

    for text in samples_text:
        cont = _continuation(text, prompt)
        pred, _ = gsm8k_eval.extract_pred_answer(cont)
        sample_values.append(pred)
        if pred is None:
            continue
        key = round(pred, VOTE_KEY_DECIMALS)
        counter[key] += 1
        representative.setdefault(key, pred)

    if not counter:
        return None, 0, sample_values

    top_key, top_count = counter.most_common(1)[0]
    return representative[top_key], top_count, sample_values


def _is_correct(value, gold):
    if value is None or gold is None:
        return False
    return abs(value - gold) <= CORRECT_TOL


# ── main ────────────────────────────────────────────────────────────────────

def main():
    args = get_args()
    seed_everything(args.seed)
    device = torch.device(
        "cuda" if args.use_gpu and torch.cuda.is_available() else "cpu"
    )

    # ── model ──────────────────────────────────────────────────────────────
    model = ReasoningGPT(_make_gpt2_args())
    saved = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    state_dict = saved.get("model", saved)
    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()

    # ── data ───────────────────────────────────────────────────────────────
    dev = _load_dev(args.rung)
    if args.limit > 0:
        dev = dev[: args.limit]
    template_labels = (
        _load_template_labels() if args.rung == "multiarith" else {}
    )
    print(
        f"Self-consistency eval on {len(dev)} {args.rung} dev items "
        f"(k={args.k}, T={args.temperature}, top_p={args.top_p})..."
    )

    # ── per-item evaluation ─────────────────────────────────────────────────
    n               = len(dev)
    sc_correct      = 0      # majority-vote correct
    any_correct     = 0      # pass@k: at least one of k samples correct
    no_answer       = 0      # voted prediction is None
    vote_agreements = []     # top_count / k per item

    in_tmpl_total = in_tmpl_correct = 0
    out_tmpl_total = out_tmpl_correct = 0

    for rec in dev:
        prompt = f"Question: {rec['question']}\n\nReasoning:\n"
        enc = model.tokenizer(
            prompt, return_tensors="pt", truncation=True, max_length=PROMPT_MAX_LENGTH
        )

        samples_text = [
            sample_generate(
                model,
                enc["input_ids"],
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                top_p=args.top_p,
                device=device,
            )
            for _ in range(args.k)
        ]

        voted_value, top_count, sample_values = self_consistency_predict(
            samples_text, prompt
        )
        gold = rec["gold_answer"]

        is_sc_correct = _is_correct(voted_value, gold)
        if is_sc_correct:
            sc_correct += 1
        if voted_value is None:
            no_answer += 1
        if any(_is_correct(v, gold) for v in sample_values):
            any_correct += 1
        vote_agreements.append(top_count / args.k if args.k > 0 else 0.0)

        # ── template split (multiarith only) ────────────────────────────────
        tid = rec["id"]
        if tid in template_labels:
            if template_labels[tid]:
                in_tmpl_total += 1
                in_tmpl_correct += int(is_sc_correct)
            else:
                out_tmpl_total += 1
                out_tmpl_correct += int(is_sc_correct)

    # ── aggregate ───────────────────────────────────────────────────────────
    metrics = {
        "n":                        n,
        "k":                        args.k,
        "temperature":              args.temperature,
        "top_p":                    args.top_p,
        "self_consistency_accuracy": (sc_correct / n) if n else 0.0,
        "any_correct_rate":         (any_correct / n) if n else 0.0,
        "mean_vote_agreement":      (sum(vote_agreements) / n) if n else 0.0,
        "no_answer_rate":           (no_answer / n) if n else 0.0,
        "checkpoint":               args.checkpoint,
        "rung":                     args.rung,
        "decoding":                 f"self_consistency(k={args.k},T={args.temperature})",
    }
    if in_tmpl_total:
        metrics["in_template_accuracy"] = in_tmpl_correct / in_tmpl_total
    if out_tmpl_total:
        metrics["held_out_template_accuracy"] = out_tmpl_correct / out_tmpl_total

    # ── write ───────────────────────────────────────────────────────────────
    os.makedirs(args.out_dir, exist_ok=True)
    name = os.path.splitext(os.path.basename(args.checkpoint.rstrip("/\\")))[0]
    out_path = os.path.join(
        args.out_dir, f"{name}_{args.rung}_sc_k{args.k}_eval.json"
    )
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print(json.dumps(metrics, indent=2))
    print(f"Metrics -> {out_path}")


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=str, required=True,
                   help=".pt checkpoint with a 'model' key.")
    p.add_argument("--rung", choices=["multiarith", "gsm8k"], default="gsm8k")
    p.add_argument("--k", type=int, default=DEFAULT_K,
                   help="Number of samples per question.")
    p.add_argument("--temperature", type=float, default=DEFAULT_TEMPERATURE)
    p.add_argument("--top_p", type=float, default=DEFAULT_TOP_P)
    p.add_argument("--limit", type=int, default=0,
                   help="If >0, only evaluate the first N dev items.")
    p.add_argument("--max_new_tokens", type=int, default=DEFAULT_MAX_NEW_TOK)
    p.add_argument("--out_dir", default=DEFAULT_OUT_DIR)
    p.add_argument("--use_gpu", action="store_true")
    p.add_argument("--seed", type=int, default=DEFAULT_SEED)
    return p.parse_args()


if __name__ == "__main__":
    main()
