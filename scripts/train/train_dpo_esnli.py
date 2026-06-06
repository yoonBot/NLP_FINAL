#!/usr/bin/env python3
"""
DPO fine-tuning on top of a CoT-SFT checkpoint for e-SNLI.

Requires: pip install trl transformers
Usage:
    python train_dpo_esnli.py --sft_checkpoint 2_1e-05-esnli.pt --use_gpu
"""

import argparse
import json
import os
import torch
from datasets import Dataset
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from trl import DPOTrainer, DPOConfig

from reasoning_generation_esnli import seed_everything

DPO_PAIRS = os.path.join("data", "esnli_dpo_pairs.jsonl")
OUT_DIR = os.path.join("outputs", "dpo_esnli_model")

def load_pairs(path, limit=0):
    with open(path, "r", encoding="utf-8") as f:
        rows = [json.loads(line) for line in f]
    if limit:
        rows = rows[:limit]
    return Dataset.from_list(rows)

def main():
    args = get_args()
    seed_everything(args.seed)

    # Load SFT weights into HuggingFace GPT2LMHeadModel for TRL compatibility
    saved = torch.load(args.sft_checkpoint, weights_only=False)
    model_args = saved["args"]

    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token

    # Policy model (will be trained)
    policy = GPT2LMHeadModel.from_pretrained(model_args.model_size)
    policy.load_state_dict(
        {k.replace("gpt.", ""): v for k, v in saved["model"].items()
         if not k.startswith("gpt.tokenizer")},
        strict=False,
    )

    # Reference model (frozen copy of SFT)
    ref_model = GPT2LMHeadModel.from_pretrained(model_args.model_size)
    ref_model.load_state_dict(policy.state_dict())
    for param in ref_model.parameters():
        param.requires_grad = False

    dataset = load_pairs(DPO_PAIRS, limit=args.limit)

    training_args = DPOConfig(
        output_dir=OUT_DIR,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        learning_rate=args.lr,
        beta=args.beta,
        logging_steps=10,
        save_strategy="epoch",
        fp16=args.use_gpu and torch.cuda.is_available(),
        report_to="none",
        remove_unused_columns=False,
        max_length=512,
    )

    trainer = DPOTrainer(
        model=policy,
        ref_model=ref_model,
        args=training_args,
        train_dataset=dataset,
        processing_class=tokenizer,
    )

    trainer.train()
    trainer.save_model(OUT_DIR)
    tokenizer.save_pretrained(OUT_DIR)
    print(f"\nDPO model saved -> {OUT_DIR}")

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sft_checkpoint", type=str, required=True,
                        help="Path to SFT .pt checkpoint (used as starting weights)")
    parser.add_argument("--use_gpu", action="store_true")
    parser.add_argument("--seed", type=int, default=11711)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-6)
    parser.add_argument("--beta", type=float, default=0.1,
                        help="DPO beta (KL regularization strength)")
    parser.add_argument("--limit", type=int, default=0,
                        help="If >0, only use first N pairs (smoke test).")
    return parser.parse_args()

if __name__ == "__main__":
    main()
