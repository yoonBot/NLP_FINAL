#!/usr/bin/env python3
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

'''
Reasoning generation for e-SNLI.
'''

import os
import argparse
import random
import torch
import numpy as np
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import GPT2Tokenizer
from einops import rearrange

from gpt_datasets_esnli import ESNLIDataset
from models.gpt2 import GPT2Model
from optimizer import AdamW

TQDM_DISABLE = False

def seed_everything(seed=11711):
  random.seed(seed)
  np.random.seed(seed)
  torch.manual_seed(seed)
  torch.cuda.manual_seed(seed)
  torch.cuda.manual_seed_all(seed)
  torch.backends.cudnn.benchmark = False
  torch.backends.cudnn.deterministic = True

class ESNLIGPT(nn.Module):
  """GPT-2 model for e-SNLI reasoning generation."""

  def __init__(self, args):
    super().__init__()
    self.gpt = GPT2Model.from_pretrained(model=args.model_size, d=args.d, l=args.l, num_heads=args.num_heads)
    self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    self.tokenizer.pad_token = self.tokenizer.eos_token

    for param in self.gpt.parameters():
      param.requires_grad = True

  def forward(self, input_ids, attention_mask):
    outputs = self.gpt(input_ids, attention_mask)
    hidden_states = outputs['last_hidden_state']
    logits = self.gpt.hidden_state_to_token(hidden_states)
    return logits

  def get_device(self):
    for param in self.gpt.parameters():
      return param.device

  @torch.no_grad()
  def generate(self, encoding, temperature=0.3, top_p=0.9, max_length=128):
    token_ids = encoding.to(self.get_device())
    attention_mask = torch.ones(
        token_ids.shape,
        dtype=torch.int64
    ).to(self.get_device())

    max_context_length = self.gpt.pos_embedding.num_embeddings
    available_tokens = max_context_length - token_ids.size(1)

    if available_tokens <= 0:
        generated_output = self.tokenizer.decode(
            token_ids[0].cpu().numpy().tolist()
        )
        return token_ids, generated_output

    for _ in range(min(max_length, available_tokens)):
      if token_ids.size(1) >= max_context_length:
        break

      logits_sequence = self.forward(token_ids, attention_mask)
      logits_last_token = logits_sequence[:, -1, :] / temperature

      probs = torch.nn.functional.softmax(logits_last_token, dim=-1)

      sorted_probs, sorted_indices = torch.sort(probs, descending=True)
      cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
      top_p_mask = cumulative_probs <= top_p
      top_p_mask[..., 1:] = top_p_mask[..., :-1].clone()
      top_p_mask[..., 0] = True
      filtered_probs = sorted_probs * top_p_mask
      filtered_probs /= filtered_probs.sum(dim=-1, keepdim=True)

      sampled_index = torch.multinomial(filtered_probs, 1)
      sampled_token = sorted_indices.gather(dim=-1, index=sampled_index)

      if sampled_token.item() == self.tokenizer.eos_token_id:
        break

      token_ids = torch.cat([token_ids, sampled_token], dim=1)
      attention_mask = torch.cat(
        [attention_mask, torch.ones((1, 1), dtype=torch.int64).to(self.get_device())], dim=1
      )

    generated_output = self.tokenizer.decode(token_ids[0].cpu().numpy().tolist())
    return token_ids, generated_output

def save_model(model, optimizer, args, filepath):
  save_info = {
    'model': model.state_dict(),
    'optim': optimizer.state_dict(),
    'args': args,
    'system_rng': random.getstate(),
    'numpy_rng': np.random.get_state(),
    'torch_rng': torch.random.get_rng_state(),
  }
  torch.save(save_info, filepath)
  print(f"save the model to {filepath}")

def train(args):
  device = torch.device('cuda') if args.use_gpu else torch.device('cpu')
  
  # Load e-SNLI datasets
  esnli_dataset = ESNLIDataset(args.reasoning_path, mask_prompt=args.mask_prompt)
  esnli_dataloader = DataLoader(esnli_dataset, shuffle=True, batch_size=args.batch_size,
                                    collate_fn=esnli_dataset.collate_fn)

  held_out_esnli_dataset = ESNLIDataset(args.held_out_reasoning_path, mask_prompt=False)
  print("Held-out examples:", len(held_out_esnli_dataset))

  model = ESNLIGPT(args)
  model = model.to(device)

  lr = args.lr
  optimizer = AdamW(model.parameters(), lr=lr)

  start_epoch = 0
  if args.resume_epoch >= 0:
    ckpt = os.path.join(args.output_dir, f'{args.resume_epoch}_{args.filepath}')
    saved = torch.load(ckpt, weights_only=False)
    model.load_state_dict(saved['model'])
    optimizer.load_state_dict(saved['optim'])
    random.setstate(saved['system_rng'])
    np.random.set_state(saved['numpy_rng'])
    torch.random.set_rng_state(saved['torch_rng'])
    start_epoch = args.resume_epoch + 1
    print(f"Resumed from epoch {args.resume_epoch} ({ckpt})")

  for epoch in range(start_epoch, args.epochs):
    model.train()
    train_loss = 0
    num_batches = 0

    for batch in tqdm(esnli_dataloader, desc=f'train-{epoch}', disable=TQDM_DISABLE):
      b_ids, b_mask = batch['token_ids'], batch['attention_mask']
      b_ids = b_ids.to(device)
      b_mask = b_mask.to(device)

      optimizer.zero_grad()
      logits = model(b_ids, b_mask)
      logits = rearrange(logits[:, :-1].contiguous(), 'b t d -> (b t) d')
      labels = b_ids[:, 1:].clone()
      labels[b_mask[:, 1:] == 0] = -100

      # Mask prompt if enabled
      if args.mask_prompt and 'prompt_lens' in batch:
        prompt_lens = batch['prompt_lens']
        for i, p_len in enumerate(prompt_lens):
          if p_len > 1:
            labels[i, :p_len - 1] = -100

      loss = F.cross_entropy(
        logits,
        labels.flatten(),
        ignore_index=-100,
        reduction='mean'
      )
      loss.backward()
      optimizer.step()

      train_loss += loss.item()
      num_batches += 1

    train_loss = train_loss / num_batches
    print(f"Epoch {epoch}: train loss :: {train_loss :.3f}.")
    print('Generating several output reasoning sequences...')
    
    model.eval()
    if len(held_out_esnli_dataset) > 0:
      batch = held_out_esnli_dataset[0]
      encoding = model.tokenizer(
        batch[1],
        return_tensors='pt',
        padding=True,
        truncation=True,
        max_length=512
      ).to(device)

      _, output = model.generate(
        encoding['input_ids'],
        temperature=args.temperature,
        top_p=args.top_p
      )
      print("Generated Sample:")
      print(output)
      print("\n\n")

    save_model(model, optimizer, args, os.path.join(args.output_dir, f'{epoch}_{args.filepath}'))

@torch.no_grad()
def generate_submission_reasonings(args):
  device = torch.device('cuda') if args.use_gpu else torch.device('cpu')
  saved = torch.load(os.path.join(args.output_dir, f'{args.epochs-1}_{args.filepath}'), weights_only=False)

  model = ESNLIGPT(saved['args'])
  model.load_state_dict(saved['model'])
  model = model.to(device)
  model.eval()

  held_out_esnli_dataset = ESNLIDataset(args.held_out_reasoning_path)
  print("Held-out examples:", len(held_out_esnli_dataset))

  generated_reasonings = []
  for batch in held_out_esnli_dataset:
    reasoning_id = batch[0]
    encoding = model.tokenizer(batch[1], return_tensors='pt', padding=False, truncation=True, max_length=512).to(device)

    _, generated_text = model.generate(
        encoding['input_ids'],
        temperature=args.temperature,
        top_p=args.top_p
    )

    full_reasoning = f'{generated_text}\n\n'
    generated_reasonings.append((reasoning_id, full_reasoning))

  os.makedirs(os.path.dirname(args.reasoning_out), exist_ok=True)
  with open(args.reasoning_out, "w+", encoding="utf-8") as f:
    f.write(f"--Generated Reasonings-- \n\n")
    for reasoning in generated_reasonings:
      f.write(f"\n{reasoning[0]}\n")
      f.write(reasoning[1])

def get_args():
  parser = argparse.ArgumentParser()
  parser.add_argument("--reasoning_path", type=str, default="data/esnli_small_train.txt")
  parser.add_argument("--held_out_reasoning_path", type=str, default="data/esnli_small_held_out.txt")
  parser.add_argument("--reasoning_out", type=str, default="outputs/generated_esnli.txt")
  parser.add_argument("--seed", type=int, default=11711)
  parser.add_argument("--epochs", type=int, default=3)
  parser.add_argument("--use_gpu", action='store_true')
  parser.add_argument("--mask_prompt", action='store_true', help="Whether to mask prompt in SFT loss")

  # Generation parameters.
  parser.add_argument("--temperature", type=float, help="softmax temperature.", default=0.7)
  parser.add_argument("--top_p", type=float, help="Cumulative probability distribution for nucleus sampling.",
                      default=0.9)

  parser.add_argument("--batch_size", help='The training batch size.', type=int, default=8)
  parser.add_argument("--lr", type=float, help="learning rate", default=1e-5)
  parser.add_argument("--model_size", type=str, help="The model size as specified on hugging face.",
                      choices=['gpt2', 'gpt2-medium', 'gpt2-large'], default='gpt2')
  parser.add_argument("--output_dir", type=str, default=".", help="Directory to save checkpoints (use /mnt/... on cluster).")
  parser.add_argument("--resume_epoch", type=int, default=-1, help="Resume training from this epoch (-1 = start fresh).")

  args = parser.parse_args()
  return args

def add_arguments(args):
  if args.model_size == 'gpt2':
    args.d = 768
    args.l = 12
    args.num_heads = 12
  elif args.model_size == 'gpt2-medium':
    args.d = 1024
    args.l = 24
    args.num_heads = 16
  elif args.model_size == 'gpt2-large':
    args.d = 1280
    args.l = 36
    args.num_heads = 20
  else:
    raise Exception(f'{args.model_size} is not supported.')
  return args

if __name__ == "__main__":
  args = get_args()
  args = add_arguments(args)
  args.filepath = f'{args.epochs}-{args.lr}-esnli.pt'  # Save path (basename only; output_dir is prepended at save/load time).
  os.makedirs(args.output_dir, exist_ok=True)
  seed_everything(args.seed)
  train(args)
  generate_submission_reasonings(args)
