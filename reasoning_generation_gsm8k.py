'''
Reasoning generation starter code.

Running:
  `python reasoning_generation.py --use_gpu`

trains your ReasoningGPT model and writes the required submission files.
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

from gpt_datasets import (
  ReasoningDataset,
)
from models.gpt2 import GPT2Model

from optimizer import AdamW

TQDM_DISABLE = False


# Fix the random seed.
def seed_everything(seed=11711):
  random.seed(seed)
  np.random.seed(seed)
  torch.manual_seed(seed)
  torch.cuda.manual_seed(seed)
  torch.cuda.manual_seed_all(seed)
  torch.backends.cudnn.benchmark = False
  torch.backends.cudnn.deterministic = True


class ReasoningGPT(nn.Module):
  """GPT-2 model for GSM8K reasoning generation."""

  def __init__(self, args):
    super().__init__()
    self.gpt = GPT2Model.from_pretrained(model=args.model_size, d=args.d, l=args.l, num_heads=args.num_heads)
    self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    self.tokenizer.pad_token = self.tokenizer.eos_token

    # By default, fine-tune the full model. TODO: this is maybe not ideal.
    for param in self.gpt.parameters():
      param.requires_grad = True

  def forward(self, input_ids, attention_mask):
    """
    Returns logits for every token position in the sequence (shape: [batch, seq_len, vocab_size]),
    enabling next-token prediction loss across the full reasoning chain.
    """
    ### YOUR CODE HERE

    # call the gpt model to get the hidden states for each token in the sequence. This is of size [batch_size, seq_len, hidden_size].  
    outputs = self.gpt(input_ids, attention_mask)

    # Get the hidden states for each token in the sequence. This is of size [batch_size, seq_len, hidden_size].
    hidden_states = outputs['last_hidden_state']

    # Get the logits for each token in the sequence. This is of size [batch_size, seq_len, vocab_size].
    logits = self.gpt.hidden_state_to_token(hidden_states)

    return logits
  
    #raise NotImplementedError


  def get_device(self):
    for param in self.gpt.parameters():
      return param.device

  @torch.no_grad()
  def generate(self, encoding, temperature=0.7, top_p=0.9, max_length=128):

    token_ids = encoding.to(self.get_device())
    attention_mask = torch.ones(
        token_ids.shape,
        dtype=torch.int64
    ).to(self.get_device())

    max_context_length = self.gpt.pos_embedding.num_embeddings
    available_tokens = max_context_length - token_ids.size(1)
    print(
      f"Prompt length: {token_ids.size(1)}, "
      f"available generation tokens: {available_tokens}"
    )

    if available_tokens <= 0:
        generated_output = self.tokenizer.decode(
            token_ids[0].cpu().numpy().tolist()
        )
        return token_ids, generated_output

    for _ in range(min(max_length, available_tokens)):
      # Forward pass to get logits
      if token_ids.size(1) >= max_context_length:
        break

      logits_sequence = self.forward(token_ids, attention_mask)
      logits_last_token = logits_sequence[:, -1, :] / temperature  # Apply temperature scaling

      # Convert logits to probabilities
      probs = torch.nn.functional.softmax(logits_last_token, dim=-1)

      
      # Top-p (nucleus) sampling
      sorted_probs, sorted_indices = torch.sort(probs, descending=True)
      cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
      top_p_mask = cumulative_probs <= top_p
      top_p_mask[..., 1:] = top_p_mask[..., :-1].clone()  # Shift mask right for proper thresholding
      top_p_mask[..., 0] = True  # Always include the highest probability token
      filtered_probs = sorted_probs * top_p_mask  # Zero out unlikely tokens
      filtered_probs /= filtered_probs.sum(dim=-1, keepdim=True)  # Normalize probabilities

      # Sample from filtered distribution
      sampled_index = torch.multinomial(filtered_probs, 1)
      sampled_token = sorted_indices.gather(dim=-1, index=sampled_index)

      # Stop if end-of-sequence token is reached
      if sampled_token.item() == self.tokenizer.eos_token_id:
        print("EOS generated at length", token_ids.size(1))
        break

      # Append sampled token
      token_ids = torch.cat([token_ids, sampled_token], dim=1)
      attention_mask = torch.cat(
        [attention_mask, torch.ones((1, 1), dtype=torch.int64).to(self.get_device())], dim=1
      )

    generated_output = self.tokenizer.decode(token_ids[0].cpu().numpy().tolist())

    print(
      "Prompt length:", encoding.size(1),
      "Final length:", token_ids.size(1),
      "Generated tokens:", token_ids.size(1) - encoding.size(1)
    )

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
  """Train GPT-2 for reasoning generation on GSM8K."""
  device = torch.device('cuda') if args.use_gpu else torch.device('cpu')
  # Create the data and its corresponding datasets and dataloader.
  reasoning_dataset = ReasoningDataset(args.reasoning_path, mask_prompt=args.mask_prompt)
  print(reasoning_dataset.examples[0])
  reasoning_dataloader = DataLoader(reasoning_dataset, shuffle=True, batch_size=args.batch_size,
                                    collate_fn=reasoning_dataset.collate_fn)

  # Held-out dataset: Question + "Reasoning:" prompt only, no gold answer.
  held_out_reasoning_dataset = ReasoningDataset(args.held_out_reasoning_path)
  print("Held-out examples:", len(held_out_reasoning_dataset))

  model = ReasoningGPT(args)
  model = model.to(device)

  lr = args.lr
  optimizer = AdamW(model.parameters(), lr=lr)

  # Run for the specified number of epochs.
  for epoch in range(args.epochs):
    model.train()
    train_loss = 0
    num_batches = 0

    for batch in tqdm(reasoning_dataloader, desc=f'train-{epoch}', disable=TQDM_DISABLE):
      # Get the input and move it to the gpu (I do not recommend training this model on CPU).
      b_ids, b_mask = batch['token_ids'], batch['attention_mask']
      b_ids = b_ids.to(device)
      b_mask = b_mask.to(device)

      # Compute the loss, gradients, and update the model's parameters.
      optimizer.zero_grad()
      logits = model(b_ids, b_mask)
      logits = rearrange(logits[:, :-1].contiguous(), 'b t d -> (b t) d')  # Ignore the last prediction in the sequence.
      labels = b_ids[:, 1:].clone()
      labels[b_mask[:, 1:] == 0] = -100  # Set padding token labels to -100 so they are ignored in the loss.

      # Prompt-loss masking: train only on the reasoning completion, not the
      # question tokens. labels are shifted by 1, so the boundary is p_len - 1.
      if args.mask_prompt and 'prompt_lens' in batch:
        for i, p_len in enumerate(batch['prompt_lens']):
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
    batch = held_out_reasoning_dataset[0]
    ids = model.tokenizer(batch[1])["input_ids"]
    print("Original token length:", len(ids))
    encoding = model.tokenizer(
      batch[1],
      return_tensors='pt',
      padding=True,
      truncation=True,
      max_length=512
    ).to(device)

    output = model.generate(
      encoding['input_ids'],
      temperature=args.temperature,
      top_p=args.top_p
    )

    print(output[1])

    print("\n\n")

    # TODO: consider early stopping to prevent overfitting.
    save_model(model, optimizer, args, f'{epoch}_{args.filepath}')


@torch.no_grad()
def generate_submission_reasonings(args):
  device = torch.device('cuda') if args.use_gpu else torch.device('cpu')
  saved = torch.load(f'{args.epochs-1}_{args.filepath}', weights_only=False)

  model = ReasoningGPT(saved['args'])
  model.load_state_dict(saved['model'])
  model = model.to(device)
  model.eval()

  # Create the held-out dataset: these only have the first 3 lines. Your job is to fill in the rest!
  held_out_reasoning_dataset = ReasoningDataset(args.held_out_reasoning_path)
  print("Held-out examples:", len(held_out_reasoning_dataset))

  generated_reasonings = []
  for batch in held_out_reasoning_dataset:
    reasoning_id = batch[0]
    encoding = model.tokenizer(batch[1], return_tensors='pt', padding=False, truncation=True, max_length=900).to(device)

    token_ids, generated_text = model.generate(
        encoding['input_ids'],
        temperature=args.temperature,
        top_p=args.top_p
    )

    print(
      "Prompt length:",
      encoding['input_ids'].size(1),
      "Final length:",
      token_ids.size(1),
      "Generated tokens:",
      token_ids.size(1) - encoding['input_ids'].size(1)
    )

    full_reasoning = f'{generated_text}\n\n'

    generated_reasonings.append(
        (reasoning_id, full_reasoning)
    )

    print(f'{generated_text}\n\n')

  os.makedirs(
    os.path.dirname(args.reasoning_out),
    exist_ok=True
  )

  with open(args.reasoning_out, "w+") as f:
    f.write(f"--Generated Reasonings-- \n\n")
    for reasoning in generated_reasonings:
      f.write(f"\n{reasoning[0]}\n")
      f.write(reasoning[1])


def get_args():
  parser = argparse.ArgumentParser()

  parser.add_argument("--reasoning_path", type=str, default="data/gsm8k_small_train.txt")
  parser.add_argument("--held_out_reasoning_path", type=str, default="data/gsm8k_small_held_out.txt")
  parser.add_argument("--reasoning_out", type=str, default="outputs/generated_reasoning.txt")

  parser.add_argument("--seed", type=int, default=11711)
  parser.add_argument("--epochs", type=int, default=10)
  parser.add_argument("--use_gpu", action='store_true')
  parser.add_argument("--mask_prompt", action='store_true',
                      help="Train only on the reasoning completion (mask question tokens).")

  # Generation parameters.
  parser.add_argument("--temperature", type=float, help="softmax temperature.", default=0.7)
  parser.add_argument("--top_p", type=float, help="Cumulative probability distribution for nucleus sampling.",
                      default=0.9)

  parser.add_argument("--batch_size", help='The training batch size.', type=int, default=8)
  parser.add_argument("--lr", type=float, help="learning rate", default=1e-5)
  parser.add_argument("--model_size", type=str, help="The model size as specified on hugging face.",
                      choices=['gpt2', 'gpt2-medium', 'gpt2-large'], default='gpt2')

  args = parser.parse_args()
  return args


def add_arguments(args):
  """Add arguments that are deterministic on model size."""
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
  args.filepath = f'{args.epochs}-{args.lr}-reasoning.pt'  # Save path.
  seed_everything(args.seed)  # Fix the seed for reproducibility.
  train(args)
  generate_submission_reasonings(args)