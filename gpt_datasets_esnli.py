#!/usr/bin/env python3
import re
import torch
from torch.utils.data import Dataset
from transformers import GPT2Tokenizer

class ESNLIDataset(Dataset):
    def __init__(self, file_path, mask_prompt=False):
        self.max_length = 512
        self.mask_prompt = mask_prompt
        self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        self.tokenizer.pad_token = self.tokenizer.eos_token

        self.examples = self._load_examples(file_path)
        print(f"Loaded e-SNLI examples: {len(self.examples)}")

        self.has_eos = "<|endoftext|>" in "".join(self.examples[:10]) # simple check

    def _load_examples(self, file_path):
        with open(file_path, 'r', encoding='utf-8') as f:
            text = f.read()

        # Split examples by <|endoftext|>
        if "<|endoftext|>" in text:
            examples = [
                e.strip() + self.tokenizer.eos_token
                for e in text.split("<|endoftext|>")
                if e.strip()
            ]
        else:
            examples = re.split(r'\n\s*\d+\s*\n', text)
            examples = [e.strip() for e in examples if e.strip()]

        return examples

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return (idx, self.examples[idx])

    def collate_fn(self, all_data):
        idx = [example[0] for example in all_data]
        texts = [example[1] for example in all_data]

        encoding = self.tokenizer(
            texts,
            return_tensors='pt',
            padding=True,
            truncation=True,
            max_length=self.max_length
        )

        prompt_lens = []
        if self.mask_prompt:
            for text in texts:
                # Find prompt up to "Explanation:\n"
                prompt_part = text
                if "Explanation:\n" in text:
                    prompt_part = text.split("Explanation:\n")[0] + "Explanation:\n"
                
                # Tokenize prompt to get its length
                prompt_enc = self.tokenizer(prompt_part)
                prompt_lens.append(len(prompt_enc["input_ids"]))

        batched_data = {
            'token_ids': torch.LongTensor(encoding['input_ids']),
            'attention_mask': torch.LongTensor(encoding['attention_mask']),
            'sent_ids': idx
        }
        if self.mask_prompt:
            batched_data['prompt_lens'] = prompt_lens

        return batched_data
