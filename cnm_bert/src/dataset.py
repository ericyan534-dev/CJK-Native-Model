"""Whole Word Masking dataset and collator for CNM-BERT."""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional

import jieba
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset

from .tokenization_cnm import CNMBertTokenizer


class CNMTextDataset(Dataset):
    """A minimal dataset that reads raw lines of text."""

    def __init__(self, path: str):
        with open(path, "r", encoding="utf-8") as f:
            self.samples = [line.strip() for line in f if line.strip()]

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> str:
        return self.samples[idx]


@dataclass
class DataCollatorForWWM:
    tokenizer: CNMBertTokenizer
    mlm_probability: float = 0.15

    def __call__(self, batch: List[str]) -> Dict[str, torch.Tensor]:
        tokenized = [
            self.tokenizer(list(text), is_split_into_words=True, return_tensors="pt", return_struct_ids=True)
            for text in batch
        ]

        input_ids = pad_sequence(
            [item["input_ids"].squeeze(0) for item in tokenized],
            batch_first=True,
            padding_value=self.tokenizer.pad_token_id,
        )
        struct_trees = pad_sequence(
            [item["struct_ids"].squeeze(0) for item in tokenized],
            batch_first=True,
            padding_value=0,
        )
        struct_indices = pad_sequence(
            [item["struct_indices"].squeeze(0) for item in tokenized],
            batch_first=True,
            padding_value=0,
        )
        labels = input_ids.clone()

        word_boundaries = [self._segment(text) for text in batch]
        masked_indices = torch.zeros_like(input_ids, dtype=torch.bool)
        for row_idx, (tokens, boundaries) in enumerate(zip(tokenized, word_boundaries)):
            # Account for [CLS] at position 0 and [SEP] at the final position.
            seq_len = tokens["input_ids"].size(1)
            max_maskable = max(seq_len - 1, 1)
            mask_positions = self._choose_words(boundaries)
            for start, end in mask_positions:
                adj_start = min(start + 1, max_maskable)
                adj_end = min(end + 1, max_maskable)
                if adj_start < adj_end:
                    masked_indices[row_idx, adj_start:adj_end] = True

        labels[~masked_indices] = -100
        # Apply masking
        # 80% [MASK]
        mask_token_id = self.tokenizer.mask_token_id
        replace_with_mask = masked_indices & (torch.rand_like(input_ids.float()) < 0.8)
        input_ids[replace_with_mask] = mask_token_id
        # 10% random
        random_replace = masked_indices & (~replace_with_mask) & (torch.rand_like(input_ids.float()) < 0.5)
        random_words = torch.randint(len(self.tokenizer), labels.shape, dtype=torch.long)
        input_ids[random_replace] = random_words[random_replace]
        # 10% original left unchanged
        attention_mask = (input_ids != self.tokenizer.pad_token_id).long()
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "struct_indices": struct_indices,
            "struct_ids": struct_trees,
        }

    def _segment(self, text: str) -> List[tuple]:
        words = jieba.lcut(text)
        boundaries: List[tuple] = []
        cursor = 0
        for w in words:
            start, end = cursor, cursor + len(w)
            boundaries.append((start, end))
            cursor = end
        return boundaries

    def _choose_words(self, boundaries: List[tuple]) -> List[tuple]:
        candidates = [b for b in boundaries if random.random() < self.mlm_probability]
        return candidates


__all__ = ["CNMTextDataset", "DataCollatorForWWM"]
