"""Whole Word Masking dataset and collator for CNM-BERT."""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional

import jieba
import torch
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
        input_ids = torch.cat([item["input_ids"] for item in tokenized], dim=0)
        struct_trees = torch.cat([item["struct_ids"] for item in tokenized], dim=0)
        struct_indices = torch.cat([item["struct_indices"] for item in tokenized], dim=0)
        labels = input_ids.clone()

        word_boundaries = [self._segment(text) for text in batch]
        masked_indices = torch.zeros_like(input_ids, dtype=torch.bool)
        flat_offset = 0
        for tokens, boundaries in zip(tokenized, word_boundaries):
            seq_len = tokens["input_ids"].size(1)
            mask_positions = self._choose_words(boundaries)
            for start, end in mask_positions:
                # Offset by 1 to skip [CLS] and ensure we do not mask [SEP].
                start_idx = max(1, start + 1)
                end_idx = min(seq_len - 1, end + 1)
                if start_idx < end_idx:
                    masked_indices[flat_offset, start_idx:end_idx] = True
            flat_offset += 1

        labels[~masked_indices] = -100
        # Apply masking
        probability_matrix = torch.full(labels.shape, self.mlm_probability)
        special_tokens_mask = torch.zeros_like(labels, dtype=torch.bool)
        probability_matrix.masked_fill_(special_tokens_mask, value=0.0)
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
