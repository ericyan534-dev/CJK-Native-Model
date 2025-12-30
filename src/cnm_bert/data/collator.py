"""Data collator for Whole Word Masking (WWM) with jieba."""

import random
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import torch
import jieba
from transformers import PreTrainedTokenizerBase


@dataclass
class WWMDataCollator:
    """Data collator for Whole Word Masking (WWM).

    This collator performs Whole Word Masking using jieba for Chinese
    word segmentation. Instead of masking individual characters randomly,
    it masks entire words, which is more linguistically meaningful.

    Args:
        tokenizer: CNMBertTokenizer instance
        mlm_probability: Probability of masking tokens (default: 0.15)
        max_seq_length: Maximum sequence length (default: 512)
    """

    tokenizer: PreTrainedTokenizerBase
    mlm_probability: float = 0.15
    max_seq_length: int = 512

    def __post_init__(self):
        """Initialize jieba."""
        # Pre-load jieba dictionary for better performance
        jieba.initialize()

    def __call__(self, examples: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        """Collate batch of examples with WWM.

        Args:
            examples: List of examples, each with 'text' field

        Returns:
            Dictionary with input_ids, struct_ids, attention_mask, labels
        """
        # Extract texts - handle both dict and direct string formats
        texts = []
        for example in examples:
            if isinstance(example, dict):
                if "text" in example:
                    texts.append(example["text"])
                elif "input_ids" in example:
                    # Already tokenized - decode it
                    texts.append(self.tokenizer.decode(example["input_ids"], skip_special_tokens=True))
                else:
                    # Debug: print what keys are available
                    raise ValueError(f"Example dict missing 'text' or 'input_ids' key. Available keys: {list(example.keys())}")
            elif isinstance(example, str):
                texts.append(example)
            else:
                raise ValueError(f"Unexpected example type: {type(example)}, value: {example}")

        # Tokenize batch
        batch_encoding = self.tokenizer(
            texts,
            padding="max_length",
            truncation=True,
            max_length=self.max_seq_length,
            return_tensors="pt",
            return_struct_ids=True,
        )

        # Apply WWM to create labels
        input_ids = batch_encoding["input_ids"].clone()
        labels = input_ids.clone()

        # Process each sequence in batch
        for i, text in enumerate(texts):
            # Apply WWM masking
            input_ids[i], labels[i] = self._apply_wwm(
                text,
                input_ids[i],
                labels[i]
            )

        # Create output dictionary
        batch = {
            "input_ids": input_ids,
            "struct_ids": batch_encoding["struct_ids"],
            "attention_mask": batch_encoding["attention_mask"],
            "labels": labels,
        }

        # Add token_type_ids if present
        if "token_type_ids" in batch_encoding:
            batch["token_type_ids"] = batch_encoding["token_type_ids"]

        return batch

    def _apply_wwm(
        self,
        text: str,
        input_ids: torch.Tensor,
        labels: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply Whole Word Masking to a single sequence.

        Args:
            text: Original text
            input_ids: Token IDs (seq_len,)
            labels: Label IDs (seq_len,)

        Returns:
            Tuple of (masked_input_ids, labels)
        """
        # Segment text into words using jieba
        words = list(jieba.cut(text))

        # Convert to tokens
        tokens = self.tokenizer.tokenize(text)

        # Build word-to-token mapping
        word_boundaries = self._get_word_boundaries(words, tokens)

        # Select words to mask
        if len(word_boundaries) == 0:
            # No words to mask, return unchanged
            return input_ids, labels
        num_to_mask = max(1, int(len(word_boundaries) * self.mlm_probability))
        num_to_mask = min(num_to_mask, len(word_boundaries))  # Don't try to sample more than available
        words_to_mask = random.sample(range(len(word_boundaries)), num_to_mask)

        # Initialize labels with -100 (ignore in loss)
        labels[:] = -100

        # Mask selected words
        for word_idx in words_to_mask:
            start_pos, end_pos = word_boundaries[word_idx]

            # Adjust for [CLS] token
            start_pos += 1
            end_pos += 1

            # Ensure within bounds
            if end_pos >= len(input_ids):
                continue

            # Apply masking strategy (80% [MASK], 10% random, 10% unchanged)
            for pos in range(start_pos, end_pos):
                labels[pos] = input_ids[pos].clone()

                prob = random.random()
                if prob < 0.8:
                    # 80%: Replace with [MASK]
                    input_ids[pos] = self.tokenizer.mask_token_id
                elif prob < 0.9:
                    # 10%: Replace with random token
                    input_ids[pos] = random.randint(0, self.tokenizer.vocab_size - 1)
                # else: 10%: Keep original token

        return input_ids, labels

    def _get_word_boundaries(
        self,
        words: List[str],
        tokens: List[str]
    ) -> List[Tuple[int, int]]:
        """Get token position boundaries for each word.

        Args:
            words: List of segmented words
            tokens: List of character tokens

        Returns:
            List of (start_pos, end_pos) tuples for each word
        """
        boundaries = []
        token_idx = 0

        for word in words:
            # Count how many tokens this word spans
            word_tokens = list(word)  # Character-level for Chinese
            start_pos = token_idx
            end_pos = token_idx + len(word_tokens)

            boundaries.append((start_pos, end_pos))
            token_idx = end_pos

        return boundaries


@dataclass
class SimpleMLMCollator:
    """Simple MLM collator without word segmentation.

    This is a fallback collator that performs random token masking
    without considering word boundaries. Useful for debugging or
    when jieba is not available.

    Args:
        tokenizer: Tokenizer instance
        mlm_probability: Probability of masking (default: 0.15)
        max_seq_length: Maximum sequence length (default: 512)
    """

    tokenizer: PreTrainedTokenizerBase
    mlm_probability: float = 0.15
    max_seq_length: int = 512

    def __call__(self, examples: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        """Collate batch with simple random masking.

        Args:
            examples: List of examples with 'text' field

        Returns:
            Dictionary with input_ids, struct_ids, attention_mask, labels
        """
        # Extract texts
        texts = [example["text"] for example in examples]

        # Tokenize
        batch_encoding = self.tokenizer(
            texts,
            padding="max_length",
            truncation=True,
            max_length=self.max_seq_length,
            return_tensors="pt",
            return_struct_ids=True,
        )

        # Clone for masking
        input_ids = batch_encoding["input_ids"].clone()
        labels = input_ids.clone()

        # Create probability matrix for masking
        probability_matrix = torch.full(labels.shape, self.mlm_probability)

        # Don't mask special tokens
        special_tokens_mask = [
            self.tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True)
            for val in labels.tolist()
        ]
        probability_matrix.masked_fill_(torch.tensor(special_tokens_mask, dtype=torch.bool), value=0.0)

        # Don't mask padding
        padding_mask = labels.eq(self.tokenizer.pad_token_id)
        probability_matrix.masked_fill_(padding_mask, value=0.0)

        # Sample masked positions
        masked_indices = torch.bernoulli(probability_matrix).bool()

        # Set labels (only compute loss on masked tokens)
        labels[~masked_indices] = -100

        # 80% of time, replace with [MASK]
        indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked_indices
        input_ids[indices_replaced] = self.tokenizer.mask_token_id

        # 10% of time, replace with random token
        indices_random = torch.bernoulli(torch.full(labels.shape, 0.5)).bool() & masked_indices & ~indices_replaced
        random_words = torch.randint(self.tokenizer.vocab_size, labels.shape, dtype=torch.long)
        input_ids[indices_random] = random_words[indices_random]

        # 10% of time, keep original

        return {
            "input_ids": input_ids,
            "struct_ids": batch_encoding["struct_ids"],
            "attention_mask": batch_encoding["attention_mask"],
            "labels": labels,
        }
