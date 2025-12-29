"""CNM-BERT tokenizer with structural information.

This tokenizer extends BertTokenizer to provide struct_ids that map
tokens to their IDS tree structures.
"""

import json
from pathlib import Path
from typing import Dict, List, Optional, Union

import torch
from transformers import BertTokenizer


def _is_cjk_character(char: str) -> bool:
    """Check if character is CJK Unified Ideograph.

    Args:
        char: Character to check

    Returns:
        True if CJK character
    """
    if len(char) != 1:
        return False
    cp = ord(char)
    return (
        0x4E00 <= cp <= 0x9FFF or      # CJK Unified Ideographs
        0x3400 <= cp <= 0x4DBF or      # CJK Extension A
        0x20000 <= cp <= 0x2A6DF or    # CJK Extension B
        0x2A700 <= cp <= 0x2B73F or    # CJK Extension C
        0x2B740 <= cp <= 0x2B81F or    # CJK Extension D
        0x2B820 <= cp <= 0x2CEAF or    # CJK Extension E
        0x2CEB0 <= cp <= 0x2EBEF       # CJK Extension F
    )


class CNMBertTokenizer(BertTokenizer):
    """Tokenizer for CNM-BERT with structural information.

    This tokenizer extends BertTokenizer to:
    1. Enforce character-level tokenization for Chinese
    2. Provide struct_ids that map to IDS tree structures
    3. Cache structural information for efficient lookup

    Args:
        *args: Arguments passed to BertTokenizer
        struct_path: Path to char_to_ids_tree.json file
        **kwargs: Keyword arguments passed to BertTokenizer
    """

    def __init__(
        self,
        *args,
        struct_path: Optional[str] = None,
        **kwargs
    ):
        super().__init__(*args, **kwargs)

        # Load structural tree map
        if struct_path:
            struct_path = Path(struct_path)
            if struct_path.exists():
                with open(struct_path, "r", encoding="utf-8") as f:
                    self.char_to_tree = json.load(f)
            else:
                raise ValueError(f"Structural tree file not found: {struct_path}")
        else:
            self.char_to_tree = {}

        # Build struct vocabulary
        # Index 0: [NONE] (for special tokens)
        # Index 1: [UNK_STRUCT] (for unknown characters)
        # Index 2+: characters with IDS trees
        self.struct_vocab = ["[NONE]", "[UNK_STRUCT]"] + sorted(self.char_to_tree.keys())
        self.struct_index_by_char = {ch: idx for idx, ch in enumerate(self.struct_vocab)}
        self.struct_index_to_char = list(self.struct_vocab)

    def _tokenize(self, text: str, **kwargs) -> List[str]:
        """Tokenize text character-by-character for Chinese.

        For Chinese text, we override BERT's WordPiece tokenization
        to enforce character-level splitting.

        Args:
            text: Input text
            **kwargs: Additional arguments

        Returns:
            List of tokens (characters)
        """
        output_tokens = []

        for char in text:
            if _is_cjk_character(char):
                # Character-level for CJK
                if char in self.vocab:
                    output_tokens.append(char)
                else:
                    output_tokens.append(self.unk_token)
            else:
                # Use standard tokenization for non-CJK
                # (punctuation, English, numbers, etc.)
                if char.strip():  # Skip whitespace
                    if char in self.vocab:
                        output_tokens.append(char)
                    else:
                        output_tokens.append(self.unk_token)

        return output_tokens

    def convert_tokens_to_struct_ids(self, tokens: List[str]) -> List[int]:
        """Convert tokens to structural indices.

        Args:
            tokens: List of tokens

        Returns:
            List of structural indices
        """
        struct_ids = []
        for token in tokens:
            # Special tokens get index 0 ([NONE])
            if token in self.all_special_tokens:
                struct_ids.append(0)
            # Known characters with IDS trees
            elif token in self.struct_index_by_char:
                struct_ids.append(self.struct_index_by_char[token])
            # Unknown structural characters get index 1 ([UNK_STRUCT])
            else:
                struct_ids.append(1)

        return struct_ids

    def encode_plus(
        self,
        text: Union[str, List[str]],
        text_pair: Optional[Union[str, List[str]]] = None,
        add_special_tokens: bool = True,
        padding: Union[bool, str] = False,
        truncation: Union[bool, str] = False,
        max_length: Optional[int] = None,
        stride: int = 0,
        return_tensors: Optional[str] = None,
        return_struct_ids: bool = True,
        **kwargs
    ) -> Dict[str, Union[List[int], torch.Tensor]]:
        """Encode text with structural information.

        Args:
            text: Input text or list of texts
            text_pair: Optional second sequence (for sentence pairs)
            add_special_tokens: Whether to add [CLS], [SEP]
            padding: Padding strategy
            truncation: Truncation strategy
            max_length: Maximum sequence length
            stride: Stride for splitting long sequences
            return_tensors: Return type ('pt' for PyTorch)
            return_struct_ids: Whether to include struct_ids
            **kwargs: Additional arguments

        Returns:
            Dictionary with input_ids, attention_mask, struct_ids, etc.
        """
        # Call parent encode_plus
        encoding = super().encode_plus(
            text=text,
            text_pair=text_pair,
            add_special_tokens=add_special_tokens,
            padding=padding,
            truncation=truncation,
            max_length=max_length,
            stride=stride,
            return_tensors=None,  # Get lists first
            **kwargs
        )

        # Add struct_ids if requested
        if return_struct_ids:
            # Convert input_ids back to tokens
            tokens = self.convert_ids_to_tokens(encoding["input_ids"])
            struct_ids = self.convert_tokens_to_struct_ids(tokens)
            encoding["struct_ids"] = struct_ids

        # Convert to tensors if requested
        if return_tensors == "pt":
            for key in encoding:
                if isinstance(encoding[key], list):
                    encoding[key] = torch.tensor([encoding[key]])

        return encoding

    def __call__(
        self,
        text: Union[str, List[str], List[List[str]]],
        text_pair: Optional[Union[str, List[str], List[List[str]]]] = None,
        add_special_tokens: bool = True,
        padding: Union[bool, str] = False,
        truncation: Union[bool, str] = False,
        max_length: Optional[int] = None,
        stride: int = 0,
        return_tensors: Optional[str] = None,
        return_struct_ids: bool = True,
        **kwargs
    ) -> Dict[str, Union[List[int], torch.Tensor]]:
        """Tokenize and encode text (main entry point).

        Args:
            text: Input text(s)
            text_pair: Optional second sequence(s)
            add_special_tokens: Whether to add [CLS], [SEP]
            padding: Padding strategy
            truncation: Truncation strategy
            max_length: Maximum sequence length
            stride: Stride for splitting
            return_tensors: Return type ('pt' for PyTorch)
            return_struct_ids: Whether to include struct_ids
            **kwargs: Additional arguments

        Returns:
            Dictionary with input_ids, attention_mask, struct_ids, etc.
        """
        # Handle batch inputs
        is_batched = isinstance(text, (list, tuple)) and text and isinstance(text[0], (list, tuple, str))

        if is_batched:
            # Batch processing
            encodings = {
                "input_ids": [],
                "attention_mask": [],
                "token_type_ids": [],
            }
            if return_struct_ids:
                encodings["struct_ids"] = []

            for i in range(len(text)):
                single_text = text[i]
                single_text_pair = text_pair[i] if text_pair else None

                single_encoding = self.encode_plus(
                    text=single_text,
                    text_pair=single_text_pair,
                    add_special_tokens=add_special_tokens,
                    padding=False,  # Batch padding done later
                    truncation=truncation,
                    max_length=max_length,
                    stride=stride,
                    return_tensors=None,
                    return_struct_ids=return_struct_ids,
                    **kwargs
                )

                for key in single_encoding:
                    encodings[key].append(single_encoding[key])

            # Apply batch padding
            if padding:
                encodings = self.pad(
                    encodings,
                    padding=padding,
                    max_length=max_length,
                    return_tensors=return_tensors
                )
            elif return_tensors == "pt":
                # Convert to tensors
                for key in encodings:
                    encodings[key] = torch.tensor(encodings[key])

            return encodings

        else:
            # Single sequence
            return self.encode_plus(
                text=text,
                text_pair=text_pair,
                add_special_tokens=add_special_tokens,
                padding=padding,
                truncation=truncation,
                max_length=max_length,
                stride=stride,
                return_tensors=return_tensors,
                return_struct_ids=return_struct_ids,
                **kwargs
            )

    def pad(
        self,
        encoded_inputs: Dict[str, List[List[int]]],
        padding: Union[bool, str] = True,
        max_length: Optional[int] = None,
        pad_to_multiple_of: Optional[int] = None,
        return_attention_mask: Optional[bool] = None,
        return_tensors: Optional[str] = None,
        verbose: bool = True,
    ) -> Dict[str, Union[List[List[int]], torch.Tensor]]:
        """Pad encoded inputs (including struct_ids).

        Args:
            encoded_inputs: Dictionary with input_ids, struct_ids, etc.
            padding: Padding strategy
            max_length: Maximum length
            pad_to_multiple_of: Pad to multiple of this value
            return_attention_mask: Whether to return attention mask
            return_tensors: Return type ('pt' for PyTorch)
            verbose: Whether to print warnings

        Returns:
            Padded dictionary
        """
        # Call parent padding for standard fields
        padded = super().pad(
            encoded_inputs={k: v for k, v in encoded_inputs.items() if k != "struct_ids"},
            padding=padding,
            max_length=max_length,
            pad_to_multiple_of=pad_to_multiple_of,
            return_attention_mask=return_attention_mask,
            return_tensors=None,  # Convert later
            verbose=verbose,
        )

        # Pad struct_ids if present
        if "struct_ids" in encoded_inputs:
            struct_ids = encoded_inputs["struct_ids"]

            # Determine target length
            if max_length is not None:
                target_length = max_length
            else:
                target_length = max(len(ids) for ids in struct_ids)

            # Pad each sequence
            padded_struct_ids = []
            for ids in struct_ids:
                padding_length = target_length - len(ids)
                padded_ids = ids + [0] * padding_length  # Pad with 0 ([NONE])
                padded_struct_ids.append(padded_ids)

            padded["struct_ids"] = padded_struct_ids

        # Convert to tensors if requested
        if return_tensors == "pt":
            for key in padded:
                if isinstance(padded[key], list):
                    padded[key] = torch.tensor(padded[key])

        return padded

    def save_pretrained(self, save_directory: Union[str, Path], **kwargs):
        """Save tokenizer and structural information.

        Args:
            save_directory: Directory to save tokenizer
            **kwargs: Additional arguments
        """
        super().save_pretrained(save_directory, **kwargs)

        # Save structural information
        save_directory = Path(save_directory)
        struct_file = save_directory / "char_to_ids_tree.json"

        with open(struct_file, "w", encoding="utf-8") as f:
            json.dump(self.char_to_tree, f, ensure_ascii=False, indent=2)

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *args, **kwargs):
        """Load tokenizer from pretrained.

        Args:
            pretrained_model_name_or_path: Model name or path
            *args: Additional arguments
            **kwargs: Keyword arguments

        Returns:
            Loaded tokenizer
        """
        # Check for struct_path in directory
        model_path = Path(pretrained_model_name_or_path)
        if model_path.is_dir():
            struct_file = model_path / "char_to_ids_tree.json"
            if struct_file.exists() and "struct_path" not in kwargs:
                kwargs["struct_path"] = str(struct_file)

        return super().from_pretrained(pretrained_model_name_or_path, *args, **kwargs)
