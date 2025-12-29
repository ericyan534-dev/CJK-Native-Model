"""Unit tests for CNMBertTokenizer."""

import pytest
import tempfile
import json
from pathlib import Path

from transformers import BertTokenizer
from cnm_bert.tokenization import CNMBertTokenizer


class TestCNMBertTokenizer:
    """Test CNMBertTokenizer class."""

    @pytest.fixture
    def sample_tree_map(self):
        """Create sample IDS tree map."""
        return {
            "好": {"op": "⿰", "children": [{"leaf": "女"}, {"leaf": "子"}]},
            "草": {"op": "⿱", "children": [
                {"op": "⿰", "children": [{"leaf": "艹"}, {"leaf": "艹"}]},
                {"leaf": "早"}
            ]},
        }

    @pytest.fixture
    def tokenizer(self, sample_tree_map):
        """Create tokenizer with sample tree map."""
        # Create temporary tree file
        with tempfile.NamedTemporaryFile(mode="w", delete=False, encoding="utf-8", suffix=".json") as f:
            json.dump(sample_tree_map, f, ensure_ascii=False)
            tree_path = Path(f.name)

        try:
            # Create tokenizer (using bert-base-chinese vocabulary)
            tokenizer = CNMBertTokenizer.from_pretrained(
                "bert-base-chinese",
                struct_path=str(tree_path)
            )
            yield tokenizer
        finally:
            tree_path.unlink()

    def test_tokenizer_creation(self, tokenizer):
        """Test tokenizer creation."""
        assert tokenizer is not None
        assert hasattr(tokenizer, "char_to_tree")
        assert hasattr(tokenizer, "struct_vocab")
        assert len(tokenizer.struct_vocab) >= 2  # At least [NONE] and [UNK_STRUCT]

    def test_character_level_tokenization(self, tokenizer):
        """Test character-level tokenization for Chinese."""
        text = "今天天气很好"
        tokens = tokenizer.tokenize(text)

        # Should split into characters
        assert len(tokens) == len(text)
        for char, token in zip(text, tokens):
            assert char == token

    def test_encode_with_struct_ids(self, tokenizer):
        """Test encoding with structural IDs."""
        text = "好"
        encoding = tokenizer(text, return_tensors="pt", return_struct_ids=True)

        assert "input_ids" in encoding
        assert "struct_ids" in encoding
        assert "attention_mask" in encoding

        # Check shapes match
        assert encoding["input_ids"].shape == encoding["struct_ids"].shape

    def test_struct_ids_mapping(self, tokenizer):
        """Test structural ID mapping."""
        text = "好"
        encoding = tokenizer(text, return_struct_ids=True)

        struct_ids = encoding["struct_ids"][0]  # Remove batch dimension

        # [CLS] token should have struct_id=0 ([NONE])
        assert struct_ids[0] == 0

        # Character "好" should have non-zero struct_id (it has a tree)
        char_struct_id = struct_ids[1]
        assert char_struct_id > 1  # Not [NONE] or [UNK_STRUCT]

        # [SEP] token should have struct_id=0 ([NONE])
        assert struct_ids[-1] == 0

    def test_padding(self, tokenizer):
        """Test padding with struct_ids."""
        texts = ["好", "今天很好"]
        encoding = tokenizer(
            texts,
            padding=True,
            return_tensors="pt",
            return_struct_ids=True
        )

        # Check batch dimension
        assert encoding["input_ids"].shape[0] == 2
        assert encoding["struct_ids"].shape[0] == 2

        # Check same length
        assert encoding["input_ids"].shape[1] == encoding["struct_ids"].shape[1]

        # Check padding tokens have struct_id=0
        # Second sequence is shorter, so it should have padding
        seq_len = encoding["input_ids"].shape[1]
        # Padding tokens should have struct_id=0
        padding_mask = encoding["attention_mask"][0] == 0
        if padding_mask.any():
            assert (encoding["struct_ids"][0][padding_mask] == 0).all()

    def test_unknown_character(self, tokenizer):
        """Test handling of characters without IDS trees."""
        # Use a character not in tree_map
        text = "你"
        encoding = tokenizer(text, return_struct_ids=True)

        struct_ids = encoding["struct_ids"][0]
        char_struct_id = struct_ids[1]  # After [CLS]

        # Should be [UNK_STRUCT] (index 1) or [NONE] (index 0)
        assert char_struct_id <= 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
