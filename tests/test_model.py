"""Unit tests for CNM model."""

import pytest
import tempfile
import json
import torch
from pathlib import Path

from cnm_bert.modeling import CNMConfig, CNMModel, CNMForMaskedLM, TreeMLPEncoder


class TestTreeMLPEncoder:
    """Test TreeMLPEncoder class."""

    @pytest.fixture
    def sample_tree_map(self):
        """Sample IDS tree map."""
        return {
            "好": {"op": "⿰", "children": [{"leaf": "女"}, {"leaf": "子"}]},
            "草": {"op": "⿱", "children": [
                {"op": "⿰", "children": [{"leaf": "艹"}, {"leaf": "艹"}]},
                {"leaf": "早"}
            ]},
        }

    @pytest.fixture
    def encoder(self, sample_tree_map):
        """Create tree encoder."""
        struct_vocab = ["[NONE]", "[UNK_STRUCT]", "好", "草"]
        return TreeMLPEncoder(
            tree_map=sample_tree_map,
            struct_index_to_char=struct_vocab,
            struct_dim=64
        )

    def test_encoder_creation(self, encoder):
        """Test encoder creation."""
        assert encoder is not None
        assert encoder.struct_dim == 64
        assert len(encoder.component_vocab) > 0
        assert len(encoder.operator_vocab) > 0

    def test_forward_pass(self, encoder):
        """Test forward pass."""
        # Create dummy struct_ids
        batch_size, seq_len = 2, 5
        struct_ids = torch.randint(0, 4, (batch_size, seq_len))

        # Forward pass
        output = encoder(struct_ids)

        # Check output shape
        assert output.shape == (batch_size, seq_len, encoder.struct_dim)

    def test_unique_batching(self, encoder):
        """Test that unique character batching works."""
        # Create struct_ids with repeated characters
        struct_ids = torch.tensor([
            [0, 2, 2, 3, 0],  # [NONE], 好, 好, 草, [NONE]
            [0, 3, 2, 2, 0],  # [NONE], 草, 好, 好, [NONE]
        ])

        # Should process 好 and 草 once each
        output = encoder(struct_ids)

        assert output.shape == (2, 5, encoder.struct_dim)
        # Same character should have same embedding
        assert torch.allclose(output[0, 1], output[0, 2])  # Both are 好
        assert torch.allclose(output[0, 1], output[1, 2])  # Same 好 across batches


class TestCNMModel:
    """Test CNMModel class."""

    @pytest.fixture
    def config(self):
        """Create test config."""
        # Create temporary tree file
        tree_map = {
            "好": {"op": "⿰", "children": [{"leaf": "女"}, {"leaf": "子"}]},
        }

        with tempfile.NamedTemporaryFile(mode="w", delete=False, encoding="utf-8", suffix=".json") as f:
            json.dump(tree_map, f, ensure_ascii=False)
            tree_path = Path(f.name)

        config = CNMConfig(
            vocab_size=100,
            hidden_size=128,
            num_hidden_layers=2,
            num_attention_heads=2,
            intermediate_size=256,
            struct_dim=64,
            tree_path=str(tree_path),
            max_position_embeddings=128,
        )

        yield config

        # Cleanup
        tree_path.unlink()

    def test_model_creation(self, config):
        """Test model creation."""
        model = CNMModel(config)
        assert model is not None
        assert isinstance(model.tree_encoder, TreeMLPEncoder)

    def test_forward_pass(self, config):
        """Test model forward pass."""
        model = CNMModel(config)
        model.eval()

        # Create dummy inputs
        batch_size, seq_len = 2, 10
        input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))
        struct_ids = torch.zeros(batch_size, seq_len, dtype=torch.long)
        attention_mask = torch.ones(batch_size, seq_len)

        # Forward pass
        with torch.no_grad():
            outputs = model(
                input_ids=input_ids,
                struct_ids=struct_ids,
                attention_mask=attention_mask,
            )

        # Check outputs
        assert outputs.last_hidden_state.shape == (batch_size, seq_len, config.hidden_size)

    def test_gradient_flow(self, config):
        """Test that gradients flow through model."""
        model = CNMModel(config)
        model.train()

        # Create dummy inputs
        batch_size, seq_len = 2, 10
        input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))
        struct_ids = torch.zeros(batch_size, seq_len, dtype=torch.long)
        attention_mask = torch.ones(batch_size, seq_len)

        # Forward pass
        outputs = model(
            input_ids=input_ids,
            struct_ids=struct_ids,
            attention_mask=attention_mask,
        )

        # Compute dummy loss
        loss = outputs.last_hidden_state.sum()

        # Backward pass
        loss.backward()

        # Check that gradients exist
        for name, param in model.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"No gradient for {name}"


class TestCNMForMaskedLM:
    """Test CNMForMaskedLM class."""

    @pytest.fixture
    def model(self):
        """Create MLM model."""
        tree_map = {
            "好": {"op": "⿰", "children": [{"leaf": "女"}, {"leaf": "子"}]},
        }

        with tempfile.NamedTemporaryFile(mode="w", delete=False, encoding="utf-8", suffix=".json") as f:
            json.dump(tree_map, f, ensure_ascii=False)
            tree_path = Path(f.name)

        config = CNMConfig(
            vocab_size=100,
            hidden_size=128,
            num_hidden_layers=2,
            num_attention_heads=2,
            intermediate_size=256,
            struct_dim=64,
            tree_path=str(tree_path),
            max_position_embeddings=128,
        )

        model = CNMForMaskedLM(config)

        yield model

        # Cleanup
        tree_path.unlink()

    def test_mlm_forward_with_labels(self, model):
        """Test MLM forward pass with labels."""
        model.eval()

        batch_size, seq_len = 2, 10
        input_ids = torch.randint(0, model.config.vocab_size, (batch_size, seq_len))
        struct_ids = torch.zeros(batch_size, seq_len, dtype=torch.long)
        attention_mask = torch.ones(batch_size, seq_len)
        labels = input_ids.clone()
        labels[:, 5:] = -100  # Mask out second half

        with torch.no_grad():
            outputs = model(
                input_ids=input_ids,
                struct_ids=struct_ids,
                attention_mask=attention_mask,
                labels=labels,
            )

        # Check that loss is computed
        assert outputs.loss is not None
        assert outputs.loss.item() > 0

        # Check logits shape
        assert outputs.logits.shape == (batch_size, seq_len, model.config.vocab_size)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
