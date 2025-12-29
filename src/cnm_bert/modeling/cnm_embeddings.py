"""CNM embedding layer that fuses BERT embeddings with structural information."""

import torch
import torch.nn as nn
from typing import Optional

from .tree_encoder import TreeMLPEncoder
from .configuration_cnm import CNMConfig


class CNMEmbeddings(nn.Module):
    """CNM embedding layer combining BERT and structural embeddings.

    This module extends standard BERT embeddings by fusing them with
    structural information from the Tree-MLP encoder. The fusion can
    be done via concatenation, addition, or gating.

    Args:
        config: CNM configuration
        tree_encoder: TreeMLPEncoder instance
    """

    def __init__(self, config: CNMConfig, tree_encoder: TreeMLPEncoder):
        super().__init__()
        self.config = config

        # Standard BERT embedding components
        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size, padding_token_id=config.pad_token_id)
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        self.token_type_embeddings = nn.Embedding(config.type_vocab_size, config.hidden_size)

        # Structural encoder
        self.tree_encoder = tree_encoder

        # Fusion layer
        self.fusion_strategy = config.fusion_strategy
        if self.fusion_strategy == "concat":
            # Concatenate and project back to hidden_size
            self.fusion_proj = nn.Linear(config.hidden_size + config.struct_dim, config.hidden_size)
        elif self.fusion_strategy == "add":
            # Add after projecting struct_dim to hidden_size
            self.struct_proj = nn.Linear(config.struct_dim, config.hidden_size)
        elif self.fusion_strategy == "gate":
            # Gated fusion
            self.struct_proj = nn.Linear(config.struct_dim, config.hidden_size)
            self.gate = nn.Linear(config.hidden_size + config.struct_dim, config.hidden_size)
        else:
            raise ValueError(f"Unknown fusion strategy: {self.fusion_strategy}")

        # Layer normalization and dropout
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.fusion_dropout)

        # Register buffer for position_ids (same as BERT)
        self.register_buffer(
            "position_ids",
            torch.arange(config.max_position_embeddings).expand((1, -1)),
            persistent=False
        )
        self.register_buffer(
            "token_type_ids",
            torch.zeros(self.position_ids.size(), dtype=torch.long),
            persistent=False
        )

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        struct_ids: Optional[torch.LongTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values_length: int = 0,
    ) -> torch.Tensor:
        """Forward pass combining BERT and structural embeddings.

        Args:
            input_ids: Token IDs (batch_size, seq_len)
            struct_ids: Structural indices (batch_size, seq_len)
            token_type_ids: Segment IDs (batch_size, seq_len)
            position_ids: Position IDs (batch_size, seq_len)
            past_key_values_length: Length of past key values (for generation)

        Returns:
            Fused embeddings of shape (batch_size, seq_len, hidden_size)
        """
        if input_ids is not None:
            input_shape = input_ids.size()
        else:
            raise ValueError("input_ids must be provided")

        seq_length = input_shape[1]
        batch_size = input_shape[0]

        # Get position IDs
        if position_ids is None:
            position_ids = self.position_ids[:, past_key_values_length : seq_length + past_key_values_length]

        # Get token type IDs
        if token_type_ids is None:
            if hasattr(self, "token_type_ids"):
                buffered_token_type_ids = self.token_type_ids[:, :seq_length]
                buffered_token_type_ids_expanded = buffered_token_type_ids.expand(batch_size, seq_length)
                token_type_ids = buffered_token_type_ids_expanded
            else:
                token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=self.position_ids.device)

        # Standard BERT embeddings
        word_embeddings = self.word_embeddings(input_ids)
        position_embeddings = self.position_embeddings(position_ids)
        token_type_embeddings = self.token_type_embeddings(token_type_ids)

        bert_embeddings = word_embeddings + position_embeddings + token_type_embeddings

        # Structural embeddings
        if struct_ids is not None:
            struct_embeddings = self.tree_encoder(struct_ids)
        else:
            # If no struct_ids, use zero embeddings
            struct_embeddings = torch.zeros(
                batch_size,
                seq_length,
                self.config.struct_dim,
                device=bert_embeddings.device
            )

        # Fusion
        if self.fusion_strategy == "concat":
            # Concatenate and project
            fused = torch.cat([bert_embeddings, struct_embeddings], dim=-1)
            fused = self.fusion_proj(fused)

        elif self.fusion_strategy == "add":
            # Project structural embeddings and add
            struct_projected = self.struct_proj(struct_embeddings)
            fused = bert_embeddings + struct_projected

        elif self.fusion_strategy == "gate":
            # Gated fusion
            struct_projected = self.struct_proj(struct_embeddings)
            concat = torch.cat([bert_embeddings, struct_embeddings], dim=-1)
            gate_weights = torch.sigmoid(self.gate(concat))
            fused = gate_weights * bert_embeddings + (1 - gate_weights) * struct_projected

        # Layer norm and dropout
        fused = self.LayerNorm(fused)
        fused = self.dropout(fused)

        return fused
