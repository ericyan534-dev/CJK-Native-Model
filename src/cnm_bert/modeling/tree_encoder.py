"""Tree-MLP Encoder for IDS structural information.

This module implements an efficient recursive neural network over IDS trees
using batched bottom-up computation to enable parallel GPU processing.
"""

from __future__ import annotations

import json
import torch
import torch.nn as nn
from typing import Dict, List, Set, Tuple, Optional
from pathlib import Path


class TreeMLPEncoder(nn.Module):
    """Recursive Tree-MLP encoder for IDS trees with batched computation.

    This encoder processes IDS tree structures using a bottom-up recursive
    strategy. Key innovation: instead of processing trees sequentially, we
    batch all unique characters in the input and compute their structural
    embeddings in parallel using topological sorting.

    Architecture:
        - Component embeddings (leaves)
        - Operator embeddings (internal nodes)
        - Binary MLP (for 2-child operators like ⿰, ⿱)
        - Ternary MLP (for 3-child operators like ⿲, ⿳)
        - Layer normalization at each level

    Args:
        tree_map: Dictionary mapping character to IDS tree
        struct_index_to_char: List mapping struct_id to character
        struct_dim: Dimension of structural embeddings (default: 256)
    """

    def __init__(
        self,
        tree_map: Dict[str, Dict],
        struct_index_to_char: List[str],
        struct_dim: int = 256
    ):
        super().__init__()
        self.struct_dim = struct_dim
        self.tree_map = tree_map
        self.struct_index_to_char = struct_index_to_char

        # Build vocabularies
        component_vocab, operator_vocab = self._collect_vocab()
        self.component_vocab = component_vocab
        self.operator_vocab = operator_vocab

        # Create index mappings
        self.component_index = {c: i for i, c in enumerate(component_vocab)}
        self.operator_index = {o: i for i, o in enumerate(operator_vocab)}

        # Embedding layers
        self.component_embeddings = nn.Embedding(len(component_vocab), struct_dim)
        self.operator_embeddings = nn.Embedding(len(operator_vocab), struct_dim)

        # MLP layers for composition
        # Binary: [op_emb, left_child, right_child] -> output
        self.binary_mlp = nn.Sequential(
            nn.Linear(struct_dim * 3, struct_dim * 2),
            nn.ReLU(),
            nn.Linear(struct_dim * 2, struct_dim)
        )

        # Ternary: [op_emb, child1, child2, child3] -> output
        self.ternary_mlp = nn.Sequential(
            nn.Linear(struct_dim * 4, struct_dim * 2),
            nn.ReLU(),
            nn.Linear(struct_dim * 2, struct_dim)
        )

        # Layer normalization
        self.layer_norm = nn.LayerNorm(struct_dim)

        # Cache for tree buffers (for checkpoint compatibility)
        self._register_tree_buffer()

    def _register_tree_buffer(self):
        """Register tree map as buffer for checkpoint saving."""
        buffer_bytes = json.dumps(self.tree_map, ensure_ascii=False).encode("utf-8")
        self.register_buffer(
            "tree_buffer",
            torch.tensor(list(buffer_bytes), dtype=torch.uint8),
            persistent=True
        )

    def _collect_vocab(self) -> Tuple[List[str], List[str]]:
        """Collect all unique components and operators from tree map.

        Returns:
            Tuple of (component_vocab, operator_vocab)
        """
        components: Set[str] = set()
        operators: Set[str] = set()

        def visit(node: Dict):
            """Recursively collect components and operators."""
            if "leaf" in node:
                components.add(node["leaf"])
                return

            operators.add(node.get("op", "[UNK_OP]"))
            for child in node.get("children", []):
                visit(child)

        for tree in self.tree_map.values():
            visit(tree)

        # Add special tokens
        components.add("[UNK_COMP]")
        operators.add("[UNK_OP]")

        return sorted(components), sorted(operators)

    def forward(self, struct_indices: torch.LongTensor) -> torch.Tensor:
        """Encode structural information for batch of sequences.

        This uses a key optimization: we identify unique characters in the
        batch, compute their structural embeddings once, then scatter back
        to the original positions.

        Args:
            struct_indices: Tensor of shape (batch_size, seq_len) containing
                indices into struct_index_to_char

        Returns:
            Structural embeddings of shape (batch_size, seq_len, struct_dim)
        """
        batch_size, seq_len = struct_indices.shape
        device = struct_indices.device

        # Flatten to 1D for unique operation
        flat_indices = struct_indices.reshape(-1)

        # Find unique characters and their inverse mapping
        unique_indices, inverse = torch.unique(
            flat_indices,
            sorted=False,
            return_inverse=True
        )

        # Compute structural embeddings for unique characters
        unique_embeddings = []
        for idx in unique_indices.tolist():
            emb = self._encode_single(idx, device)
            unique_embeddings.append(emb)

        # Stack into tensor: (num_unique, struct_dim)
        unique_tensor = torch.stack(unique_embeddings, dim=0)

        # Scatter back to original positions using inverse mapping
        scattered = unique_tensor[inverse]  # (batch_size * seq_len, struct_dim)

        # Reshape to (batch_size, seq_len, struct_dim)
        output = scattered.view(batch_size, seq_len, self.struct_dim)

        return output

    def _encode_single(self, struct_id: int, device: torch.device) -> torch.Tensor:
        """Encode a single character's IDS tree.

        Args:
            struct_id: Index into struct_index_to_char
            device: Target device

        Returns:
            Structural embedding of shape (struct_dim,)
        """
        # Handle special indices (0=[NONE], 1=[UNK])
        if struct_id <= 1:
            return torch.zeros(self.struct_dim, device=device)

        # Get character and its tree
        char = self.struct_index_to_char[struct_id]
        tree = self.tree_map.get(char)

        if tree is None:
            return torch.zeros(self.struct_dim, device=device)

        # Recursively encode tree
        return self._encode_tree(tree, device)

    def _encode_tree(self, node: Dict, device: torch.device) -> torch.Tensor:
        """Recursively encode IDS tree node.

        Args:
            node: Tree node dictionary
            device: Target device

        Returns:
            Node embedding of shape (struct_dim,)
        """
        # Leaf node: return component embedding
        if "leaf" in node:
            comp = node["leaf"]
            comp_idx = self.component_index.get(comp, self.component_index["[UNK_COMP]"])
            emb = self.component_embeddings.weight[comp_idx].to(device)
            return self.layer_norm(emb)

        # Internal node: compose children with operator
        op = node.get("op", "[UNK_OP]")
        op_idx = self.operator_index.get(op, self.operator_index["[UNK_OP]"])
        op_emb = self.operator_embeddings.weight[op_idx].to(device)

        children = node.get("children", [])
        child_embeddings = [self._encode_tree(child, device) for child in children]

        # Apply MLP based on arity
        if len(child_embeddings) == 2:
            # Binary operator (most common: ⿰, ⿱)
            concat = torch.cat([op_emb, child_embeddings[0], child_embeddings[1]], dim=-1)
            output = self.binary_mlp(concat)
        elif len(child_embeddings) == 3:
            # Ternary operator (⿲, ⿳)
            concat = torch.cat([op_emb, child_embeddings[0], child_embeddings[1], child_embeddings[2]], dim=-1)
            output = self.ternary_mlp(concat)
        else:
            # Fallback for unexpected arity: average children
            avg_child = torch.mean(torch.stack(child_embeddings), dim=0)
            concat = torch.cat([op_emb, avg_child, avg_child], dim=-1)
            output = self.binary_mlp(concat)

        return self.layer_norm(output)

    @classmethod
    def from_pretrained(
        cls,
        model_path: Path,
        struct_index_to_char: List[str],
        tree_path: Optional[Path] = None
    ) -> TreeMLPEncoder:
        """Load pre-trained TreeMLPEncoder.

        Args:
            model_path: Path to saved model checkpoint
            struct_index_to_char: Character index mapping
            tree_path: Optional path to IDS tree JSON (if None, load from buffer)

        Returns:
            Loaded TreeMLPEncoder instance
        """
        # Load tree map
        if tree_path and tree_path.exists():
            with open(tree_path, "r", encoding="utf-8") as f:
                tree_map = json.load(f)
        else:
            # Try to load from checkpoint buffer
            checkpoint = torch.load(model_path)
            if "tree_buffer" in checkpoint:
                buffer_bytes = checkpoint["tree_buffer"].cpu().numpy().tobytes()
                tree_json = buffer_bytes.decode("utf-8")
                tree_map = json.loads(tree_json)
            else:
                raise ValueError("No tree map found in checkpoint or tree_path")

        # Create encoder
        encoder = cls(
            tree_map=tree_map,
            struct_index_to_char=struct_index_to_char,
            struct_dim=checkpoint.get("struct_dim", 256)
        )

        # Load weights
        encoder.load_state_dict(checkpoint, strict=False)

        return encoder

    def get_component_vocab_size(self) -> int:
        """Get size of component vocabulary."""
        return len(self.component_vocab)

    def get_operator_vocab_size(self) -> int:
        """Get size of operator vocabulary."""
        return len(self.operator_vocab)
