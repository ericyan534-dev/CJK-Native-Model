"""CNM tokenizer that augments BertTokenizer with structural tree tensors."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import torch
from transformers import BertTokenizer


def _is_cjk(char: str) -> bool:
    if len(char) != 1:
        return False
    cp = ord(char)
    return (
        0x4E00 <= cp <= 0x9FFF
        or 0x3400 <= cp <= 0x4DBF
        or 0x20000 <= cp <= 0x2A6DF
        or 0x2A700 <= cp <= 0x2B73F
        or 0x2B740 <= cp <= 0x2B81F
        or 0x2B820 <= cp <= 0x2CEAF
        or 0x2CEB0 <= cp <= 0x2EBEF
    )


class CNMBertTokenizer(BertTokenizer):
    """Character-level tokenizer with cached structural tensors.

    The tokenizer enforces pure character-level tokenization while precomputing a
    table that maps each vocabulary token to a padded tree tensor of shape
    ``[max_depth, max_branching]``. To reduce runtime allocations, the call output
    includes both the structural tensor per token and the index mapping back to
    the cached table (so the caller can move the cache to GPU once and reuse it).
    """

    def __init__(
        self,
        *args,
        struct_path: Optional[str] = None,
        max_depth_cap: int = 16,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        # BERT-style models default to a 512 token limit; clamp the tokenizer to that
        # when the provided vocab/config leaves the max length unset (which defaults to
        # a very large sentinel value).
        if self.model_max_length and self.model_max_length > 512:
            self.model_max_length = 512
        if struct_path is None:
            struct_path = str(Path(__file__).resolve().parent.parent / "data" / "char_to_ids_tree.json")
        self.struct_path = Path(struct_path)
        self.char_to_tree: Dict[str, Dict] = {}
        if self.struct_path.exists():
            with self.struct_path.open("r", encoding="utf-8") as f:
                self.char_to_tree = json.load(f)

        # Structural label vocabulary (operators + components + sentinels).
        operators, components = self._collect_struct_symbols(self.char_to_tree)
        self.struct_labels: List[str] = ["[NONE]", "[UNK_STRUCT]"] + sorted(operators) + sorted(components)
        self.struct_label_to_id = {lbl: i for i, lbl in enumerate(self.struct_labels)}

        # Character index vocabulary for cached tree lookup.
        self.struct_vocab: List[str] = ["[NONE]", "[UNK_STRUCT]"] + sorted(self.char_to_tree.keys())
        self.struct_index_by_char = {ch: idx for idx, ch in enumerate(self.struct_vocab)}
        self.struct_index_to_char = list(self.struct_vocab)

        # Precompute cached structural tensors per vocab token.
        self.max_depth = 1
        self.max_branching = 1
        self.struct_tensor = self._build_struct_tensor(max_depth_cap=max_depth_cap)
        self.struct_index_table = self._build_struct_index_table()

    def _collect_struct_symbols(self, tree_map: Dict[str, Dict]) -> Tuple[set, set]:
        operators, components = set(), set()

        def visit(node: Dict):
            if "leaf" in node:
                components.add(node["leaf"])
                return
            operators.add(node.get("op", "[UNK_OP]"))
            for child in node.get("children", []):
                visit(child)

        for tree in tree_map.values():
            visit(tree)
        return operators, components

    def _tree_depth_and_branch(self, node: Dict, depth: int = 1) -> Tuple[int, int]:
        """Compute max depth and branching factor for padding."""

        if "leaf" in node:
            return depth, 1
        depths, branchings = [], []
        for child in node.get("children", []):
            d, b = self._tree_depth_and_branch(child, depth + 1)
            depths.append(d)
            branchings.append(b)
        max_depth = max(depths) if depths else depth
        max_branch = max([len(node.get("children", []))] + branchings) if branchings else len(node.get("children", []))
        return max_depth, max_branch

    def _build_struct_tensor(self, max_depth_cap: int) -> torch.Tensor:
        """Precompute padded tree tensors for every vocab token."""

        depths: List[int] = []
        branching: List[int] = []
        for tree in self.char_to_tree.values():
            d, b = self._tree_depth_and_branch(tree)
            depths.append(d)
            branching.append(b)
        self.max_depth = min(max(depths, default=1), max_depth_cap)
        self.max_branching = max(branching, default=1)
        self.level_width = self.max_branching ** (self.max_depth - 1) if self.max_depth > 0 else 1

        struct_tensors: List[torch.Tensor] = [torch.empty(0) for _ in range(self.vocab_size)]
        for token, idx in self.vocab.items():
            tree_tensor = self._tensorize_tree(token)
            struct_tensors[idx] = tree_tensor
        return torch.stack(struct_tensors, dim=0)

    def _tensorize_tree(self, token: str) -> torch.Tensor:
        pad_id = self.struct_label_to_id["[NONE]"]
        unk_id = self.struct_label_to_id["[UNK_STRUCT]"]
        tensor = torch.full((self.max_depth, self.level_width), pad_id, dtype=torch.long)

        if len(token) != 1 or not _is_cjk(token):
            tensor[0, 0] = pad_id
            return tensor
        tree = self.char_to_tree.get(token)
        if not tree:
            tensor[0, 0] = unk_id
            return tensor

        def fill(node: Dict, depth: int, offset: int):
            if depth >= self.max_depth:
                return
            if offset >= self.level_width:
                return
            if "leaf" in node:
                tensor[depth, offset] = self.struct_label_to_id.get(node["leaf"], unk_id)
                return
            tensor[depth, offset] = self.struct_label_to_id.get(node.get("op", "[UNK_STRUCT]"), unk_id)
            children = node.get("children", [])
            for idx, child in enumerate(children[: self.max_branching]):
                fill(child, depth + 1, offset * self.max_branching + idx)

        fill(tree, 0, 0)
        return tensor

    def _build_struct_index_table(self) -> torch.Tensor:
        ids: List[int] = [0 for _ in range(self.vocab_size)]
        for token, idx in self.vocab.items():
            ids[idx] = self._struct_index_for_token(token)
        return torch.tensor(ids, dtype=torch.long)

    def _struct_index_for_token(self, token: str) -> int:
        if len(token) != 1:
            return 0
        if not _is_cjk(token):
            return 0
        return self.struct_index_by_char.get(token, 1)

    def get_struct_index(self, token_id: int) -> int:
        return int(self.struct_index_table[token_id])

    def _encode_chars(self, text: str) -> List[str]:
        return list(text)

    def __call__(self, *args, return_struct_ids: bool = False, **kwargs):
        # Force character-level tokenization by splitting raw strings into char list.
        if args and isinstance(args[0], str):
            kwargs["text"] = self._encode_chars(args[0])
            args = ()
            kwargs["is_split_into_words"] = True
        elif kwargs.get("text") is not None and isinstance(kwargs["text"], str):
            kwargs["text"] = self._encode_chars(kwargs["text"])
            kwargs["is_split_into_words"] = True

        encoded = super().__call__(*args, **kwargs)
        if not return_struct_ids:
            return encoded

        input_ids: torch.Tensor = encoded["input_ids"]
        flat_ids = input_ids.view(-1)
        struct_indices = torch.tensor([self.get_struct_index(int(tid)) for tid in flat_ids], dtype=torch.long)
        struct_indices = struct_indices.view_as(input_ids)
        # Gather precomputed tree tensors for each token.
        struct_trees = self.struct_tensor[flat_ids].view(
            *input_ids.shape, self.max_depth, self.level_width
        )
        encoded["struct_indices"] = struct_indices
        encoded["struct_ids"] = struct_trees
        return encoded

    @property
    def struct_size(self) -> int:
        return len(self.struct_vocab)


__all__ = ["CNMBertTokenizer"]
