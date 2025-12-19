"""CNM tokenizer that augments BertTokenizer with structural indices."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional

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
    """Character-level tokenizer with structural index lookup."""

    def __init__(self, *args, struct_path: Optional[str] = None, **kwargs):
        super().__init__(*args, **kwargs)
        if struct_path is None:
            struct_path = str(Path(__file__).resolve().parent.parent / "data" / "char_to_ids_tree.json")
        self.struct_path = Path(struct_path)
        self.char_to_tree: Dict[str, Dict] = {}
        if self.struct_path.exists():
            with self.struct_path.open("r", encoding="utf-8") as f:
                self.char_to_tree = json.load(f)
        self.struct_vocab: List[str] = ["[NONE]", "[UNK_STRUCT]"] + sorted(self.char_to_tree.keys())
        self.struct_index_by_char = {ch: idx for idx, ch in enumerate(self.struct_vocab)}
        self.struct_index_to_char = list(self.struct_vocab)
        self.struct_id_table = self._build_struct_id_table()

    def _build_struct_id_table(self) -> torch.Tensor:
        ids: List[int] = []
        for token in self.vocab:
            ids.append(self._struct_index_for_token(token))
        return torch.tensor(ids, dtype=torch.long)

    def _struct_index_for_token(self, token: str) -> int:
        if len(token) != 1:
            return 0
        if not _is_cjk(token):
            return 0
        return self.struct_index_by_char.get(token, 1)

    def get_struct_id(self, token_id: int) -> int:
        return int(self.struct_id_table[token_id])

    def __call__(self, *args, return_struct_ids: bool = False, **kwargs):
        encoded = super().__call__(*args, **kwargs)
        if not return_struct_ids:
            return encoded
        input_ids = encoded["input_ids"]
        struct_ids = torch.zeros_like(input_ids)
        for idx, token_id in enumerate(input_ids.view(-1)):
            struct_ids.view(-1)[idx] = self.get_struct_id(int(token_id))
        encoded["struct_ids"] = struct_ids.view_as(input_ids)
        return encoded

    @property
    def struct_size(self) -> int:
        return len(self.struct_vocab)


__all__ = ["CNMBertTokenizer"]
