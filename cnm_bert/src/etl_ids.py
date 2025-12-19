"""IDS ETL pipeline for CNM-BERT.

This script canonicalizes BabelStone IDS data into a deterministic JSON mapping
from character to a parsed IDS tree. The implementation follows the project
plan: filter out PUA code points, select the shallowest IDS when multiple are
available, and emit a recursive operator/children structure.
"""

from __future__ import annotations

import argparse
import json
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

# Common IDS operator set. The list is limited to standard IDC operators used in
# BabelStone IDS definitions.
IDC_OPERATORS = {
    "⿰",
    "⿱",
    "⿲",
    "⿳",
    "⿴",
    "⿵",
    "⿶",
    "⿷",
    "⿸",
    "⿹",
    "⿺",
    "⿻",
}

# Operator arity hints. Most operators are binary; ⿲/⿳ are ternary.
IDC_ARITY = {
    "⿰": 2,
    "⿱": 2,
    "⿲": 3,
    "⿳": 3,
    "⿴": 2,
    "⿵": 2,
    "⿶": 2,
    "⿷": 2,
    "⿸": 2,
    "⿹": 2,
    "⿺": 2,
    "⿻": 2,
}

PUA_RANGES: List[Tuple[int, int]] = [
    (0xE000, 0xF8FF),
    (0xF0000, 0xFFFFD),
    (0x100000, 0x10FFFD),
]


@dataclass
class TreeNode:
    """Simple tree node representation."""

    operator: Optional[str]
    children: Optional[List["TreeNode"]] = None
    leaf: Optional[str] = None

    @property
    def is_leaf(self) -> bool:
        return self.leaf is not None

    def to_dict(self) -> Dict:
        if self.is_leaf:
            return {"leaf": self.leaf}
        return {
            "op": self.operator,
            "children": [child.to_dict() for child in self.children or []],
        }


def contains_pua(text: str) -> bool:
    codepoints = [ord(ch) for ch in text]
    for cp in codepoints:
        for start, end in PUA_RANGES:
            if start <= cp <= end:
                return True
    return False


def tree_depth(node: TreeNode) -> int:
    if node.is_leaf:
        return 1
    return 1 + max(tree_depth(child) for child in (node.children or []))


def parse_ids_expression(expr: str, idx: int = 0) -> Tuple[TreeNode, int]:
    """Parse an IDS expression into a tree.

    The parser consumes characters from ``expr`` starting at ``idx``. IDS uses a
    prefix operator format where the operator is followed by its children (which
    may be operators themselves).
    """

    if idx >= len(expr):
        raise ValueError("Unexpected end of IDS expression")

    char = expr[idx]
    if char in IDC_OPERATORS:
        arity = IDC_ARITY.get(char, 2)
        children: List[TreeNode] = []
        next_idx = idx + 1
        for _ in range(arity):
            child, next_idx = parse_ids_expression(expr, next_idx)
            children.append(child)
        return TreeNode(operator=char, children=children), next_idx

    return TreeNode(operator=None, leaf=char), idx + 1


def parse_ids_line(line: str) -> Optional[Tuple[str, TreeNode]]:
    line = line.strip()
    if not line or "\t" not in line:
        return None
    char, expr = line.split("\t", maxsplit=1)
    if contains_pua(line):
        return None
    try:
        tree, consumed = parse_ids_expression(expr)
    except ValueError:
        return None
    if consumed != len(expr):
        # Skip malformed entries with trailing junk.
        return None
    return char, tree


def select_best(trees: List[TreeNode]) -> TreeNode:
    # Select the tree with the smallest depth, then the fewest nodes.
    best = None
    best_score = (1e9, 1e9)
    for tree in trees:
        depth = tree_depth(tree)
        size = count_nodes(tree)
        score = (depth, size)
        if score < best_score:
            best = tree
            best_score = score
    assert best is not None
    return best


def count_nodes(node: TreeNode) -> int:
    if node.is_leaf:
        return 1
    return 1 + sum(count_nodes(child) for child in node.children or [])


def load_ids(path: Path) -> Dict[str, TreeNode]:
    candidates: Dict[str, List[TreeNode]] = {}
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            parsed = parse_ids_line(line)
            if parsed is None:
                continue
            ch, tree = parsed
            candidates.setdefault(ch, []).append(tree)

    cleaned: Dict[str, TreeNode] = {}
    for ch, trees in candidates.items():
        best_tree = select_best(trees)
        cleaned[ch] = best_tree
    return cleaned


def save_json(mapping: Dict[str, TreeNode], output: Path) -> None:
    serializable = {ch: tree.to_dict() for ch, tree in mapping.items()}
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as f:
        json.dump(serializable, f, ensure_ascii=False, indent=2)


def validate_depth(mapping: Dict[str, TreeNode], max_depth: int = 16) -> None:
    for ch, tree in mapping.items():
        depth = tree_depth(tree)
        if depth > max_depth:
            raise ValueError(f"Tree for {ch} exceeds maximum depth {max_depth}: {depth}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Canonicalize IDS data for CNM-BERT")
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("cnm_bert/assets/ids.txt"),
        help="Path to BabelStone IDS text file (tab-separated char and IDS)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("cnm_bert/data/char_to_ids_tree.json"),
        help="Path to write the canonicalized JSON mapping",
    )
    args = parser.parse_args()

    mapping = load_ids(args.input)
    validate_depth(mapping)
    save_json(mapping, args.output)
    print(f"Saved {len(mapping)} IDS entries to {args.output}")


if __name__ == "__main__":
    main()
