"""IDS (Ideographic Description Sequence) parser with canonicalization.

This module parses BabelStone IDS data and produces deterministic, canonical
tree representations of Chinese character composition. It implements heuristic
algorithms to resolve ambiguity when multiple IDS exist for the same character.
"""

from __future__ import annotations

import json
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple
from collections import defaultdict

from ..utils.logging import setup_logger

logger = setup_logger(__name__)


# Standard IDS operators (Ideographic Description Characters)
IDC_OPERATORS = {
    "⿰",  # Left-to-right
    "⿱",  # Top-to-bottom
    "⿲",  # Left-to-middle-to-right (ternary)
    "⿳",  # Top-to-middle-to-bottom (ternary)
    "⿴",  # Surround from outside
    "⿵",  # Surround from above
    "⿶",  # Surround from below
    "⿷",  # Surround from left
    "⿸",  # Surround from upper-left
    "⿹",  # Surround from upper-right
    "⿺",  # Surround from lower-left
    "⿻",  # Overlaid
}

# Operator arity mapping
IDC_ARITY = {
    "⿰": 2, "⿱": 2, "⿲": 3, "⿳": 3,
    "⿴": 2, "⿵": 2, "⿶": 2, "⿷": 2,
    "⿸": 2, "⿹": 2, "⿺": 2, "⿻": 2,
}

# Standard operators (prefer these in ambiguity resolution)
STANDARD_OPERATORS = {"⿰", "⿱"}

# Private Use Area (PUA) ranges to filter out
PUA_RANGES: List[Tuple[int, int]] = [
    (0xE000, 0xF8FF),      # BMP PUA
    (0xF0000, 0xFFFFD),    # Supplementary PUA-A
    (0x100000, 0x10FFFD),  # Supplementary PUA-B
]


@dataclass
class TreeNode:
    """Represents a node in the IDS tree structure.

    Attributes:
        operator: IDS operator (e.g., ⿰, ⿱) if internal node
        children: List of child nodes if internal node
        leaf: Leaf character if leaf node
    """
    operator: Optional[str] = None
    children: Optional[List[TreeNode]] = None
    leaf: Optional[str] = None

    @property
    def is_leaf(self) -> bool:
        """Check if this is a leaf node."""
        return self.leaf is not None

    def to_dict(self) -> Dict:
        """Convert tree to dictionary representation."""
        if self.is_leaf:
            return {"leaf": self.leaf}
        return {
            "op": self.operator,
            "children": [child.to_dict() for child in (self.children or [])]
        }

    def depth(self) -> int:
        """Compute depth of tree."""
        if self.is_leaf:
            return 1
        return 1 + max(child.depth() for child in (self.children or []))

    def node_count(self) -> int:
        """Count total nodes in tree."""
        if self.is_leaf:
            return 1
        return 1 + sum(child.node_count() for child in (self.children or []))

    def has_standard_ops(self) -> bool:
        """Check if tree uses only standard operators (⿰, ⿱)."""
        if self.is_leaf:
            return True
        if self.operator not in STANDARD_OPERATORS:
            return False
        return all(child.has_standard_ops() for child in (self.children or []))

    def operator_sequence(self) -> str:
        """Get lexicographic sequence of operators (for tie-breaking)."""
        if self.is_leaf:
            return ""
        ops = [self.operator or ""]
        for child in (self.children or []):
            ops.append(child.operator_sequence())
        return "".join(ops)


class IDSParser:
    """Parser for IDS expressions with canonicalization."""

    def __init__(self):
        self.stats = {
            "total_lines": 0,
            "parsed": 0,
            "skipped_pua": 0,
            "skipped_malformed": 0,
            "multiple_ids": 0,
            "circular_refs": 0,
        }

    def contains_pua(self, text: str) -> bool:
        """Check if text contains Private Use Area characters.

        Args:
            text: Text to check

        Returns:
            True if PUA characters found
        """
        for char in text:
            codepoint = ord(char)
            for start, end in PUA_RANGES:
                if start <= codepoint <= end:
                    return True
        return False

    def parse_ids_expression(self, expr: str, idx: int = 0) -> Tuple[TreeNode, int]:
        """Parse an IDS expression into a tree structure.

        IDS uses prefix notation: operator followed by operands.
        Example: ⿰女子 means "女 left-of 子"

        Args:
            expr: IDS expression string
            idx: Current position in expression

        Returns:
            Tuple of (parsed TreeNode, next position)

        Raises:
            ValueError: If expression is malformed
        """
        if idx >= len(expr):
            raise ValueError(f"Unexpected end of IDS expression at position {idx}")

        char = expr[idx]

        # Check if this is an operator
        if char in IDC_OPERATORS:
            arity = IDC_ARITY.get(char, 2)
            children: List[TreeNode] = []
            next_idx = idx + 1

            # Parse children according to operator arity
            for i in range(arity):
                try:
                    child, next_idx = self.parse_ids_expression(expr, next_idx)
                    children.append(child)
                except ValueError as e:
                    raise ValueError(f"Failed to parse child {i+1} of operator {char}: {e}")

            return TreeNode(operator=char, children=children), next_idx

        # Leaf node (component character)
        return TreeNode(leaf=char), idx + 1

    def parse_ids_line(self, line: str) -> Optional[Tuple[str, TreeNode]]:
        """Parse a single line from BabelStone IDS file.

        Format: CHARACTER<tab>IDS_EXPRESSION
        Example: 好\t⿰女子

        Args:
            line: Line from IDS file

        Returns:
            Tuple of (character, tree) or None if line should be skipped
        """
        line = line.strip()
        if not line or line.startswith("#"):
            return None

        # Split on tab
        if "\t" not in line:
            self.stats["skipped_malformed"] += 1
            return None

        parts = line.split("\t", maxsplit=1)
        if len(parts) != 2:
            self.stats["skipped_malformed"] += 1
            return None

        char, expr = parts

        # Filter PUA characters
        if self.contains_pua(line):
            self.stats["skipped_pua"] += 1
            return None

        # Parse expression
        try:
            tree, consumed = self.parse_ids_expression(expr)
        except ValueError as e:
            logger.debug(f"Failed to parse IDS for '{char}': {e}")
            self.stats["skipped_malformed"] += 1
            return None

        # Verify entire expression was consumed
        if consumed != len(expr):
            logger.debug(f"Trailing characters in IDS for '{char}': {expr[consumed:]}")
            self.stats["skipped_malformed"] += 1
            return None

        self.stats["parsed"] += 1
        return char, tree

    def canonicalize_trees(self, trees: List[TreeNode]) -> TreeNode:
        """Select canonical tree when multiple IDS exist for same character.

        Heuristic priority (from plan):
        1. Prefer shallower trees (minimum depth)
        2. Prefer standard operators (⿰, ⿱)
        3. Prefer fewer nodes
        4. Tie-break by lexicographic order of operator sequence

        Args:
            trees: List of candidate trees

        Returns:
            Canonical tree
        """
        if len(trees) == 1:
            return trees[0]

        self.stats["multiple_ids"] += 1

        # Score each tree
        scored = []
        for tree in trees:
            score = (
                tree.depth(),                          # Lower is better
                not tree.has_standard_ops(),           # False (0) is better
                tree.node_count(),                     # Lower is better
                tree.operator_sequence(),              # Lexicographic
            )
            scored.append((score, tree))

        # Sort and return best
        scored.sort(key=lambda x: x[0])
        return scored[0][1]

    def detect_circular_references(
        self,
        char_to_trees: Dict[str, TreeNode]
    ) -> Set[str]:
        """Detect circular references in IDS definitions.

        Example: If A is defined as ⿰BA and B is defined as ⿰CA,
        we have a circular reference through A.

        Args:
            char_to_trees: Mapping of character to IDS tree

        Returns:
            Set of characters involved in circular references
        """
        circular = set()

        def has_cycle(char: str, visited: Set[str], stack: Set[str]) -> bool:
            """DFS to detect cycles."""
            if char in stack:
                return True
            if char in visited:
                return False

            visited.add(char)
            stack.add(char)

            tree = char_to_trees.get(char)
            if tree:
                # Check all leaf nodes in tree
                leaves = self._get_leaves(tree)
                for leaf in leaves:
                    if leaf in char_to_trees and has_cycle(leaf, visited, stack):
                        circular.add(char)
                        return True

            stack.remove(char)
            return False

        visited = set()
        for char in char_to_trees:
            if char not in visited:
                has_cycle(char, visited, set())

        if circular:
            self.stats["circular_refs"] = len(circular)
            logger.warning(f"Detected {len(circular)} characters with circular references")

        return circular

    def _get_leaves(self, node: TreeNode) -> List[str]:
        """Extract all leaf characters from tree."""
        if node.is_leaf:
            return [node.leaf] if node.leaf else []

        leaves = []
        for child in (node.children or []):
            leaves.extend(self._get_leaves(child))
        return leaves

    def parse_file(self, ids_file: Path) -> Dict[str, Dict]:
        """Parse entire BabelStone IDS file.

        Args:
            ids_file: Path to IDS file

        Returns:
            Dictionary mapping character to canonical IDS tree (as dict)
        """
        logger.info(f"Parsing IDS file: {ids_file}")

        # Group by character (may have multiple IDS per character)
        char_to_trees: Dict[str, List[TreeNode]] = defaultdict(list)

        with open(ids_file, "r", encoding="utf-8") as f:
            for line in f:
                self.stats["total_lines"] += 1
                result = self.parse_ids_line(line)
                if result:
                    char, tree = result
                    char_to_trees[char].append(tree)

        # Canonicalize: select best tree for each character
        canonical: Dict[str, TreeNode] = {}
        for char, trees in char_to_trees.items():
            canonical[char] = self.canonicalize_trees(trees)

        # Detect and remove circular references
        circular = self.detect_circular_references(canonical)
        for char in circular:
            del canonical[char]

        # Convert to dictionary format
        result = {char: tree.to_dict() for char, tree in canonical.items()}

        # Log statistics
        logger.info(f"IDS parsing complete:")
        logger.info(f"  Total lines: {self.stats['total_lines']}")
        logger.info(f"  Successfully parsed: {self.stats['parsed']}")
        logger.info(f"  Skipped (PUA): {self.stats['skipped_pua']}")
        logger.info(f"  Skipped (malformed): {self.stats['skipped_malformed']}")
        logger.info(f"  Characters with multiple IDS: {self.stats['multiple_ids']}")
        logger.info(f"  Circular references removed: {self.stats['circular_refs']}")
        logger.info(f"  Final canonical characters: {len(result)}")

        return result

    def compute_statistics(self, char_to_tree: Dict[str, Dict]) -> Dict[str, any]:
        """Compute statistics about IDS lexicon (for paper).

        Args:
            char_to_tree: Canonical IDS mapping

        Returns:
            Dictionary of statistics
        """
        depths = []
        node_counts = []
        operators = defaultdict(int)
        components = set()

        def analyze_tree(node_dict: Dict):
            """Recursively analyze tree."""
            if "leaf" in node_dict:
                components.add(node_dict["leaf"])
                return 1, 1  # depth=1, nodes=1

            op = node_dict.get("op")
            if op:
                operators[op] += 1

            child_depths = []
            total_nodes = 1
            for child in node_dict.get("children", []):
                depth, nodes = analyze_tree(child)
                child_depths.append(depth)
                total_nodes += nodes

            max_depth = 1 + max(child_depths) if child_depths else 1
            return max_depth, total_nodes

        for tree_dict in char_to_tree.values():
            depth, nodes = analyze_tree(tree_dict)
            depths.append(depth)
            node_counts.append(nodes)

        stats = {
            "total_characters": len(char_to_tree),
            "total_components": len(components),
            "total_operators": len(operators),
            "depth_min": min(depths) if depths else 0,
            "depth_max": max(depths) if depths else 0,
            "depth_mean": sum(depths) / len(depths) if depths else 0,
            "nodes_min": min(node_counts) if node_counts else 0,
            "nodes_max": max(node_counts) if node_counts else 0,
            "nodes_mean": sum(node_counts) / len(node_counts) if node_counts else 0,
            "operator_distribution": dict(operators),
            "sample_components": sorted(list(components))[:50],
        }

        return stats


def parse_ids_file(
    ids_file: Path,
    output_file: Path,
    vocab_file: Optional[Path] = None
) -> Dict[str, Dict]:
    """Convenience function to parse IDS file and save output.

    Args:
        ids_file: Path to BabelStone IDS file
        output_file: Path to save canonical JSON
        vocab_file: Optional BERT vocab file to check coverage

    Returns:
        Canonical IDS mapping
    """
    parser = IDSParser()
    char_to_tree = parser.parse_file(ids_file)

    # Save output
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(char_to_tree, f, ensure_ascii=False, indent=2)
    logger.info(f"Saved canonical IDS to: {output_file}")

    # Check vocabulary coverage if provided
    if vocab_file and vocab_file.exists():
        with open(vocab_file, "r", encoding="utf-8") as f:
            vocab = set(line.strip() for line in f)

        # Count coverage
        covered = sum(1 for token in vocab if token in char_to_tree)
        coverage = 100 * covered / len(vocab) if vocab else 0
        logger.info(f"Vocabulary coverage: {covered}/{len(vocab)} ({coverage:.2f}%)")

    # Compute and log statistics
    stats = parser.compute_statistics(char_to_tree)
    logger.info(f"IDS Statistics:")
    logger.info(f"  Characters: {stats['total_characters']}")
    logger.info(f"  Components: {stats['total_components']}")
    logger.info(f"  Operators: {stats['total_operators']}")
    logger.info(f"  Depth range: [{stats['depth_min']}, {stats['depth_max']}], mean={stats['depth_mean']:.2f}")
    logger.info(f"  Nodes range: [{stats['nodes_min']}, {stats['nodes_max']}], mean={stats['nodes_mean']:.2f}")

    return char_to_tree


def canonicalize_ids(ids_file: Path, output_file: Path) -> None:
    """Legacy wrapper for parse_ids_file."""
    parse_ids_file(ids_file, output_file)
