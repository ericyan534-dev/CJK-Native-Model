"""Heuristics for IDS ambiguity resolution and tree optimization.

This module contains additional heuristic algorithms that can be applied
during or after IDS parsing to improve tree quality.
"""

from typing import Dict, List, Tuple
from collections import defaultdict

from .ids_parser import TreeNode


def compute_component_frequency(char_to_tree: Dict[str, Dict]) -> Dict[str, int]:
    """Compute frequency of each component across all IDS trees.

    Args:
        char_to_tree: Mapping of character to IDS tree

    Returns:
        Dictionary mapping component to frequency
    """
    freq = defaultdict(int)

    def count_components(node: Dict):
        """Recursively count components."""
        if "leaf" in node:
            freq[node["leaf"]] += 1
        else:
            for child in node.get("children", []):
                count_components(child)

    for tree in char_to_tree.values():
        count_components(tree)

    return dict(freq)


def identify_common_substructures(
    char_to_tree: Dict[str, Dict],
    min_frequency: int = 10
) -> List[Tuple[str, int]]:
    """Identify frequently occurring substructures in IDS trees.

    This can be used to create a hierarchy of components or to
    identify good candidates for shared embeddings.

    Args:
        char_to_tree: Mapping of character to IDS tree
        min_frequency: Minimum frequency threshold

    Returns:
        List of (substructure_repr, frequency) tuples
    """
    substructure_freq = defaultdict(int)

    def tree_signature(node: Dict) -> str:
        """Create string signature of tree structure."""
        if "leaf" in node:
            return node["leaf"]
        op = node.get("op", "")
        children = [tree_signature(child) for child in node.get("children", [])]
        return f"{op}({'|'.join(children)})"

    def extract_substructures(node: Dict):
        """Recursively extract all substructures."""
        sig = tree_signature(node)
        substructure_freq[sig] += 1

        if "children" in node:
            for child in node["children"]:
                extract_substructures(child)

    for tree in char_to_tree.values():
        extract_substructures(tree)

    # Filter by frequency and sort
    common = [(sig, freq) for sig, freq in substructure_freq.items() if freq >= min_frequency]
    common.sort(key=lambda x: x[1], reverse=True)

    return common


def validate_tree_consistency(char_to_tree: Dict[str, Dict]) -> List[str]:
    """Validate that IDS trees are internally consistent.

    Checks:
    1. All operators have correct arity
    2. No null/empty nodes
    3. Leaves contain valid characters

    Args:
        char_to_tree: Mapping of character to IDS tree

    Returns:
        List of validation error messages (empty if all valid)
    """
    errors = []

    # Arity map
    arity_map = {
        "⿰": 2, "⿱": 2, "⿲": 3, "⿳": 3,
        "⿴": 2, "⿵": 2, "⿶": 2, "⿷": 2,
        "⿸": 2, "⿹": 2, "⿺": 2, "⿻": 2,
    }

    def validate_node(char: str, node: Dict, path: str = "root"):
        """Recursively validate tree node."""
        # Check for empty node
        if not node:
            errors.append(f"Character '{char}': Empty node at {path}")
            return

        # Leaf node
        if "leaf" in node:
            if not node["leaf"]:
                errors.append(f"Character '{char}': Empty leaf at {path}")
            return

        # Internal node
        if "op" not in node:
            errors.append(f"Character '{char}': Missing operator at {path}")
            return

        op = node["op"]
        if op not in arity_map:
            errors.append(f"Character '{char}': Unknown operator '{op}' at {path}")
            return

        expected_arity = arity_map[op]
        children = node.get("children", [])

        if len(children) != expected_arity:
            errors.append(
                f"Character '{char}': Operator '{op}' expects {expected_arity} children, "
                f"got {len(children)} at {path}"
            )

        # Recursively validate children
        for i, child in enumerate(children):
            validate_node(char, child, f"{path}.{op}[{i}]")

    for char, tree in char_to_tree.items():
        validate_node(char, tree)

    return errors


def prune_rare_components(
    char_to_tree: Dict[str, Dict],
    min_frequency: int = 5
) -> Dict[str, Dict]:
    """Replace rare components with [UNK_COMP] marker.

    This can help reduce vocabulary size and improve generalization
    for very rare components that may not have enough training data.

    Args:
        char_to_tree: Mapping of character to IDS tree
        min_frequency: Minimum component frequency to keep

    Returns:
        Updated mapping with rare components replaced
    """
    # Compute component frequencies
    freq = compute_component_frequency(char_to_tree)

    # Identify rare components
    rare = {comp for comp, count in freq.items() if count < min_frequency}

    if not rare:
        return char_to_tree

    def replace_rare(node: Dict) -> Dict:
        """Recursively replace rare components."""
        if "leaf" in node:
            if node["leaf"] in rare:
                return {"leaf": "[UNK_COMP]"}
            return node

        return {
            "op": node["op"],
            "children": [replace_rare(child) for child in node.get("children", [])]
        }

    # Create updated mapping
    updated = {char: replace_rare(tree) for char, tree in char_to_tree.items()}

    return updated
