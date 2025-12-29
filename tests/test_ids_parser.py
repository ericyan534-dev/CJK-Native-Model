"""Unit tests for IDS parser."""

import pytest
import tempfile
from pathlib import Path

from cnm_bert.etl.ids_parser import IDSParser, TreeNode


class TestTreeNode:
    """Test TreeNode class."""

    def test_leaf_node(self):
        """Test leaf node creation."""
        node = TreeNode(leaf="女")
        assert node.is_leaf
        assert node.depth() == 1
        assert node.node_count() == 1
        assert node.to_dict() == {"leaf": "女"}

    def test_binary_node(self):
        """Test binary operator node."""
        left = TreeNode(leaf="女")
        right = TreeNode(leaf="子")
        node = TreeNode(operator="⿰", children=[left, right])

        assert not node.is_leaf
        assert node.depth() == 2
        assert node.node_count() == 3
        assert node.to_dict() == {
            "op": "⿰",
            "children": [{"leaf": "女"}, {"leaf": "子"}]
        }

    def test_nested_tree(self):
        """Test nested tree structure."""
        # 草 = ⿱(⿰(艹,艹), 早)
        top_left = TreeNode(leaf="艹")
        top_right = TreeNode(leaf="艹")
        top = TreeNode(operator="⿰", children=[top_left, top_right])
        bottom = TreeNode(leaf="早")
        root = TreeNode(operator="⿱", children=[top, bottom])

        assert root.depth() == 3
        assert root.node_count() == 6


class TestIDSParser:
    """Test IDSParser class."""

    def test_parse_simple_expression(self):
        """Test parsing simple IDS expression."""
        parser = IDSParser()
        tree, consumed = parser.parse_ids_expression("⿰女子")

        assert consumed == 3
        assert tree.operator == "⿰"
        assert len(tree.children) == 2
        assert tree.children[0].leaf == "女"
        assert tree.children[1].leaf == "子"

    def test_parse_nested_expression(self):
        """Test parsing nested IDS expression."""
        parser = IDSParser()
        tree, consumed = parser.parse_ids_expression("⿱⿰艹艹早")

        assert consumed == 5
        assert tree.operator == "⿱"
        assert tree.children[0].operator == "⿰"
        assert tree.children[1].leaf == "早"

    def test_parse_ternary_operator(self):
        """Test parsing ternary operator."""
        parser = IDSParser()
        tree, consumed = parser.parse_ids_expression("⿲木木木")

        assert consumed == 4
        assert tree.operator == "⿲"
        assert len(tree.children) == 3

    def test_parse_line(self):
        """Test parsing full IDS line."""
        parser = IDSParser()
        result = parser.parse_ids_line("好\t⿰女子")

        assert result is not None
        char, tree = result
        assert char == "好"
        assert tree.operator == "⿰"

    def test_skip_pua_characters(self):
        """Test that PUA characters are skipped."""
        parser = IDSParser()
        # PUA character in expression
        result = parser.parse_ids_line("好\t⿰\uE000子")

        assert result is None
        assert parser.stats["skipped_pua"] == 1

    def test_canonicalize_trees(self):
        """Test tree canonicalization."""
        parser = IDSParser()

        # Shallow tree
        shallow = TreeNode(operator="⿰", children=[
            TreeNode(leaf="女"),
            TreeNode(leaf="子")
        ])

        # Deep tree
        deep = TreeNode(operator="⿱", children=[
            TreeNode(operator="⿰", children=[
                TreeNode(leaf="女"),
                TreeNode(leaf="子")
            ]),
            TreeNode(leaf="一")
        ])

        canonical = parser.canonicalize_trees([shallow, deep])
        assert canonical == shallow  # Prefer shallower tree

    def test_parse_file(self):
        """Test parsing entire file."""
        # Create temporary IDS file
        with tempfile.NamedTemporaryFile(mode="w", delete=False, encoding="utf-8") as f:
            f.write("好\t⿰女子\n")
            f.write("草\t⿱⿰艹艹早\n")
            f.write("# Comment line\n")
            f.write("\n")
            temp_path = Path(f.name)

        try:
            parser = IDSParser()
            result = parser.parse_file(temp_path)

            assert "好" in result
            assert "草" in result
            assert result["好"]["op"] == "⿰"
            assert result["草"]["op"] == "⿱"

        finally:
            temp_path.unlink()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
