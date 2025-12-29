#!/usr/bin/env python3
"""Debug script to diagnose IDS file parsing issues."""

import sys
from pathlib import Path

def analyze_ids_file(ids_file: Path, num_samples: int = 20):
    """Analyze IDS file format to diagnose parsing issues."""

    print("=" * 70)
    print("IDS File Format Diagnostic")
    print("=" * 70)
    print()

    # Check file exists
    if not ids_file.exists():
        print(f"ERROR: File not found: {ids_file}")
        return

    # Read file
    with open(ids_file, "r", encoding="utf-8") as f:
        lines = f.readlines()

    print(f"Total lines: {len(lines)}")
    print()

    # Sample lines
    print("First 20 lines:")
    print("-" * 70)
    for i, line in enumerate(lines[:num_samples]):
        line_repr = repr(line.rstrip('\n'))
        print(f"Line {i+1}: {line_repr}")
    print()

    # Analyze line formats
    formats = {
        "empty": 0,
        "comment": 0,
        "no_tab": 0,
        "one_tab": 0,
        "multiple_tabs": 0,
        "has_ids_operator": 0,
    }

    # IDS operators
    ids_operators = {"⿰", "⿱", "⿲", "⿳", "⿴", "⿵", "⿶", "⿷", "⿸", "⿹", "⿺", "⿻"}

    sample_valid = []
    sample_invalid = []

    for line in lines:
        line = line.strip()

        if not line:
            formats["empty"] += 1
            continue

        if line.startswith("#"):
            formats["comment"] += 1
            continue

        tab_count = line.count("\t")
        if tab_count == 0:
            formats["no_tab"] += 1
            if len(sample_invalid) < 5:
                sample_invalid.append(repr(line))
        elif tab_count == 1:
            formats["one_tab"] += 1
            # Check if has IDS operator
            if any(op in line for op in ids_operators):
                formats["has_ids_operator"] += 1
                if len(sample_valid) < 5:
                    sample_valid.append(line)
        else:
            formats["multiple_tabs"] += 1

    print("Format Analysis:")
    print("-" * 70)
    for fmt, count in formats.items():
        print(f"  {fmt}: {count}")
    print()

    print("Sample VALID lines (with tab and IDS operator):")
    print("-" * 70)
    for line in sample_valid:
        parts = line.split("\t")
        if len(parts) == 2:
            char, expr = parts
            print(f"  Char: {repr(char)} (len={len(char)})")
            print(f"  IDS:  {repr(expr)} (len={len(expr)})")
            print()
    print()

    print("Sample INVALID lines (no tab):")
    print("-" * 70)
    for line in sample_invalid[:5]:
        print(f"  {line}")
    print()

    # Check for BOM or special characters
    first_line = lines[0] if lines else ""
    if first_line.startswith('\ufeff'):
        print("⚠️  WARNING: File starts with UTF-8 BOM")

    # Check tab character type
    if lines:
        for line in lines[:100]:
            if '\t' in line:
                print(f"Tab character found (ord={ord(chr(9))}): ✓")
                break
        else:
            print("⚠️  WARNING: No tab characters found in first 100 lines")

    print()
    print("=" * 70)
    print("Diagnosis Complete")
    print("=" * 70)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python debug_ids_format.py <ids_file>")
        sys.exit(1)

    ids_file = Path(sys.argv[1])
    analyze_ids_file(ids_file)
