"""Build canonical IDS lexicon from BabelStone IDS database.

This script parses the IDS file, applies canonicalization heuristics,
and produces the char_to_ids_tree.json file used by CNM-BERT.
"""

import argparse
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from cnm_bert.etl.ids_parser import parse_ids_file
from cnm_bert.utils.logging import setup_logger

logger = setup_logger(__name__)


def main():
    parser = argparse.ArgumentParser(
        description="Build canonical IDS lexicon for CNM-BERT"
    )
    parser.add_argument(
        "--ids_file",
        type=Path,
        required=True,
        help="Path to BabelStone IDS file (ids.txt)"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/processed/char_to_ids_tree.json"),
        help="Output JSON file (default: data/processed/char_to_ids_tree.json)"
    )
    parser.add_argument(
        "--vocab_file",
        type=Path,
        help="Optional BERT vocab file to check coverage"
    )
    parser.add_argument(
        "--save_stats",
        type=Path,
        help="Optional path to save statistics JSON"
    )
    args = parser.parse_args()

    # Validate input
    if not args.ids_file.exists():
        logger.error(f"IDS file not found: {args.ids_file}")
        logger.info("Download it with: python scripts/download_ids.py")
        return 1

    # Parse and canonicalize
    logger.info("=" * 70)
    logger.info("Building canonical IDS lexicon")
    logger.info("=" * 70)

    try:
        char_to_tree = parse_ids_file(
            args.ids_file,
            args.output,
            vocab_file=args.vocab_file
        )

        # Save statistics if requested
        if args.save_stats:
            import json
            from cnm_bert.etl.ids_parser import IDSParser

            parser_obj = IDSParser()
            stats = parser_obj.compute_statistics(char_to_tree)

            args.save_stats.parent.mkdir(parents=True, exist_ok=True)
            with open(args.save_stats, "w", encoding="utf-8") as f:
                json.dump(stats, f, ensure_ascii=False, indent=2)
            logger.info(f"Statistics saved to: {args.save_stats}")

        logger.info("=" * 70)
        logger.info("IDS lexicon build complete!")
        logger.info(f"Output: {args.output}")
        logger.info("=" * 70)

        return 0

    except Exception as e:
        logger.error(f"Error building IDS lexicon: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    exit(main())
