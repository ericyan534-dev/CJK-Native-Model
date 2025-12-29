"""Prepare pre-training corpus by cleaning and filtering.

This script processes raw text files (Wikipedia, Common Crawl, etc.)
and produces a clean corpus ready for pre-training.
"""

import argparse
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from cnm_bert.etl.corpus_preprocessing import CorpusPreprocessor
from cnm_bert.utils.logging import setup_logger

logger = setup_logger(__name__)


def main():
    parser = argparse.ArgumentParser(
        description="Prepare pre-training corpus for CNM-BERT"
    )
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Input raw corpus file"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/pretrain/corpus_clean.txt"),
        help="Output cleaned corpus file (default: data/pretrain/corpus_clean.txt)"
    )
    parser.add_argument(
        "--min_length",
        type=int,
        default=10,
        help="Minimum sentence length in characters (default: 10)"
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=512,
        help="Maximum sentence length in characters (default: 512)"
    )
    parser.add_argument(
        "--no_dedup",
        action="store_true",
        help="Disable deduplication"
    )
    parser.add_argument(
        "--no_normalize",
        action="store_true",
        help="Disable punctuation normalization"
    )
    parser.add_argument(
        "--stats_output",
        type=Path,
        help="Optional path to save corpus statistics JSON"
    )
    args = parser.parse_args()

    # Validate input
    if not args.input.exists():
        logger.error(f"Input file not found: {args.input}")
        return 1

    # Process corpus
    logger.info("=" * 70)
    logger.info("Preparing pre-training corpus")
    logger.info("=" * 70)
    logger.info(f"Input: {args.input}")
    logger.info(f"Output: {args.output}")
    logger.info(f"Min length: {args.min_length}")
    logger.info(f"Max length: {args.max_length}")
    logger.info(f"Deduplication: {not args.no_dedup}")
    logger.info(f"Normalization: {not args.no_normalize}")
    logger.info("=" * 70)

    try:
        preprocessor = CorpusPreprocessor(
            min_length=args.min_length,
            max_length=args.max_length,
            remove_duplicates=not args.no_dedup,
            normalize_punctuation=not args.no_normalize
        )

        # Process file
        preprocessor.process_file(args.input, args.output)

        # Compute and save statistics
        stats = preprocessor.compute_corpus_statistics(args.output)

        if args.stats_output:
            import json
            args.stats_output.parent.mkdir(parents=True, exist_ok=True)
            with open(args.stats_output, "w", encoding="utf-8") as f:
                # Convert Counter to serializable format
                stats_serializable = {
                    k: (dict(v) if hasattr(v, "items") else v)
                    for k, v in stats.items()
                }
                json.dump(stats_serializable, f, ensure_ascii=False, indent=2)
            logger.info(f"Statistics saved to: {args.stats_output}")

        logger.info("=" * 70)
        logger.info("Corpus preparation complete!")
        logger.info(f"Output: {args.output}")
        logger.info("=" * 70)

        return 0

    except Exception as e:
        logger.error(f"Error preparing corpus: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    exit(main())
