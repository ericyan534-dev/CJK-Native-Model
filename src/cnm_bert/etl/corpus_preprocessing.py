"""Corpus preprocessing utilities for pre-training."""

import re
import unicodedata
from pathlib import Path
from typing import Iterator, Optional, Set
from collections import Counter

from ..utils.logging import setup_logger

logger = setup_logger(__name__)


class CorpusPreprocessor:
    """Preprocessor for Chinese text corpora."""

    def __init__(
        self,
        min_length: int = 10,
        max_length: int = 512,
        remove_duplicates: bool = True,
        normalize_punctuation: bool = True,
    ):
        """Initialize corpus preprocessor.

        Args:
            min_length: Minimum sentence length (characters)
            max_length: Maximum sentence length (characters)
            remove_duplicates: Whether to deduplicate sentences
            normalize_punctuation: Whether to normalize full/half-width
        """
        self.min_length = min_length
        self.max_length = max_length
        self.remove_duplicates = remove_duplicates
        self.normalize_punctuation = normalize_punctuation
        self.seen_sentences: Set[str] = set() if remove_duplicates else None

        self.stats = {
            "total_lines": 0,
            "kept": 0,
            "filtered_empty": 0,
            "filtered_length": 0,
            "filtered_duplicate": 0,
            "filtered_non_chinese": 0,
        }

    def is_chinese_dominant(self, text: str, threshold: float = 0.5) -> bool:
        """Check if text is predominantly Chinese characters.

        Args:
            text: Input text
            threshold: Minimum ratio of Chinese characters

        Returns:
            True if Chinese character ratio >= threshold
        """
        if not text:
            return False

        chinese_count = sum(1 for char in text if self._is_cjk(char))
        ratio = chinese_count / len(text)
        return ratio >= threshold

    @staticmethod
    def _is_cjk(char: str) -> bool:
        """Check if character is CJK Unified Ideograph."""
        if len(char) != 1:
            return False
        cp = ord(char)
        return (
            0x4E00 <= cp <= 0x9FFF or      # CJK Unified Ideographs
            0x3400 <= cp <= 0x4DBF or      # CJK Extension A
            0x20000 <= cp <= 0x2A6DF or    # CJK Extension B
            0x2A700 <= cp <= 0x2B73F or    # CJK Extension C
            0x2B740 <= cp <= 0x2B81F or    # CJK Extension D
            0x2B820 <= cp <= 0x2CEAF or    # CJK Extension E
            0x2CEB0 <= cp <= 0x2EBEF       # CJK Extension F
        )

    def normalize_text(self, text: str) -> str:
        """Normalize text (punctuation, whitespace, etc.).

        Args:
            text: Input text

        Returns:
            Normalized text
        """
        # Remove control characters
        text = "".join(char for char in text if unicodedata.category(char) != "Cc")

        # Normalize whitespace
        text = re.sub(r"\s+", " ", text)

        # Normalize punctuation (full-width <-> half-width)
        if self.normalize_punctuation:
            # Convert full-width ASCII to half-width
            text = unicodedata.normalize("NFKC", text)

        # Strip leading/trailing whitespace
        text = text.strip()

        return text

    def clean_sentence(self, sentence: str) -> Optional[str]:
        """Clean and validate a single sentence.

        Args:
            sentence: Input sentence

        Returns:
            Cleaned sentence or None if should be filtered
        """
        # Normalize
        sentence = self.normalize_text(sentence)

        # Filter empty
        if not sentence:
            self.stats["filtered_empty"] += 1
            return None

        # Filter by length
        if len(sentence) < self.min_length or len(sentence) > self.max_length:
            self.stats["filtered_length"] += 1
            return None

        # Filter non-Chinese
        if not self.is_chinese_dominant(sentence):
            self.stats["filtered_non_chinese"] += 1
            return None

        # Deduplicate
        if self.remove_duplicates:
            if sentence in self.seen_sentences:
                self.stats["filtered_duplicate"] += 1
                return None
            self.seen_sentences.add(sentence)

        self.stats["kept"] += 1
        return sentence

    def process_file(
        self,
        input_file: Path,
        output_file: Path,
        encoding: str = "utf-8"
    ) -> None:
        """Process corpus file line-by-line.

        Args:
            input_file: Input text file
            output_file: Output cleaned file
            encoding: File encoding
        """
        logger.info(f"Processing corpus: {input_file}")

        output_file.parent.mkdir(parents=True, exist_ok=True)

        with open(input_file, "r", encoding=encoding) as f_in, \
             open(output_file, "w", encoding=encoding) as f_out:

            for line in f_in:
                self.stats["total_lines"] += 1

                # Progress logging
                if self.stats["total_lines"] % 100000 == 0:
                    logger.info(f"Processed {self.stats['total_lines']} lines...")

                cleaned = self.clean_sentence(line)
                if cleaned:
                    f_out.write(cleaned + "\n")

        # Log statistics
        logger.info(f"Preprocessing complete:")
        logger.info(f"  Total lines: {self.stats['total_lines']}")
        logger.info(f"  Kept: {self.stats['kept']}")
        logger.info(f"  Filtered (empty): {self.stats['filtered_empty']}")
        logger.info(f"  Filtered (length): {self.stats['filtered_length']}")
        logger.info(f"  Filtered (non-Chinese): {self.stats['filtered_non_chinese']}")
        logger.info(f"  Filtered (duplicate): {self.stats['filtered_duplicate']}")
        logger.info(f"Saved to: {output_file}")

    def compute_corpus_statistics(self, corpus_file: Path) -> dict:
        """Compute statistics about processed corpus.

        Args:
            corpus_file: Path to corpus file

        Returns:
            Dictionary of statistics
        """
        logger.info(f"Computing corpus statistics: {corpus_file}")

        char_counter = Counter()
        sentence_lengths = []
        total_sentences = 0

        with open(corpus_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    total_sentences += 1
                    sentence_lengths.append(len(line))
                    char_counter.update(line)

        stats = {
            "total_sentences": total_sentences,
            "total_characters": sum(char_counter.values()),
            "unique_characters": len(char_counter),
            "avg_sentence_length": sum(sentence_lengths) / len(sentence_lengths) if sentence_lengths else 0,
            "min_sentence_length": min(sentence_lengths) if sentence_lengths else 0,
            "max_sentence_length": max(sentence_lengths) if sentence_lengths else 0,
            "top_50_chars": char_counter.most_common(50),
        }

        logger.info(f"Corpus statistics:")
        logger.info(f"  Sentences: {stats['total_sentences']:,}")
        logger.info(f"  Characters: {stats['total_characters']:,}")
        logger.info(f"  Unique chars: {stats['unique_characters']:,}")
        logger.info(f"  Avg length: {stats['avg_sentence_length']:.2f}")

        return stats


def clean_corpus(
    input_file: Path,
    output_file: Path,
    min_length: int = 10,
    max_length: int = 512,
    remove_duplicates: bool = True
) -> None:
    """Convenience function for corpus cleaning.

    Args:
        input_file: Input corpus file
        output_file: Output cleaned file
        min_length: Minimum sentence length
        max_length: Maximum sentence length
        remove_duplicates: Whether to deduplicate
    """
    preprocessor = CorpusPreprocessor(
        min_length=min_length,
        max_length=max_length,
        remove_duplicates=remove_duplicates
    )
    preprocessor.process_file(input_file, output_file)
    preprocessor.compute_corpus_statistics(output_file)
