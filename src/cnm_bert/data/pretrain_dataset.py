"""Pre-training dataset for CNM-BERT.

Designed to work correctly with multi-GPU distributed training using proper
pickling strategies to avoid serializing large datasets across processes.
"""

from pathlib import Path
from typing import Dict, Iterator, Optional, List

from torch.utils.data import Dataset, IterableDataset


class PreTrainingDataset(Dataset):
    """Line-by-line text dataset for pre-training with DDP support.

    Each line in the file is treated as a separate document.
    Uses __getstate__/__setstate__ to avoid pickling millions of text lines.

    Args:
        file_path: Path to text file (one sentence per line)
        max_samples: Optional limit on number of samples (for debugging)
    """

    def __init__(
        self,
        file_path: Path,
        max_samples: Optional[int] = None
    ):
        # Store as string to ensure pickling works
        self.file_path_str = str(Path(file_path).resolve())
        self.max_samples = max_samples
        self._lines = None  # Will be loaded lazily
        self._length = None  # Cache length

        # Load data immediately in main process
        self._load_data()

    def _load_data(self):
        """Load text data from file.

        Called during __init__ and after unpickling in worker processes.
        """
        if self._lines is not None:
            return  # Already loaded

        file_path_obj = Path(self.file_path_str)
        if not file_path_obj.exists():
            raise FileNotFoundError(f"Corpus file not found: {self.file_path_str}")

        # Load all lines into memory (fast random access)
        with open(self.file_path_str, "r", encoding="utf-8") as f:
            self._lines = [line.strip() for line in f if line.strip()]

        if not self._lines:
            raise ValueError(f"Corpus file is empty: {self.file_path_str}")

        if self.max_samples:
            self._lines = self._lines[:self.max_samples]

        # Cache length
        self._length = len(self._lines)

    @property
    def lines(self) -> List[str]:
        """Lazy-load lines if not available (handles unpickling)."""
        if self._lines is None:
            self._load_data()
        return self._lines

    def __getstate__(self):
        """Prepare state for pickling (multi-GPU serialization).

        Only pickle the file path and settings, NOT the text data.
        """
        return {
            'file_path_str': self.file_path_str,
            'max_samples': self.max_samples,
            '_length': self._length,
        }

    def __setstate__(self, state):
        """Restore from pickle (in worker process).

        Data will be reloaded lazily on first access.
        """
        self.file_path_str = state['file_path_str']
        self.max_samples = state['max_samples']
        self._length = state.get('_length')
        self._lines = None  # Will be loaded on first access

    def __len__(self) -> int:
        """Return dataset length using cached value if available."""
        if self._length is not None:
            return self._length
        return len(self.lines)

    def __getitem__(self, idx: int) -> Dict[str, str]:
        """Get item at index via property (handles lazy loading)."""
        lines = self.lines  # Access via property

        if idx < 0 or idx >= len(lines):
            raise IndexError(f"Index {idx} out of range for dataset of size {len(lines)}")

        text = lines[idx]
        if not text:
            raise ValueError(f"Empty text at index {idx}")

        return {"text": text}


class StreamingPreTrainingDataset(IterableDataset):
    """Streaming dataset for large corpora.

    This dataset streams lines from disk without loading the entire
    file into memory. Useful for very large corpora.

    Args:
        file_path: Path to text file
        max_samples: Optional limit on samples (for debugging)
    """

    def __init__(
        self,
        file_path: Path,
        max_samples: Optional[int] = None
    ):
        self.file_path = Path(file_path)
        self.max_samples = max_samples

        if not self.file_path.exists():
            raise FileNotFoundError(f"Corpus file not found: {self.file_path}")

    def __iter__(self) -> Iterator[Dict[str, str]]:
        """Iterate through lines in file."""
        with open(self.file_path, "r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                if self.max_samples and i >= self.max_samples:
                    break

                line = line.strip()
                if line:
                    yield {"text": line}


class MultiFilePreTrainingDataset(IterableDataset):
    """Dataset that streams from multiple corpus files.

    Args:
        file_paths: List of corpus file paths
        max_samples: Optional limit on total samples
    """

    def __init__(
        self,
        file_paths: list[Path],
        max_samples: Optional[int] = None
    ):
        self.file_paths = [Path(p) for p in file_paths]
        self.max_samples = max_samples

        # Validate files exist
        for path in self.file_paths:
            if not path.exists():
                raise FileNotFoundError(f"Corpus file not found: {path}")

    def __iter__(self) -> Iterator[Dict[str, str]]:
        """Iterate through all files."""
        count = 0

        for file_path in self.file_paths:
            with open(file_path, "r", encoding="utf-8") as f:
                for line in f:
                    if self.max_samples and count >= self.max_samples:
                        return

                    line = line.strip()
                    if line:
                        yield {"text": line}
                        count += 1
