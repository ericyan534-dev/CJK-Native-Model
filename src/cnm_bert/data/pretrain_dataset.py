"""Pre-training dataset for CNM-BERT."""

from pathlib import Path
from typing import Dict, Iterator, Optional

from torch.utils.data import Dataset, IterableDataset


class PreTrainingDataset(Dataset):
    """Line-by-line text dataset for pre-training.

    Each line in the file is treated as a separate document.

    Args:
        file_path: Path to text file (one sentence per line)
        max_samples: Optional limit on number of samples (for debugging)
    """

    def __init__(
        self,
        file_path: Path,
        max_samples: Optional[int] = None
    ):
        self.file_path = Path(file_path)
        self.max_samples = max_samples
        self.lines = None  # Will be loaded lazily
        self._load_data()

    def _load_data(self):
        """Load data from file. Called in __init__ and __getstate__ for pickling."""
        if not self.file_path.exists():
            raise FileNotFoundError(f"Corpus file not found: {self.file_path}")

        # Load all lines into memory (fast random access)
        with open(self.file_path, "r", encoding="utf-8") as f:
            self.lines = [line.strip() for line in f if line.strip()]

        if not self.lines:
            raise ValueError(f"Corpus file is empty: {self.file_path}")

        if self.max_samples:
            self.lines = self.lines[:self.max_samples]

    def __getstate__(self):
        """Custom pickling to handle multiprocessing."""
        # Return state without the lines data - will be reloaded in worker
        return {'file_path': self.file_path, 'max_samples': self.max_samples}

    def __setstate__(self, state):
        """Custom unpickling to reload data in worker processes."""
        self.file_path = state['file_path']
        self.max_samples = state['max_samples']
        self.lines = None
        self._load_data()

    def __len__(self) -> int:
        if self.lines is None:
            self._load_data()
        return len(self.lines)

    def __getitem__(self, idx: int) -> Dict[str, str]:
        if self.lines is None:
            self._load_data()

        if idx < 0 or idx >= len(self.lines):
            raise IndexError(f"Index {idx} out of range for dataset of size {len(self.lines)}")

        text = self.lines[idx]
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
