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

        if not self.file_path.exists():
            raise FileNotFoundError(f"Corpus file not found: {self.file_path}")

        # Load all lines into memory (fast random access)
        with open(self.file_path, "r", encoding="utf-8") as f:
            self.lines = [line.strip() for line in f if line.strip()]

        if max_samples:
            self.lines = self.lines[:max_samples]

    def __len__(self) -> int:
        return len(self.lines)

    def __getitem__(self, idx: int) -> Dict[str, str]:
        return {"text": self.lines[idx]}


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
