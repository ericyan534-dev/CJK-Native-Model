"""Rock-solid dataset implementation for CNM-BERT pre-training.

This dataset is designed to work correctly with multi-GPU distributed training.
It uses __getstate__/__setstate__ to avoid pickling large text data across processes.
"""

from pathlib import Path
from typing import Optional, List
import torch
from torch.utils.data import Dataset


class TextLineDataset(Dataset):
    """Simple, bulletproof text dataset for distributed training.

    Designed to work with PyTorch DDP and Accelerate:
    - Only pickles file path (not millions of text lines)
    - Each worker process reloads data from disk
    - Lazy loading via property ensures data availability

    Every attribute is a basic Python type (str, list, int) for compatibility.
    """

    def __init__(self, file_path: str, max_samples: Optional[int] = None):
        """Initialize dataset.

        Args:
            file_path: Path to text file (one line per example)
            max_samples: Optional limit on examples
        """
        # Store everything as basic types (picklable)
        self.file_path = str(file_path)
        self.max_samples = max_samples
        self._texts = None  # Will be loaded lazily
        self._length = None  # Cache length for efficiency

        # Load data immediately in main process
        self._load_data()

    def _load_data(self):
        """Load text data from file.

        Called during __init__ and after unpickling in worker processes.
        """
        if self._texts is not None:
            return  # Already loaded

        with open(self.file_path, 'r', encoding='utf-8') as f:
            self._texts = [line.strip() for line in f if line.strip()]

        if not self._texts:
            raise ValueError(f"No text found in {self.file_path}")

        if self.max_samples:
            self._texts = self._texts[:self.max_samples]

        # Cache length
        self._length = len(self._texts)

    @property
    def texts(self) -> List[str]:
        """Lazy-load texts if not available (handles unpickling).

        This property ensures data is always available, even after
        the dataset is unpickled in a worker process.
        """
        if self._texts is None:
            self._load_data()
        return self._texts

    def __getstate__(self):
        """Prepare state for pickling (multi-GPU serialization).

        Only pickle the file path and settings, NOT the text data.
        This prevents sending millions of strings across processes.
        Each worker will reload from disk instead.
        """
        return {
            'file_path': self.file_path,
            'max_samples': self.max_samples,
            '_length': self._length,  # Cache length to avoid reload for __len__
        }

    def __setstate__(self, state):
        """Restore from pickle (in worker process).

        Reinitialize with file path and settings.
        Data will be reloaded lazily on first access.
        """
        self.file_path = state['file_path']
        self.max_samples = state['max_samples']
        self._length = state.get('_length')  # Use cached length if available
        self._texts = None  # Will be loaded on first __getitem__ access

    def __len__(self):
        """Return dataset length.

        Uses cached length to avoid reloading data just for size check.
        """
        if self._length is not None:
            return self._length
        return len(self.texts)  # Falls back to loading if needed

    def __getitem__(self, idx):
        """Get item at index.

        Returns:
            dict: {"text": str} format expected by collator
        """
        # Access via property (handles lazy loading)
        return {"text": self.texts[idx]}
