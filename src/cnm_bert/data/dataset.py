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
        import os

        # CRITICAL: Store ABSOLUTE path for multi-process compatibility
        # Worker processes may have different working directories
        self.file_path = os.path.abspath(str(file_path))
        self.max_samples = max_samples
        self._texts = None  # Will be loaded lazily
        self._length = None  # Cache length for efficiency

        # Load data immediately in main process
        self._load_data()

    def _load_data(self):
        """Load text data from file.

        Called during __init__ and after unpickling in worker processes.
        """
        import os

        if self._texts is not None:
            return  # Already loaded

        # Verify file exists (critical for debugging in worker processes)
        if not os.path.exists(self.file_path):
            raise FileNotFoundError(
                f"Dataset file not found: {self.file_path}\n"
                f"Current working directory: {os.getcwd()}\n"
                f"This usually means the file path is not absolute or the file was moved."
            )

        try:
            with open(self.file_path, 'r', encoding='utf-8') as f:
                self._texts = [line.strip() for line in f if line.strip()]
        except Exception as e:
            raise RuntimeError(
                f"Failed to load dataset from {self.file_path}\n"
                f"Error: {e}\n"
                f"Current working directory: {os.getcwd()}"
            ) from e

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
            import sys
            print(f"[DATASET texts property] _texts is None, calling _load_data()", file=sys.stderr)
            try:
                self._load_data()
                print(f"[DATASET texts property] _load_data() succeeded, loaded {len(self._texts)} items", file=sys.stderr)
            except Exception as e:
                print(f"[DATASET texts property] _load_data() FAILED: {e}", file=sys.stderr)
                import traceback
                traceback.print_exc(file=sys.stderr)
                raise
        return self._texts

    def __getstate__(self):
        """Prepare state for pickling (multi-GPU serialization).

        Only pickle the file path and settings, NOT the text data.
        This prevents sending millions of strings across processes.
        Each worker will reload from disk instead.
        """
        import sys
        import os
        print(f"[DATASET __getstate__] Pickling dataset", file=sys.stderr)
        print(f"[DATASET __getstate__] file_path: {self.file_path}", file=sys.stderr)
        print(f"[DATASET __getstate__] file exists: {os.path.exists(self.file_path)}", file=sys.stderr)
        print(f"[DATASET __getstate__] _length: {self._length}", file=sys.stderr)

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
        import sys
        import os
        print(f"[DATASET __setstate__] Unpickling dataset in worker process", file=sys.stderr)
        print(f"[DATASET __setstate__] PID: {os.getpid()}", file=sys.stderr)
        print(f"[DATASET __setstate__] Current dir: {os.getcwd()}", file=sys.stderr)

        self.file_path = state['file_path']
        self.max_samples = state['max_samples']
        self._length = state.get('_length')  # Use cached length if available
        self._texts = None  # Will be loaded on first __getitem__ access

        print(f"[DATASET __setstate__] Restored file_path: {self.file_path}", file=sys.stderr)
        print(f"[DATASET __setstate__] File exists: {os.path.exists(self.file_path)}", file=sys.stderr)
        print(f"[DATASET __setstate__] _texts is None: {self._texts is None}", file=sys.stderr)

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
        import sys
        import traceback

        try:
            # Debug logging for first few items
            if idx < 3:
                print(f"[DATASET __getitem__] Called with idx={idx}", file=sys.stderr)
                print(f"[DATASET __getitem__] self._texts is None: {self._texts is None}", file=sys.stderr)
                print(f"[DATASET __getitem__] hasattr(self, '_texts'): {hasattr(self, '_texts')}", file=sys.stderr)

            # Access via property (handles lazy loading)
            texts = self.texts

            if idx < 3:
                print(f"[DATASET __getitem__] texts length: {len(texts)}", file=sys.stderr)
                print(f"[DATASET __getitem__] Accessing texts[{idx}]", file=sys.stderr)

            result = {"text": texts[idx]}

            if idx < 3:
                print(f"[DATASET __getitem__] Returning: {result}", file=sys.stderr)

            return result

        except Exception as e:
            print(f"[DATASET __getitem__] EXCEPTION at idx={idx}: {e}", file=sys.stderr)
            print(f"[DATASET __getitem__] Traceback:", file=sys.stderr)
            traceback.print_exc(file=sys.stderr)
            print(f"[DATASET __getitem__] self.__dict__ = {self.__dict__}", file=sys.stderr)
            # Re-raise instead of silently returning {}
            raise
