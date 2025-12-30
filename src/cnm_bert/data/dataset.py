"""Rock-solid dataset implementation for CNM-BERT pre-training."""

from pathlib import Path
from typing import Optional
import torch
from torch.utils.data import Dataset


class TextLineDataset(Dataset):
    """Simple, bulletproof text dataset.

    Loads all text lines into memory. No lazy loading, no pickling issues.
    Every attribute is a basic Python type (str, list, int).
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

        # Load all data into memory immediately
        with open(self.file_path, 'r', encoding='utf-8') as f:
            self.texts = [line.strip() for line in f if line.strip()]

        if not self.texts:
            raise ValueError(f"No text found in {self.file_path}")

        if max_samples:
            self.texts = self.texts[:max_samples]

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        # Return plain dict with plain string
        return {"text": self.texts[idx]}
