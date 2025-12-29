"""Metrics computation utilities."""

import torch
import numpy as np
from typing import Dict, Union


def compute_perplexity(loss: Union[float, torch.Tensor]) -> float:
    """Compute perplexity from cross-entropy loss.

    Args:
        loss: Cross-entropy loss value

    Returns:
        Perplexity score
    """
    if isinstance(loss, torch.Tensor):
        loss = loss.item()
    return np.exp(loss)


def compute_mlm_accuracy(predictions: torch.Tensor, labels: torch.Tensor, ignore_index: int = -100) -> Dict[str, float]:
    """Compute masked language modeling accuracy.

    Args:
        predictions: Model predictions (batch_size, seq_len, vocab_size)
        labels: Ground truth labels (batch_size, seq_len)
        ignore_index: Index to ignore in accuracy computation (default: -100)

    Returns:
        Dictionary with accuracy metrics
    """
    # Get predicted token IDs
    pred_ids = predictions.argmax(dim=-1)

    # Create mask for valid positions (not ignored)
    mask = labels != ignore_index

    # Compute accuracy only on masked positions
    correct = (pred_ids == labels) & mask
    accuracy = correct.sum().float() / mask.sum().float()

    # Top-5 accuracy
    top5_preds = predictions.topk(5, dim=-1).indices
    top5_correct = (top5_preds == labels.unsqueeze(-1)).any(dim=-1) & mask
    top5_accuracy = top5_correct.sum().float() / mask.sum().float()

    return {
        "accuracy": accuracy.item(),
        "top5_accuracy": top5_accuracy.item(),
        "num_masked_tokens": mask.sum().item(),
    }
