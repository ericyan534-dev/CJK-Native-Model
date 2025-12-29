"""Utility functions for logging, metrics, and helpers."""

from .logging import setup_logger
from .metrics import compute_perplexity, compute_mlm_accuracy

# W&B callback (optional import)
try:
    from .wandb_callback import CNMWandbCallback, setup_wandb_training
    __all__ = ["setup_logger", "compute_perplexity", "compute_mlm_accuracy", "CNMWandbCallback", "setup_wandb_training"]
except ImportError:
    __all__ = ["setup_logger", "compute_perplexity", "compute_mlm_accuracy"]
