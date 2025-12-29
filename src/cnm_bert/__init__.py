"""CNM-BERT: Compositional Network Model for Chinese Language Understanding.

This package implements a dual-path encoder that integrates orthographic structural
information via Ideographic Description Sequences (IDS) into BERT embeddings.
"""

__version__ = "0.1.0"

from .modeling.configuration_cnm import CNMConfig
from .modeling.cnm_model import CNMModel, CNMForMaskedLM
from .tokenization.tokenization_cnm import CNMBertTokenizer

__all__ = [
    "CNMConfig",
    "CNMModel",
    "CNMForMaskedLM",
    "CNMBertTokenizer",
]
