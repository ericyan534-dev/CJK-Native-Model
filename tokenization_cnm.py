"""Compatibility shim for importing the CNM tokenizer.

This module allows legacy scripts that import ``tokenization_cnm`` directly
(e.g., ``from tokenization_cnm import CNMTokenizer``) to work without modifying
PYTHONPATH. It simply re-exports the tokenizer implementation defined in
``cnm_bert.src.tokenization_cnm``.
"""

from cnm_bert.src.tokenization_cnm import CNMBertTokenizer

# Backwards-compatible alias used in older scripts.
CNMTokenizer = CNMBertTokenizer

__all__ = ["CNMBertTokenizer", "CNMTokenizer"]
