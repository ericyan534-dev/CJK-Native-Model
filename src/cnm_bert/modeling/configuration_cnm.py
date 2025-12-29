"""CNM-BERT configuration."""

from typing import Optional
from transformers import BertConfig


class CNMConfig(BertConfig):
    """Configuration class for CNM-BERT model.

    This extends BertConfig with additional parameters for structural
    encoding via IDS trees.

    Args:
        struct_dim: Dimension of structural embeddings (default: 256)
        tree_path: Path to IDS tree JSON file
        fusion_strategy: How to combine BERT and structural embeddings
            ("concat", "add", or "gate") (default: "concat")
        fusion_dropout: Dropout rate for fusion layer (default: 0.1)
        freeze_bert_encoder: Whether to freeze BERT encoder during training (default: False)
        **kwargs: Additional arguments passed to BertConfig
    """

    model_type = "cnm-bert"

    def __init__(
        self,
        struct_dim: int = 256,
        tree_path: Optional[str] = None,
        fusion_strategy: str = "concat",
        fusion_dropout: float = 0.1,
        freeze_bert_encoder: bool = False,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.struct_dim = struct_dim
        self.tree_path = tree_path
        self.fusion_strategy = fusion_strategy
        self.fusion_dropout = fusion_dropout
        self.freeze_bert_encoder = freeze_bert_encoder
