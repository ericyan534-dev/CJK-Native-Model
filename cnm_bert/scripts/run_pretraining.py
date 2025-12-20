"""Pre-training entrypoint for CNM-BERT."""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


# Ensure the repository root is importable when running this script directly.
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch
from transformers import HfArgumentParser, Trainer, TrainingArguments, set_seed
from transformers.trainer_callback import TrainerCallback

from cnm_bert.src.dataset import CNMTextDataset, DataCollatorForWWM
from cnm_bert.src.modeling_cnm import CNMForMaskedLM
from cnm_bert.src.tokenization_cnm import CNMBertTokenizer


def freeze_bert(model: CNMForMaskedLM, freeze: bool = True) -> None:
    for name, param in model.bert.named_parameters():
        if "embeddings" in name:
            param.requires_grad = True
        else:
            param.requires_grad = not freeze


class UnfreezeAfterEpoch(TrainerCallback):
    def __init__(self, model: CNMForMaskedLM, unfreeze_at: float = 1.0):
        self.model = model
        self.unfreeze_at = unfreeze_at

    def on_epoch_begin(self, args, state, control, **kwargs):
        if state.epoch is not None and state.epoch >= self.unfreeze_at:
            freeze_bert(self.model, freeze=False)
        return control


@dataclass
class ModelArguments:
    tree_path: str = field(default="cnm_bert/data/char_to_ids_tree.json")
    struct_dim: int = field(default=256)


@dataclass
class DataArguments:
    dataset_path: str = field(default="cnm_bert/assets/wiki_zh_sample.txt")
    vocab_path: Optional[str] = field(default="cnm_bert/assets/vocab.txt")



def main():
    parser = HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    set_seed(training_args.seed)

    tokenizer = CNMBertTokenizer(vocab_file=data_args.vocab_path, struct_path=model_args.tree_path)
    dataset = CNMTextDataset(data_args.dataset_path)
    data_collator = DataCollatorForWWM(tokenizer=tokenizer)

    model = CNMForMaskedLM.from_pretrained(
        "bert-base-chinese",
        tree_path=Path(model_args.tree_path),
        tokenizer_struct_vocab=tokenizer.struct_index_to_char,
        struct_dim=model_args.struct_dim,
    )

    freeze_bert(model, freeze=True)
    callbacks = [UnfreezeAfterEpoch(model, unfreeze_at=1.0)]

    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=dataset,
        tokenizer=tokenizer,
        callbacks=callbacks,
    )

    trainer.train()


if __name__ == "__main__":
    main()
