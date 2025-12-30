"""Pre-training script for CNM-BERT with distributed training support.

This script trains CNM-BERT using masked language modeling (MLM) with
Whole Word Masking (WWM). Supports multi-GPU training via PyTorch DDP.

Example usage:
    # Single GPU
    python scripts/03_pretrain.py --config experiments/configs/pretrain_base.yaml

    # Multi-GPU (8 GPUs)
    torchrun --nproc_per_node=8 scripts/03_pretrain.py --config experiments/configs/pretrain_base.yaml
"""

import argparse
import sys
import yaml
from pathlib import Path

import torch
from transformers import (
    TrainingArguments,
    Trainer,
    set_seed,
)

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from cnm_bert import CNMConfig, CNMForMaskedLM, CNMBertTokenizer
from cnm_bert.data import PreTrainingDataset, WWMDataCollator
from cnm_bert.utils import setup_logger, compute_perplexity

logger = setup_logger(__name__)


def load_config(config_path: Path) -> dict:
    """Load training configuration from YAML file.

    Args:
        config_path: Path to config YAML

    Returns:
        Configuration dictionary
    """
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config


def create_model(config: dict) -> CNMForMaskedLM:
    """Create CNM-BERT model.

    Args:
        config: Configuration dictionary

    Returns:
        CNMForMaskedLM model
    """
    model_config = config.get("model", {})

    # Check if we should initialize from BERT
    if model_config.get("init_from_bert", False):
        logger.info("Initializing from pre-trained BERT...")
        model = CNMForMaskedLM.from_bert_pretrained(
            bert_model_name=model_config.get("bert_model_name", "bert-base-chinese"),
            tree_path=Path(model_config["tree_path"]),
            struct_dim=model_config.get("struct_dim", 256),
            fusion_strategy=model_config.get("fusion_strategy", "concat"),
            freeze_bert_encoder=model_config.get("freeze_bert_encoder", False),
        )
    else:
        logger.info("Initializing CNM-BERT from scratch...")
        cnm_config = CNMConfig(
            vocab_size=model_config.get("vocab_size", 21128),
            hidden_size=model_config.get("hidden_size", 768),
            num_hidden_layers=model_config.get("num_hidden_layers", 12),
            num_attention_heads=model_config.get("num_attention_heads", 12),
            intermediate_size=model_config.get("intermediate_size", 3072),
            struct_dim=model_config.get("struct_dim", 256),
            tree_path=model_config["tree_path"],
            fusion_strategy=model_config.get("fusion_strategy", "concat"),
            freeze_bert_encoder=model_config.get("freeze_bert_encoder", False),
        )
        model = CNMForMaskedLM(cnm_config)

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    logger.info(f"Model created:")
    logger.info(f"  Total parameters: {total_params:,}")
    logger.info(f"  Trainable parameters: {trainable_params:,}")

    return model


def main():
    parser = argparse.ArgumentParser(description="Pre-train CNM-BERT")
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Path to training config YAML"
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        help="Output directory (overrides config)"
    )
    parser.add_argument(
        "--local_rank",
        type=int,
        default=-1,
        help="Local rank for distributed training (set by torchrun)"
    )
    args = parser.parse_args()

    # Load configuration
    config = load_config(args.config)

    # Set seed for reproducibility
    seed = config.get("seed", 42)
    set_seed(seed)

    logger.info("=" * 70)
    logger.info("CNM-BERT Pre-training")
    logger.info("=" * 70)
    logger.info(f"Config: {args.config}")
    logger.info(f"Seed: {seed}")

    # Create tokenizer
    tokenizer_config = config.get("tokenizer", {})
    logger.info("Loading tokenizer...")
    tokenizer = CNMBertTokenizer.from_pretrained(
        tokenizer_config.get("vocab_path", "bert-base-chinese"),
        struct_path=tokenizer_config["struct_path"],
    )
    logger.info(f"Tokenizer loaded: vocab_size={tokenizer.vocab_size}")

    # Create model
    model = create_model(config)

    # Load dataset
    data_config = config.get("data", {})
    logger.info("Loading dataset...")
    train_dataset = PreTrainingDataset(
        file_path=Path(data_config["train_file"]),
        max_samples=data_config.get("max_train_samples"),
    )
    logger.info(f"Train dataset: {len(train_dataset):,} examples")

    # Validation dataset (optional)
    val_dataset = None
    if data_config.get("val_file"):
        val_dataset = PreTrainingDataset(
            file_path=Path(data_config["val_file"]),
            max_samples=data_config.get("max_val_samples"),
        )
        logger.info(f"Validation dataset: {len(val_dataset):,} examples")

    # Create data collator
    collator_config = config.get("collator", {})
    data_collator = WWMDataCollator(
        tokenizer=tokenizer,
        mlm_probability=collator_config.get("mlm_probability", 0.15),
        max_seq_length=collator_config.get("max_seq_length", 512),
    )

    # Training arguments
    training_config = config.get("training", {})
    output_dir = args.output_dir or training_config["output_dir"]

    training_args = TrainingArguments(
        output_dir=str(output_dir),
        overwrite_output_dir=True,
        num_train_epochs=training_config.get("num_train_epochs", 3),
        per_device_train_batch_size=training_config.get("per_device_train_batch_size", 32),
        per_device_eval_batch_size=training_config.get("per_device_eval_batch_size", 32),
        gradient_accumulation_steps=training_config.get("gradient_accumulation_steps", 1),
        learning_rate=training_config.get("learning_rate", 1e-4),
        weight_decay=training_config.get("weight_decay", 0.01),
        warmup_steps=training_config.get("warmup_steps", 10000),
        max_steps=training_config.get("max_steps", -1),
        logging_dir=str(Path(output_dir) / "logs"),
        logging_steps=training_config.get("logging_steps", 100),
        save_steps=training_config.get("save_steps", 10000),
        save_total_limit=training_config.get("save_total_limit", 3),
        eval_steps=training_config.get("eval_steps", 50000),
        evaluation_strategy="steps" if val_dataset else "no",
        fp16=training_config.get("fp16", True),
        dataloader_num_workers=training_config.get("num_workers", 4),
        ddp_find_unused_parameters=False,
        report_to=["tensorboard", "wandb"] if training_config.get("use_wandb", False) else ["tensorboard"],
        run_name=training_config.get("run_name", "cnm-bert-pretrain"),
        seed=seed,
    )

    logger.info("Training configuration:")
    logger.info(f"  Output directory: {output_dir}")
    logger.info(f"  Batch size per device: {training_args.per_device_train_batch_size}")
    logger.info(f"  Gradient accumulation steps: {training_args.gradient_accumulation_steps}")
    logger.info(f"  Total batch size: {training_args.per_device_train_batch_size * training_args.gradient_accumulation_steps * training_args.world_size}")
    logger.info(f"  Learning rate: {training_args.learning_rate}")
    logger.info(f"  Warmup steps: {training_args.warmup_steps}")
    logger.info(f"  Max steps: {training_args.max_steps}")
    logger.info(f"  FP16: {training_args.fp16}")
    logger.info(f"  World size: {training_args.world_size}")

    # Create Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
    )

    # Train
    logger.info("=" * 70)
    logger.info("Starting training...")
    logger.info("=" * 70)

    train_result = trainer.train()

    # Save final model
    logger.info("Saving final model...")
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)

    # Log metrics
    metrics = train_result.metrics
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)

    logger.info("=" * 70)
    logger.info("Training complete!")
    logger.info(f"Final model saved to: {output_dir}")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
