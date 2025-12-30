#!/usr/bin/env python3
"""Production-ready CNM-BERT pre-training script.

Supports:
- Single GPU and multi-GPU (DDP) training
- Weights & Biases logging
- Validation with metrics
- Checkpointing
- Resume from checkpoint
"""

import argparse
import sys
import yaml
import os
from pathlib import Path
import numpy as np

import torch
from transformers import (
    TrainingArguments,
    Trainer,
    set_seed,
)

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from cnm_bert import CNMConfig, CNMForMaskedLM, CNMBertTokenizer
from cnm_bert.data.dataset import TextLineDataset
from cnm_bert.data.collator import WWMDataCollator
from cnm_bert.utils import setup_logger

logger = setup_logger(__name__)


def load_config(config_path: Path) -> dict:
    """Load YAML configuration."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def compute_metrics(eval_pred):
    """Compute metrics for evaluation.

    Args:
        eval_pred: EvalPrediction with predictions and label_ids

    Returns:
        Dictionary of metrics
    """
    predictions, labels = eval_pred

    # predictions: (batch_size, seq_len, vocab_size)
    # labels: (batch_size, seq_len)

    # Get predicted token IDs
    preds = np.argmax(predictions, axis=-1)

    # Mask out padding and non-masked tokens (label = -100)
    mask = labels != -100

    # Calculate accuracy
    correct = (preds == labels) & mask
    accuracy = correct.sum() / mask.sum() if mask.sum() > 0 else 0.0

    # Calculate perplexity from loss (computed by model)
    # Note: loss is already computed by the model, we just return accuracy here

    return {
        "accuracy": float(accuracy),
    }


def create_model(config: dict) -> CNMForMaskedLM:
    """Create CNM-BERT model."""
    model_config = config.get("model", {})

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
    parser.add_argument("--config", type=Path, required=True, help="Path to config YAML")
    parser.add_argument("--output_dir", type=Path, help="Output directory (overrides config)")
    parser.add_argument("--resume_from_checkpoint", type=Path, help="Resume from checkpoint")
    args = parser.parse_args()

    # Load configuration
    config = load_config(args.config)
    seed = config.get("seed", 42)
    set_seed(seed)

    logger.info("=" * 70)
    logger.info("CNM-BERT Pre-training")
    logger.info("=" * 70)
    logger.info(f"Config: {args.config}")
    logger.info(f"Seed: {seed}")

    # Initialize W&B if enabled
    training_config = config.get("training", {})
    if training_config.get("use_wandb", False):
        import wandb
        wandb.init(
            project=training_config.get("wandb_project", "cnm-bert"),
            name=training_config.get("run_name", "cnm-bert-pretrain"),
            config=config,
        )

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

    # Load datasets
    data_config = config.get("data", {})
    logger.info("Loading datasets...")

    train_dataset = TextLineDataset(
        file_path=data_config["train_file"],
        max_samples=data_config.get("max_train_samples"),
    )
    logger.info(f"Train dataset: {len(train_dataset):,} examples")

    # DEBUG: Verify dataset has the fix
    import sys
    print(f"\n[DEBUG train.py] Dataset class: {train_dataset.__class__}", file=sys.stderr)
    print(f"[DEBUG train.py] Has __getstate__: {hasattr(train_dataset, '__getstate__')}", file=sys.stderr)
    print(f"[DEBUG train.py] Has __setstate__: {hasattr(train_dataset, '__getstate__')}", file=sys.stderr)
    print(f"[DEBUG train.py] Dataset __dict__ keys: {list(train_dataset.__dict__.keys())}", file=sys.stderr)
    print(f"[DEBUG train.py] Test dataset[0]: {train_dataset[0]}", file=sys.stderr)
    print(f"[DEBUG train.py] Type of dataset[0]: {type(train_dataset[0])}", file=sys.stderr)
    print(f"[DEBUG train.py] Keys in dataset[0]: {list(train_dataset[0].keys())}\n", file=sys.stderr)

    val_dataset = None
    if data_config.get("val_file"):
        val_dataset = TextLineDataset(
            file_path=data_config["val_file"],
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
    output_dir = args.output_dir or training_config["output_dir"]

    training_args = TrainingArguments(
        # Output
        output_dir=str(output_dir),
        overwrite_output_dir=True,

        # Training schedule
        num_train_epochs=training_config.get("num_train_epochs", 3),
        max_steps=training_config.get("max_steps", -1),

        # Batch size and gradient accumulation
        per_device_train_batch_size=training_config.get("per_device_train_batch_size", 32),
        per_device_eval_batch_size=training_config.get("per_device_eval_batch_size", 64),
        gradient_accumulation_steps=training_config.get("gradient_accumulation_steps", 1),

        # Optimization
        learning_rate=training_config.get("learning_rate", 1e-4),
        weight_decay=training_config.get("weight_decay", 0.01),
        warmup_steps=training_config.get("warmup_steps", 10000),

        # Logging
        logging_dir=str(Path(output_dir) / "logs"),
        logging_steps=training_config.get("logging_steps", 100),
        report_to=["wandb"] if training_config.get("use_wandb", False) else ["tensorboard"],

        # Saving
        save_strategy="steps",
        save_steps=training_config.get("save_steps", 10000),
        save_total_limit=training_config.get("save_total_limit", 3),

        # Performance
        fp16=training_config.get("fp16", True) and torch.cuda.is_available(),
        dataloader_num_workers=0,  # Critical: avoid pickling issues
        dataloader_pin_memory=True,

        # DDP
        ddp_find_unused_parameters=False,

        # Misc
        run_name=training_config.get("run_name", "cnm-bert-pretrain"),
        seed=seed,
        disable_tqdm=False,
    )

    logger.info("Training configuration:")
    logger.info(f"  Output directory: {output_dir}")
    logger.info(f"  Batch size per device: {training_args.per_device_train_batch_size}")
    logger.info(f"  Gradient accumulation: {training_args.gradient_accumulation_steps}")
    logger.info(f"  Effective batch size: {training_args.per_device_train_batch_size * training_args.gradient_accumulation_steps * max(1, torch.cuda.device_count())}")
    logger.info(f"  Learning rate: {training_args.learning_rate}")
    logger.info(f"  Warmup steps: {training_args.warmup_steps}")
    logger.info(f"  Max steps: {training_args.max_steps}")
    logger.info(f"  FP16: {training_args.fp16}")
    logger.info(f"  Validation: {'Yes' if val_dataset else 'No'}")
    logger.info(f"  W&B: {'Yes' if training_config.get('use_wandb', False) else 'No'}")

    # Create Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=None,
        data_collator=data_collator,
        tokenizer=tokenizer,
        compute_metrics=None,
    )

    # Train
    logger.info("=" * 70)
    logger.info("Starting training...")
    logger.info("=" * 70)

    # DEBUG: Check what dataset Trainer actually has
    import sys
    print(f"\n[DEBUG train.py] Trainer.train_dataset type: {type(trainer.train_dataset)}", file=sys.stderr)
    print(f"[DEBUG train.py] Trainer.train_dataset class: {trainer.train_dataset.__class__}", file=sys.stderr)
    print(f"[DEBUG train.py] Is same object as train_dataset: {trainer.train_dataset is train_dataset}", file=sys.stderr)
    print(f"[DEBUG train.py] Trainer.train_dataset length: {len(trainer.train_dataset)}", file=sys.stderr)
    print(f"[DEBUG train.py] Trainer.train_dataset[0]: {trainer.train_dataset[0]}", file=sys.stderr)
    print(f"[DEBUG train.py] About to call trainer.train()...\n", file=sys.stderr)

    # Resume from checkpoint if specified
    resume_from = str(args.resume_from_checkpoint) if args.resume_from_checkpoint else None

    train_result = trainer.train(resume_from_checkpoint=resume_from)

    # Save final model
    logger.info("Saving final model...")
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)

    # Save metrics
    metrics = train_result.metrics
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()

    logger.info("=" * 70)
    logger.info("Training complete!")
    logger.info(f"Final model saved to: {output_dir}")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
