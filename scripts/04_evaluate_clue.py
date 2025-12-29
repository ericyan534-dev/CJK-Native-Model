"""Evaluation script for CLUE benchmarks.

This script fine-tunes and evaluates CNM-BERT on CLUE benchmark tasks:
- TNEWS: Text classification (15 classes)
- AFQMC: Sentence pair matching
- CLUEWSC: Winograd Schema Challenge
- CSL: Keyword recognition
- CMRC2018: Reading comprehension

Example usage:
    python scripts/04_evaluate_clue.py \
        --model_path experiments/logs/cnm_bert_base/checkpoint-best \
        --tasks tnews afqmc \
        --output_dir experiments/results
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List

import torch
import numpy as np
from datasets import load_dataset
from transformers import (
    TrainingArguments,
    Trainer,
    set_seed,
)
from sklearn.metrics import accuracy_score, f1_score

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from cnm_bert import CNMForMaskedLM, CNMBertTokenizer
from cnm_bert.utils import setup_logger

logger = setup_logger(__name__)


TASK_CONFIGS = {
    "tnews": {
        "name": "TNEWS",
        "metric": "accuracy",
        "num_labels": 15,
        "description": "Text classification (news categories)",
    },
    "afqmc": {
        "name": "AFQMC",
        "metric": "accuracy",
        "num_labels": 2,
        "description": "Sentence pair matching",
    },
    "cluewsc": {
        "name": "CLUEWSC",
        "metric": "accuracy",
        "num_labels": 2,
        "description": "Winograd Schema Challenge",
    },
    "csl": {
        "name": "CSL",
        "metric": "accuracy",
        "num_labels": 2,
        "description": "Keyword recognition",
    },
    "cmrc2018": {
        "name": "CMRC2018",
        "metric": "f1",
        "num_labels": None,  # Span extraction task
        "description": "Reading comprehension",
    },
}


def compute_metrics_classification(eval_pred):
    """Compute metrics for classification tasks.

    Args:
        eval_pred: Tuple of (predictions, labels)

    Returns:
        Dictionary of metrics
    """
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=-1)

    accuracy = accuracy_score(labels, predictions)
    f1 = f1_score(labels, predictions, average="macro")

    return {
        "accuracy": accuracy,
        "f1": f1,
    }


def load_clue_dataset(task: str, data_dir: Path):
    """Load CLUE dataset for specific task.

    Args:
        task: Task name (tnews, afqmc, etc.)
        data_dir: Directory containing CLUE data

    Returns:
        Dataset splits dictionary
    """
    logger.info(f"Loading CLUE dataset: {task}")

    # Check if local files exist
    task_dir = data_dir / task
    if task_dir.exists():
        # Load from local files
        dataset = load_dataset(
            "json",
            data_files={
                "train": str(task_dir / "train.json"),
                "validation": str(task_dir / "dev.json"),
                "test": str(task_dir / "test.json"),
            }
        )
    else:
        # Try to load from HuggingFace datasets
        try:
            dataset = load_dataset("clue", task)
        except Exception as e:
            logger.error(f"Failed to load dataset for {task}: {e}")
            logger.info(f"Please download CLUE datasets to {data_dir}")
            raise

    logger.info(f"Dataset loaded: {task}")
    logger.info(f"  Train: {len(dataset['train'])} examples")
    logger.info(f"  Dev: {len(dataset['validation'])} examples")
    if "test" in dataset:
        logger.info(f"  Test: {len(dataset['test'])} examples")

    return dataset


def preprocess_function_classification(examples, tokenizer, max_length=512):
    """Preprocess examples for classification tasks.

    Args:
        examples: Batch of examples
        tokenizer: Tokenizer instance
        max_length: Maximum sequence length

    Returns:
        Tokenized inputs
    """
    # Different tasks have different input fields
    if "sentence" in examples:
        # Single sentence tasks (TNEWS, CSL)
        texts = examples["sentence"]
    elif "sentence1" in examples:
        # Sentence pair tasks (AFQMC)
        texts = examples["sentence1"]
        texts_pair = examples["sentence2"]
    else:
        raise ValueError("Unknown input format")

    # Tokenize
    if "sentence1" in examples:
        result = tokenizer(
            texts,
            texts_pair,
            padding="max_length",
            max_length=max_length,
            truncation=True,
            return_struct_ids=True,
        )
    else:
        result = tokenizer(
            texts,
            padding="max_length",
            max_length=max_length,
            truncation=True,
            return_struct_ids=True,
        )

    # Add labels
    if "label" in examples:
        result["labels"] = examples["label"]

    return result


def fine_tune_and_evaluate(
    model_path: Path,
    task: str,
    data_dir: Path,
    output_dir: Path,
    num_epochs: int = 3,
    batch_size: int = 32,
    learning_rate: float = 2e-5,
    seed: int = 42,
) -> Dict:
    """Fine-tune and evaluate model on a CLUE task.

    Args:
        model_path: Path to pre-trained CNM-BERT
        task: Task name
        data_dir: Directory containing CLUE data
        output_dir: Output directory for results
        num_epochs: Number of fine-tuning epochs
        batch_size: Batch size
        learning_rate: Learning rate
        seed: Random seed

    Returns:
        Dictionary of evaluation metrics
    """
    set_seed(seed)

    logger.info("=" * 70)
    logger.info(f"Fine-tuning on {TASK_CONFIGS[task]['name']}")
    logger.info("=" * 70)

    # Load model and tokenizer
    logger.info(f"Loading model from {model_path}")
    tokenizer = CNMBertTokenizer.from_pretrained(model_path)

    # For classification tasks, we need a sequence classification head
    # For now, we'll use the MLM model and add classification head
    # In production, create CNMForSequenceClassification
    from transformers import BertForSequenceClassification, BertConfig

    # Create classification model from CNM checkpoint
    # NOTE: This is a simplified version. In production, create CNMForSequenceClassification
    config = BertConfig.from_pretrained(model_path)
    config.num_labels = TASK_CONFIGS[task]["num_labels"]
    model = BertForSequenceClassification(config)

    # TODO: Load CNM weights properly (requires CNMForSequenceClassification class)
    logger.warning("Using BERT classification head. For full CNM support, implement CNMForSequenceClassification")

    # Load dataset
    dataset = load_clue_dataset(task, data_dir)

    # Preprocess
    tokenized_dataset = dataset.map(
        lambda examples: preprocess_function_classification(examples, tokenizer),
        batched=True,
        remove_columns=dataset["train"].column_names,
    )

    # Training arguments
    task_output_dir = output_dir / task
    task_output_dir.mkdir(parents=True, exist_ok=True)

    training_args = TrainingArguments(
        output_dir=str(task_output_dir),
        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size * 2,
        learning_rate=learning_rate,
        weight_decay=0.01,
        warmup_ratio=0.1,
        logging_steps=100,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        fp16=torch.cuda.is_available(),
        seed=seed,
    )

    # Create trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["validation"],
        compute_metrics=compute_metrics_classification,
    )

    # Train
    logger.info("Starting fine-tuning...")
    train_result = trainer.train()

    # Evaluate on test set
    logger.info("Evaluating on test set...")
    if "test" in tokenized_dataset:
        test_results = trainer.evaluate(tokenized_dataset["test"])
    else:
        test_results = trainer.evaluate(tokenized_dataset["validation"])

    # Save results
    results = {
        "task": task,
        "train_metrics": train_result.metrics,
        "test_metrics": test_results,
        "hyperparameters": {
            "num_epochs": num_epochs,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "seed": seed,
        }
    }

    results_file = task_output_dir / "results.json"
    with open(results_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    logger.info(f"Results saved to {results_file}")
    logger.info(f"Test accuracy: {test_results.get('eval_accuracy', 0):.4f}")
    logger.info(f"Test F1: {test_results.get('eval_f1', 0):.4f}")

    return results


def main():
    parser = argparse.ArgumentParser(description="Evaluate CNM-BERT on CLUE benchmarks")
    parser.add_argument(
        "--model_path",
        type=Path,
        required=True,
        help="Path to pre-trained CNM-BERT checkpoint"
    )
    parser.add_argument(
        "--tasks",
        nargs="+",
        default=["tnews", "afqmc"],
        choices=list(TASK_CONFIGS.keys()),
        help="CLUE tasks to evaluate on"
    )
    parser.add_argument(
        "--data_dir",
        type=Path,
        default=Path("data/downstream/clue"),
        help="Directory containing CLUE datasets"
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("experiments/results"),
        help="Output directory for results"
    )
    parser.add_argument(
        "--num_epochs",
        type=int,
        default=3,
        help="Number of fine-tuning epochs"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Batch size for fine-tuning"
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=2e-5,
        help="Learning rate for fine-tuning"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed"
    )
    parser.add_argument(
        "--num_runs",
        type=int,
        default=1,
        help="Number of runs with different seeds (for statistical significance)"
    )
    args = parser.parse_args()

    logger.info("=" * 70)
    logger.info("CLUE Benchmark Evaluation")
    logger.info("=" * 70)
    logger.info(f"Model: {args.model_path}")
    logger.info(f"Tasks: {', '.join(args.tasks)}")
    logger.info(f"Output: {args.output_dir}")
    logger.info(f"Runs per task: {args.num_runs}")
    logger.info("=" * 70)

    # Run evaluation for each task
    all_results = {}

    for task in args.tasks:
        task_results = []

        for run_idx in range(args.num_runs):
            seed = args.seed + run_idx
            logger.info(f"\nRun {run_idx + 1}/{args.num_runs} for {task} (seed={seed})")

            try:
                results = fine_tune_and_evaluate(
                    model_path=args.model_path,
                    task=task,
                    data_dir=args.data_dir,
                    output_dir=args.output_dir / f"run_{run_idx}",
                    num_epochs=args.num_epochs,
                    batch_size=args.batch_size,
                    learning_rate=args.learning_rate,
                    seed=seed,
                )
                task_results.append(results)

            except Exception as e:
                logger.error(f"Error evaluating {task}: {e}", exc_info=True)
                continue

        all_results[task] = task_results

    # Save aggregated results
    summary_file = args.output_dir / "summary.json"
    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)

    logger.info("=" * 70)
    logger.info("Evaluation complete!")
    logger.info(f"Summary saved to {summary_file}")
    logger.info("=" * 70)

    # Print summary table
    logger.info("\nResults Summary:")
    logger.info("-" * 70)
    logger.info(f"{'Task':<15} {'Accuracy':<15} {'F1':<15}")
    logger.info("-" * 70)

    for task, runs in all_results.items():
        if runs:
            accuracies = [r["test_metrics"].get("eval_accuracy", 0) for r in runs]
            f1s = [r["test_metrics"].get("eval_f1", 0) for r in runs]
            acc_mean = np.mean(accuracies)
            acc_std = np.std(accuracies)
            f1_mean = np.mean(f1s)
            f1_std = np.std(f1s)
            logger.info(f"{task:<15} {acc_mean:.4f}±{acc_std:.4f}   {f1_mean:.4f}±{f1_std:.4f}")

    logger.info("-" * 70)


if __name__ == "__main__":
    main()
