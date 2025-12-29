"""Custom Weights & Biases callback for CNM-BERT training.

This callback provides enhanced logging and visualization for W&B including:
- Detailed training metrics
- GPU utilization tracking
- Sample predictions
- Model architecture visualization
- Learning rate schedules
"""

import os
import torch
import numpy as np
from typing import Optional, Dict, Any
from transformers import TrainerCallback, TrainingArguments, TrainerState, TrainerControl

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False


class CNMWandbCallback(TrainerCallback):
    """Enhanced W&B callback for CNM-BERT training.

    This callback extends the default W&B integration with:
    - GPU memory tracking
    - Structural embedding statistics
    - Sample token predictions
    - Attention visualization (periodic)
    - Custom metrics (perplexity, etc.)
    """

    def __init__(self):
        if not WANDB_AVAILABLE:
            raise ImportError("wandb is not installed. Install with: pip install wandb")

        self.log_model_architecture = True
        self.log_predictions_every = 1000  # Log predictions every N steps

    def on_init_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        """Called at the end of Trainer initialization."""
        if wandb.run is None:
            return

        # Log model architecture details
        model = kwargs.get("model")
        if model and self.log_model_architecture:
            config = model.config if hasattr(model, "config") else None
            if config:
                # Log architecture parameters
                wandb.config.update({
                    "model_type": "cnm-bert",
                    "vocab_size": config.vocab_size,
                    "hidden_size": config.hidden_size,
                    "num_hidden_layers": config.num_hidden_layers,
                    "num_attention_heads": config.num_attention_heads,
                    "intermediate_size": config.intermediate_size,
                    "struct_dim": getattr(config, "struct_dim", None),
                    "fusion_strategy": getattr(config, "fusion_strategy", None),
                    "max_position_embeddings": config.max_position_embeddings,
                })

                # Calculate total parameters
                total_params = sum(p.numel() for p in model.parameters())
                trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

                wandb.config.update({
                    "total_parameters": total_params,
                    "trainable_parameters": trainable_params,
                    "frozen_parameters": total_params - trainable_params,
                })

                # Log to summary
                wandb.run.summary["total_parameters"] = total_params
                wandb.run.summary["trainable_parameters"] = trainable_params

                self.log_model_architecture = False  # Only log once

        # Log training configuration
        wandb.config.update({
            "learning_rate": args.learning_rate,
            "train_batch_size": args.per_device_train_batch_size * args.gradient_accumulation_steps * args.world_size,
            "per_device_train_batch_size": args.per_device_train_batch_size,
            "gradient_accumulation_steps": args.gradient_accumulation_steps,
            "num_train_epochs": args.num_train_epochs,
            "max_steps": args.max_steps,
            "warmup_steps": args.warmup_steps,
            "weight_decay": args.weight_decay,
            "fp16": args.fp16,
            "world_size": args.world_size,
        })

    def on_log(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, logs: Optional[Dict[str, Any]] = None, **kwargs):
        """Called when logging occurs."""
        if wandb.run is None or logs is None:
            return

        # Add GPU memory stats
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                memory_allocated = torch.cuda.memory_allocated(i) / 1024**3  # GB
                memory_reserved = torch.cuda.memory_reserved(i) / 1024**3  # GB
                memory_free = (torch.cuda.get_device_properties(i).total_memory - torch.cuda.memory_allocated(i)) / 1024**3

                logs[f"gpu_{i}/memory_allocated_gb"] = memory_allocated
                logs[f"gpu_{i}/memory_reserved_gb"] = memory_reserved
                logs[f"gpu_{i}/memory_free_gb"] = memory_free

        # Calculate perplexity from loss
        if "loss" in logs:
            logs["perplexity"] = np.exp(logs["loss"])

        if "eval_loss" in logs:
            logs["eval_perplexity"] = np.exp(logs["eval_loss"])

        # Add throughput metrics
        if "steps" in logs and state.global_step > 0:
            # Calculate samples per second
            if hasattr(state, "log_history") and len(state.log_history) > 1:
                # Estimate throughput based on recent logs
                batch_size = args.per_device_train_batch_size * args.gradient_accumulation_steps * args.world_size
                # This is approximate - actual throughput is calculated by HF Trainer
                logs["throughput/global_batch_size"] = batch_size

    def on_step_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        """Called at the end of each training step."""
        if wandb.run is None:
            return

        # Periodically log sample predictions
        if state.global_step % self.log_predictions_every == 0:
            self._log_sample_predictions(state.global_step, **kwargs)

    def _log_sample_predictions(self, step: int, **kwargs):
        """Log sample model predictions to W&B."""
        model = kwargs.get("model")
        tokenizer = kwargs.get("tokenizer")

        if model is None or tokenizer is None:
            return

        try:
            model.eval()
            with torch.no_grad():
                # Create sample input
                sample_text = "今天天气很好"
                inputs = tokenizer(
                    sample_text,
                    return_tensors="pt",
                    return_struct_ids=True
                )

                # Move to device
                device = next(model.parameters()).device
                inputs = {k: v.to(device) for k, v in inputs.items()}

                # Get predictions
                outputs = model(**inputs)
                predictions = torch.argmax(outputs.logits, dim=-1)

                # Decode predictions
                predicted_tokens = tokenizer.convert_ids_to_tokens(predictions[0].cpu().tolist())

                # Create a table for W&B
                table = wandb.Table(columns=["Step", "Input", "Predicted_Tokens"])
                table.add_data(step, sample_text, " ".join(predicted_tokens[:20]))  # First 20 tokens

                wandb.log({"predictions/sample": table}, step=step)

            model.train()
        except Exception as e:
            # Don't fail training if prediction logging fails
            print(f"Warning: Failed to log sample predictions: {e}")

    def on_train_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        """Called at the end of training."""
        if wandb.run is None:
            return

        # Log final training summary
        wandb.run.summary["final_step"] = state.global_step
        wandb.run.summary["final_epoch"] = state.epoch

        # Log final metrics
        if state.log_history:
            final_metrics = state.log_history[-1]
            if "loss" in final_metrics:
                wandb.run.summary["final_loss"] = final_metrics["loss"]
                wandb.run.summary["final_perplexity"] = np.exp(final_metrics["loss"])

        wandb.log({"training/status": "completed"})


def setup_wandb_training(
    project: str = "cnm-bert",
    name: Optional[str] = None,
    config: Optional[Dict] = None,
    tags: Optional[list] = None,
    notes: Optional[str] = None,
):
    """Initialize W&B for training run.

    Args:
        project: W&B project name
        name: Run name (auto-generated if None)
        config: Configuration dictionary
        tags: List of tags
        notes: Run description

    Returns:
        W&B run object
    """
    if not WANDB_AVAILABLE:
        raise ImportError("wandb is not installed. Install with: pip install wandb")

    # Check for environment variables
    project = os.getenv("WANDB_PROJECT", project)
    entity = os.getenv("WANDB_ENTITY", None)

    # Default tags
    if tags is None:
        tags = ["cnm-bert", "pre-training"]
        if os.getenv("WANDB_TAGS"):
            tags.extend(os.getenv("WANDB_TAGS").split(","))

    # Initialize run
    run = wandb.init(
        project=project,
        entity=entity,
        name=name,
        config=config,
        tags=tags,
        notes=notes,
        resume="allow",  # Allow resuming from checkpoints
    )

    # Log system info
    if torch.cuda.is_available():
        gpu_info = []
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            gpu_info.append({
                "name": props.name,
                "total_memory_gb": props.total_memory / 1024**3,
                "sm_count": props.multi_processor_count,
            })

        wandb.config.update({"gpu_info": gpu_info})

    return run
