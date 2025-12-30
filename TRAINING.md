# CNM-BERT Training Guide

Production-ready training pipeline for ACL 2026 submission.

## Quick Start

### 1. Validate Setup (REQUIRED)

**Always run this first** to ensure everything works:

```bash
python scripts/validate_setup.py
```

This validates:
- ✓ Dataset loading
- ✓ Tokenizer initialization
- ✓ Model loading from BERT
- ✓ Data collator
- ✓ Forward pass
- ✓ DataLoader iteration
- ✓ Training arguments
- ✓ W&B integration

**Only proceed if all tests pass.**

### 2. Single GPU Training

```bash
python scripts/train.py \
    --config experiments/configs/pretrain_base.yaml \
    --output_dir experiments/runs/run1
```

### 3. Multi-GPU Training (8x H100)

```bash
torchrun --nproc_per_node=8 scripts/train.py \
    --config experiments/configs/pretrain_base.yaml \
    --output_dir experiments/runs/run1
```

### 4. Resume from Checkpoint

```bash
python scripts/train.py \
    --config experiments/configs/pretrain_base.yaml \
    --resume_from_checkpoint experiments/runs/run1/checkpoint-10000
```

## Configuration

Edit `experiments/configs/pretrain_base.yaml`:

```yaml
# Model
model:
  init_from_bert: true
  struct_dim: 256
  fusion_strategy: "concat"  # concat, add, or gate

# Data
data:
  train_file: "data/pretrain/corpus_clean.txt"
  val_file: null  # Optional validation file
  max_train_samples: null  # null = use all data

# Training
training:
  per_device_train_batch_size: 32
  gradient_accumulation_steps: 1
  learning_rate: 1.0e-4
  max_steps: 1000000
  warmup_steps: 10000

  # Evaluation (optional)
  eval_steps: 10000

  # Checkpointing
  save_steps: 10000
  save_total_limit: 3

  # W&B (optional)
  use_wandb: true
  wandb_project: "cnm-bert"
  run_name: "base-pretrain"
```

## Weights & Biases

### Setup

1. Install: `pip install wandb`
2. Login: `wandb login`
3. Enable in config: `use_wandb: true`

### Features

The training script automatically logs:
- Training loss
- Evaluation metrics (accuracy, loss)
- Learning rate schedule
- GPU memory usage
- Model checkpoints

View at: `https://wandb.ai/YOUR_USERNAME/cnm-bert`

## Validation

To evaluate a trained model:

```python
from transformers import Trainer
from cnm_bert import CNMForMaskedLM, CNMBertTokenizer
from cnm_bert.data import TextLineDataset, WWMDataCollator

# Load model
model = CNMForMaskedLM.from_pretrained("experiments/runs/run1")
tokenizer = CNMBertTokenizer.from_pretrained("experiments/runs/run1")

# Load validation data
val_dataset = TextLineDataset("data/pretrain/val.txt")
collator = WWMDataCollator(tokenizer=tokenizer)

# Evaluate
trainer = Trainer(
    model=model,
    eval_dataset=val_dataset,
    data_collator=collator,
    compute_metrics=compute_metrics,
)
metrics = trainer.evaluate()
print(metrics)
```

## Troubleshooting

### Issue: "evaluation_strategy not recognized"

**Fix**: You're using old transformers. Update:
```bash
pip install transformers==4.38.2 --upgrade
```

### Issue: "Dataset returning empty dicts"

**Fix**: Reinstall package:
```bash
pip uninstall cnm-bert -y
pip install -e .
```

### Issue: Training hangs at "Loading dataset..."

**Cause**: Dataset file is too large or has encoding issues.

**Fix**:
1. Check file: `head -100 data/pretrain/corpus_clean.txt`
2. Verify encoding: `file data/pretrain/corpus_clean.txt`
3. Test with small sample: `max_train_samples: 1000`

### Issue: W&B not logging

**Fix**:
```bash
# Re-login
wandb login

# Check config
python -c "import wandb; print(wandb.api.api_key)"
```

## Performance

### Expected Throughput (8x H100 80GB)

- Batch size: 32/GPU × 8 GPUs = 256 global
- Throughput: ~5,000-8,000 samples/sec
- Time per 10K steps: ~4-6 hours
- Total training (1M steps): ~400-600 GPU-hours

### Memory Usage

- Model: ~410MB
- Batch (512 seq len, bs=32): ~8GB/GPU
- Peak: ~12GB/GPU with FP16

### Optimization Tips

1. **Increase batch size**: If <70% GPU utilization, increase `per_device_train_batch_size`
2. **Gradient accumulation**: If OOM, increase `gradient_accumulation_steps` and decrease batch size
3. **Mixed precision**: Always use `fp16: true` on modern GPUs

## File Structure

```
experiments/
├── configs/
│   └── pretrain_base.yaml          # Training configuration
├── runs/
│   └── run1/                        # Output directory
│       ├── checkpoint-10000/        # Checkpoints
│       ├── logs/                    # TensorBoard logs
│       ├── trainer_state.json       # Training state
│       ├── config.json              # Model config
│       ├── pytorch_model.bin        # Model weights
│       └── vocab.txt                # Tokenizer vocab
```

## Citation

```bibtex
@inproceedings{cnm-bert-2026,
  title={CNM-BERT: Compositional Network Model for Chinese Pre-trained Language Models},
  author={...},
  booktitle={Proceedings of ACL 2026},
  year={2026}
}
```
