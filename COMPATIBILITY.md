# Library Compatibility Notes

This project uses **Anaconda environment** with specific library versions.

## Environment Versions (from environment.yml)

```yaml
Python: 3.10
PyTorch: 2.2.0
transformers: 4.38.2
tokenizers: 0.15.2
datasets: 2.16.1
```

## Critical Compatibility Notes

### transformers 4.38.2

**TrainingArguments parameter changes:**

✅ **CORRECT for 4.38.2:**
```python
TrainingArguments(
    evaluation_strategy="steps",  # Use this
    save_strategy="steps",
    # ...
)
```

❌ **WRONG (for newer versions only):**
```python
TrainingArguments(
    eval_strategy="steps",  # Don't use - only in 4.40+
    # ...
)
```

**Parameter naming history:**
- transformers < 4.19: `evaluation_strategy`
- transformers 4.19-4.39: `evaluation_strategy` (our version: 4.38.2)
- transformers 4.40+: `eval_strategy` (new name)

### PyTorch 2.2.0

- ✅ Full CUDA 12.1 support
- ✅ Native support for H100 GPUs
- ✅ Distributed training (DDP) works correctly

## Validation

Always test in the **conda environment**, not base Python:

```bash
# Activate conda env
conda activate cnm-bert

# Verify versions
python -c "import transformers; print(transformers.__version__)"  # Should be 4.38.2
python -c "import torch; print(torch.__version__)"  # Should be 2.2.0+cu121

# Run validation
python scripts/validate_setup.py
```

## Common Issues

### Issue: "eval_strategy not recognized"

**Cause:** Using parameter name from newer transformers versions.

**Fix:** Use `evaluation_strategy` instead of `eval_strategy` for transformers 4.38.2.

### Issue: "CUDA version mismatch"

**Cause:** System CUDA doesn't match PyTorch CUDA.

**Fix:** PyTorch 2.2.0 is built with CUDA 12.1. Ensure your system has CUDA 12.x:
```bash
nvidia-smi  # Check driver supports CUDA 12.1+
```

### Issue: Different behavior in base vs conda env

**Cause:** Base environment has different library versions.

**Fix:** ALWAYS use the conda environment:
```bash
conda activate cnm-bert
```

## Installation

To recreate the environment:

```bash
# Remove old environment (if exists)
conda env remove -n cnm-bert

# Create new environment
conda env create -f environment.yml

# Activate
conda activate cnm-bert

# Install package
pip install -e .
```

## Dependencies

The project explicitly pins versions to ensure reproducibility:

| Package | Version | Purpose |
|---------|---------|---------|
| transformers | 4.38.2 | HuggingFace Transformers |
| torch | 2.2.0 | PyTorch with CUDA 12.1 |
| tokenizers | 0.15.2 | Fast tokenizers |
| datasets | 2.16.1 | Dataset loading |
| wandb | 0.16.3 | Experiment tracking |
| jieba | 0.42.1 | Chinese word segmentation |

## Testing Compatibility

Before training, always run:

```bash
conda activate cnm-bert
python scripts/validate_setup.py
```

This tests:
1. ✓ Dataset loading with correct pickling
2. ✓ Tokenizer with transformers 4.38.2
3. ✓ Model loading and BERT weight transfer
4. ✓ Data collator with jieba
5. ✓ Forward pass on GPU
6. ✓ DataLoader iteration
7. ✓ TrainingArguments with correct parameters
8. ✓ W&B integration (if installed)

All must pass before training.
