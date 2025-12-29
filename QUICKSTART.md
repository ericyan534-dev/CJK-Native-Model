# CNM-BERT Quick Start Guide

This guide will help you set up and run CNM-BERT for ACL 2026 submission.

## System Requirements

- **Hardware**: 8x NVIDIA H100 80GB GPUs (or similar)
- **OS**: Linux (Ubuntu 20.04+ recommended)
- **Python**: 3.9-3.11
- **CUDA**: 12.1+

## Installation

### Step 1: Clone Repository

```bash
git clone https://github.com/yourusername/cnm-bert.git
cd cnm-bert
```

### Step 2: Set Up Environment

#### Option A: Using Conda (Recommended)

```bash
# Create environment
conda env create -f environment.yml

# Activate environment
conda activate cnm-bert

# Install package in editable mode
pip install -e .
```

#### Option B: Using pip

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
pip install -e .
```

### Step 3: Verify Installation

```bash
# Run tests
pytest tests/ -v

# Check imports
python -c "from cnm_bert import CNMForMaskedLM; print('✓ Installation successful!')"
```

## Data Preparation

### Phase 1: Download IDS Database

```bash
# Download BabelStone IDS
python scripts/download_ids.py --output data/raw/ids.txt
```

### Phase 2: Build IDS Lexicon

```bash
# Parse IDS and create canonical tree mapping
python scripts/01_build_ids_lexicon.py \
    --ids_file data/raw/ids.txt \
    --output data/processed/char_to_ids_tree.json \
    --save_stats data/processed/ids_stats.json
```

**Expected Output**:
- `data/processed/char_to_ids_tree.json`: ~20K characters with IDS trees
- Coverage of `bert-base-chinese` vocabulary: >95%
- Mean tree depth: <5, max depth: <15

### Phase 3: Download Pre-training Corpus

#### Wikipedia-zh (Primary Corpus)

```bash
# Download and extract Wikipedia-zh
python scripts/download_wikipedia.py \
    --output data/raw/wiki_zh.txt \
    --keep-dump
```

This will:
1. Download latest zhwiki dump (~2GB compressed)
2. Extract plain text using WikiExtractor
3. Merge into single text file (~10GB)
4. Take ~2-4 hours depending on network

#### Common Crawl (Optional, for larger corpus)

```bash
# Download Chinese text from Common Crawl
python scripts/download_common_crawl.py \
    --output data/raw/cc_zh.txt \
    --max-pages 10000
```

**Note**: Common Crawl download is experimental. Wikipedia-zh alone is sufficient for ACL paper.

### Phase 4: Preprocess Corpus

```bash
# Clean and deduplicate
python scripts/02_prepare_corpus.py \
    --input data/raw/wiki_zh.txt \
    --output data/pretrain/corpus_clean.txt \
    --min_length 10 \
    --max_length 512 \
    --stats_output data/pretrain/corpus_stats.json
```

**Expected Output**:
- Clean corpus: ~100M sentences
- Average sentence length: ~50-80 characters
- Vocabulary: ~10K unique characters

## Pre-training

### Single GPU (Debugging)

```bash
# Small-scale test (1 GPU, limited data)
python scripts/03_pretrain.py \
    --config experiments/configs/pretrain_base.yaml \
    --output_dir experiments/debug
```

### Multi-GPU (Full Training)

```bash
# 8x H100 training
torchrun --nproc_per_node=8 scripts/03_pretrain.py \
    --config experiments/configs/pretrain_base.yaml \
    --output_dir experiments/logs/cnm_bert_base
```

**Training Configuration** (in `pretrain_base.yaml`):
- Global batch size: 256 (32 per GPU × 8 GPUs)
- Learning rate: 1e-4 with 10K warmup steps
- Max steps: 1M (~2-3 weeks on 8x H100)
- FP16: Enabled
- Checkpoints: Every 10K steps

**Monitoring**:
```bash
# TensorBoard
tensorboard --logdir experiments/logs/cnm_bert_base/logs

# Weights & Biases (if enabled)
# View at https://wandb.ai/your-project/cnm-bert-base-pretrain
```

### Curriculum Learning (Optional)

For more stable training, use 2-phase curriculum:

**Phase 1**: Freeze BERT encoder, train only Tree-MLP + Fusion (5K steps)

```bash
# Edit config: set freeze_bert_encoder: true
torchrun --nproc_per_node=8 scripts/03_pretrain.py \
    --config experiments/configs/pretrain_curriculum_phase1.yaml
```

**Phase 2**: Unfreeze all, end-to-end training

```bash
# Edit config: set freeze_bert_encoder: false
# Initialize from phase 1 checkpoint
torchrun --nproc_per_node=8 scripts/03_pretrain.py \
    --config experiments/configs/pretrain_curriculum_phase2.yaml
```

## Evaluation on CLUE

### Download CLUE Benchmarks

```bash
# Download CLUE datasets
mkdir -p data/downstream/clue
cd data/downstream/clue

# Download from official CLUE repo
# https://github.com/CLUEbenchmark/CLUE
# Or use HuggingFace datasets (automatic download)
```

### Run Evaluation

```bash
# Evaluate on TNEWS and AFQMC with 5 random seeds
python scripts/04_evaluate_clue.py \
    --model_path experiments/logs/cnm_bert_base/checkpoint-best \
    --tasks tnews afqmc cluewsc csl \
    --output_dir experiments/results \
    --num_runs 5
```

### Expected Results (Targets for ACL Paper)

| Task | BERT-base-chinese | CNM-BERT (Target) |
|------|-------------------|-------------------|
| TNEWS | ~56.0% | **>57.0%** |
| AFQMC | ~73.0% | **>74.0%** |
| CLUEWSC | ~60.0% | **>61.0%** |
| CSL | ~80.0% | **>80.5%** |

**Goal**: Outperform BERT on ≥3/5 tasks with statistical significance (p < 0.05)

## Ablation Studies

### Ablation 1: No Structural Encoding (Should collapse to BERT)

```bash
torchrun --nproc_per_node=8 scripts/03_pretrain.py \
    --config experiments/configs/pretrain_ablation_no_struct.yaml
```

### Ablation 2: Different Fusion Strategies

Edit `pretrain_base.yaml` and set `fusion_strategy` to:
- `"concat"` (default)
- `"add"`
- `"gate"`

### Ablation 3: Structural Dimensions

Edit `struct_dim` in config:
- 128 (lightweight)
- 256 (default)
- 512 (high-capacity)

## Analysis for Paper

### 1. Low-Frequency Character Performance

```python
from cnm_bert.analysis import analyze_low_frequency_chars

analyze_low_frequency_chars(
    model_path="experiments/logs/cnm_bert_base",
    corpus_path="data/pretrain/corpus_clean.txt",
    output_dir="analysis/low_freq"
)
```

### 2. Structural Similarity Visualization

```python
from cnm_bert.analysis import visualize_embeddings

visualize_embeddings(
    model_path="experiments/logs/cnm_bert_base",
    output_path="analysis/figures/embedding_tsne.pdf"
)
```

### 3. Attention Map Analysis

See `analysis/notebooks/attention_visualization.ipynb`

## Troubleshooting

### Issue: OOM (Out of Memory)

**Solution**: Reduce `per_device_train_batch_size` or enable `gradient_checkpointing`:

```yaml
# In training config
per_device_train_batch_size: 16  # Reduce from 32
gradient_checkpointing: true
```

### Issue: Slow Training

**Checklist**:
- [ ] FP16 enabled?
- [ ] Using fast dataloader (`num_workers: 4`)?
- [ ] Tree-MLP batching working? (Check unique chars per batch)

### Issue: Loss Not Decreasing

**Debugging Steps**:
1. Run overfitting test on 1000 sentences
2. Check learning rate (too high/low?)
3. Verify struct_ids alignment with input_ids
4. Check gradient norms

### Issue: Tests Failing

```bash
# Install test dependencies
pip install pytest pytest-cov

# Run specific test
pytest tests/test_model.py::TestCNMModel::test_forward_pass -v

# Check imports
python -c "from cnm_bert import CNMForMaskedLM"
```

## Paper Writing Checklist

- [ ] Main results table (Table 1): CLUE benchmark scores
- [ ] Ablation table (Table 2): Component ablations
- [ ] Hyperparameter table (Table 3): Struct_dim experiments
- [ ] Figure 2: Training loss curves (BERT vs CNM)
- [ ] Figure 3: Low-freq character performance
- [ ] Figure 4: t-SNE visualization of embeddings
- [ ] Appendix A: Reproducibility checklist (100%)
- [ ] Appendix B: IDS heuristics algorithm
- [ ] Appendix C: Full results (all seeds)
- [ ] Code release: GitHub repo public on acceptance
- [ ] Model release: HuggingFace Hub checkpoint

## Timeline to Submission

Assuming ACL 2026 deadline: ~May 2026

| Week | Tasks |
|------|-------|
| 1-2 | Data preparation + IDS lexicon build |
| 3-5 | Pre-training (1M steps on 8x H100) |
| 6-7 | CLUE evaluation + ablations |
| 8-9 | Analysis + visualization |
| 10-12 | Paper writing + experiments refinement |
| 13-14 | Internal review + revisions |
| 15 | Final checks + submission |

**Total**: ~15 weeks (~4 months)

## Support

For issues or questions:
- GitHub Issues: https://github.com/yourusername/cnm-bert/issues
- Email: your-email@domain.com
- ACL 2026 Submission: Track at https://softconf.com/acl2026

## Citation (Pre-print)

If you use this code before ACL 2026 publication:

```bibtex
@misc{cnmbert2025,
  title={CNM-BERT: Integrating Orthographic Compositionality into Chinese Pre-trained Language Models via Ideographic Description Sequences},
  author={Your Name},
  year={2025},
  note={Under review at ACL 2026}
}
```

---

**Good luck with your ACL 2026 submission!** 🚀
