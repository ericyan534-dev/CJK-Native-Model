# CNM-BERT: Compositional Network Model for Chinese PLM

<div align="center">

**Integrating Orthographic Compositionality into Chinese Pre-trained Language Models via Ideographic Description Sequences**

[Paper](link-tbd) | [Model](link-tbd) | [Data](link-tbd)

</div>

## Overview

CNM-BERT is a novel Chinese pre-trained language model that explicitly models the compositional structure of Chinese characters using Ideographic Description Sequences (IDS). Unlike standard BERT which treats characters as atomic tokens, CNM-BERT integrates structural information through a dual-path encoder architecture.

### Key Features

- **Structural Awareness**: Leverages IDS trees to model character composition (e.g., 好 = 女 + 子)
- **Efficient Tree-MLP**: Batched bottom-up encoding with <20% training overhead
- **HuggingFace Compatible**: Drop-in replacement for `bert-base-chinese`
- **Publication-Grade**: Designed for ACL 2026 submission with full reproducibility

### Architecture

```
Input: "今天天气很好"
         ↓
    ┌────────┴────────┐
    │                 │
BERT Embedding    Tree-MLP Encoder
(Standard)        (IDS Structure)
    │                 │
    └────────┬────────┘
             ↓
       Fusion Layer
             ↓
    12-Layer Transformer
             ↓
        MLM Head
```

## Installation

### From Source

```bash
git clone https://github.com/yourusername/cnm-bert.git
cd cnm-bert
pip install -e .
```

### Requirements

- Python >= 3.9
- PyTorch >= 2.2.0
- Transformers >= 4.38.0
- 8x H100 80GB GPUs (for full pre-training)

## Quick Start

### 1. Download Data

```bash
# Download BabelStone IDS database
python scripts/download_ids.py --output data/raw/ids.txt

# Download Wikipedia-zh corpus
python scripts/download_wikipedia.py --output data/raw/wiki_zh.txt

# Download Common Crawl Chinese (optional)
python scripts/download_common_crawl.py --output data/raw/cc_zh.txt
```

### 2. Build IDS Lexicon

```bash
python scripts/01_build_ids_lexicon.py \
    --ids_file data/raw/ids.txt \
    --output data/processed/char_to_ids_tree.json \
    --vocab_file data/raw/vocab.txt
```

### 3. Prepare Pre-training Corpus

```bash
python scripts/02_prepare_corpus.py \
    --input data/raw/wiki_zh.txt \
    --output data/pretrain/wiki_zh_processed.txt \
    --min_length 10 \
    --max_length 512
```

### 4. Pre-train CNM-BERT

```bash
# Multi-GPU training with DDP
torchrun --nproc_per_node=8 scripts/03_pretrain.py \
    --config experiments/configs/pretrain_base.yaml \
    --output_dir experiments/logs/cnm_bert_base \
    --logging_dir experiments/logs/cnm_bert_base/tensorboard
```

### 5. Evaluate on CLUE

```bash
python scripts/04_evaluate_clue.py \
    --model_path experiments/logs/cnm_bert_base/checkpoint-best \
    --tasks tnews afqmc cluewsc csl cmrc2018 \
    --output_dir experiments/results
```

## Model Usage

```python
from cnm_bert import CNMBertTokenizer, CNMForMaskedLM

# Load pre-trained model
tokenizer = CNMBertTokenizer.from_pretrained("cnm-bert-base-chinese")
model = CNMForMaskedLM.from_pretrained("cnm-bert-base-chinese")

# Tokenize with structural information
text = "今天天气很好"
inputs = tokenizer(text, return_tensors="pt", return_struct_ids=True)

# inputs contains:
# - input_ids: standard BERT token IDs
# - struct_ids: pointers to IDS tree structures
# - attention_mask: attention mask

# Forward pass
outputs = model(**inputs)
predictions = outputs.logits.argmax(dim=-1)
```

## Project Structure

```
cnm-bert/
├── src/cnm_bert/           # Core library
│   ├── etl/                # IDS parsing & corpus preprocessing
│   ├── modeling/           # Model architecture
│   ├── tokenization/       # Tokenizer with struct_ids
│   ├── data/               # Dataset & collator
│   └── utils/              # Logging, metrics
├── scripts/                # Training & evaluation scripts
├── experiments/            # Configs & logs
├── tests/                  # Unit tests
├── data/                   # Data directory
└── analysis/               # Notebooks & visualizations
```

## Experiments & Results

### Main Results (CLUE Benchmark)

| Model | TNEWS | AFQMC | CLUEWSC | CSL | CMRC (F1) | Avg |
|-------|-------|-------|---------|-----|-----------|-----|
| BERT-base-chinese | TBD | TBD | TBD | TBD | TBD | TBD |
| CNM-BERT (Ours) | **TBD** | **TBD** | **TBD** | **TBD** | **TBD** | **TBD** |

### Ablation Studies

See [experiments/results/ablations.md](experiments/results/ablations.md) for detailed ablation analysis.

## Citation

If you use CNM-BERT in your research, please cite:

```bibtex
@inproceedings{cnmbert2026,
  title={CNM-BERT: Integrating Orthographic Compositionality into Chinese Pre-trained Language Models via Ideographic Description Sequences},
  author={Your Name and Co-authors},
  booktitle={Proceedings of the 64th Annual Meeting of the Association for Computational Linguistics (ACL)},
  year={2026}
}
```

## License

Apache License 2.0

## Acknowledgments

- BabelStone IDS Database
- HuggingFace Transformers
- CLUE Benchmark

## Contact

For questions or issues, please open an issue on GitHub or contact [your-email@domain.com].
