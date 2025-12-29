#!/bin/bash
# Setup script for CNM-BERT development environment

set -e

echo "================================"
echo "CNM-BERT Setup Script"
echo "================================"

# Check if conda is available
if ! command -v conda &> /dev/null; then
    echo "Error: conda not found. Please install Anaconda or Miniconda first."
    exit 1
fi

# Create conda environment
echo "Creating conda environment..."
conda env create -f environment.yml

echo ""
echo "Environment created successfully!"
echo ""
echo "To activate the environment, run:"
echo "  conda activate cnm-bert"
echo ""
echo "Next steps:"
echo "  1. Download BabelStone IDS database:"
echo "     python scripts/download_ids.py"
echo ""
echo "  2. Build IDS lexicon:"
echo "     python scripts/01_build_ids_lexicon.py --ids_file data/raw/ids.txt --output data/processed/char_to_ids_tree.json"
echo ""
echo "  3. Download Wikipedia corpus:"
echo "     python scripts/download_wikipedia.py"
echo ""
echo "  4. Prepare corpus:"
echo "     python scripts/02_prepare_corpus.py --input data/raw/wiki_zh.txt --output data/pretrain/corpus_clean.txt"
echo ""
echo "  5. Start pre-training:"
echo "     torchrun --nproc_per_node=8 scripts/03_pretrain.py --config experiments/configs/pretrain_base.yaml"
echo ""
