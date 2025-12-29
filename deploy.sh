#!/bin/bash
# Automated Deployment Script for CNM-BERT on Alibaba Cloud
# This script automates the entire deployment process from environment setup to training

set -e  # Exit on error

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
GITHUB_REPO="https://github.com/ericyan534-dev/CJK-Native-Model.git"
PROJECT_DIR="$HOME/CJK-Native-Model"
CONDA_ENV_NAME="cnm-bert"

# Logging functions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Progress indicator
print_progress() {
    local current=$1
    local total=$2
    local step_name=$3
    echo ""
    echo "=========================================="
    echo "STEP $current/$total: $step_name"
    echo "=========================================="
}

# Main deployment steps
TOTAL_STEPS=11
CURRENT_STEP=0

# Parse command line arguments
SKIP_MINICONDA=false
SKIP_CLONE=false
SKIP_DATA=false
SKIP_WANDB=false
TEST_MODE=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --skip-miniconda)
            SKIP_MINICONDA=true
            shift
            ;;
        --skip-clone)
            SKIP_CLONE=true
            shift
            ;;
        --skip-data)
            SKIP_DATA=true
            shift
            ;;
        --skip-wandb)
            SKIP_WANDB=true
            shift
            ;;
        --test-mode)
            TEST_MODE=true
            shift
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: $0 [--skip-miniconda] [--skip-clone] [--skip-data] [--skip-wandb] [--test-mode]"
            exit 1
            ;;
    esac
done

log_info "CNM-BERT Automated Deployment Script"
log_info "====================================="
echo ""

# STEP 1: Check prerequisites
((CURRENT_STEP++))
print_progress $CURRENT_STEP $TOTAL_STEPS "Checking Prerequisites"

log_info "Checking system requirements..."

# Check GPU availability
if ! command_exists nvidia-smi; then
    log_error "nvidia-smi not found. NVIDIA drivers not installed?"
    exit 1
fi

GPU_COUNT=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
log_success "Found $GPU_COUNT GPU(s)"

if [ "$GPU_COUNT" -ne 8 ]; then
    log_warning "Expected 8 GPUs but found $GPU_COUNT"
fi

# Check disk space (need at least 200GB)
AVAILABLE_SPACE=$(df -BG . | tail -1 | awk '{print $4}' | sed 's/G//')
log_info "Available disk space: ${AVAILABLE_SPACE}GB"

if [ "$AVAILABLE_SPACE" -lt 200 ]; then
    log_error "Insufficient disk space. Need at least 200GB, have ${AVAILABLE_SPACE}GB"
    exit 1
fi

log_success "Prerequisites check passed"

# STEP 2: Install Miniconda
((CURRENT_STEP++))
print_progress $CURRENT_STEP $TOTAL_STEPS "Installing Miniconda"

if [ "$SKIP_MINICONDA" = true ]; then
    log_warning "Skipping Miniconda installation"
elif command_exists conda; then
    log_success "Conda already installed: $(conda --version)"
else
    log_info "Downloading Miniconda installer..."
    cd ~
    wget -q --show-progress https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh

    log_info "Installing Miniconda..."
    bash Miniconda3-latest-Linux-x86_64.sh -b -p $HOME/miniconda3

    log_info "Initializing conda..."
    $HOME/miniconda3/bin/conda init bash

    # Source bashrc to make conda available
    source ~/.bashrc

    rm Miniconda3-latest-Linux-x86_64.sh
    log_success "Miniconda installed successfully"
fi

# Ensure conda is in PATH
export PATH="$HOME/miniconda3/bin:$PATH"

# STEP 3: Clone Repository
((CURRENT_STEP++))
print_progress $CURRENT_STEP $TOTAL_STEPS "Cloning Repository"

if [ "$SKIP_CLONE" = true ]; then
    log_warning "Skipping repository clone"
    cd $PROJECT_DIR
elif [ -d "$PROJECT_DIR" ]; then
    log_warning "Repository already exists at $PROJECT_DIR"
    log_info "Pulling latest changes..."
    cd $PROJECT_DIR
    git pull
else
    log_info "Cloning repository from $GITHUB_REPO..."
    git clone $GITHUB_REPO $PROJECT_DIR
    cd $PROJECT_DIR
    log_success "Repository cloned successfully"
fi

# STEP 4: Create Conda Environment
((CURRENT_STEP++))
print_progress $CURRENT_STEP $TOTAL_STEPS "Creating Conda Environment"

if conda env list | grep -q "^${CONDA_ENV_NAME} "; then
    log_warning "Conda environment '$CONDA_ENV_NAME' already exists"
    log_info "Activating existing environment..."
else
    log_info "Creating conda environment from environment.yml..."
    conda env create -f environment.yml
    log_success "Conda environment created"
fi

# Activate environment
log_info "Activating environment..."
source $HOME/miniconda3/bin/activate $CONDA_ENV_NAME

# STEP 5: Install Package
((CURRENT_STEP++))
print_progress $CURRENT_STEP $TOTAL_STEPS "Installing CNM-BERT Package"

log_info "Installing package in editable mode..."
pip install -e . --quiet

log_info "Verifying installation..."
python -c "from cnm_bert import CNMForMaskedLM; print('✓ Import successful')"
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU count: {torch.cuda.device_count()}')"

log_success "Package installed successfully"

# STEP 6: Setup Weights & Biases
((CURRENT_STEP++))
print_progress $CURRENT_STEP $TOTAL_STEPS "Setting up Weights & Biases"

if [ "$SKIP_WANDB" = true ]; then
    log_warning "Skipping W&B setup"
else
    log_info "Setting up Weights & Biases integration..."
    python scripts/setup_wandb.py
    log_success "W&B setup complete"
fi

# STEP 7: Download BabelStone IDS
((CURRENT_STEP++))
print_progress $CURRENT_STEP $TOTAL_STEPS "Downloading BabelStone IDS"

if [ "$SKIP_DATA" = true ]; then
    log_warning "Skipping data download steps"
elif [ -f "data/raw/ids.txt" ]; then
    log_success "IDS file already exists"
else
    log_info "Downloading BabelStone IDS database..."
    python scripts/download_ids.py --output data/raw/ids.txt
    log_success "IDS database downloaded"
fi

# STEP 8: Build IDS Lexicon
((CURRENT_STEP++))
print_progress $CURRENT_STEP $TOTAL_STEPS "Building IDS Lexicon"

if [ "$SKIP_DATA" = true ]; then
    log_warning "Skipping IDS lexicon build"
elif [ -f "data/processed/char_to_ids_tree.json" ]; then
    log_success "IDS lexicon already exists"
else
    log_info "Downloading BERT vocabulary..."
    python -c "from transformers import BertTokenizer; tok = BertTokenizer.from_pretrained('bert-base-chinese'); tok.save_vocabulary('data/raw/')"

    log_info "Building IDS lexicon (this may take 5-10 minutes)..."
    python scripts/01_build_ids_lexicon.py \
        --ids_file data/raw/ids.txt \
        --output data/processed/char_to_ids_tree.json \
        --vocab_file data/raw/vocab.txt \
        --save_stats data/processed/ids_stats.json

    log_info "Checking IDS coverage..."
    TOTAL_CHARS=$(cat data/processed/ids_stats.json | grep -oP '"total_characters":\s*\K\d+')
    log_success "IDS lexicon built with $TOTAL_CHARS characters"
fi

# STEP 9: Download Wikipedia Corpus
((CURRENT_STEP++))
print_progress $CURRENT_STEP $TOTAL_STEPS "Downloading Wikipedia Corpus"

if [ "$SKIP_DATA" = true ]; then
    log_warning "Skipping Wikipedia download"
elif [ -f "data/raw/wiki_zh.txt" ]; then
    log_success "Wikipedia corpus already exists"
else
    if [ "$TEST_MODE" = true ]; then
        log_warning "TEST MODE: Skipping Wikipedia download (use small test corpus instead)"
        # Create a small test corpus
        echo "今天天气很好" > data/raw/wiki_zh.txt
        for i in {1..1000}; do
            echo "这是一个测试句子，用于快速测试训练流程。" >> data/raw/wiki_zh.txt
        done
        log_success "Created test corpus with 1000 sentences"
    else
        log_info "Downloading Wikipedia-zh corpus (this will take 2-4 hours)..."
        log_warning "This is the longest step. Consider running in tmux/screen."
        python scripts/download_wikipedia.py \
            --output data/raw/wiki_zh.txt \
            --keep-dump
        log_success "Wikipedia corpus downloaded"
    fi
fi

# STEP 10: Preprocess Corpus
((CURRENT_STEP++))
print_progress $CURRENT_STEP $TOTAL_STEPS "Preprocessing Corpus"

if [ "$SKIP_DATA" = true ]; then
    log_warning "Skipping corpus preprocessing"
elif [ -f "data/pretrain/corpus_clean.txt" ]; then
    log_success "Preprocessed corpus already exists"
else
    log_info "Preprocessing corpus (this may take 1-2 hours)..."
    python scripts/02_prepare_corpus.py \
        --input data/raw/wiki_zh.txt \
        --output data/pretrain/corpus_clean.txt \
        --min_length 10 \
        --max_length 512 \
        --stats_output data/pretrain/corpus_stats.json

    SENTENCE_COUNT=$(wc -l < data/pretrain/corpus_clean.txt)
    log_success "Corpus preprocessed: $SENTENCE_COUNT sentences"
fi

# STEP 11: Deployment Summary
((CURRENT_STEP++))
print_progress $CURRENT_STEP $TOTAL_STEPS "Deployment Complete"

echo ""
log_success "=========================================="
log_success "CNM-BERT Deployment Successful!"
log_success "=========================================="
echo ""

log_info "Deployment Summary:"
echo "  - Project directory: $PROJECT_DIR"
echo "  - Conda environment: $CONDA_ENV_NAME"
echo "  - GPUs available: $GPU_COUNT"
echo "  - IDS lexicon: data/processed/char_to_ids_tree.json"
echo "  - Training corpus: data/pretrain/corpus_clean.txt"
echo ""

log_info "Next Steps:"
echo ""
echo "1. Test single GPU training (sanity check):"
echo "   cd $PROJECT_DIR"
echo "   conda activate $CONDA_ENV_NAME"
echo "   CUDA_VISIBLE_DEVICES=0 python scripts/03_pretrain.py \\"
echo "       --config experiments/configs/pretrain_base.yaml \\"
echo "       --output_dir experiments/debug"
echo ""
echo "2. Start full multi-GPU training:"
echo "   tmux new -s training"
echo "   conda activate $CONDA_ENV_NAME"
echo "   torchrun --nproc_per_node=8 scripts/03_pretrain.py \\"
echo "       --config experiments/configs/pretrain_base.yaml \\"
echo "       --output_dir experiments/logs/cnm_bert_base"
echo ""
echo "3. Monitor training:"
echo "   - TensorBoard: tensorboard --logdir experiments/logs/cnm_bert_base/logs --port 6006"
echo "   - Weights & Biases: https://wandb.ai/your-username/cnm-bert"
echo "   - GPU usage: watch -n 1 nvidia-smi"
echo ""

log_info "Detach from tmux: Ctrl+B, then D"
log_info "Reattach to tmux: tmux attach -t training"
echo ""

if [ "$TEST_MODE" = true ]; then
    log_warning "TEST MODE was enabled - using small test corpus"
    log_warning "For production, run without --test-mode flag"
fi

log_success "Deployment completed successfully! 🚀"
