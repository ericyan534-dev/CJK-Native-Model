#!/usr/bin/env python3
"""Comprehensive end-to-end test of the entire training pipeline."""

import sys
from pathlib import Path
import torch

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from cnm_bert import CNMConfig, CNMForMaskedLM, CNMBertTokenizer
from cnm_bert.data import PreTrainingDataset, WWMDataCollator
from torch.utils.data import DataLoader

def test_full_pipeline():
    """Test the complete training pipeline end-to-end."""
    print("=" * 70)
    print("COMPREHENSIVE PIPELINE TEST")
    print("=" * 70)

    # 1. Load tokenizer
    print("\n[1/7] Loading tokenizer...")
    try:
        tokenizer = CNMBertTokenizer.from_pretrained(
            "bert-base-chinese",
            struct_path="data/processed/char_to_ids_tree.json"
        )
        print(f"   ✓ Tokenizer loaded: vocab_size={tokenizer.vocab_size}")
    except Exception as e:
        print(f"   ✗ FAILED: {e}")
        return False

    # 2. Load model
    print("\n[2/7] Creating model...")
    try:
        model = CNMForMaskedLM.from_bert_pretrained(
            bert_model_name="bert-base-chinese",
            tree_path=Path("data/processed/char_to_ids_tree.json"),
            struct_dim=256,
            fusion_strategy="concat",
        )
        total_params = sum(p.numel() for p in model.parameters())
        print(f"   ✓ Model created: {total_params:,} parameters")
    except Exception as e:
        print(f"   ✗ FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False

    # 3. Load dataset
    print("\n[3/7] Loading dataset...")
    try:
        dataset = PreTrainingDataset(
            file_path=Path("data/pretrain/corpus_clean.txt"),
            max_samples=100  # Small sample for testing
        )
        print(f"   ✓ Dataset loaded: {len(dataset)} examples")
    except Exception as e:
        print(f"   ✗ FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False

    # 4. Create data collator
    print("\n[4/7] Creating data collator...")
    try:
        collator = WWMDataCollator(
            tokenizer=tokenizer,
            mlm_probability=0.15,
            max_seq_length=128
        )
        print(f"   ✓ Collator created")
    except Exception as e:
        print(f"   ✗ FAILED: {e}")
        return False

    # 5. Test single batch
    print("\n[5/7] Testing single batch...")
    try:
        batch_examples = [dataset[i] for i in range(4)]
        batch = collator(batch_examples)
        print(f"   ✓ Single batch successful")
        print(f"     Batch keys: {list(batch.keys())}")
        for key, val in batch.items():
            print(f"     {key}: {val.shape}")
    except Exception as e:
        print(f"   ✗ FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False

    # 6. Test DataLoader with num_workers=0
    print("\n[6/7] Testing DataLoader (num_workers=0)...")
    try:
        dataloader = DataLoader(
            dataset,
            batch_size=4,
            collate_fn=collator,
            num_workers=0,
            shuffle=False
        )

        for i, batch in enumerate(dataloader):
            if i >= 2:  # Test 2 batches
                break
            print(f"   ✓ Batch {i+1}: {batch['input_ids'].shape}")

        print(f"   ✓ DataLoader test passed")
    except Exception as e:
        print(f"   ✗ FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False

    # 7. Test forward pass
    print("\n[7/7] Testing model forward pass...")
    try:
        model.eval()
        with torch.no_grad():
            outputs = model(**batch)

        print(f"   ✓ Forward pass successful")
        print(f"     Loss: {outputs.loss.item():.4f}")
        print(f"     Logits shape: {outputs.logits.shape}")
    except Exception as e:
        print(f"   ✗ FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False

    print("\n" + "=" * 70)
    print("ALL TESTS PASSED ✓")
    print("=" * 70)
    print("\nYou can now run full training:")
    print("  CUDA_VISIBLE_DEVICES=0 python scripts/03_pretrain.py \\")
    print("      --config experiments/configs/pretrain_base.yaml \\")
    print("      --output_dir experiments/debug")
    print("=" * 70)
    return True

if __name__ == "__main__":
    success = test_full_pipeline()
    sys.exit(0 if success else 1)
