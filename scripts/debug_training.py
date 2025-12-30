#!/usr/bin/env python3
"""Debug script to test the training data pipeline."""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from cnm_bert import CNMBertTokenizer
from cnm_bert.data import PreTrainingDataset, WWMDataCollator

def test_pipeline():
    """Test the entire data pipeline."""
    print("=" * 70)
    print("Testing CNM-BERT Data Pipeline")
    print("=" * 70)

    # 1. Test dataset
    print("\n1. Testing Dataset...")
    dataset = PreTrainingDataset(
        file_path=Path("data/pretrain/corpus_clean.txt"),
        max_samples=10
    )
    print(f"   Dataset size: {len(dataset)}")

    # Test first example
    example = dataset[0]
    print(f"   First example type: {type(example)}")
    print(f"   First example keys: {list(example.keys()) if isinstance(example, dict) else 'N/A'}")
    print(f"   First example: {example}")

    # 2. Test tokenizer
    print("\n2. Testing Tokenizer...")
    tokenizer = CNMBertTokenizer.from_pretrained(
        "bert-base-chinese",
        struct_path="data/processed/char_to_ids_tree.json"
    )
    print(f"   Tokenizer loaded: vocab_size={tokenizer.vocab_size}")

    # 3. Test collator
    print("\n3. Testing Collator...")
    collator = WWMDataCollator(
        tokenizer=tokenizer,
        mlm_probability=0.15,
        max_seq_length=128
    )

    # Test with a batch
    batch_examples = [dataset[i] for i in range(min(4, len(dataset)))]
    print(f"   Batch size: {len(batch_examples)}")
    print(f"   Batch example types: {[type(ex) for ex in batch_examples]}")

    try:
        batch = collator(batch_examples)
        print(f"   ✓ Collator succeeded!")
        print(f"   Batch keys: {list(batch.keys())}")
        print(f"   Batch shapes:")
        for key, val in batch.items():
            print(f"     {key}: {val.shape}")
    except Exception as e:
        print(f"   ✗ Collator failed: {e}")
        import traceback
        traceback.print_exc()
        return False

    print("\n" + "=" * 70)
    print("Pipeline Test: SUCCESS")
    print("=" * 70)
    return True

if __name__ == "__main__":
    success = test_pipeline()
    sys.exit(0 if success else 1)
