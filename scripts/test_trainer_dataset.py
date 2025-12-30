#!/usr/bin/env python3
"""Test script to diagnose dataset issues with Trainer/Accelerate.

This script tests each layer of the training pipeline to isolate where
the dataset breaks and returns empty dicts.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from cnm_bert.data.dataset import TextLineDataset
from cnm_bert.data.collator import WWMDataCollator
from cnm_bert.tokenization import CNMBertTokenizer
from transformers import Trainer, TrainingArguments
from torch.utils.data import DataLoader
import torch


def test_step_1_dataset():
    """Test basic dataset creation and access."""
    print("=" * 80)
    print("STEP 1: Create dataset")
    print("=" * 80)

    # Create small test file
    test_file = '/tmp/test_train.txt'
    with open(test_file, 'w') as f:
        for i in range(100):
            f.write(f"这是第{i}行测试数据，用于验证数据加载是否正常工作。\n")

    dataset = TextLineDataset(test_file, max_samples=10)
    print(f"✓ Dataset created")
    print(f"  Length: {len(dataset)}")
    print(f"  First item: {dataset[0]}")
    print(f"  Type: {type(dataset[0])}")
    print(f"  Keys: {list(dataset[0].keys())}")

    return dataset


def test_step_2_pickle(dataset):
    """Test pickling and unpickling."""
    print("\n" + "=" * 80)
    print("STEP 2: Test pickle/unpickle")
    print("=" * 80)

    import pickle

    print(f"  Pickling...")
    pickled = pickle.dumps(dataset)
    print(f"  ✓ Pickled successfully, size: {len(pickled)} bytes")

    print(f"  Unpickling...")
    ds2 = pickle.loads(pickled)
    print(f"  ✓ Unpickled successfully")
    print(f"  Length: {len(ds2)}")
    print(f"  First item: {ds2[0]}")
    print(f"  Has 'text' key: {'text' in ds2[0]}")

    return ds2


def test_step_3_tokenizer():
    """Test tokenizer creation."""
    print("\n" + "=" * 80)
    print("STEP 3: Create tokenizer")
    print("=" * 80)

    tokenizer = CNMBertTokenizer.from_pretrained(
        "bert-base-chinese",
        struct_path="data/processed/char_to_ids_tree.json"
    )
    print(f"  ✓ Tokenizer created")
    print(f"  Vocab size: {tokenizer.vocab_size}")

    return tokenizer


def test_step_4_collator(dataset, tokenizer):
    """Test data collator."""
    print("\n" + "=" * 80)
    print("STEP 4: Test collator directly")
    print("=" * 80)

    collator = WWMDataCollator(tokenizer=tokenizer)
    print(f"  ✓ Collator created")

    batch = [dataset[0], dataset[1], dataset[2]]
    print(f"  Batch before collator:")
    for i, item in enumerate(batch):
        print(f"    [{i}] type={type(item)}, keys={list(item.keys()) if isinstance(item, dict) else 'N/A'}")

    try:
        collated = collator(batch)
        print(f"  ✓ Collated successfully")
        print(f"  Collated keys: {list(collated.keys())}")
    except Exception as e:
        print(f"  ✗ Collator failed: {e}")
        import traceback
        traceback.print_exc()
        raise

    return collator


def test_step_5_dataloader(dataset, collator):
    """Test PyTorch DataLoader."""
    print("\n" + "=" * 80)
    print("STEP 5: Test DataLoader (num_workers=0)")
    print("=" * 80)

    dataloader = DataLoader(
        dataset,
        batch_size=4,
        collate_fn=collator,
        num_workers=0,  # Single process
    )
    print(f"  ✓ DataLoader created")

    print(f"  Iterating...")
    for i, batch in enumerate(dataloader):
        print(f"    Batch {i}: keys={list(batch.keys())}, batch_size={len(batch['input_ids'])}")
        if i >= 2:
            break
    print(f"  ✓ DataLoader iteration successful")

    return dataloader


def test_step_6_accelerate(dataloader):
    """Test with Accelerate prepare."""
    print("\n" + "=" * 80)
    print("STEP 6: Test with Accelerate prepare")
    print("=" * 80)

    from accelerate import Accelerator

    accelerator = Accelerator()
    print(f"  Accelerator initialized")
    print(f"  Num processes: {accelerator.num_processes}")
    print(f"  Process index: {accelerator.process_index}")
    print(f"  Device: {accelerator.device}")

    prepared_dataloader = accelerator.prepare(dataloader)
    print(f"  ✓ DataLoader prepared")

    print(f"  Iterating prepared DataLoader...")
    for i, batch in enumerate(prepared_dataloader):
        print(f"    Batch {i}: keys={list(batch.keys())}, batch_size={len(batch['input_ids'])}")
        if i >= 2:
            break
    print(f"  ✓ Prepared DataLoader iteration successful")


def test_step_7_multiprocessing(dataset, collator):
    """Test with num_workers > 0 to simulate multiprocessing."""
    print("\n" + "=" * 80)
    print("STEP 7: Test DataLoader with num_workers=2 (multiprocessing)")
    print("=" * 80)

    dataloader = DataLoader(
        dataset,
        batch_size=4,
        collate_fn=collator,
        num_workers=2,  # Use multiprocessing
    )
    print(f"  ✓ DataLoader created with num_workers=2")

    print(f"  Iterating...")
    try:
        for i, batch in enumerate(dataloader):
            print(f"    Batch {i}: keys={list(batch.keys())}, batch_size={len(batch['input_ids'])}")
            if i >= 2:
                break
        print(f"  ✓ Multiprocessing DataLoader iteration successful")
    except Exception as e:
        print(f"  ✗ Multiprocessing DataLoader failed: {e}")
        import traceback
        traceback.print_exc()
        print(f"\n  This suggests the issue is with multiprocessing/pickling in worker processes")


def main():
    """Run all tests."""
    print("\n" + "=" * 80)
    print("DATASET DIAGNOSTIC TEST SUITE")
    print("=" * 80)
    print()

    try:
        # Step 1: Basic dataset
        dataset = test_step_1_dataset()

        # Step 2: Pickle test
        dataset = test_step_2_pickle(dataset)

        # Step 3: Tokenizer
        tokenizer = test_step_3_tokenizer()

        # Step 4: Collator
        collator = test_step_4_collator(dataset, tokenizer)

        # Step 5: DataLoader (single process)
        dataloader = test_step_5_dataloader(dataset, collator)

        # Step 6: Accelerate
        test_step_6_accelerate(dataloader)

        # Step 7: Multiprocessing
        test_step_7_multiprocessing(dataset, collator)

        print("\n" + "=" * 80)
        print("ALL TESTS PASSED!")
        print("=" * 80)
        print("\nIf training still fails, the issue is likely in how Trainer/Accelerate")
        print("initializes or wraps the dataset in distributed mode.")

    except Exception as e:
        print("\n" + "=" * 80)
        print(f"TEST FAILED: {e}")
        print("=" * 80)
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
