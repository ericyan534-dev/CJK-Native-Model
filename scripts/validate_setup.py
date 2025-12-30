#!/usr/bin/env python3
"""Comprehensive validation script for CNM-BERT training setup.

Validates:
1. Dataset loading and format
2. Tokenizer initialization
3. Model loading from BERT
4. Data collator functionality
5. Single batch forward pass
6. DataLoader iteration
7. Training argument setup
8. W&B integration (if enabled)
"""

import sys
from pathlib import Path
import torch

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from cnm_bert import CNMForMaskedLM, CNMBertTokenizer
from cnm_bert.data import TextLineDataset, WWMDataCollator


def print_section(title):
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)


def test_dataset():
    print_section("1/8: Dataset Loading")

    dataset = TextLineDataset(
        file_path="data/pretrain/corpus_clean.txt",
        max_samples=100
    )

    print(f"✓ Dataset loaded: {len(dataset)} examples")

    # Test first example
    example = dataset[0]
    assert isinstance(example, dict), f"Expected dict, got {type(example)}"
    assert "text" in example, f"Missing 'text' key. Keys: {list(example.keys())}"
    assert isinstance(example["text"], str), f"text should be str, got {type(example['text'])}"

    print(f"✓ Example format valid")
    print(f"  First text: {example['text'][:50]}...")

    return dataset


def test_tokenizer():
    print_section("2/8: Tokenizer")

    tokenizer = CNMBertTokenizer.from_pretrained(
        "bert-base-chinese",
        struct_path="data/processed/char_to_ids_tree.json"
    )

    print(f"✓ Tokenizer loaded")
    print(f"  Vocab size: {tokenizer.vocab_size}")
    print(f"  Special tokens: {tokenizer.all_special_tokens}")

    # Test tokenization
    text = "今天天气很好"
    encoding = tokenizer(
        text,
        return_tensors="pt",
        return_struct_ids=True,
        padding="max_length",
        max_length=128,
        truncation=True
    )

    assert "input_ids" in encoding
    assert "struct_ids" in encoding
    assert "attention_mask" in encoding

    print(f"✓ Tokenization works")
    print(f"  input_ids shape: {encoding['input_ids'].shape}")
    print(f"  struct_ids shape: {encoding['struct_ids'].shape}")

    return tokenizer


def test_model():
    print_section("3/8: Model Loading")

    model = CNMForMaskedLM.from_bert_pretrained(
        bert_model_name="bert-base-chinese",
        tree_path=Path("data/processed/char_to_ids_tree.json"),
        struct_dim=256,
        fusion_strategy="concat",
    )

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"✓ Model loaded")
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")

    return model


def test_collator(tokenizer, dataset):
    print_section("4/8: Data Collator")

    collator = WWMDataCollator(
        tokenizer=tokenizer,
        mlm_probability=0.15,
        max_seq_length=128
    )

    # Test with batch of examples
    batch_examples = [dataset[i] for i in range(4)]
    batch = collator(batch_examples)

    print(f"✓ Collator works")
    print(f"  Batch keys: {list(batch.keys())}")
    for key, val in batch.items():
        print(f"    {key}: {val.shape}")

    # Verify batch format
    assert "input_ids" in batch
    assert "struct_ids" in batch
    assert "attention_mask" in batch
    assert "labels" in batch

    print(f"✓ Batch format valid")

    return collator, batch


def test_forward_pass(model, batch):
    print_section("5/8: Forward Pass")

    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    batch = {k: v.to(device) for k, v in batch.items()}

    with torch.no_grad():
        outputs = model(**batch)

    print(f"✓ Forward pass successful")
    print(f"  Loss: {outputs.loss.item():.4f}")
    print(f"  Logits shape: {outputs.logits.shape}")
    print(f"  Device: {device}")


def test_dataloader(dataset, collator):
    print_section("6/8: DataLoader")

    from torch.utils.data import DataLoader

    dataloader = DataLoader(
        dataset,
        batch_size=4,
        collate_fn=collator,
        num_workers=0,
        shuffle=False
    )

    batches_tested = 0
    for batch in dataloader:
        batches_tested += 1
        if batches_tested >= 3:
            break

    print(f"✓ DataLoader works")
    print(f"  Tested {batches_tested} batches successfully")


def test_training_args():
    print_section("7/8: Training Arguments")

    from transformers import TrainingArguments

    training_args = TrainingArguments(
        output_dir="experiments/test",
        per_device_train_batch_size=8,
        gradient_accumulation_steps=1,
        learning_rate=1e-4,
        warmup_steps=100,
        max_steps=1000,
        logging_steps=10,
        save_steps=100,
        evaluation_strategy="steps",  # transformers 4.38.2 uses evaluation_strategy
        eval_steps=100,
        fp16=torch.cuda.is_available(),
        dataloader_num_workers=0,
        report_to=["tensorboard"],
    )

    print(f"✓ Training arguments valid")
    print(f"  Output dir: {training_args.output_dir}")
    print(f"  Batch size: {training_args.per_device_train_batch_size}")
    print(f"  FP16: {training_args.fp16}")


def test_wandb():
    print_section("8/8: W&B Integration (Optional)")

    try:
        import wandb
        print(f"✓ W&B installed: {wandb.__version__}")
        print(f"  To use W&B, run: wandb login")
    except ImportError:
        print(f"⚠ W&B not installed (optional)")
        print(f"  Install with: pip install wandb")


def main():
    print("\n" + "=" * 70)
    print("  CNM-BERT TRAINING SETUP VALIDATION")
    print("=" * 70)

    try:
        dataset = test_dataset()
        tokenizer = test_tokenizer()
        model = test_model()
        collator, batch = test_collator(tokenizer, dataset)
        test_forward_pass(model, batch)
        test_dataloader(dataset, collator)
        test_training_args()
        test_wandb()

        print("\n" + "=" * 70)
        print("  ✓✓✓ ALL VALIDATION TESTS PASSED ✓✓✓")
        print("=" * 70)
        print("\nYou can now run training:")
        print("  python scripts/train.py --config experiments/configs/pretrain_base.yaml")
        print("\nFor multi-GPU:")
        print("  torchrun --nproc_per_node=8 scripts/train.py --config experiments/configs/pretrain_base.yaml")
        print("=" * 70)

        return 0

    except Exception as e:
        print("\n" + "=" * 70)
        print(f"  ✗✗✗ VALIDATION FAILED ✗✗✗")
        print("=" * 70)
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
