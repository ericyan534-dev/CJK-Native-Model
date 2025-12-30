#!/usr/bin/env python3
"""Simple training script that WORKS - no Trainer complexity."""

import sys
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from cnm_bert import CNMForMaskedLM, CNMBertTokenizer
from cnm_bert.data import WWMDataCollator

def load_corpus(file_path):
    """Load corpus into memory."""
    with open(file_path, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip()]
    return lines

def main():
    print("=" * 70)
    print("SIMPLE TRAINING SCRIPT (NO TRAINER)")
    print("=" * 70)

    # Config
    corpus_file = "data/pretrain/corpus_clean.txt"
    output_dir = "experiments/debug"
    batch_size = 32
    max_steps = 1000
    learning_rate = 1e-4
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. Load corpus
    print("\n[1/5] Loading corpus...")
    texts = load_corpus(corpus_file)
    print(f"   Loaded {len(texts):,} examples")

    # 2. Load tokenizer
    print("\n[2/5] Loading tokenizer...")
    tokenizer = CNMBertTokenizer.from_pretrained(
        "bert-base-chinese",
        struct_path="data/processed/char_to_ids_tree.json"
    )
    print(f"   Tokenizer: vocab_size={tokenizer.vocab_size}")

    # 3. Load model
    print("\n[3/5] Loading model...")
    model = CNMForMaskedLM.from_bert_pretrained(
        bert_model_name="bert-base-chinese",
        tree_path=Path("data/processed/char_to_ids_tree.json"),
        struct_dim=256,
        fusion_strategy="concat",
    )
    model = model.to(device)
    print(f"   Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # 4. Create data collator
    print("\n[4/5] Creating data collator...")
    collator = WWMDataCollator(
        tokenizer=tokenizer,
        mlm_probability=0.15,
        max_seq_length=512
    )

    # 5. Training loop
    print("\n[5/5] Training...")
    print("=" * 70)

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    model.train()

    global_step = 0
    total_loss = 0
    pbar = tqdm(total=max_steps, desc="Training")

    while global_step < max_steps:
        # Sample random batch
        import random
        batch_texts = random.sample(texts, min(batch_size, len(texts)))
        batch_dicts = [{"text": text} for text in batch_texts]

        # Collate
        batch = collator(batch_dicts)

        # Move to device
        batch = {k: v.to(device) for k, v in batch.items()}

        # Forward
        outputs = model(**batch)
        loss = outputs.loss

        # Backward
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        # Log
        total_loss += loss.item()
        global_step += 1
        pbar.update(1)
        pbar.set_postfix({"loss": f"{loss.item():.4f}", "avg_loss": f"{total_loss/global_step:.4f}"})

        # Save checkpoint
        if global_step % 100 == 0:
            Path(output_dir).mkdir(parents=True, exist_ok=True)
            torch.save({
                'step': global_step,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': total_loss / global_step,
            }, f"{output_dir}/checkpoint-{global_step}.pt")

    pbar.close()
    print("\n" + "=" * 70)
    print(f"Training complete! Avg loss: {total_loss/global_step:.4f}")
    print(f"Checkpoints saved to: {output_dir}")
    print("=" * 70)

if __name__ == "__main__":
    main()
