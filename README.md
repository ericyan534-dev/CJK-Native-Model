# CNM-BERT (Compositional Network Model)

This repository implements the CNM-BERT prototype described in the research plan. The focus is on injecting explicit IDS-based structural information into a standard `bert-base-chinese` backbone without altering the vocabulary.

## Repository Layout

```
cnm_bert/
├── assets/                  # External artifacts (download BabelStone IDS + vocab here)
├── data/                    # Generated structural lexicon JSON
├── src/                     # Library code (ETL, tokenizer, model, dataset)
├── scripts/                 # Training entry points
├── requirements.txt         # Minimal runtime dependencies
```

## Quickstart

1. Place the external resources:
   - `cnm_bert/assets/ids.txt`: BabelStone IDS file.
   - `cnm_bert/assets/vocab.txt`: `bert-base-chinese` vocabulary.
   - `cnm_bert/assets/wiki_zh_sample.txt`: Small text slice for debugging.
2. Build the structural lexicon:
   ```bash
   python cnm_bert/src/etl_ids.py --input cnm_bert/assets/ids.txt --output cnm_bert/data/char_to_ids_tree.json
   ```
3. Run a debugging pre-train loop (overfits the sample):
   ```bash
   python cnm_bert/scripts/run_pretraining.py \
     --output_dir outputs \
     --per_device_train_batch_size 2 \
     --num_train_epochs 1 \
     --learning_rate 5e-5 \
     --logging_steps 10
   ```

## Notes

- The tokenizer (`CNMBertTokenizer`) pre-computes structural indices for every token in the BERT vocabulary for fast lookup.
- The model (`CNMForMaskedLM`) fuses Tree-MLP structural embeddings with the original BERT embeddings and supports optional encoder freezing for stability during early training.
- The data collator performs Whole Word Masking (WWM) using jieba segmentation and emits `struct_ids` aligned with `input_ids`.
