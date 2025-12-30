#!/usr/bin/env python3
"""Custom Trainer with extensive debugging to diagnose dataset issues."""

import sys
from transformers import Trainer
from torch.utils.data import DataLoader


class DebugTrainer(Trainer):
    """Trainer with debugging hooks to trace dataset/dataloader issues."""

    def get_train_dataloader(self):
        """Override to add debugging."""
        print("\n" + "="*80, file=sys.stderr)
        print("[DEBUG_TRAINER] get_train_dataloader() called", file=sys.stderr)
        print("="*80, file=sys.stderr)

        # Check dataset before creating dataloader
        print(f"\n[DEBUG_TRAINER] self.train_dataset type: {type(self.train_dataset)}", file=sys.stderr)
        print(f"[DEBUG_TRAINER] self.train_dataset class: {self.train_dataset.__class__}", file=sys.stderr)
        print(f"[DEBUG_TRAINER] self.train_dataset length: {len(self.train_dataset)}", file=sys.stderr)

        # Test direct access
        try:
            item = self.train_dataset[0]
            print(f"[DEBUG_TRAINER] Direct access dataset[0]: {item}", file=sys.stderr)
            print(f"[DEBUG_TRAINER] Keys: {list(item.keys())}", file=sys.stderr)
        except Exception as e:
            print(f"[DEBUG_TRAINER] ERROR accessing dataset[0]: {e}", file=sys.stderr)
            import traceback
            traceback.print_exc(file=sys.stderr)

        # Call parent implementation
        print(f"\n[DEBUG_TRAINER] Calling parent get_train_dataloader()...", file=sys.stderr)
        dataloader = super().get_train_dataloader()

        print(f"[DEBUG_TRAINER] Dataloader created: {type(dataloader)}", file=sys.stderr)
        print(f"[DEBUG_TRAINER] Dataloader.dataset type: {type(dataloader.dataset)}", file=sys.stderr)
        print(f"[DEBUG_TRAINER] Dataloader.dataset class: {dataloader.dataset.__class__}", file=sys.stderr)

        # Check if dataset is wrapped
        if hasattr(dataloader.dataset, 'dataset'):
            print(f"[DEBUG_TRAINER] Dataset is wrapped!", file=sys.stderr)
            print(f"[DEBUG_TRAINER] Inner dataset type: {type(dataloader.dataset.dataset)}", file=sys.stderr)
            print(f"[DEBUG_TRAINER] Inner dataset class: {dataloader.dataset.dataset.__class__}", file=sys.stderr)

        # Try to access item from dataloader's dataset
        try:
            item = dataloader.dataset[0]
            print(f"[DEBUG_TRAINER] Dataloader.dataset[0]: {item}", file=sys.stderr)
            print(f"[DEBUG_TRAINER] Keys: {list(item.keys()) if isinstance(item, dict) else 'NOT A DICT'}", file=sys.stderr)
        except Exception as e:
            print(f"[DEBUG_TRAINER] ERROR accessing dataloader.dataset[0]: {e}", file=sys.stderr)
            import traceback
            traceback.print_exc(file=sys.stderr)

        print(f"[DEBUG_TRAINER] Returning dataloader\n", file=sys.stderr)
        return dataloader

    def training_step(self, model, inputs):
        """Override to debug inputs."""
        # Only log first few steps
        if self.state.global_step < 3:
            print(f"\n[DEBUG_TRAINER] training_step() called, step={self.state.global_step}", file=sys.stderr)
            print(f"[DEBUG_TRAINER] inputs keys: {list(inputs.keys())}", file=sys.stderr)
            if 'input_ids' in inputs:
                print(f"[DEBUG_TRAINER] input_ids shape: {inputs['input_ids'].shape}", file=sys.stderr)
                print(f"[DEBUG_TRAINER] First 3 input_ids: {inputs['input_ids'][:3]}", file=sys.stderr)
            if 'labels' in inputs:
                print(f"[DEBUG_TRAINER] labels shape: {inputs['labels'].shape}", file=sys.stderr)

        return super().training_step(model, inputs)
