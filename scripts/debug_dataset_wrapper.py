#!/usr/bin/env python3
"""Dataset wrapper with extensive logging for debugging."""

import sys
from torch.utils.data import Dataset


class DebugDatasetWrapper(Dataset):
    """Wrapper that logs all dataset access."""

    def __init__(self, dataset):
        """Wrap a dataset with logging.

        Args:
            dataset: The dataset to wrap
        """
        self.dataset = dataset
        self._access_count = 0
        print(f"\n[DEBUG_WRAPPER] Created wrapper around {type(dataset)}", file=sys.stderr)
        print(f"[DEBUG_WRAPPER] Dataset length: {len(dataset)}", file=sys.stderr)
        print(f"[DEBUG_WRAPPER] Dataset class: {dataset.__class__}", file=sys.stderr)

    def __len__(self):
        """Return length of wrapped dataset."""
        length = len(self.dataset)
        if self._access_count < 5:  # Only log first few calls
            print(f"[DEBUG_WRAPPER] __len__() called, returning {length}", file=sys.stderr)
        return length

    def __getitem__(self, idx):
        """Get item from wrapped dataset with logging."""
        self._access_count += 1

        # Always log to catch the issue
        print(f"\n[DEBUG_WRAPPER] __getitem__({idx}) called (call #{self._access_count})", file=sys.stderr)

        try:
            item = self.dataset[idx]
            print(f"[DEBUG_WRAPPER] Got item: type={type(item)}, ", file=sys.stderr, end='')

            if isinstance(item, dict):
                print(f"keys={list(item.keys())}", file=sys.stderr)
                if len(str(item)) < 200:
                    print(f"[DEBUG_WRAPPER] Item value: {item}", file=sys.stderr)
            else:
                print(f"value={item}", file=sys.stderr)

            return item

        except Exception as e:
            print(f"[DEBUG_WRAPPER] ERROR getting item {idx}: {e}", file=sys.stderr)
            import traceback
            traceback.print_exc(file=sys.stderr)
            raise

    def __getstate__(self):
        """Handle pickling."""
        print(f"[DEBUG_WRAPPER] __getstate__() called", file=sys.stderr)
        # Forward to wrapped dataset if it has __getstate__
        if hasattr(self.dataset, '__getstate__'):
            return {
                'dataset_state': self.dataset.__getstate__(),
                '_access_count': self._access_count,
            }
        else:
            return {
                'dataset': self.dataset,
                '_access_count': self._access_count,
            }

    def __setstate__(self, state):
        """Handle unpickling."""
        print(f"[DEBUG_WRAPPER] __setstate__() called in PID {__import__('os').getpid()}", file=sys.stderr)

        if 'dataset_state' in state:
            # Reconstruct wrapped dataset
            from cnm_bert.data.dataset import TextLineDataset
            self.dataset = TextLineDataset.__new__(TextLineDataset)
            self.dataset.__setstate__(state['dataset_state'])
        else:
            self.dataset = state['dataset']

        self._access_count = state['_access_count']
        print(f"[DEBUG_WRAPPER] Restored wrapper, dataset type: {type(self.dataset)}", file=sys.stderr)
