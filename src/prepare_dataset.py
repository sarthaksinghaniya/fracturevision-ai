#!/usr/bin/env python3
"""
Compatibility wrapper for split creation from src/.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from create_dataset_splits import create_splits


def prepare_dataset():
    return create_splits()


if __name__ == "__main__":
    if not prepare_dataset():
        sys.exit(1)
