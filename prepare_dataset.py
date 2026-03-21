#!/usr/bin/env python3
"""
Compatibility wrapper for dataset preparation.
"""

import sys

from create_dataset_splits import create_splits


def prepare_datasets():
    return create_splits()


if __name__ == "__main__":
    if not prepare_datasets():
        sys.exit(1)
