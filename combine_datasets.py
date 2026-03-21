#!/usr/bin/env python3
"""
Combine Multiple Classification Datasets

Combines multiple converted YOLO classification datasets into a single unified dataset.
"""

import os
import shutil
from pathlib import Path
from collections import defaultdict

def combine_classification_datasets(dataset_dirs, output_dir):
    """
    Combine multiple classification datasets into one.

    Args:
        dataset_dirs: List of dataset directory paths
        output_dir: Output directory for combined dataset
    """

    output_path = Path(output_dir)
    splits = ['train', 'valid', 'test']

    # Create output directories
    for split in splits:
        split_dir = output_path / split
        split_dir.mkdir(parents=True, exist_ok=True)

    print(f"Combining {len(dataset_dirs)} datasets into {output_dir}")

    total_stats = defaultdict(lambda: defaultdict(int))
    global_class_mapping = {}  # Map original classes to unified classes
    next_class_id = 0

    for dataset_idx, dataset_dir in enumerate(dataset_dirs):
        dataset_path = Path(dataset_dir)
        print(f"\nProcessing dataset {dataset_idx + 1}: {dataset_dir}")

        # Process each split
        for split in splits:
            split_dir = dataset_path / split
            if not split_dir.exists():
                print(f"  Warning: {split} split not found in {dataset_dir}")
                continue

            output_split_dir = output_path / split

            # Process each class directory
            for class_dir in split_dir.iterdir():
                if not class_dir.is_dir():
                    continue

                class_name = class_dir.name  # e.g., "class_0", "class_1"

                # Get unified class ID
                if class_name not in global_class_mapping:
                    global_class_mapping[class_name] = f"class_{next_class_id}"
                    next_class_id += 1

                unified_class_name = global_class_mapping[class_name]
                unified_class_dir = output_split_dir / unified_class_name
                unified_class_dir.mkdir(exist_ok=True)

                # Copy all images from this class
                image_count = 0
                for img_file in class_dir.glob('*.jpg'):
                    # Create unique filename to avoid conflicts
                    unique_name = f"ds{dataset_idx}_{img_file.name}"
                    output_img = unified_class_dir / unique_name
                    shutil.copy2(img_file, output_img)
                    image_count += 1

                total_stats[split][unified_class_name] += image_count
                print(f"    {split}/{class_name} -> {unified_class_name}: {image_count} images")

    # Print final statistics
    print("\nCombined Dataset Statistics:")
    print("=" * 50)

    for split in splits:
        print(f"\n{split.upper()} SPLIT:")
        split_total = 0
        for class_name, count in sorted(total_stats[split].items()):
            print(f"  {class_name}: {count}")
            split_total += count
        print(f"  Total: {split_total}")

    # Save class mapping
    mapping_file = output_path / 'class_mapping.yaml'
    with open(mapping_file, 'w') as f:
        import yaml
        yaml.dump(dict(global_class_mapping), f, default_flow_style=False)

    print(f"\nClass mapping saved to: {mapping_file}")
    print(f"Combined dataset created in: {output_dir}")
    print(f"Total classes: {len(global_class_mapping)}")

    return dict(global_class_mapping)

if __name__ == '__main__':
    # Dataset directories to combine
    dataset_dirs = [
        'dataset_classification_v2i',
        'dataset_classification_v4',
        'dataset_classification_yolo8'
    ]

    output_dir = 'dataset_combined'

    combine_classification_datasets(dataset_dirs, output_dir)
