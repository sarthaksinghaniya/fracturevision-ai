#!/usr/bin/env python3
"""
Auto-Dataset Discovery and Preparation Script

Automatically discovers images from datasets/ directory, handles multiple formats,
creates train/val/test splits, and prepares classification-ready dataset.
"""

import os
import shutil
import random
from pathlib import Path
from collections import defaultdict
import yaml

def find_images_recursive(base_dir, extensions=('.jpg', '.jpeg', '.png')):
    """
    Recursively find all images in directory tree.

    Args:
        base_dir: Root directory to search
        extensions: Tuple of file extensions to include

    Returns:
        dict: {class_name: [image_paths]}
    """
    base_path = Path(base_dir)
    class_images = defaultdict(list)

    if not base_path.exists():
        print(f"Warning: Directory {base_dir} does not exist")
        return class_images

    print(f"Scanning for images in: {base_dir}")

    # Find all image files recursively
    for ext in extensions:
        for img_path in base_path.rglob(f'*{ext}'):
            # Get relative path from base_dir
            rel_path = img_path.relative_to(base_path)

            # Infer class from parent directory name
            class_name = rel_path.parts[0] if len(rel_path.parts) > 1 else 'unknown'

            # Skip if in train/val/test subdirs (already split)
            if class_name in ['train', 'valid', 'test', 'val']:
                # Go up one level for class name
                if len(rel_path.parts) > 2:
                    class_name = rel_path.parts[1]
                else:
                    continue

            class_images[class_name].append(str(img_path))

    return dict(class_images)

def detect_dataset_format(base_dir):
    """
    Detect if dataset is already split or needs splitting.

    Args:
        base_dir: Dataset directory

    Returns:
        str: 'split' if train/val/test exist, 'unsplit' otherwise
    """
    base_path = Path(base_dir)

    # Check for standard split structure
    split_dirs = ['train', 'valid', 'test']
    has_splits = all((base_path / split).exists() for split in split_dirs)

    if has_splits:
        # Check if splits contain class subdirectories
        train_dir = base_path / 'train'
        if train_dir.exists():
            class_dirs = [d for d in train_dir.iterdir() if d.is_dir()]
            if class_dirs:
                return 'split'

    return 'unsplit'

def convert_yolo_to_classification_if_needed(base_dir):
    """
    Check for YOLO format and convert to classification if found.

    Args:
        base_dir: Dataset directory
    """
    base_path = Path(base_dir)

    # Look for YOLO structure (images/ and labels/ directories)
    yolo_dirs = []
    for item in base_path.iterdir():
        if item.is_dir():
            images_dir = item / 'images'
            labels_dir = item / 'labels'
            if images_dir.exists() and labels_dir.exists():
                yolo_dirs.append(str(item))

    if yolo_dirs:
        print(f"Found YOLO datasets: {yolo_dirs}")
        print("Converting to classification format...")

        # Import conversion function
        try:
            from yolo_to_classification import convert_yolo_to_classification

            for yolo_dir in yolo_dirs:
                output_dir = str(base_path / f"{Path(yolo_dir).name}_classification")
                convert_yolo_to_classification(yolo_dir, output_dir)
                print(f"Converted {yolo_dir} -> {output_dir}")

        except ImportError:
            print("Warning: yolo_to_classification.py not found, skipping YOLO conversion")

def create_train_val_test_split(class_images, output_dir, train_ratio=0.7, val_ratio=0.1, test_ratio=0.2):
    """
    Create train/val/test splits from unsplit dataset.

    Args:
        class_images: dict of {class_name: [image_paths]}
        output_dir: Output directory for splits
        train_ratio, val_ratio, test_ratio: Split ratios
    """
    output_path = Path(output_dir)
    splits = ['train', 'valid', 'test']

    # Create split directories
    for split in splits:
        split_dir = output_path / split
        split_dir.mkdir(parents=True, exist_ok=True)

    print("Creating train/val/test splits...")

    total_images = 0
    split_counts = {split: defaultdict(int) for split in splits}

    for class_name, image_paths in class_images.items():
        if not image_paths:
            continue

        # Shuffle images for random split
        random.shuffle(image_paths)

        n_total = len(image_paths)
        n_train = int(n_total * train_ratio)
        n_val = int(n_total * val_ratio)
        n_test = n_total - n_train - n_val

        # Ensure at least one image per split if possible
        if n_total >= 3:
            splits_indices = [n_train, n_train + n_val, n_total]
        elif n_total == 2:
            splits_indices = [1, 1, 2]
        else:
            splits_indices = [1, 1, 1]

        split_data = {
            'train': image_paths[:splits_indices[0]],
            'valid': image_paths[splits_indices[0]:splits_indices[1]],
            'test': image_paths[splits_indices[1]:]
        }

        # Copy images to split directories
        for split, images in split_data.items():
            if not images:
                continue

            class_dir = output_path / split / class_name
            class_dir.mkdir(exist_ok=True)

            for img_path in images:
                img_name = Path(img_path).name
                shutil.copy2(img_path, class_dir / img_name)
                split_counts[split][class_name] += 1
                total_images += 1

        print(f"  {class_name}: {n_train} train, {n_val} val, {n_test} test")

    # Print summary
    print(f"\nSplit Summary:")
    for split in splits:
        total_split = sum(split_counts[split].values())
        print(f"  {split.upper()}: {total_split} images")
        if split_counts[split]:
            for class_name, count in sorted(split_counts[split].items()):
                print(f"    {class_name}: {count}")

    print(f"\nTotal images processed: {total_images}")

def prepare_dataset(datasets_dir='datasets', output_dir='dataset_prepared'):
    """
    Main function to prepare dataset for training.

    Args:
        datasets_dir: Input datasets directory
        output_dir: Output prepared dataset directory
    """
    print("🔍 Auto-Dataset Discovery and Preparation")
    print("=" * 50)

    # Convert YOLO datasets if found
    convert_yolo_to_classification_if_needed(datasets_dir)

    # Find all images
    class_images = find_images_recursive(datasets_dir)

    if not class_images:
        print("❌ ERROR: No images found in datasets directory!")
        print("Please ensure images are in datasets/ directory with proper structure.")
        return False

    total_images = sum(len(images) for images in class_images.values())
    print(f"✅ Found {total_images} images in {len(class_images)} classes:")
    for class_name, images in sorted(class_images.items()):
        print(f"  {class_name}: {len(images)} images")

    # Check if already split
    dataset_format = detect_dataset_format(datasets_dir)

    if dataset_format == 'split':
        print("\n✅ Dataset already split into train/val/test")
        # Just copy to output_dir if needed
        output_path = Path(output_dir)
        if not output_path.exists():
            print("Copying existing split to output directory...")
            shutil.copytree(datasets_dir, output_dir)
    else:
        print("\n🔀 Dataset not split, creating automatic train/val/test splits...")
        create_train_val_test_split(class_images, output_dir)

    # Save class mapping
    class_mapping = {class_name: idx for idx, class_name in enumerate(sorted(class_images.keys()))}
    mapping_file = Path(output_dir) / 'class_mapping.yaml'
    with open(mapping_file, 'w') as f:
        yaml.dump(class_mapping, f, default_flow_style=False)

    print(f"\n✅ Dataset prepared successfully!")
    print(f"📁 Output directory: {output_dir}")
    print(f"🏷️  Classes: {list(class_mapping.keys())}")
    print(f"📊 Class mapping saved to: {mapping_file}")

    return True

if __name__ == '__main__':
    # Set random seed for reproducible splits
    random.seed(42)

    success = prepare_dataset()
    if not success:
        exit(1)
