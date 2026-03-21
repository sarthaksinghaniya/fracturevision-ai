#!/usr/bin/env python3
"""
Dataset split creation for binary or multi-class fracture classification.
"""

import os
import pickle
import sys
from collections import Counter

import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.utils import class_weight

from src.pipeline_utils import (
    deduplicate_records,
    load_config,
    resolve_task_metadata,
    save_dataset_metadata,
    scan_dataset_records,
)


def _compute_normalization_stats(image_paths):
    means = []
    stds = []

    for image_path in image_paths[: min(500, len(image_paths))]:
        image = cv2.imread(image_path)
        if image is None:
            continue
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) / 255.0
        means.append(np.mean(image, axis=(0, 1)))
        stds.append(np.std(image, axis=(0, 1)))

    if not means:
        return {
            "mean": np.array([0.485, 0.456, 0.406]),
            "std": np.array([0.229, 0.224, 0.225]),
        }

    return {"mean": np.mean(means, axis=0), "std": np.mean(stds, axis=0)}


def create_splits():
    print("=" * 60)
    print("Creating Dataset Splits")
    print("=" * 60)

    config = load_config()
    output_dir = os.path.join("outputs", "splits")
    os.makedirs(output_dir, exist_ok=True)

    records = scan_dataset_records("datasets", config=config)
    if not records:
        print("ERROR: No dataset records found under datasets/")
        return False

    metadata = resolve_task_metadata(records, config=config)
    records = deduplicate_records(metadata["records"])
    metadata = resolve_task_metadata(records, config=config)

    print(f"Detected task type: {metadata['task_type']}")
    print(f"Class names: {metadata['class_names']}")
    print(f"Class distribution: {metadata['class_distribution']}")

    if len(records) < 30:
        print("ERROR: Need at least 30 images to create reliable splits.")
        return False

    counts = Counter(record["label"] for record in records)
    if len(counts) < 2:
        print("ERROR: Need at least 2 classes in the dataset.")
        return False

    min_class_count = min(counts.values())
    if min_class_count < 3:
        print(f"ERROR: Smallest class has only {min_class_count} samples. Need at least 3.")
        return False

    image_paths = [record["image_path"] for record in records]
    labels = [record["label"] for record in records]
    label_names = [record["label_name"] for record in records]

    try:
        x_train, x_temp, y_train, y_temp, name_train, name_temp = train_test_split(
            image_paths,
            labels,
            label_names,
            test_size=0.3,
            stratify=labels,
            random_state=config.get("random_seed", 42),
        )
        x_val, x_test, y_val, y_test, name_val, name_test = train_test_split(
            x_temp,
            y_temp,
            name_temp,
            test_size=2 / 3,
            stratify=y_temp,
            random_state=config.get("random_seed", 42),
        )
    except ValueError as exc:
        print(f"ERROR during stratified split creation: {exc}")
        return False

    splits = {
        "train": list(zip(x_train, y_train)),
        "val": list(zip(x_val, y_val)),
        "test": list(zip(x_test, y_test)),
    }
    with open(os.path.join(output_dir, "dataset_splits.pkl"), "wb") as handle:
        pickle.dump(splits, handle)

    class_ids = np.array(sorted(set(y_train)))
    weights = class_weight.compute_class_weight(class_weight="balanced", classes=class_ids, y=np.array(y_train))
    weights = weights / np.mean(weights)
    class_weights = {int(class_id): float(weight) for class_id, weight in zip(class_ids, weights)}
    with open(os.path.join(output_dir, "class_weights.pkl"), "wb") as handle:
        pickle.dump(class_weights, handle)

    norm_stats = _compute_normalization_stats(x_train)
    with open(os.path.join(output_dir, "normalization_stats.pkl"), "wb") as handle:
        pickle.dump(norm_stats, handle)

    dataset_metadata = {
        "task_type": metadata["task_type"],
        "class_names": metadata["class_names"],
        "num_classes": metadata["num_classes"],
        "class_to_idx": metadata["class_to_idx"],
        "class_distribution": metadata["class_distribution"],
        "train_distribution": dict(Counter(name_train)),
        "val_distribution": dict(Counter(name_val)),
        "test_distribution": dict(Counter(name_test)),
    }
    save_dataset_metadata(dataset_metadata, os.path.join(output_dir, "dataset_metadata.yaml"))

    print(f"Train split: {len(x_train)} images {dict(Counter(name_train))}")
    print(f"Val split: {len(x_val)} images {dict(Counter(name_val))}")
    print(f"Test split: {len(x_test)} images {dict(Counter(name_test))}")
    print(f"Class weights: {class_weights}")
    print("Dataset splits created successfully.")
    return True


if __name__ == "__main__":
    if not create_splits():
        sys.exit(1)
