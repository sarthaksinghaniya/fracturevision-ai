#!/usr/bin/env python3
"""
Convert YOLO-format fracture datasets into classification folders.
"""

import argparse
import shutil
from collections import Counter
from pathlib import Path

import yaml

from src.pipeline_utils import (
    DEFAULT_BINARY_CLASS_NAMES,
    load_config,
    normalize_label_name,
    resolve_task_metadata,
)


def load_class_mapping(yaml_file):
    if not yaml_file or not Path(yaml_file).exists():
        return {}
    with open(yaml_file, "r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    names = data.get("names", {})
    if isinstance(names, list):
        return {index: name for index, name in enumerate(names)}
    return {int(index): name for index, name in names.items()}


def parse_yolo_label(label_path):
    with open(label_path, "r", encoding="utf-8") as handle:
        for line in handle:
            parts = line.strip().split()
            if len(parts) >= 5:
                return int(parts[0])
    return None


def resolve_class_name(class_id, config, class_mapping):
    fracture_mapping = {int(key): normalize_label_name(value) for key, value in (config.get("fracture_mapping") or {}).items()}
    if class_id in fracture_mapping:
        return fracture_mapping[class_id]

    raw_name = normalize_label_name(class_mapping.get(class_id, f"class-{class_id}"))
    if raw_name in {"simple", "comminuted", "spiral", "greenstick", "stress", "non-fracture"}:
        return raw_name
    return "non-fracture" if raw_name in {"humerus", "normal"} else "fracture"


def convert_yolo_to_classification(input_dir, output_dir, class_mapping=None):
    config = load_config()
    class_mapping = class_mapping or {}
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    splits = ["train", "valid", "test"]

    provisional_records = []
    for split in splits:
        images_dir = input_path / split / "images"
        labels_dir = input_path / split / "labels"
        if not images_dir.exists() or not labels_dir.exists():
            continue

        for image_path in sorted(images_dir.iterdir()):
            if image_path.suffix.lower() not in {".jpg", ".jpeg", ".png", ".bmp"}:
                continue
            label_path = labels_dir / f"{image_path.stem}.txt"
            if not label_path.exists():
                continue

            class_id = parse_yolo_label(label_path)
            if class_id is None:
                continue

            provisional_records.append(
                {
                    "image_path": str(image_path),
                    "label_name": resolve_class_name(class_id, config, class_mapping),
                    "split": split,
                }
            )

    metadata = resolve_task_metadata(
        [{"image_path": item["image_path"], "label_name": item["label_name"]} for item in provisional_records],
        config=config,
    )
    active_classes = metadata["class_names"] if metadata["task_type"] == "multiclass" else DEFAULT_BINARY_CLASS_NAMES

    for split in splits:
        for class_name in active_classes:
            (output_path / split / class_name).mkdir(parents=True, exist_ok=True)

    class_counts = Counter()
    for item in provisional_records:
        class_name = item["label_name"]
        if metadata["task_type"] == "binary":
            class_name = "non-fracture" if class_name == "non-fracture" else "fracture"

        destination = output_path / item["split"] / class_name / Path(item["image_path"]).name
        shutil.copy2(item["image_path"], destination)
        class_counts[class_name] += 1

    with open(output_path / "class_mapping.yaml", "w", encoding="utf-8") as handle:
        yaml.safe_dump(dict(class_counts), handle, sort_keys=True)

    print(f"Converted {sum(class_counts.values())} images")
    print(f"Task type: {metadata['task_type']}")
    print(f"Classes: {active_classes}")
    print(dict(class_counts))


def main():
    parser = argparse.ArgumentParser(description="Convert YOLO dataset to classification format")
    parser.add_argument("--input_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--yaml_file")
    args = parser.parse_args()

    convert_yolo_to_classification(
        args.input_dir,
        args.output_dir,
        load_class_mapping(args.yaml_file) if args.yaml_file else {},
    )


if __name__ == "__main__":
    main()
