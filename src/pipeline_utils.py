import os
import pickle
from collections import Counter
from pathlib import Path

import yaml


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp"}
DEFAULT_BINARY_CLASS_NAMES = ["non-fracture", "fracture"]
DEFAULT_CONFIG_CANDIDATES = [
    Path("config.yaml"),
    Path("configs") / "config.yaml",
]


def resolve_config_path(config_path=None):
    if config_path:
        path = Path(config_path)
        if path.exists():
            return path

    for candidate in DEFAULT_CONFIG_CANDIDATES:
        if candidate.exists():
            return candidate

    raise FileNotFoundError("Could not find config.yaml in the project root or configs/")


def load_config(config_path=None):
    path = resolve_config_path(config_path)
    with open(path, "r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle) or {}

    int_keys = ["batch_size", "epochs", "image_size", "num_classes", "random_seed"]
    float_keys = [
        "learning_rate",
        "lr",
        "weight_decay",
        "early_stopping_delta",
        "max_grad_norm",
        "scheduler_factor",
        "scheduler_min_lr",
    ]
    bool_keys = ["pretrained", "use_class_weights", "use_mixed_precision"]

    for key in int_keys:
        if key in config and config[key] is not None:
            config[key] = int(config[key])

    for key in float_keys:
        if key in config and config[key] is not None:
            config[key] = float(config[key])

    for key in bool_keys:
        if key in config and isinstance(config[key], str):
            config[key] = config[key].strip().lower() == "true"

    if "lr" not in config and "learning_rate" in config:
        config["lr"] = float(config["learning_rate"])
    if "learning_rate" not in config and "lr" in config:
        config["learning_rate"] = float(config["lr"])

    config["config_path"] = str(path)
    return config


def normalize_label_name(label_name):
    normalized = str(label_name).strip().lower().replace("_", "-")
    normalized = normalized.replace(" ", "-")
    while "--" in normalized:
        normalized = normalized.replace("--", "-")
    return normalized


def _normalized_set(values):
    return {normalize_label_name(value) for value in values}


def _load_yolo_names(dataset_dir):
    data_yaml = Path(dataset_dir) / "data.yaml"
    if not data_yaml.exists():
        return {}

    with open(data_yaml, "r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}

    names = data.get("names", {})
    if isinstance(names, list):
        return {int(index): str(name) for index, name in enumerate(names)}
    if isinstance(names, dict):
        return {int(index): str(name) for index, name in names.items()}
    return {}


def _extract_first_class_id(label_path):
    try:
        with open(label_path, "r", encoding="utf-8") as handle:
            for line in handle:
                parts = line.strip().split()
                if len(parts) >= 5:
                    return int(parts[0])
    except (OSError, ValueError):
        return None
    return None


def _build_fracture_mapping(config):
    mapping = {}
    for key, value in (config.get("fracture_mapping") or {}).items():
        try:
            mapping[int(key)] = normalize_label_name(value)
        except (TypeError, ValueError):
            continue
    return mapping


def _label_from_folder_name(folder_name):
    name = normalize_label_name(folder_name)
    if name in {"not-fractured", "not-fracture", "normal", "healthy"}:
        return "non-fracture"
    if name in {"fractured", "fracture"}:
        return "fracture"
    return name


def _binary_label_from_name(class_name):
    name = normalize_label_name(class_name)
    if name == "non-fracture":
        return "non-fracture"
    return "fracture"


def _resolve_yolo_label(class_id, yolo_names, fracture_mapping):
    if class_id in fracture_mapping:
        return fracture_mapping[class_id]

    raw_name = normalize_label_name(yolo_names.get(class_id, f"class-{class_id}"))
    if raw_name in {"non-fracture", "fracture", "simple", "comminuted", "spiral", "greenstick", "stress"}:
        return raw_name

    return _binary_label_from_name(raw_name if raw_name else f"class-{class_id}")


def scan_dataset_records(datasets_dir="datasets", config=None):
    config = config or {}
    fracture_mapping = _build_fracture_mapping(config)
    records = []
    datasets_root = Path(datasets_dir)

    if not datasets_root.exists():
        return records

    for dataset_dir in sorted(path for path in datasets_root.iterdir() if path.is_dir()):
        yolo_names = _load_yolo_names(dataset_dir)
        split_dirs = [dataset_dir / split for split in ("train", "valid", "test")]
        split_dirs = [split_dir for split_dir in split_dirs if split_dir.exists()]
        if not split_dirs:
            split_dirs = [dataset_dir]

        for split_dir in split_dirs:
            images_dir = split_dir / "images"
            labels_dir = split_dir / "labels"

            if images_dir.exists() and labels_dir.exists():
                for image_path in sorted(images_dir.iterdir()):
                    if image_path.suffix.lower() not in IMAGE_EXTENSIONS:
                        continue

                    label_path = labels_dir / f"{image_path.stem}.txt"
                    if not label_path.exists():
                        continue

                    class_id = _extract_first_class_id(label_path)
                    if class_id is None:
                        continue

                    label_name = _resolve_yolo_label(class_id, yolo_names, fracture_mapping)
                    records.append(
                        {
                            "image_path": str(image_path),
                            "label_name": label_name,
                            "source_dataset": dataset_dir.name,
                            "source_split": split_dir.name,
                            "source_type": "yolo",
                            "source_class_id": class_id,
                        }
                    )
                continue

            class_dirs = [path for path in split_dir.iterdir() if path.is_dir()]
            for class_dir in sorted(class_dirs):
                label_name = _label_from_folder_name(class_dir.name)
                for image_path in sorted(class_dir.iterdir()):
                    if image_path.suffix.lower() not in IMAGE_EXTENSIONS:
                        continue
                    records.append(
                        {
                            "image_path": str(image_path),
                            "label_name": label_name,
                            "source_dataset": dataset_dir.name,
                            "source_split": split_dir.name,
                            "source_type": "folder",
                        }
                    )

    return records


def resolve_task_metadata(records, config=None):
    config = config or {}
    configured_class_names = [normalize_label_name(name) for name in config.get("class_names", [])]
    discovered = sorted({normalize_label_name(record["label_name"]) for record in records})
    non_binary_labels = [label for label in discovered if label not in set(DEFAULT_BINARY_CLASS_NAMES)]

    if non_binary_labels:
        ordered_class_names = [name for name in configured_class_names if name in discovered]
        ordered_class_names.extend(name for name in discovered if name not in ordered_class_names)
        task_type = "multiclass"
    else:
        ordered_class_names = DEFAULT_BINARY_CLASS_NAMES.copy()
        task_type = "binary"

    class_to_idx = {name: index for index, name in enumerate(ordered_class_names)}
    normalized_records = []
    for record in records:
        label_name = normalize_label_name(record["label_name"])
        if task_type == "binary":
            label_name = _binary_label_from_name(label_name)

        updated_record = dict(record)
        updated_record["label_name"] = label_name
        updated_record["label"] = class_to_idx[label_name]
        normalized_records.append(updated_record)

    counts = Counter(record["label_name"] for record in normalized_records)
    return {
        "task_type": task_type,
        "class_names": ordered_class_names,
        "num_classes": len(ordered_class_names),
        "class_to_idx": class_to_idx,
        "class_distribution": dict(counts),
        "records": normalized_records,
    }


def deduplicate_records(records):
    deduplicated = {}
    for record in records:
        image_name = Path(record["image_path"]).name
        logical_name = image_name.split(".rf.")[0]
        existing = deduplicated.get(logical_name)
        if existing is None or record["label_name"] != "non-fracture":
            deduplicated[logical_name] = record
    return list(deduplicated.values())


def save_dataset_metadata(metadata, output_path):
    output = {
        key: value
        for key, value in metadata.items()
        if key != "records"
    }
    with open(output_path, "w", encoding="utf-8") as handle:
        yaml.safe_dump(output, handle, sort_keys=False)


def load_dataset_metadata(metadata_path):
    with open(metadata_path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def load_pickle(path):
    with open(path, "rb") as handle:
        return pickle.load(handle)
