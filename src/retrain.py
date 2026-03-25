import json
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


def _normalize_label(label):
    return str(label).strip().lower().replace("_", "-").replace(" ", "-")


class FeedbackDataset(Dataset):
    """Dataset for retraining from user feedback records."""

    def __init__(
        self,
        feedback_path="feedback/feedback_data.json",
        class_to_idx=None,
        class_names=None,
        transform=None,
        image_size=224,
        strict=False,
    ):
        self.feedback_path = Path(feedback_path)
        self.project_root = Path(__file__).resolve().parents[1]
        self.strict = strict
        self.transform = transform or transforms.Compose(
            [
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
            ]
        )

        records = self._load_feedback_records()
        self.class_to_idx = self._build_class_index(records, class_to_idx=class_to_idx, class_names=class_names)
        self.samples = self._build_samples(records)

    def _load_feedback_records(self):
        path = self.feedback_path
        if not path.is_absolute():
            path = self.project_root / path

        if not path.exists():
            return []

        with path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
        return data if isinstance(data, list) else []

    def _build_class_index(self, records, class_to_idx=None, class_names=None):
        if class_to_idx is not None:
            return {_normalize_label(name): int(index) for name, index in class_to_idx.items()}

        if class_names is not None:
            return {_normalize_label(name): index for index, name in enumerate(class_names)}

        labels = sorted({_normalize_label(record.get("correct", "")) for record in records if record.get("correct")})
        return {label: index for index, label in enumerate(labels)}

    def _resolve_image_path(self, image_value):
        raw_path = Path(str(image_value))
        candidates = []
        if raw_path.is_absolute():
            candidates.append(raw_path)
        else:
            candidates.append(self.project_root / raw_path)
            candidates.append(self.project_root / "feedback" / raw_path.name)

        for candidate in candidates:
            if candidate.exists():
                return candidate
        return None

    def _build_samples(self, records):
        samples = []
        for record in records:
            image_value = record.get("image")
            correct_label = record.get("correct")
            if not image_value or correct_label is None:
                if self.strict:
                    raise ValueError(f"Invalid feedback entry: {record}")
                continue

            normalized_label = _normalize_label(correct_label)
            if normalized_label not in self.class_to_idx:
                if self.strict:
                    raise ValueError(f"Unknown label '{correct_label}' in feedback entry: {record}")
                continue

            image_path = self._resolve_image_path(image_value)
            if image_path is None:
                if self.strict:
                    raise FileNotFoundError(f"Image not found for feedback entry: {record}")
                continue

            samples.append((image_path, self.class_to_idx[normalized_label]))
        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        image_path, label = self.samples[idx]
        image = Image.open(image_path).convert("RGB")
        image_tensor = self.transform(image)
        return image_tensor, torch.tensor(label, dtype=torch.long)
