import json
from pathlib import Path

import torch
from PIL import Image
from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torchvision import transforms


def _normalize_label(label):
    return str(label).strip().lower().replace("_", "-").replace(" ", "-")


def should_retrain(feedback_count, threshold=20):
    return int(feedback_count) >= int(threshold)


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


def retrain(model, dataset, epochs=3, learning_rate=1e-5, batch_size=8, device=None, min_feedback_samples=20):
    """Lightweight retraining loop for feedback data."""
    dataset_size = len(dataset)
    if dataset_size == 0:
        print("[retrain] No feedback samples found. Skipping retraining.")
        return model
    if not should_retrain(dataset_size, threshold=min_feedback_samples):
        print(
            f"[retrain] Not enough feedback samples ({dataset_size}/{min_feedback_samples}). "
            "Skipping retraining."
        )
        return model

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    effective_batch_size = max(1, min(batch_size, dataset_size))
    dataloader = DataLoader(dataset, batch_size=effective_batch_size, shuffle=True)
    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=learning_rate)
    epoch_count = max(2, min(int(epochs), 3))

    model = model.to(device)
    model.train()
    print(
        f"[retrain] Starting retraining | samples={dataset_size}, "
        f"batch_size={effective_batch_size}, epochs={epoch_count}, lr={learning_rate}"
    )

    for epoch in range(epoch_count):
        running_loss = 0.0
        seen = 0
        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.to(device)
            if labels.ndim > 1:
                labels = labels.squeeze(-1)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            batch_len = images.size(0)
            running_loss += loss.item() * batch_len
            seen += batch_len

        avg_loss = running_loss / max(1, seen)
        print(f"[retrain] Epoch {epoch + 1}/{epoch_count} | loss={avg_loss:.4f}")

    output_path = Path(__file__).resolve().parents[1] / "models" / "model_updated.pth"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), output_path)
    print(f"[retrain] Saved updated model to: {output_path}")
    return model
