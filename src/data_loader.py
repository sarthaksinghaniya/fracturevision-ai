import os
from collections import Counter

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import transforms

from src.pipeline_utils import load_dataset_metadata, load_pickle, normalize_label_name


class AddGaussianNoise:
    def __init__(self, std=0.03):
        self.std = std

    def __call__(self, tensor):
        noise = torch.randn_like(tensor) * self.std
        return torch.clamp(tensor + noise, 0.0, 1.0)


class FractureDataset(Dataset):
    def __init__(self, items, mean, std, image_size, class_names, hard_example_indices=None, is_train=False):
        self.items = items
        self.is_train = is_train
        self.class_names = class_names
        self.hard_example_indices = set(hard_example_indices or [])
        self.greenstick_index = self._find_class_index("greenstick")

        self.base_transform = transforms.Compose(
            [
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std),
            ]
        )
        self.train_transform = transforms.Compose(
            [
                transforms.RandomResizedCrop(image_size, scale=(0.85, 1.0)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(12),
                transforms.ColorJitter(brightness=0.15, contrast=0.15),
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std),
            ]
        )
        self.greenstick_transform = transforms.Compose(
            [
                transforms.RandomResizedCrop(image_size, scale=(0.88, 1.0), ratio=(0.95, 1.05)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(10),
                transforms.ColorJitter(brightness=0.08, contrast=0.08),
                transforms.ToTensor(),
                AddGaussianNoise(std=0.015),
                transforms.Normalize(mean=mean, std=std),
            ]
        )
        self.hard_example_transform = transforms.Compose(
            [
                transforms.RandomResizedCrop(image_size, scale=(0.75, 1.0)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(15),
                transforms.ColorJitter(brightness=0.2, contrast=0.2),
                transforms.ToTensor(),
                AddGaussianNoise(std=0.02),
                transforms.Normalize(mean=mean, std=std),
            ]
        )

    def _find_class_index(self, class_name):
        target = normalize_label_name(class_name)
        for index, candidate in enumerate(self.class_names):
            if normalize_label_name(candidate) == target:
                return index
        return None

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        image_path, label = self.items[idx]
        image = Image.open(image_path).convert("RGB")

        if not self.is_train:
            image = self.base_transform(image)
        elif self.greenstick_index is not None and label == self.greenstick_index:
            image = self.greenstick_transform(image)
        elif idx in self.hard_example_indices:
            image = self.hard_example_transform(image)
        else:
            image = self.train_transform(image)

        return image, torch.tensor(label, dtype=torch.long), torch.tensor(idx, dtype=torch.long)


class TrainingDataModule:
    def __init__(self, config):
        self.config = config
        self.splits, self.norm_stats, self.metadata, self.class_weights = self.load_split_artifacts()
        self.class_names = self.metadata["class_names"]
        self.train_labels = [label for _, label in self.splits["train"]]
        self.greenstick_index = self._find_class_index("greenstick")
        self.non_fracture_index = self._find_class_index("non-fracture")
        self.spiral_index = self._find_class_index("spiral")
        self.hard_example_boosts = np.ones(len(self.splits["train"]), dtype=np.float64)
        self.greenstick_boost = float(config.get("greenstick_class_boost", 1.75))
        self.non_fracture_confusion_boost = float(config.get("non_fracture_confusion_boost", 1.5))
        self.hard_example_boost = float(config.get("hard_example_boost", 2.0))

        self.mean = self.norm_stats["mean"].tolist() if hasattr(self.norm_stats["mean"], "tolist") else list(self.norm_stats["mean"])
        self.std = self.norm_stats["std"].tolist() if hasattr(self.norm_stats["std"], "tolist") else list(self.norm_stats["std"])

        self.val_dataset = FractureDataset(
            self.splits["val"],
            mean=self.mean,
            std=self.std,
            image_size=config["image_size"],
            class_names=self.class_names,
            is_train=False,
        )
        self.test_dataset = FractureDataset(
            self.splits["test"],
            mean=self.mean,
            std=self.std,
            image_size=config["image_size"],
            class_names=self.class_names,
            is_train=False,
        )

    @staticmethod
    def load_split_artifacts(base_dir="outputs/splits"):
        splits = load_pickle(os.path.join(base_dir, "dataset_splits.pkl"))
        norm_stats = load_pickle(os.path.join(base_dir, "normalization_stats.pkl"))
        metadata = load_dataset_metadata(os.path.join(base_dir, "dataset_metadata.yaml"))
        class_weights = load_pickle(os.path.join(base_dir, "class_weights.pkl"))
        return splits, norm_stats, metadata, class_weights

    def _find_class_index(self, class_name):
        target = normalize_label_name(class_name)
        for index, candidate in enumerate(self.class_names):
            if normalize_label_name(candidate) == target:
                return index
        return None

    def _compute_sample_weights(self):
        label_counts = Counter(self.train_labels)
        weights = []
        for index, (_, label) in enumerate(self.splits["train"]):
            weight = 1.0
            if self.non_fracture_index is not None and label == self.non_fracture_index:
                weight *= self.non_fracture_confusion_boost
            if self.greenstick_index is not None and label == self.greenstick_index:
                weight *= float(self.config.get("greenstick_sampler_boost", 2.5))
            else:
                class_count = label_counts[label]
                max_count = max(label_counts.values())
                weight *= max(1.0, np.sqrt(max_count / max(class_count, 1)))
            weight *= self.hard_example_boosts[index]
            weights.append(weight)
        return np.array(weights, dtype=np.float64)

    def verify_label_integrity(self):
        issues = []
        class_count = len(self.class_names)
        greenstick_count = 0
        for image_path, label in self.splits["train"]:
            if not os.path.exists(image_path):
                issues.append(f"Missing file: {image_path}")
            if label < 0 or label >= class_count:
                issues.append(f"Out-of-range label {label} for {image_path}")
            if self.greenstick_index is not None and label == self.greenstick_index:
                greenstick_count += 1
        return {
            "greenstick_train_count": greenstick_count,
            "issue_count": len(issues),
            "issues": issues[:10],
        }

    def _build_train_dataset(self):
        hard_indices = set(np.where(self.hard_example_boosts > 1.0)[0].tolist())
        return FractureDataset(
            self.splits["train"],
            mean=self.mean,
            std=self.std,
            image_size=self.config["image_size"],
            class_names=self.class_names,
            hard_example_indices=hard_indices,
            is_train=True,
        )

    def create_train_loader(self):
        sample_weights = self._compute_sample_weights()
        generator = torch.Generator().manual_seed(int(self.config.get("random_seed", 42)))
        sampler = WeightedRandomSampler(
            weights=torch.tensor(sample_weights, dtype=torch.double),
            num_samples=len(self.splits["train"]),
            replacement=True,
            generator=generator,
        )

        preview_generator = torch.Generator().manual_seed(int(self.config.get("random_seed", 42)))
        preview_sampler = WeightedRandomSampler(
            weights=torch.tensor(sample_weights, dtype=torch.double),
            num_samples=len(self.splits["train"]),
            replacement=True,
            generator=preview_generator,
        )
        sampled_indices = list(preview_sampler)
        sampled_labels = [self.train_labels[index] for index in sampled_indices]
        sampler_distribution = dict(sorted(Counter(sampled_labels).items()))

        train_dataset = self._build_train_dataset()
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config["batch_size"],
            sampler=sampler,
            shuffle=False,
            num_workers=int(self.config.get("num_workers", 0)),
        )

        runtime_metadata = dict(self.metadata)
        runtime_metadata["class_weights"] = {int(key): float(value) for key, value in self.class_weights.items()}
        runtime_metadata["train_distribution_indices"] = dict(sorted(Counter(self.train_labels).items()))
        runtime_metadata["sampler_distribution_indices"] = sampler_distribution
        runtime_metadata["sample_weights_summary"] = {
            "min": float(sample_weights.min()),
            "max": float(sample_weights.max()),
            "mean": float(sample_weights.mean()),
        }
        runtime_metadata["hard_example_count"] = int(np.sum(self.hard_example_boosts > 1.0))
        runtime_metadata["greenstick_index"] = self.greenstick_index
        runtime_metadata["non_fracture_index"] = self.non_fracture_index
        runtime_metadata["spiral_index"] = self.spiral_index
        runtime_metadata["label_integrity"] = self.verify_label_integrity()
        return train_loader, runtime_metadata

    def get_eval_loaders(self):
        val_loader = DataLoader(
            self.val_dataset,
            batch_size=self.config["batch_size"],
            shuffle=False,
            num_workers=int(self.config.get("num_workers", 0)),
        )
        test_loader = DataLoader(
            self.test_dataset,
            batch_size=self.config["batch_size"],
            shuffle=False,
            num_workers=int(self.config.get("num_workers", 0)),
        )
        return val_loader, test_loader

    def update_hard_examples(self, indices, true_labels, pred_labels):
        self.hard_example_boosts = np.maximum(self.hard_example_boosts * 0.9, 1.0)
        for sample_index, true_label, pred_label in zip(indices, true_labels, pred_labels):
            should_boost = False
            if self.non_fracture_index is not None and true_label == self.non_fracture_index and pred_label != self.non_fracture_index:
                should_boost = True
            if self.greenstick_index is not None and true_label == self.greenstick_index and pred_label != self.greenstick_index:
                should_boost = True
            if should_boost:
                self.hard_example_boosts[int(sample_index)] *= self.hard_example_boost


def get_dataloaders(config):
    data_module = TrainingDataModule(config)
    train_loader, metadata = data_module.create_train_loader()
    val_loader, test_loader = data_module.get_eval_loaders()
    return train_loader, val_loader, test_loader, metadata
