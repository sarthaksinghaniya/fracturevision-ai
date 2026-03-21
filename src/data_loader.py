import os
from collections import Counter

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import transforms

from src.pipeline_utils import load_dataset_metadata, load_pickle


class FractureDataset(Dataset):
    def __init__(self, items, mean, std, image_size, minority_classes=None, is_train=False):
        self.items = items
        self.is_train = is_train
        self.minority_classes = set(minority_classes or [])
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
        self.minority_transform = transforms.Compose(
            [
                transforms.RandomResizedCrop(image_size, scale=(0.7, 1.0)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(15),
                transforms.ColorJitter(brightness=0.25, contrast=0.25),
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std),
            ]
        )

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        image_path, label = self.items[idx]
        image = Image.open(image_path).convert("RGB")

        if not self.is_train:
            image = self.base_transform(image)
        elif label in self.minority_classes:
            image = self.minority_transform(image)
        else:
            image = self.train_transform(image)

        return image, torch.tensor(label, dtype=torch.long)


def load_split_artifacts(base_dir="outputs/splits"):
    splits = load_pickle(os.path.join(base_dir, "dataset_splits.pkl"))
    norm_stats = load_pickle(os.path.join(base_dir, "normalization_stats.pkl"))
    metadata = load_dataset_metadata(os.path.join(base_dir, "dataset_metadata.yaml"))
    class_weights = load_pickle(os.path.join(base_dir, "class_weights.pkl"))
    return splits, norm_stats, metadata, class_weights


def _identify_minority_classes(train_labels):
    counts = Counter(train_labels)
    if not counts:
        return set()
    median_count = float(np.median(list(counts.values())))
    return {label for label, count in counts.items() if count <= median_count}


def _build_weighted_sampler(train_items, class_weights, random_seed):
    labels = [label for _, label in train_items]
    sample_weights = torch.tensor([class_weights[label] for label in labels], dtype=torch.double)
    generator = torch.Generator().manual_seed(random_seed)
    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(train_items),
        replacement=True,
        generator=generator,
    )

    preview_generator = torch.Generator().manual_seed(random_seed)
    preview_sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(train_items),
        replacement=True,
        generator=preview_generator,
    )
    sampled_indices = list(preview_sampler)
    sampled_labels = [labels[index] for index in sampled_indices]
    sampler_distribution = dict(sorted(Counter(sampled_labels).items()))
    return sampler, sampler_distribution


def get_dataloaders(config):
    splits, norm_stats, metadata, class_weights = load_split_artifacts()
    train_labels = [label for _, label in splits["train"]]
    minority_classes = _identify_minority_classes(train_labels)
    sampler, sampler_distribution = _build_weighted_sampler(
        splits["train"],
        class_weights,
        random_seed=int(config.get("random_seed", 42)),
    )

    mean = norm_stats["mean"].tolist() if hasattr(norm_stats["mean"], "tolist") else list(norm_stats["mean"])
    std = norm_stats["std"].tolist() if hasattr(norm_stats["std"], "tolist") else list(norm_stats["std"])

    train_dataset = FractureDataset(
        splits["train"],
        mean=mean,
        std=std,
        image_size=config["image_size"],
        minority_classes=minority_classes,
        is_train=True,
    )
    val_dataset = FractureDataset(
        splits["val"],
        mean=mean,
        std=std,
        image_size=config["image_size"],
        is_train=False,
    )
    test_dataset = FractureDataset(
        splits["test"],
        mean=mean,
        std=std,
        image_size=config["image_size"],
        is_train=False,
    )

    num_workers = int(config.get("num_workers", 0))
    train_loader = DataLoader(
        train_dataset,
        batch_size=config["batch_size"],
        sampler=sampler,
        shuffle=False,
        num_workers=num_workers,
    )
    val_loader = DataLoader(val_dataset, batch_size=config["batch_size"], shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=config["batch_size"], shuffle=False, num_workers=num_workers)

    runtime_metadata = dict(metadata)
    runtime_metadata["class_weights"] = {int(key): float(value) for key, value in class_weights.items()}
    runtime_metadata["train_distribution_indices"] = dict(sorted(Counter(train_labels).items()))
    runtime_metadata["minority_class_indices"] = sorted(minority_classes)
    runtime_metadata["sampler_distribution_indices"] = sampler_distribution
    return train_loader, val_loader, test_loader, runtime_metadata
