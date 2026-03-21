import csv
import os
import random
import subprocess
import sys
from collections import Counter

import matplotlib

matplotlib.use("Agg")
import numpy as np
import torch
import torch.optim as optim
import yaml
from sklearn.metrics import precision_recall_fscore_support
from torch.nn import functional as F
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from models.model import FractureClassifier
from src.data_loader import get_dataloaders
from src.losses import FocalLoss
from src.pipeline_utils import load_config


class EarlyStopping:
    def __init__(self, patience=5, delta=0.0, path="outputs/models/best_model.pth"):
        self.patience = patience
        self.delta = delta
        self.path = path
        self.counter = 0
        self.best_score = None
        self.early_stop = False

    def __call__(self, metric_value, model):
        score = metric_value
        if self.best_score is None or score > self.best_score + self.delta:
            self.best_score = score
            self.counter = 0
            os.makedirs(os.path.dirname(self.path), exist_ok=True)
            torch.save(model.state_dict(), self.path)
            return

        self.counter += 1
        if self.counter >= self.patience:
            self.early_stop = True


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def ensure_splits_exist():
    splits_path = os.path.join("outputs", "splits", "dataset_splits.pkl")
    if os.path.exists(splits_path):
        return True

    create_script = os.path.join(os.path.dirname(os.path.dirname(__file__)), "create_dataset_splits.py")
    result = subprocess.run([sys.executable, create_script], capture_output=True, text=True)
    if result.returncode != 0:
        print(result.stdout)
        print(result.stderr)
        return False
    return True


def save_model_metadata(config, metadata):
    model_metadata = {
        "model_name": config.get("model", "efficientnet_b0"),
        "task_type": metadata["task_type"],
        "class_names": metadata["class_names"],
        "num_classes": metadata["num_classes"],
        "config_path": config["config_path"],
    }
    with open(os.path.join("outputs", "models", "model_metadata.yaml"), "w", encoding="utf-8") as handle:
        yaml.safe_dump(model_metadata, handle, sort_keys=False)


def mixup_batch(images, labels, num_classes, alpha):
    if alpha <= 0:
        return images, F.one_hot(labels, num_classes=num_classes).float(), labels

    lam = np.random.beta(alpha, alpha)
    batch_size = images.size(0)
    indices = torch.randperm(batch_size, device=images.device)
    mixed_images = lam * images + (1.0 - lam) * images[indices]
    labels_a = F.one_hot(labels, num_classes=num_classes).float()
    labels_b = F.one_hot(labels[indices], num_classes=num_classes).float()
    mixed_targets = lam * labels_a + (1.0 - lam) * labels_b
    return mixed_images, mixed_targets, labels


def compute_metrics(labels, preds, num_classes):
    labels = np.array(labels)
    preds = np.array(preds)
    metric_labels = list(range(num_classes))
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels,
        preds,
        labels=metric_labels,
        zero_division=0,
    )
    per_class_accuracy = []
    for class_index in metric_labels:
        class_mask = labels == class_index
        if class_mask.any():
            per_class_accuracy.append(float((preds[class_mask] == labels[class_mask]).mean()))
        else:
            per_class_accuracy.append(0.0)

    return {
        "accuracy": float((preds == labels).mean()),
        "macro_f1": float(np.mean(f1)),
        "per_class_precision": precision.tolist(),
        "per_class_recall": recall.tolist(),
        "per_class_f1": f1.tolist(),
        "per_class_accuracy": per_class_accuracy,
    }


def run_epoch(model, dataloader, criterion, device, num_classes, optimizer=None, mixup_alpha=0.0, mixup_probability=0.0):
    is_training = optimizer is not None
    model.train(is_training)

    running_loss = 0.0
    all_labels = []
    all_preds = []

    context = torch.enable_grad() if is_training else torch.no_grad()
    with context:
        for images, labels in tqdm(dataloader, desc="Training" if is_training else "Validating"):
            images = images.to(device)
            labels = labels.to(device)
            hard_labels = labels

            if is_training:
                optimizer.zero_grad()
                if mixup_alpha > 0 and np.random.rand() < mixup_probability:
                    images, labels, hard_labels = mixup_batch(images, labels, num_classes, mixup_alpha)

            outputs = model(images)
            loss = criterion(outputs, labels)

            if is_training:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

            running_loss += loss.item() * images.size(0)
            all_labels.extend(hard_labels.cpu().numpy())
            all_preds.extend(torch.argmax(outputs, dim=1).cpu().numpy())

    metrics = compute_metrics(all_labels, all_preds, num_classes)
    metrics["loss"] = running_loss / len(dataloader.dataset)
    return metrics


def print_distribution(label, distribution, class_names):
    print(label)
    for class_index, count in distribution.items():
        class_name = class_names[class_index]
        print(f"  {class_name}: {count}")


def print_per_class_metrics(prefix, metrics, class_names):
    print(f"{prefix} per-class metrics:")
    for class_index, class_name in enumerate(class_names):
        print(
            f"  {class_name}: acc={metrics['per_class_accuracy'][class_index]:.4f}, "
            f"precision={metrics['per_class_precision'][class_index]:.4f}, "
            f"recall={metrics['per_class_recall'][class_index]:.4f}, "
            f"f1={metrics['per_class_f1'][class_index]:.4f}"
        )


def main():
    print("=" * 60)
    print("FractureVision-AI Training Pipeline")
    print("=" * 60)

    config = load_config()
    config.setdefault("focal_gamma", 2.0)
    config.setdefault("label_smoothing", 0.1)
    config.setdefault("mixup_alpha", 0.2)
    config.setdefault("mixup_probability", 0.3)
    config.setdefault("early_stopping_patience", 5)
    set_seed(config.get("random_seed", 42))

    os.makedirs(os.path.join("outputs", "models"), exist_ok=True)
    os.makedirs(os.path.join("outputs", "metrics"), exist_ok=True)
    os.makedirs(os.path.join("outputs", "plots"), exist_ok=True)
    os.makedirs("results", exist_ok=True)

    if not ensure_splits_exist():
        print("ERROR: Could not create dataset splits.")
        return

    train_loader, val_loader, _, metadata = get_dataloaders(config)
    class_names = metadata["class_names"]
    num_classes = metadata["num_classes"]
    class_weights = torch.tensor(
        [metadata["class_weights"][class_index] for class_index in range(num_classes)],
        dtype=torch.float32,
    )

    print_distribution("Class distribution before training:", metadata["train_distribution_indices"], class_names)
    print_distribution("Sampler distribution preview:", metadata["sampler_distribution_indices"], class_names)
    print(
        "Minority classes receiving stronger augmentation:",
        [class_names[index] for index in metadata["minority_class_indices"]],
    )
    print("Normalized class weights:", {class_names[i]: float(class_weights[i]) for i in range(num_classes)})

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = FractureClassifier(
        model_name=config.get("model", "efficientnet_b0"),
        pretrained=config.get("pretrained", True),
        num_classes=num_classes,
    ).to(device)

    criterion = FocalLoss(
        gamma=config["focal_gamma"],
        weight=class_weights.to(device) if config.get("use_class_weights", True) else None,
        label_smoothing=config.get("label_smoothing", 0.0),
    )
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config.get("learning_rate", 1e-3),
        weight_decay=config.get("weight_decay", 1e-4),
    )
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        patience=config.get("scheduler_patience", 3),
        factor=config.get("scheduler_factor", 0.5),
        min_lr=config.get("scheduler_min_lr", 1e-6),
    )
    early_stopping = EarlyStopping(
        patience=config.get("early_stopping_patience", 5),
        delta=config.get("early_stopping_delta", 0.0),
        path=os.path.join("outputs", "models", "best_model.pth"),
    )

    metrics_path = os.path.join("outputs", "metrics", "training_metrics.csv")
    with open(metrics_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            ["epoch", "train_loss", "val_loss", "train_accuracy", "val_accuracy", "train_macro_f1", "val_macro_f1"]
        )

    for epoch in range(config.get("epochs", 50)):
        print(f"\nEpoch {epoch + 1}/{config.get('epochs', 50)}")
        train_metrics = run_epoch(
            model,
            train_loader,
            criterion,
            device,
            num_classes,
            optimizer=optimizer,
            mixup_alpha=config.get("mixup_alpha", 0.0),
            mixup_probability=config.get("mixup_probability", 0.0),
        )
        val_metrics = run_epoch(model, val_loader, criterion, device, num_classes)

        print(
            f"Train loss {train_metrics['loss']:.4f} | acc {train_metrics['accuracy']:.4f} | "
            f"macro F1 {train_metrics['macro_f1']:.4f}"
        )
        print(
            f"Val loss {val_metrics['loss']:.4f} | acc {val_metrics['accuracy']:.4f} | "
            f"macro F1 {val_metrics['macro_f1']:.4f}"
        )
        print_per_class_metrics("Train", train_metrics, class_names)
        print_per_class_metrics("Val", val_metrics, class_names)

        with open(metrics_path, "a", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(
                [
                    epoch + 1,
                    train_metrics["loss"],
                    val_metrics["loss"],
                    train_metrics["accuracy"],
                    val_metrics["accuracy"],
                    train_metrics["macro_f1"],
                    val_metrics["macro_f1"],
                ]
            )

        scheduler.step(val_metrics["macro_f1"])
        early_stopping(val_metrics["macro_f1"], model)
        if early_stopping.early_stop:
            print("Early stopping triggered.")
            break

    best_model_path = os.path.join("outputs", "models", "best_model.pth")
    if os.path.exists(best_model_path):
        model.load_state_dict(torch.load(best_model_path, map_location=device))
    torch.save(model.state_dict(), os.path.join("outputs", "models", "final_model.pth"))
    save_model_metadata(config, metadata)

    evaluate_script = os.path.join(os.path.dirname(__file__), "evaluate.py")
    subprocess.run([sys.executable, evaluate_script], check=False)
    print("Training complete.")


if __name__ == "__main__":
    main()
