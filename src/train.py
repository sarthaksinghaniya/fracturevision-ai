import csv
import os
import random
import subprocess
import sys

import matplotlib

matplotlib.use("Agg")
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
import yaml
from sklearn.metrics import precision_recall_fscore_support
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from models.model import FractureClassifier, combine_head_probabilities
from src.data_loader import TrainingDataModule
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
        if self.best_score is None or metric_value > self.best_score + self.delta:
            self.best_score = metric_value
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
        "model_name": config.get("model", "efficientnet_b3"),
        "task_type": metadata["task_type"],
        "class_names": metadata["class_names"],
        "num_classes": metadata["num_classes"],
        "config_path": config["config_path"],
        "dual_head": True,
    }
    with open(os.path.join("outputs", "models", "model_metadata.yaml"), "w", encoding="utf-8") as handle:
        yaml.safe_dump(model_metadata, handle, sort_keys=False)


def mixup_batch(images, labels, num_classes, alpha, non_fracture_index):
    if alpha <= 0:
        binary_targets = (labels != non_fracture_index).float()
        return images, F.one_hot(labels, num_classes=num_classes).float(), binary_targets, labels

    lam = np.random.beta(alpha, alpha)
    indices = torch.randperm(images.size(0), device=images.device)
    mixed_images = lam * images + (1.0 - lam) * images[indices]
    mixed_labels = lam * F.one_hot(labels, num_classes=num_classes).float() + (1.0 - lam) * F.one_hot(
        labels[indices], num_classes=num_classes
    ).float()
    binary_targets = lam * (labels != non_fracture_index).float() + (1.0 - lam) * (labels[indices] != non_fracture_index).float()
    return mixed_images, mixed_labels, binary_targets, labels


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
        mask = labels == class_index
        per_class_accuracy.append(float((preds[mask] == labels[mask]).mean()) if mask.any() else 0.0)

    return {
        "accuracy": float((preds == labels).mean()),
        "macro_f1": float(np.mean(f1)),
        "per_class_precision": precision.tolist(),
        "per_class_recall": recall.tolist(),
        "per_class_f1": f1.tolist(),
        "per_class_accuracy": per_class_accuracy,
    }


def compute_total_loss(loss_fn, multitask_outputs, labels, label_targets, binary_targets):
    multi_loss = loss_fn(multitask_outputs["multi_logits"], label_targets)
    binary_loss = F.binary_cross_entropy_with_logits(multitask_outputs["binary_logits"], binary_targets.float())
    total_loss = multi_loss + 0.2 * binary_loss
    return total_loss, multi_loss, binary_loss


def run_epoch(
    model,
    dataloader,
    loss_fn,
    device,
    num_classes,
    non_fracture_index,
    optimizer=None,
    mixup_alpha=0.0,
    mixup_probability=0.0,
):
    is_training = optimizer is not None
    model.train(is_training)

    running_total_loss = 0.0
    running_multi_loss = 0.0
    running_binary_loss = 0.0
    all_labels = []
    all_preds = []
    hard_example_true = []
    hard_example_pred = []
    hard_example_indices = []

    context = torch.enable_grad() if is_training else torch.no_grad()
    with context:
        for images, labels, indices in tqdm(dataloader, desc="Training" if is_training else "Validating"):
            images = images.to(device)
            labels = labels.to(device)
            hard_labels = labels
            binary_targets = (labels != non_fracture_index).float()

            if is_training:
                optimizer.zero_grad()
                label_targets = labels
                if mixup_alpha > 0 and np.random.rand() < mixup_probability:
                    images, label_targets, binary_targets, hard_labels = mixup_batch(
                        images,
                        labels,
                        num_classes,
                        mixup_alpha,
                        non_fracture_index,
                    )
                else:
                    label_targets = labels
            else:
                label_targets = labels

            outputs = model.forward_multitask(images)
            total_loss, multi_loss, binary_loss = compute_total_loss(
                loss_fn,
                outputs,
                labels,
                label_targets,
                binary_targets,
            )

            if is_training:
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

            adjusted_probs = combine_head_probabilities(
                outputs["multi_logits"],
                outputs["binary_logits"],
                non_fracture_index,
            )
            preds = torch.argmax(adjusted_probs, dim=1)

            running_total_loss += total_loss.item() * images.size(0)
            running_multi_loss += multi_loss.item() * images.size(0)
            running_binary_loss += binary_loss.item() * images.size(0)
            all_labels.extend(hard_labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            if is_training:
                mismatches = preds != hard_labels
                if mismatches.any():
                    hard_example_indices.extend(indices[mismatches].cpu().numpy())
                    hard_example_true.extend(hard_labels[mismatches].cpu().numpy())
                    hard_example_pred.extend(preds[mismatches].cpu().numpy())

    metrics = compute_metrics(all_labels, all_preds, num_classes)
    metrics["loss"] = running_total_loss / len(dataloader.dataset)
    metrics["multi_loss"] = running_multi_loss / len(dataloader.dataset)
    metrics["binary_loss"] = running_binary_loss / len(dataloader.dataset)
    metrics["hard_example_indices"] = hard_example_indices
    metrics["hard_example_true"] = hard_example_true
    metrics["hard_example_pred"] = hard_example_pred
    return metrics


def print_distribution(title, distribution, class_names):
    print(title)
    for class_index, count in distribution.items():
        print(f"  {class_names[class_index]}: {count}")


def print_per_class_metrics(prefix, metrics, class_names):
    print(f"{prefix} per-class metrics:")
    for class_index, class_name in enumerate(class_names):
        print(
            f"  {class_name}: acc={metrics['per_class_accuracy'][class_index]:.4f}, "
            f"precision={metrics['per_class_precision'][class_index]:.4f}, "
            f"recall={metrics['per_class_recall'][class_index]:.4f}, "
            f"f1={metrics['per_class_f1'][class_index]:.4f}"
        )


def warn_low_recall(metrics, class_names, threshold=0.05):
    for class_index, recall in enumerate(metrics["per_class_recall"]):
        if recall < threshold:
            print(f"WARNING: {class_names[class_index]} recall is critically low at {recall:.4f}")


def main():
    print("=" * 60)
    print("FractureVision-AI Training Pipeline")
    print("=" * 60)

    config = load_config()
    config.setdefault("model", "efficientnet_b3")
    config.setdefault("epochs", 70)
    config.setdefault("focal_gamma", 1.5)
    config.setdefault("label_smoothing", 0.05)
    config.setdefault("mixup_alpha", 0.2)
    config.setdefault("mixup_probability", 0.3)
    config.setdefault("early_stopping_patience", 8)
    config.setdefault("classifier_dropout", 0.3)
    config.setdefault("greenstick_class_boost", 2.0)
    config.setdefault("greenstick_sampler_boost", 3.5)
    config.setdefault("hard_example_boost", 2.0)
    set_seed(config.get("random_seed", 42))

    os.makedirs(os.path.join("outputs", "models"), exist_ok=True)
    os.makedirs(os.path.join("outputs", "metrics"), exist_ok=True)
    os.makedirs(os.path.join("outputs", "plots"), exist_ok=True)

    if not ensure_splits_exist():
        print("ERROR: Could not create dataset splits.")
        return

    data_module = TrainingDataModule(config)
    val_loader, _ = data_module.get_eval_loaders()
    class_names = data_module.class_names
    num_classes = data_module.metadata["num_classes"]
    non_fracture_index = data_module.non_fracture_index if data_module.non_fracture_index is not None else 0

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    class_weights = torch.tensor(
        [data_module.class_weights[class_index] for class_index in range(num_classes)],
        dtype=torch.float32,
    )
    if data_module.greenstick_index is not None:
        class_weights[data_module.greenstick_index] *= float(config.get("greenstick_class_boost", 1.75))

    model = FractureClassifier(
        model_name=config.get("model", "efficientnet_b3"),
        pretrained=config.get("pretrained", True),
        num_classes=num_classes,
        dropout=config.get("classifier_dropout", 0.45),
    ).to(device)

    focal_loss = FocalLoss(
        gamma=config["focal_gamma"],
        weight=class_weights.to(device) if config.get("use_class_weights", True) else None,
        label_smoothing=config.get("label_smoothing", 0.0),
    )
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config.get("learning_rate", 1e-3),
        weight_decay=config.get("weight_decay", 1e-4),
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=config.get("epochs", 70),
        eta_min=config.get("scheduler_min_lr", 1e-6),
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
            [
                "epoch",
                "train_loss",
                "train_multi_loss",
                "train_binary_loss",
                "val_loss",
                "val_multi_loss",
                "val_binary_loss",
                "train_accuracy",
                "val_accuracy",
                "train_macro_f1",
                "val_macro_f1",
            ]
        )

    for epoch in range(config.get("epochs", 70)):
        train_loader, runtime_metadata = data_module.create_train_loader()
        print(f"\nEpoch {epoch + 1}/{config.get('epochs', 70)}")
        print_distribution("Class distribution before training:", runtime_metadata["train_distribution_indices"], class_names)
        print_distribution("Sampler distribution preview:", runtime_metadata["sampler_distribution_indices"], class_names)
        print(f"Sample weight summary: {runtime_metadata['sample_weights_summary']}")
        print(f"Active hard examples: {runtime_metadata['hard_example_count']}")
        print(
            "Label integrity:",
            {
                "greenstick_train_count": runtime_metadata["label_integrity"]["greenstick_train_count"],
                "issue_count": runtime_metadata["label_integrity"]["issue_count"],
            },
        )
        if runtime_metadata["label_integrity"]["issues"]:
            for issue in runtime_metadata["label_integrity"]["issues"]:
                print(f"  Integrity issue: {issue}")

        train_metrics = run_epoch(
            model,
            train_loader,
            focal_loss,
            device,
            num_classes,
            non_fracture_index,
            optimizer=optimizer,
            mixup_alpha=config.get("mixup_alpha", 0.0),
            mixup_probability=config.get("mixup_probability", 0.0),
        )
        val_metrics = run_epoch(
            model,
            val_loader,
            focal_loss,
            device,
            num_classes,
            non_fracture_index,
        )

        print(
            f"Train loss {train_metrics['loss']:.4f} | multi {train_metrics['multi_loss']:.4f} | "
            f"binary {train_metrics['binary_loss']:.4f} | macro F1 {train_metrics['macro_f1']:.4f}"
        )
        print(
            f"Val loss {val_metrics['loss']:.4f} | multi {val_metrics['multi_loss']:.4f} | "
            f"binary {val_metrics['binary_loss']:.4f} | macro F1 {val_metrics['macro_f1']:.4f}"
        )
        print_per_class_metrics("Train", train_metrics, class_names)
        print_per_class_metrics("Val", val_metrics, class_names)
        warn_low_recall(train_metrics, class_names)
        warn_low_recall(val_metrics, class_names)

        with open(metrics_path, "a", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(
                [
                    epoch + 1,
                    train_metrics["loss"],
                    train_metrics["multi_loss"],
                    train_metrics["binary_loss"],
                    val_metrics["loss"],
                    val_metrics["multi_loss"],
                    val_metrics["binary_loss"],
                    train_metrics["accuracy"],
                    val_metrics["accuracy"],
                    train_metrics["macro_f1"],
                    val_metrics["macro_f1"],
                ]
            )

        data_module.update_hard_examples(
            train_metrics["hard_example_indices"],
            train_metrics["hard_example_true"],
            train_metrics["hard_example_pred"],
        )

        scheduler.step()
        early_stopping(val_metrics["macro_f1"], model)
        if early_stopping.early_stop:
            print("Early stopping triggered.")
            break

    best_model_path = os.path.join("outputs", "models", "best_model.pth")
    if os.path.exists(best_model_path):
        model.load_state_dict(torch.load(best_model_path, map_location=device))
    torch.save(model.state_dict(), os.path.join("outputs", "models", "final_model.pth"))
    save_model_metadata(config, data_module.metadata)

    evaluate_script = os.path.join(os.path.dirname(__file__), "evaluate.py")
    subprocess.run([sys.executable, evaluate_script], check=False)
    print("Training complete.")


if __name__ == "__main__":
    main()
