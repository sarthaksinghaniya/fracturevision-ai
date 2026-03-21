import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from sklearn.metrics import classification_report, confusion_matrix, f1_score

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from models.model import FractureClassifier, combine_head_probabilities
from src.data_loader import TrainingDataModule
from src.pipeline_utils import load_config, normalize_label_name


def find_class_index(class_names, target_name):
    target = normalize_label_name(target_name)
    for index, class_name in enumerate(class_names):
        if normalize_label_name(class_name) == target:
            return index
    return None


def evaluate_model(model, dataloader, device, non_fracture_index):
    model.eval()
    all_labels = []
    all_preds = []
    all_probs = []

    with torch.no_grad():
        for images, labels, _indices in dataloader:
            images = images.to(device)
            outputs = model.forward_multitask(images)
            probs = combine_head_probabilities(outputs["multi_logits"], outputs["binary_logits"], non_fracture_index)
            preds = torch.argmax(probs, dim=1)

            all_labels.extend(labels.numpy())
            all_preds.extend(preds.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    return np.array(all_labels), np.array(all_preds), np.array(all_probs)


def plot_confusion_matrix(cm, class_names, save_path):
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
    plt.title("Confusion Matrix")
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def load_previous_summary(results_path):
    if not os.path.exists(results_path):
        return None
    try:
        return pd.read_csv(results_path).iloc[0].to_dict()
    except Exception:
        return None


def main():
    config = load_config()
    os.makedirs(os.path.join("outputs", "plots"), exist_ok=True)
    os.makedirs(os.path.join("outputs", "metrics"), exist_ok=True)

    results_path = os.path.join("outputs", "metrics", "final_results.csv")
    previous_summary = load_previous_summary(results_path)

    data_module = TrainingDataModule(config)
    _, test_loader = data_module.get_eval_loaders()
    class_names = data_module.class_names
    non_fracture_index = data_module.non_fracture_index if data_module.non_fracture_index is not None else 0

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = FractureClassifier(
        model_name=config.get("model", "efficientnet_b3"),
        pretrained=False,
        num_classes=data_module.metadata["num_classes"],
        dropout=config.get("classifier_dropout", 0.45),
    )
    model.load_state_dict(torch.load(os.path.join("outputs", "models", "best_model.pth"), map_location=device))
    model.to(device)

    true_labels, preds, probs = evaluate_model(model, test_loader, device, non_fracture_index)
    labels = list(range(data_module.metadata["num_classes"]))

    report = classification_report(
        true_labels,
        preds,
        labels=labels,
        target_names=class_names,
        output_dict=True,
        zero_division=0,
    )
    report_df = pd.DataFrame(report).transpose()
    report_df.to_csv(os.path.join("outputs", "metrics", "classification_report.csv"))

    summary = {
        "accuracy": report["accuracy"],
        "macro_precision": report["macro avg"]["precision"],
        "macro_recall": report["macro avg"]["recall"],
        "macro_f1": report["macro avg"]["f1-score"],
        "weighted_f1": report["weighted avg"]["f1-score"],
    }
    pd.DataFrame([summary]).to_csv(results_path, index=False)

    cm = confusion_matrix(true_labels, preds, labels=labels)
    plot_confusion_matrix(cm, class_names, os.path.join("outputs", "plots", "confusion_matrix.png"))

    per_class_rows = report_df.loc[class_names, ["precision", "recall", "f1-score", "support"]]
    worst_class = per_class_rows["f1-score"].idxmin()

    greenstick_index = find_class_index(class_names, "greenstick")
    spiral_index = find_class_index(class_names, "spiral")

    print("Evaluation Results:")
    for key, value in summary.items():
        print(f"{key}: {value:.4f}")
    print("Per-class metrics:")
    for class_name in class_names:
        row = per_class_rows.loc[class_name]
        print(
            f"  {class_name}: precision={row['precision']:.4f}, "
            f"recall={row['recall']:.4f}, f1={row['f1-score']:.4f}, support={int(row['support'])}"
        )
    print(f"Worst performing class by F1: {worst_class} ({per_class_rows.loc[worst_class, 'f1-score']:.4f})")

    if greenstick_index is not None:
        print(f"Confusion non-fracture -> greenstick: {int(cm[non_fracture_index, greenstick_index])}")
        print(f"Confusion greenstick -> non-fracture: {int(cm[greenstick_index, non_fracture_index])}")
    if spiral_index is not None:
        print(f"Confusion non-fracture -> spiral: {int(cm[non_fracture_index, spiral_index])}")
        print(f"Confusion spiral -> non-fracture: {int(cm[spiral_index, non_fracture_index])}")

    if previous_summary is not None:
        print("Comparison vs previous run:")
        for metric_name in ["accuracy", "macro_precision", "macro_recall", "macro_f1", "weighted_f1"]:
            delta = summary[metric_name] - float(previous_summary.get(metric_name, 0.0))
            print(f"  {metric_name}: {delta:+.4f}")

    if data_module.metadata["num_classes"] == 2:
        fracture_index = 1 - non_fracture_index
        binary_probs = probs[:, fracture_index]
        binary_results = pd.DataFrame(
            [
                {
                    "macro_f1": f1_score(true_labels, preds, average="macro"),
                    "mean_fracture_probability": float(np.mean(binary_probs)),
                }
            ]
        )
        binary_results.to_csv(os.path.join("outputs", "metrics", "binary_summary.csv"), index=False)

    print("Per-class metrics saved to outputs/metrics/classification_report.csv")
    print("Confusion matrix saved to outputs/plots/confusion_matrix.png")


if __name__ == "__main__":
    main()
