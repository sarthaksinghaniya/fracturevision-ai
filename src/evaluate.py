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

from models.model import FractureClassifier
from src.data_loader import get_dataloaders
from src.pipeline_utils import load_config


def evaluate_model(model, dataloader, device):
    model.eval()
    all_labels = []
    all_preds = []
    all_probs = []

    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)
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


def main():
    config = load_config()
    os.makedirs(os.path.join("outputs", "plots"), exist_ok=True)
    os.makedirs(os.path.join("outputs", "metrics"), exist_ok=True)

    _, _, test_loader, metadata = get_dataloaders(config)
    class_names = metadata["class_names"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = FractureClassifier(
        model_name=config.get("model", "efficientnet_b0"),
        pretrained=False,
        num_classes=metadata["num_classes"],
    )
    model.load_state_dict(torch.load(os.path.join("outputs", "models", "best_model.pth"), map_location=device))
    model.to(device)

    true_labels, preds, probs = evaluate_model(model, test_loader, device)
    labels = list(range(metadata["num_classes"]))

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
    pd.DataFrame([summary]).to_csv(os.path.join("outputs", "metrics", "final_results.csv"), index=False)

    cm = confusion_matrix(true_labels, preds, labels=labels)
    plot_confusion_matrix(cm, class_names, os.path.join("outputs", "plots", "confusion_matrix.png"))

    per_class_rows = report_df.loc[class_names, ["precision", "recall", "f1-score", "support"]]
    worst_class = per_class_rows["f1-score"].idxmin()

    if metadata["num_classes"] == 2:
        non_fracture_index = class_names.index("non-fracture")
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
    print("Per-class metrics saved to outputs/metrics/classification_report.csv")
    print("Confusion matrix saved to outputs/plots/confusion_matrix.png")


if __name__ == "__main__":
    main()
