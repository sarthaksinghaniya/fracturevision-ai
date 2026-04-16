#!/usr/bin/env python3
"""
Generate Verified Metrics from Actual Evaluation Artifacts

This script computes real metrics from saved evaluation artifacts only.
No fake or hardcoded values - all metrics are traceable to actual model outputs.
"""

import os
import json
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

def load_evaluation_artifacts(metrics_dir="outputs/metrics"):
    """
    Load actual evaluation artifacts if they exist.

    Returns:
        dict: Evaluation data or None if not available
    """
    metrics_path = Path(metrics_dir)

    artifacts = {
        'predictions_exist': False,
        'ground_truth_exist': False,
        'predictions': None,
        'ground_truth': None,
        'class_names': None
    }

    final_results_path = metrics_path / "final_results.csv"
    classification_report_path = metrics_path / "classification_report.csv"

    if final_results_path.exists():
        try:
            artifacts["final_results"] = pd.read_csv(final_results_path)
            artifacts["final_results_exist"] = True
            print(f"✓ Loaded final results from {final_results_path}")
        except Exception as exc:
            print(f"⚠️  Could not load final results: {exc}")

    if classification_report_path.exists():
        try:
            artifacts["classification_report"] = pd.read_csv(classification_report_path, index_col=0)
            artifacts["classification_report_exist"] = True
            print(f"✓ Loaded classification report from {classification_report_path}")
        except Exception as exc:
            print(f"⚠️  Could not load classification report: {exc}")

    if artifacts.get("classification_report_exist"):
        report_df = artifacts["classification_report"]
        class_names = []
        for idx in report_df.index.tolist():
            if str(idx).strip() in {"accuracy", "macro avg", "weighted avg"}:
                continue
            class_names.append(str(idx))
        artifacts["class_names"] = class_names

    return artifacts

def compute_verified_metrics(artifacts):
    """
    Compute real metrics from artifacts if available.

    Args:
        artifacts: Dict containing predictions, ground truth, etc.

    Returns:
        dict: Computed metrics or None
    """
    if not artifacts.get("final_results_exist") or not artifacts.get("classification_report_exist"):
        print("❌ Verified evaluation artifacts not found in outputs/metrics")
        return None

    final_results = artifacts["final_results"].iloc[0].to_dict()
    report_df = artifacts["classification_report"]
    class_names = artifacts.get("class_names", [])

    overall_metrics = {
        "accuracy": float(final_results.get("accuracy", 0.0)),
        "macro_precision": float(final_results.get("macro_precision", 0.0)),
        "macro_recall": float(final_results.get("macro_recall", 0.0)),
        "macro_f1": float(final_results.get("macro_f1", 0.0)),
        "weighted_f1": float(final_results.get("weighted_f1", 0.0)),
    }

    per_class_metrics = {}
    total_support = 0
    for class_name in class_names:
        if class_name not in report_df.index:
            continue
        row = report_df.loc[class_name]
        support = int(float(row.get("support", 0)))
        total_support += support
        per_class_metrics[class_name] = {
            "precision": float(row.get("precision", 0.0)),
            "recall": float(row.get("recall", 0.0)),
            "f1": float(row.get("f1-score", 0.0)),
            "support": support,
        }

    return {
        "status": "verified",
        "source": {
            "final_results_csv": "outputs/metrics/final_results.csv",
            "classification_report_csv": "outputs/metrics/classification_report.csv",
        },
        "overall_metrics": overall_metrics,
        "per_class_metrics": per_class_metrics,
        "class_names": class_names,
        "num_samples": int(total_support),
    }

def save_verified_metrics(metrics, output_dir='outputs/metrics'):
    """
    Save computed metrics to files for UI consumption.

    Args:
        metrics: Computed metrics dict
        output_dir: Output directory
    """
    if not metrics:
        # Create placeholder files indicating no verified metrics
        placeholder_content = {
            "status": "no_verified_metrics",
            "message": "No verified evaluation artifacts found. Run the evaluation pipeline to generate metrics.",
            "timestamp": pd.Timestamp.now().isoformat()
        }

        os.makedirs(output_dir, exist_ok=True)

        with open(f"{output_dir}/metrics_summary.json", 'w', encoding="utf-8") as f:
            json.dump(placeholder_content, f, indent=2)

        # Create empty CSV placeholder
        pd.DataFrame().to_csv(f"{output_dir}/per_class_metrics.csv", index=False)

        print("📄 Created placeholder files for missing metrics")
        return

    os.makedirs(output_dir, exist_ok=True)

    # Save overall metrics
    with open(f"{output_dir}/metrics_summary.json", 'w', encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    # Save per-class metrics as CSV
    if metrics.get("per_class_metrics"):
        per_class_df = pd.DataFrame.from_dict(metrics["per_class_metrics"], orient="index")
        per_class_df.index.name = "class"
        per_class_df.reset_index(inplace=True)
        per_class_df.to_csv(f"{output_dir}/per_class_metrics.csv", index=False)

    # Generate confusion matrix plot
    cm_source_path = os.path.join("outputs", "plots", "confusion_matrix.png")
    if os.path.exists(cm_source_path):
        try:
            from PIL import Image

            os.makedirs(output_dir, exist_ok=True)
            image = Image.open(cm_source_path)
            image.save(f"{output_dir}/confusion_matrix_verified.png")
        except Exception as exc:
            print(f"⚠️  Could not copy confusion matrix image: {exc}")

    print(f"✅ Saved verified metrics to {output_dir}/")

def main():
    """Main function to generate verified metrics."""
    print("🔍 Generating Verified Metrics from Evaluation Artifacts")
    print("=" * 60)

    # Load evaluation artifacts
    artifacts = load_evaluation_artifacts(metrics_dir=os.path.join("outputs", "metrics"))

    # Compute metrics if possible
    metrics = compute_verified_metrics(artifacts)

    # Save results
    save_verified_metrics(metrics)

    # Summary
    if metrics and metrics.get("status") == "verified":
        overall = metrics.get("overall_metrics", {})
        print("\n📊 Verified Metrics Summary:")
        print(f"  accuracy: {overall.get('accuracy', 0.0):.4f}")
        print(f"  macro_f1: {overall.get('macro_f1', 0.0):.4f}")
        print(f"  weighted_f1: {overall.get('weighted_f1', 0.0):.4f}")
        print(f"  classes: {len(metrics.get('class_names', []))}")
        print(f"  samples: {metrics.get('num_samples', 0)}")
    else:
        print("\n⚠️  No Verified Metrics Available")
        print("   Run the complete evaluation pipeline to generate real metrics:")
        print("   1. Train a model: cd src && python train.py")
        print("   2. Run evaluation: cd src && python evaluate.py")
        print("   3. Re-run this script: python generate_verified_metrics.py")

if __name__ == '__main__':
    main()
