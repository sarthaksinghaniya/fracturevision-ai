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

def load_evaluation_artifacts(results_dir='results'):
    """
    Load actual evaluation artifacts if they exist.

    Returns:
        dict: Evaluation data or None if not available
    """
    results_path = Path(results_dir)

    artifacts = {
        'predictions_exist': False,
        'ground_truth_exist': False,
        'predictions': None,
        'ground_truth': None,
        'class_names': None
    }

    # Check for prediction files
    pred_files = list(results_path.glob('*prediction*.csv')) + list(results_path.glob('*pred*.csv'))
    if pred_files:
        try:
            pred_df = pd.read_csv(pred_files[0])
            artifacts['predictions'] = pred_df['predictions'].values if 'predictions' in pred_df.columns else None
            artifacts['predictions_exist'] = True
            print(f"✓ Loaded predictions from {pred_files[0]}")
        except Exception as e:
            print(f"⚠️  Could not load predictions: {e}")

    # Check for ground truth
    gt_files = list(results_path.glob('*ground_truth*.csv')) + list(results_path.glob('*true*.csv'))
    if gt_files:
        try:
            gt_df = pd.read_csv(gt_files[0])
            artifacts['ground_truth'] = gt_df['ground_truth'].values if 'ground_truth' in gt_df.columns else None
            artifacts['ground_truth_exist'] = True
            print(f"✓ Loaded ground truth from {gt_files[0]}")
        except Exception as e:
            print(f"⚠️  Could not load ground truth: {e}")

    # Try to infer class names from training data
    try:
        # Look for class names in config or other files
        config_files = list(Path('.').glob('**/config*.yaml'))
        if config_files:
            import yaml
            with open(config_files[0], 'r') as f:
                config = yaml.safe_load(f)
                artifacts['class_names'] = config.get('class_names', [])
                print(f"✓ Loaded class names from config: {artifacts['class_names']}")
    except Exception as e:
        print(f"⚠️  Could not load class names: {e}")

    return artifacts

def compute_verified_metrics(artifacts):
    """
    Compute real metrics from artifacts if available.

    Args:
        artifacts: Dict containing predictions, ground truth, etc.

    Returns:
        dict: Computed metrics or None
    """
    if not artifacts['predictions_exist'] or not artifacts['ground_truth_exist']:
        print("❌ Insufficient evaluation artifacts found")
        return None

    predictions = artifacts['predictions']
    ground_truth = artifacts['ground_truth']
    class_names = artifacts.get('class_names', [])

    if len(predictions) != len(ground_truth):
        print("❌ Mismatched prediction and ground truth lengths")
        return None

    if len(predictions) == 0:
        print("❌ Empty prediction arrays")
        return None

    print(f"📊 Computing metrics for {len(predictions)} samples")

    # Compute classification report
    target_names = class_names if class_names else None
    report = classification_report(
        ground_truth,
        predictions,
        target_names=target_names,
        output_dict=True,
        zero_division=0
    )

    # Confusion matrix
    cm = confusion_matrix(ground_truth, predictions)

    # Overall metrics
    overall_metrics = {
        'accuracy': report.get('accuracy', 0),
        'macro_avg': report.get('macro avg', {}),
        'weighted_avg': report.get('weighted avg', {})
    }

    # Per-class metrics
    per_class_metrics = {}
    num_classes = len([k for k in report.keys() if k not in ['accuracy', 'macro avg', 'weighted avg']])

    for i in range(num_classes):
        class_key = str(i)
        if class_key in report:
            class_name = class_names[i] if i < len(class_names) else f"class_{i}"
            per_class_metrics[class_name] = {
                'precision': report[class_key]['precision'],
                'recall': report[class_key]['recall'],
                'f1-score': report[class_key]['f1-score'],
                'support': report[class_key]['support']
            }

    return {
        'overall_metrics': overall_metrics,
        'per_class_metrics': per_class_metrics,
        'confusion_matrix': cm.tolist(),
        'num_samples': len(predictions),
        'num_classes': num_classes,
        'class_names': class_names
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

        with open(f"{output_dir}/metrics_summary.json", 'w') as f:
            json.dump(placeholder_content, f, indent=2)

        # Create empty CSV placeholder
        pd.DataFrame().to_csv(f"{output_dir}/per_class_metrics.csv")

        print("📄 Created placeholder files for missing metrics")
        return

    os.makedirs(output_dir, exist_ok=True)

    # Save overall metrics
    with open(f"{output_dir}/metrics_summary.json", 'w') as f:
        json.dump(metrics, f, indent=2)

    # Save per-class metrics as CSV
    if metrics['per_class_metrics']:
        per_class_df = pd.DataFrame.from_dict(metrics['per_class_metrics'], orient='index')
        per_class_df.to_csv(f"{output_dir}/per_class_metrics.csv")

    # Generate confusion matrix plot
    if 'confusion_matrix' in metrics:
        cm = np.array(metrics['confusion_matrix'])
        plt.figure(figsize=(10, 8))

        class_names = metrics.get('class_names', [f"Class {i}" for i in range(cm.shape[0])])

        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=class_names, yticklabels=class_names)
        plt.title(f'Confusion Matrix (Verified - {metrics["num_samples"]} samples)')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()

        plt.savefig(f"{output_dir}/confusion_matrix_verified.png", dpi=300, bbox_inches='tight')
        plt.close()

    print(f"✅ Saved verified metrics to {output_dir}/")

def main():
    """Main function to generate verified metrics."""
    print("🔍 Generating Verified Metrics from Evaluation Artifacts")
    print("=" * 60)

    # Load evaluation artifacts
    artifacts = load_evaluation_artifacts()

    # Compute metrics if possible
    metrics = compute_verified_metrics(artifacts)

    # Save results
    save_verified_metrics(metrics)

    # Summary
    if metrics:
        print("
📊 Verified Metrics Summary:"        print(".2f"        print(".2f"        print(f"  Classes: {metrics['num_classes']}")
        print(f"  Samples: {metrics['num_samples']}")
    else:
        print("\n⚠️  No Verified Metrics Available")
        print("   Run the complete evaluation pipeline to generate real metrics:")
        print("   1. Train a model: cd src && python train.py")
        print("   2. Run evaluation: cd src && python evaluate.py")
        print("   3. Re-run this script: python generate_verified_metrics.py")

if __name__ == '__main__':
    main()
