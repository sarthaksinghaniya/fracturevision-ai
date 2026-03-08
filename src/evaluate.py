import os
import sys
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import yaml
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, roc_curve, auc
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from models.model import FractureClassifier
from src.data_loader import get_dataloaders

def evaluate_model(model, dataloader, device):
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []

    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)

            outputs = model(images).squeeze()
            probs = torch.sigmoid(outputs)
            preds = (probs > 0.5).float()

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    return np.array(all_labels), np.array(all_preds), np.array(all_probs)

def plot_confusion_matrix(cm, classes, save_path):
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig(save_path)
    plt.close()

def plot_roc_curve(true_labels, probs, save_path):
    fpr, tpr, _ = roc_curve(true_labels, probs)
    roc_auc = auc(fpr, tpr)
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    plt.savefig(save_path)
    plt.close()

def main():
    # Load config
    with open('configs/config.yaml', 'r') as f:
        config = yaml.safe_load(f)

    # Convert config values to proper types
    config['batch_size'] = int(config['batch_size'])
    config['epochs'] = int(config['epochs'])
    config['lr'] = float(config['lr'])
    config['image_size'] = int(config['image_size'])

    # Create output directories
    os.makedirs('outputs/plots', exist_ok=True)
    os.makedirs('outputs/metrics', exist_ok=True)

    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Get dataloaders
    _, _, test_loader = get_dataloaders(config)

    # Initialize model and load weights
    model = FractureClassifier(model_name=config['model'], pretrained=False)
    model.load_state_dict(torch.load('outputs/models/best_model.pth', map_location=device))
    model.to(device)

    # Evaluate
    true_labels, preds, probs = evaluate_model(model, test_loader, device)

    # Compute metrics
    accuracy = accuracy_score(true_labels, preds)
    precision = precision_score(true_labels, preds)
    recall = recall_score(true_labels, preds)
    f1 = f1_score(true_labels, preds)
    roc_auc = roc_auc_score(true_labels, probs)

    # Plot ROC curve
    plot_roc_curve(true_labels, probs, 'outputs/plots/roc_curve.png')

    # Confusion matrix
    cm = confusion_matrix(true_labels, preds)
    plot_confusion_matrix(cm, ['Not Fractured', 'Fractured'], 'outputs/plots/confusion_matrix.png')

    # Save results
    results = {
        'Accuracy': accuracy,
        'Precision': precision,
        'Recall': recall,
        'F1 Score': f1,
        'ROC AUC': roc_auc
    }

    df = pd.DataFrame([results])
    df.to_csv('outputs/metrics/final_results.csv', index=False)

    # Print results
    print("Evaluation Results:")
    for key, value in results.items():
        print(f"{key}: {value:.4f}")

    print(f"Confusion matrix saved to outputs/plots/confusion_matrix.png")
    print(f"ROC curve saved to outputs/plots/roc_curve.png")
    print(f"Results saved to outputs/metrics/final_results.csv")

if __name__ == '__main__':
    main()
