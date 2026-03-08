# FractureVision-AI

An AI-powered system for classifying bone fractures from X-ray images using deep learning, with explainability through Grad-CAM visualizations.

## Project Overview

This project implements a binary classification model to detect fractures in X-ray images. It uses EfficientNet-B0 as the backbone architecture, pre-trained on ImageNet, and fine-tuned for fracture detection. The model provides both classification predictions and visual explanations using Grad-CAM.

## Dataset

The project uses YOLO-formatted datasets located in `./datasets/` with the following structure:

```
datasets/
├── BoneFractureYolo8/
│   ├── data.yaml
│   ├── train/
│   │   ├── images/ (X-ray images)
│   │   └── labels/ (.txt files with bounding boxes and class IDs)
│   ├── valid/
│   │   ├── images/
│   │   └── labels/
│   └── test/
│       ├── images/
│       └── labels/
└── bone fracture detection.v4-v4.yolov8/
    └── (same structure)
```

The original datasets contain 7 classes: ['elbow positive', 'fingers positive', 'forearm fracture', 'humerus fracture', 'humerus', 'shoulder fracture', 'wrist positive']

For binary classification, the data loader converts them as follows:
- Classes 0-3, 5-6 (indicating fractures): labeled as "fractured" (1)
- Class 4 ('humerus'): labeled as "not fractured" (0)

The datasets are concatenated to provide more training data.

## Model Architecture

- **Backbone**: EfficientNet-B0 (pre-trained on ImageNet)
- **Classifier**: Custom binary classification head with dropout
- **Input Size**: 256x256 pixels
- **Output**: Binary classification (fractured vs. not fractured)

## Requirements

Install the required packages:

```bash
pip install -r requirements.txt
```

## Configuration

Model hyperparameters are defined in `configs/config.yaml`:

- `batch_size`: 32
- `epochs`: 20
- `lr`: 3e-4
- `image_size`: 256
- `model`: efficientnet_b0

## Training

To train the model:

```bash
python src/train.py
```

The training script will:
- Load the dataset with data augmentation
- Train the model with AdamW optimizer and Binary Cross Entropy loss
- Implement early stopping and learning rate scheduling
- Save the best model checkpoint to `outputs/checkpoints/`
- Log training progress and metrics

## Evaluation

To evaluate the trained model:

```bash
python src/evaluate.py
```

This will compute:
- Accuracy, Precision, Recall, F1-score
- ROC-AUC curve
- Confusion matrix (saved as plot)
- Save detailed results to `results/final_results.csv`

## Inference

To run inference on a single X-ray image:

```bash
python src/inference.py --image path_to_xray.png
```

The script will:
- Load the trained model
- Predict the fracture class and probability
- Generate Grad-CAM heatmap overlay
- Save visualization to `outputs/figures/`

## Explainability

The Grad-CAM module (`src/gradcam.py`) generates heatmaps that highlight regions of the X-ray image that contributed most to the model's prediction, providing visual explanations for the classification decision.

## Project Structure

```
fracturevision-ai/
├── data/
│   └── dataset/  # Dataset location
├── src/
│   ├── data_loader.py
│   ├── train.py
│   ├── evaluate.py
│   ├── inference.py
│   └── gradcam.py
├── models/
│   └── model.py
├── configs/
│   └── config.yaml
├── outputs/
│   ├── checkpoints/
│   ├── figures/
│   └── logs/
├── results/
│   ├── final_results.csv
│   └── model_performance_analysis.csv
├── notebooks/
│   └── eda.ipynb
├── presentation/
├── README.md
├── requirements.txt
└── .gitignore
```

## Features

- Modular and clean code structure
- Data augmentation with Albumentations
- Early stopping to prevent overfitting
- Learning rate scheduling
- Model checkpoint saving
- Progress bars during training
- Comprehensive evaluation metrics
- Grad-CAM explainability
- Configurable hyperparameters
