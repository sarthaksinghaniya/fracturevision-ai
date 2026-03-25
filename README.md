# 🦴 FractureVision-AI
**AI-powered multi-class bone fracture classification system for clinical decision support**

FractureVision-AI is a hackathon-level deep learning project for multi-class bone fracture classification from X-ray images. It combines an EfficientNet-B3 backbone, dual-head prediction, imbalance-aware training, and explainability tools to support faster and more reliable fracture assessment.

---

## 📌 Overview

Bone fractures are common but challenging to classify accurately from X-ray images. This project presents a deep learning-based solution that classifies fractures into multiple categories using an advanced dual-head architecture.

The goal is to assist radiologists with faster and more reliable diagnosis.

---

## 🚀 Features

- Multi-class fracture classification (6 classes)
- Dual-head architecture:
  - Multi-class classifier
  - Binary fracture detection (fracture vs non-fracture)
- Class imbalance handling:
  - Focal Loss
  - `WeightedRandomSampler`
- Targeted data augmentation
- Test Time Augmentation (TTA)
- Ensemble-ready pipeline
- Streamlit demo with deployment-safe model loading
- Optional Grad-CAM explainability (on-demand, cached, fail-safe)
- Confidence visualization (percentage + progress + color band)
- Active-learning feedback capture to JSON
- Lightweight feedback retraining utilities
- Modular and scalable design

---

## 📊 Dataset

- **6 Classes**
  - `non-fracture`
  - `simple`
  - `comminuted`
  - `spiral`
  - `greenstick`
  - `stress`
- **Preprocessing**
  - Resize: `224×224`
  - Normalization
- **Split**
  - Train: `70%`
  - Validation: `10%`
  - Test: `20%`

---

## 🧠 Model Architecture

- Backbone: **EfficientNet-B3** (pretrained on ImageNet)
- Shared feature extractor
- Dual-head output:
  1. Multi-class classification head
  2. Binary classification head (`fracture` vs `non-fracture`)

### 🔄 Flow

`Input Image → EfficientNet → Feature Layer → Multi-class Head + Binary Head → Probability Calibration → Final Prediction`

### 📌 Why Dual-Head?

The binary head acts as a prior to reduce confusion between fracture and non-fracture classes, improving overall classification reliability. This is especially useful for subtle cases where minor fracture patterns can visually overlap with normal bone structure.

![Architecture Diagram](docs/architecture.png)

---

## ⚙️ Training Details

- **Loss**
  - Focal Loss (`gamma = 1.5`)
  - Combined with binary auxiliary loss (`0.2` weight)
- **Optimizer**
  - Adam
- **Scheduler**
  - Cosine Annealing
- **Techniques**
  - Class weights
  - `WeightedRandomSampler`
  - Label smoothing (`0.05`)
  - Early stopping (`patience = 8`)

---

## 📈 Evaluation & Results

The project is optimized around **Macro F1-score** to ensure balanced performance across all fracture categories, not just the dominant classes.

### 📊 Overall Metrics

| Metric | Value |
|--------|-------|
| Accuracy | 0.7423 |
| Macro Precision | 0.7386 |
| Macro Recall | 0.6895 |
| Macro F1-score | 0.6967 |
| Weighted F1-score | 0.7353 |

### 📈 Per-Class Performance

| Class         | Precision | Recall | F1-score | Support |
|--------------|----------|--------|---------|--------|
| Non-fracture | 0.8125   | 0.6842 | 0.7429  | 19     |
| Simple       | 0.7381   | 0.7381 | 0.7381  | 42     |
| Comminuted   | 0.7576   | 0.6410 | 0.6944  | 39     |
| Spiral       | 0.6757   | 0.8621 | 0.7576  | 29     |
| Greenstick   | 0.6667   | 0.2857 | 0.4000  | 7      |
| Stress       | 0.7812   | 0.9259 | 0.8475  | 27     |

### 🔍 Key Insights

- Strong performance on major fracture classes (simple, spiral, stress)
- High recall for stress fractures (0.92), indicating robust detection
- Dual-head architecture improved non-fracture classification (F1: 0.74)
- Greenstick remains the most challenging class due to limited samples and subtle features
- Overall Macro F1-score of 0.69 reflects balanced multi-class performance

### 🔄 Confusion Analysis

- Non-fracture ↔ Greenstick confusion observed (minor overlap)
- Some misclassification between comminuted and simple fractures
- Spiral class shows strong separability with high recall (0.86)

![Confusion Matrix](docs/confusion_matrix.png)

---

## 📁 Project Structure

```text
fracturevision-ai/
├── feedback/
│   └── feedback_data.json
├── src/
│   ├── active_learning.py
│   ├── feedback.py
│   ├── gradcam.py
│   ├── predict.py
│   └── retrain.py
├── models/
│   └── model.pth
├── configs/
├── outputs/
├── docs/
│   ├── architecture.png
│   └── confusion_matrix.png
├── app.py
├── requirements.txt
├── runtime.txt
├── config.yaml
├── create_dataset_splits.py
├── prepare_dataset.py
└── README.md
```

---

## ⚡ Installation

```bash
git clone <your-repo-link>
cd project
pip install -r requirements.txt
```

---

## ▶️ Usage

### Create train/validation/test splits

```bash
python create_dataset_splits.py
```

### Train the model

```bash
python src/train.py
```

### Evaluate the model

```bash
python src/evaluate.py
```

### Run inference on a single image

```bash
python src/inference.py --image path/to/xray.jpg
```

### Launch the demo app

```bash
streamlit run app.py
```

### Capture user feedback from app

Feedback entries are appended to:

```text
feedback/feedback_data.json
```

### Retrain from collected feedback

```python
from src.retrain import FeedbackDataset, retrain
from models.model import FractureClassifier

dataset = FeedbackDataset(class_names=["non-fracture", "simple", "comminuted", "spiral", "greenstick", "stress"])
model = FractureClassifier(model_name="efficientnet_b3", pretrained=False, num_classes=6, dropout=0.3)
updated_model = retrain(model, dataset)
```

The retraining helper saves updated weights to:

```text
models/model_updated.pth
```

---

## 🔮 Future Improvements

- Better minority-class handling for greenstick fractures
- Larger and more diverse X-ray dataset
- Real-time deployment for web or clinical interfaces
- Stronger lightweight ensembles for final-stage inference

---

## 👥 Team / Author

**Team:** TechNeekX  
**Institution:** BBD University, Lucknow

### Primary Contact

- **Sarthak Singhaniya**
- GitHub: [sarthaksinghaniya](https://github.com/sarthaksinghaniya/FractureVision-AI)
- LinkedIn: _Add LinkedIn link_
- Portfolio: _Add portfolio link_
- Email: `sarthaksinghaniya789@gmail.com`

### Team Members

- Sarthak Singhaniya — Team Lead, AI/ML
- Nikhil Yadav — Design and Architecture
- Vaishnavi Choudhari — Backend
- Anshuman Soni — Media Handling
- Palak Mishra — Logistics and Research

---

## 🏁 Final Note

FractureVision-AI is designed to demonstrate practical ML engineering depth for hackathon evaluation, portfolio presentation, and technical interviews. With real evaluation metrics, explainability support, and a modular training pipeline, it showcases both research thinking and production-oriented design.

This project is intended for research and demonstration purposes and is not a substitute for professional clinical diagnosis.
