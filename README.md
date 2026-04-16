# 🦴 FractureVision-AI

**AI-powered multi-class bone fracture detection system with interpretable predictions for clinical decision support**

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-orange.svg)](https://streamlit.io/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

---

## 🎯 Hero Demo

*Interactive web application for real-time X-ray fracture analysis with Grad-CAM explainability*

---

## 🚨 Problem Statement

Emergency rooms worldwide face overwhelming caseloads of orthopedic injuries, with X-ray interpretation bottlenecks causing delayed diagnoses and suboptimal treatment. Traditional manual fracture assessment is subjective, time-intensive, and prone to human error—especially for complex fracture patterns like comminuted or greenstick fractures.

**Impact:** Delayed fracture detection can lead to complications, increased recovery time, and higher healthcare costs.

---

## 💡 Solution

FractureVision-AI delivers an intelligent, interpretable AI system that:

1. **Automates multi-class fracture classification** into 6 clinically relevant categories
2. **Provides visual explainability** through Grad-CAM heatmaps to build radiologist trust
3. **Handles class imbalance** with advanced sampling and loss techniques
4. **Supports continuous learning** through user feedback integration

Designed for research and prototyping workflows, the system aims to reduce time-to-insight while emphasizing interpretability and traceability.

---

## 🔑 Key Features

- **🔍 Multi-Class Classification**: 6 fracture types (non-fracture, simple, comminuted, spiral, greenstick, stress)
- **🎯 Dual-Head Architecture**: EfficientNet-B0 with multi-class + binary prior for robust predictions
- **📊 Grad-CAM Explainability**: Visual heatmaps showing model's decision regions
- **⚖️ Imbalance Handling**: Focal Loss + WeightedRandomSampler for minority classes
- **🔄 Active Learning**: Feedback-based retraining for continuous improvement
- **🌐 Streamlit Deployment**: User-friendly web interface for real-time inference
- **📈 Comprehensive Evaluation**: Programmatic evaluation outputs (classification report, confusion matrix)

---

## 🏗️ System Architecture

```
X-ray Image → Data Augmentation → EfficientNet-B0 → Global Pooling → Dual Heads (Multi-class + Binary) → Probability Calibration → Grad-CAM → Clinical UI
```

### Architecture Flow:
1. **Input Processing**: Image resizing, normalization, augmentation
2. **Feature Extraction**: EfficientNet-B0 pretrained backbone
3. **Dual Prediction Heads**:
   - Multi-class head (6 fracture types)
   - Binary head (fracture vs non-fracture prior)
4. **Explainability**: Grad-CAM overlays for decision transparency
5. **UI Integration**: Streamlit web app for clinical deployment

---

## 🤖 Model Details

### Backbone: EfficientNet-B0
- **Pretrained**: ImageNet weights for robust feature extraction
- **Architecture**: Inverted residual blocks with squeeze-and-excitation
- **Efficiency**: Optimized for medical imaging with low computational cost

### Dual-Head Design
- **Multi-Class Head**: Direct classification into 6 fracture categories
- **Binary Head**: Auxiliary fracture/non-fracture detection for improved reliability
- **Combined Loss**: Weighted Focal Loss (γ=1.5) + binary auxiliary loss (weight=0.2)

### Training Strategy
- **Optimizer**: AdamW with weight decay (1e-4)
- **Scheduler**: Cosine Annealing with warm restarts
- **Augmentation**: Random rotations, flips, color jittering
- **Regularization**: Dropout (0.3), label smoothing (0.05)

---

## 📊 Evaluation Status

### Current Status
This model has been trained and validated during development, but final reproducible benchmark metrics are being regenerated using a standardized evaluation pipeline.

### Why this matters
Medical AI systems require strict validation to ensure reliability. This project prioritizes reproducibility and traceability over presenting unverified performance numbers.

### How to generate metrics
- Run:
  `python src/evaluate.py`
- Outputs will be generated in:
  - `outputs/metrics_summary.json`
  - `outputs/per_class_metrics.csv`
  - `outputs/confusion_matrix.png`

### Note
All metrics shown in this project are generated programmatically from model predictions. No values are manually hardcoded.

> ⚠️ Transparency Note  
> This project intentionally avoids displaying unverified or inflated metrics.  
> All performance results are generated only after running the evaluation pipeline.

---

## 🎯 Challenges & Learnings

### Technical Challenges
- **Class Imbalance**: Greenstick fractures underrepresented, requiring focal loss and weighted sampling
- **Medical Variability**: X-ray quality variations across different imaging equipment
- **Interpretability Trade-offs**: Balancing model complexity with clinical explainability

### Engineering Learnings
- **Data Pipeline Robustness**: Auto-adaptive dataset discovery prevents manual preprocessing bottlenecks
- **Clinical Feedback Integration**: Active learning loop enables continuous model improvement
- **Production Deployment**: Streamlit-based web interface ensures accessibility for non-technical users

---

## 🚀 Future Improvements

- **Enhanced Minority Class Performance**: Targeted augmentation for greenstick fractures
- **Larger Clinical Dataset**: Multi-center X-ray collection for improved generalization
- **Real-time Deployment**: Integration with hospital PACS systems
- **Advanced Explainability**: Multi-modal explanations combining heatmaps with clinical annotations
- **Clinical Validation**: Prospective studies with radiologist ground truth

---

## 🎬 Demo & Screenshots

### Live Demo
[🔗 Launch FractureVision-AI Demo](https://fracturevision-ai.streamlit.app/) *(Deployed on Streamlit Cloud)*

Live demo deployed using Streamlit Cloud for real-time inference and visualization.

*Upload X-ray images and get instant fracture analysis with visual explainability*

### Interface Screenshots

#### Upload Interface

```plaintext
🦴 FractureVision-AI: Clinical X-ray Analysis

📤 Upload X-ray Image
[ Choose File ]   [ Upload ]

📊 Analysis Results
Prediction: Spiral Fracture
Confidence: Displayed dynamically in app
Notes: Designed for research and prototyping

🎯 Show Explanation →
```

#### Grad-CAM Explainability
*Visual heatmap showing the regions that most influenced the model prediction*

| View | Description |
| --- | --- |
| **Original X-ray** | Raw input image uploaded by the user |
| **Grad-CAM Overlay** | Heatmap overlay highlighting high-attention regions used by the model |

### Model Evaluation Visualization
Confusion matrix and per-class performance plots are generated dynamically after running the evaluation pipeline.

*(Visualization outputs are generated into `outputs/` after running evaluation.)*

---

## 🧠 Why This Project Stands Out

- Combines multi-class classification with an auxiliary binary prior to improve robustness
- Focuses on interpretability (Grad-CAM), not just aggregate metrics
- Built with a deployment-first mindset (Streamlit UI + reproducible outputs)
- Includes a feedback-driven improvement loop (rare in most student projects)

---

## ⚙️ Installation & Setup

### Prerequisites
- Python 3.8+
- CUDA-compatible GPU (recommended for training)
- 8GB+ RAM

### Quick Start

```bash
# Clone the repository
git clone https://github.com/sarthaksinghaniya/FractureVision-AI.git
cd fracturevision-ai

# Install dependencies
pip install -r requirements.txt

# Launch the web application
streamlit run app.py
```

### Advanced Usage

```bash
# Prepare custom dataset
python auto_prepare_dataset.py

# Train the model
cd src && python train.py

# Evaluate performance
cd src && python evaluate.py

# Run inference on custom image
python src/predict.py --image path/to/xray.jpg
```

### Configuration
Modify `config.yaml` for custom parameters:
- Model architecture
- Training hyperparameters
- Dataset paths
- Evaluation settings

---

## 👥 Author & Team

**FractureVision-AI** is developed by the TechNeekX team at BBD University, Lucknow.

### Primary Contact
- **Sarthak Singhaniya** - Team Lead & AI/ML Engineer
- 📧 Email: sarthaksinghaniya789@gmail.com
- 🔗 GitHub: [sarthaksinghaniya](https://github.com/sarthaksinghaniya)
- 💼 LinkedIn: [Sarthak Singhaniya](https://www.linkedin.com/in/sarthak-singhaniya-a4ab9a323)
- 🌐 Portfolio: [sarthaksinghaniya.dev](https://sarthaksinghaniya.netlify.app)

### Core Team
- **Sarthak Singhaniya** — AI/ML Engineering & Architecture
- **Nikhil Yadav** — System Design & Backend
- **Vaishnavi Choudhari** — Data Pipeline & Processing
- **Anshuman Soni** — UI/UX & Frontend Integration

---

## 📄 License & Disclaimer

**License**: MIT License - Open source for research and clinical applications.

**Medical Disclaimer**: This system is designed for research and educational purposes. It is NOT a substitute for professional medical diagnosis or treatment. Always consult qualified healthcare professionals for clinical decisions.

---

## 🏆 Acknowledgments

- Dataset sources: Roboflow Bone Fracture Detection collections
- PyTorch team for the excellent deep learning framework
- Streamlit community for deployment tools
- Medical advisors for clinical guidance and validation

---

## 🚀 Future Work

- **Clinical Trials**: Collaborate with hospitals for prospective studies
- **Real-World Deployment**: Integrate with existing clinical workflows
- **Continuous Learning**: Expand active learning capabilities for user feedback
- **Explainability Research**: Investigate novel explainability techniques for medical AI

---

*FractureVision-AI: Transforming orthopedic diagnostics through intelligent, interpretable AI.* 🏥🤖

---

**⭐ Star this repository if you find it valuable for medical AI research and clinical applications!**
