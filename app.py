import streamlit as st
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import cv2
import matplotlib.pyplot as plt
import os
import sys
import yaml

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(__file__))

from models.model import FractureClassifier

# Page configuration
st.set_page_config(
    page_title="FractureVision-AI",
    page_icon="🦴",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for medical theme
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #2E86AB;
        text-align: center;
        margin-bottom: 2rem;
    }
    .prediction-fractured {
        background-color: #FF6B6B;
        color: white;
        padding: 20px;
        border-radius: 10px;
        text-align: center;
        font-size: 1.5rem;
        font-weight: bold;
        margin: 10px 0;
    }
    .prediction-normal {
        background-color: #4ECDC4;
        color: white;
        padding: 20px;
        border-radius: 10px;
        text-align: center;
        font-size: 1.5rem;
        font-weight: bold;
        margin: 10px 0;
    }
    .confidence-score {
        font-size: 1.2rem;
        text-align: center;
        margin: 10px 0;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model():
    """Load the trained PyTorch model"""
    try:
        # Load config
        with open('configs/config.yaml', 'r') as f:
            config = yaml.safe_load(f)

        # Convert config values
        config['batch_size'] = int(config['batch_size'])
        config['epochs'] = int(config['epochs'])
        config['lr'] = float(config['lr'])
        config['image_size'] = int(config['image_size'])

        # Initialize and load model
        model = FractureClassifier(model_name=config['model'], pretrained=False)
        model_path = 'outputs/models/best_model.pth'

        if not os.path.exists(model_path):
            st.error(f"Model file not found at {model_path}. Please train the model first.")
            return None

        model.load_state_dict(torch.load(model_path, map_location='cpu'))
        model.eval()
        return model, config

    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None

def preprocess_image(image, config):
    """Preprocess the uploaded image for model input"""
    # Resize image
    image = image.resize((config['image_size'], config['image_size']))

    # Convert to RGB if necessary
    if image.mode != 'RGB':
        image = image.convert('RGB')

    # Convert to numpy array
    image_array = np.array(image)

    # Apply transforms
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Convert to tensor
    image_tensor = transform(image_array)

    return image_tensor.unsqueeze(0), image_array

def generate_gradcam(model, image_tensor, original_image, config):
    """Generate Grad-CAM heatmap"""
    try:
        from pytorch_grad_cam import GradCAM
        from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
        from pytorch_grad_cam.utils.image import show_cam_on_image

        # Get target layer (EfficientNet-B0 conv head)
        target_layer = model.model.conv_head

        # Create GradCAM
        cam = GradCAM(model=model, target_layers=[target_layer])

        # Get prediction
        with torch.no_grad():
            output = model(image_tensor)
            pred_class = torch.argmax(output, dim=1).item()

        # Generate CAM
        targets = [ClassifierOutputTarget(pred_class)]
        grayscale_cam = cam(input_tensor=image_tensor, targets=targets)[0]

        # Normalize original image for overlay
        original_normalized = original_image.astype(np.float32) / 255.0

        # Create overlay
        cam_image = show_cam_on_image(original_normalized, grayscale_cam, use_rgb=True)

        return cam_image, pred_class

    except Exception as e:
        st.error(f"Error generating Grad-CAM: {str(e)}")
        return None, None

def create_probability_chart(probabilities):
    """Create a bar chart for prediction probabilities"""
    classes = ['Not Fractured', 'Fractured']
    colors = ['#4ECDC4', '#FF6B6B']

    fig, ax = plt.subplots(figsize=(8, 4))
    bars = ax.bar(classes, probabilities, color=colors, alpha=0.7)

    ax.set_ylabel('Probability')
    ax.set_title('Prediction Probabilities')
    ax.set_ylim(0, 1)

    # Add value labels on bars
    for bar, prob in zip(bars, probabilities):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{prob:.3f}', ha='center', va='bottom', fontweight='bold')

    plt.tight_layout()
    return fig

def main():
    # Load model
    model_data = load_model()
    if model_data is None:
        return

    model, config = model_data

    # Title and description
    st.markdown('<h1 class="main-header">🦴 FractureVision-AI</h1>', unsafe_allow_html=True)
    st.markdown("""
    ### Automated Bone Fracture Detection from X-ray Images

    Upload an X-ray image to detect bone fractures using advanced deep learning.
    The system provides fracture classification with explainable AI visualizations.
    """)

    # Sidebar with information
    with st.sidebar:
        st.header("ℹ️ About")
        st.markdown("""
        **FractureVision-AI** uses a deep learning model trained on thousands of X-ray images
        to automatically detect bone fractures.

        **Features:**
        - Real-time fracture detection
        - Explainable AI with Grad-CAM
        - High accuracy medical imaging
        - User-friendly interface
        """)

        st.header("⚠️ Medical Disclaimer")
        st.markdown("""
        This tool is for educational and research purposes only.
        Always consult qualified medical professionals for clinical diagnosis.
        """)

    # File uploader
    st.header("📤 Upload X-ray Image")
    uploaded_file = st.file_uploader(
        "Choose an X-ray image (JPG, JPEG, PNG)",
        type=['jpg', 'jpeg', 'png'],
        help="Upload a clear X-ray image for fracture analysis"
    )

    if uploaded_file is not None:
        # Load and display image
        image = Image.open(uploaded_file)

        # Create two columns
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("📷 Uploaded X-ray Image")
            st.image(image, caption="Uploaded Image", use_column_width=True)

        with col2:
            st.subheader("🔍 Analysis Results")

            # Show processing message
            with st.spinner("Analyzing image..."):
                # Preprocess image
                image_tensor, image_array = preprocess_image(image, config)

                # Make prediction
                with torch.no_grad():
                    outputs = model(image_tensor)
                    probabilities = torch.softmax(outputs, dim=1)[0]
                    pred_class = torch.argmax(outputs, dim=1).item()
                    confidence = probabilities[pred_class].item()

                # Generate Grad-CAM
                gradcam_result, gradcam_pred = generate_gradcam(model, image_tensor, image_array, config)

            # Display prediction
            if pred_class == 1:  # Fractured
                st.markdown("""
                <div class="prediction-fractured">
                    🚨 FRACTURE DETECTED
                </div>
                """, unsafe_allow_html=True)
                st.error("⚠️ Potential bone fracture detected. Please consult a medical professional immediately.")
            else:  # Not Fractured
                st.markdown("""
                <div class="prediction-normal">
                    ✅ NO FRACTURE DETECTED
                </div>
                """, unsafe_allow_html=True)
                st.success("✅ No fracture detected in the X-ray image.")

            # Confidence score
            st.markdown(f"""
            <div class="confidence-score">
                **Confidence Score:** {confidence:.1%}
            </div>
            """, unsafe_allow_html=True)

            # Probability bar chart
            st.subheader("📊 Prediction Probabilities")
            prob_chart = create_probability_chart(probabilities.numpy())
            st.pyplot(prob_chart)

        # Grad-CAM visualization (full width)
        if gradcam_result is not None:
            st.header("🔥 Explainable AI - Grad-CAM Heatmap")
            st.markdown("""
            The heatmap shows which regions of the X-ray contributed most to the model's prediction.
            Red areas indicate high importance for the predicted class.
            """)

            # Display Grad-CAM
            st.image(gradcam_result, caption="Grad-CAM Heatmap Overlay", use_column_width=True)

            # Additional explanation
            if pred_class == 1:
                st.info("🔴 Red regions highlight potential fracture areas that influenced the model's decision.")
            else:
                st.info("🔵 The model focused on normal bone structures, confirming no fracture detected.")
        else:
            st.warning("Grad-CAM visualization could not be generated. Prediction is still valid.")

    else:
        # Default message when no image uploaded
        st.info("👆 Please upload an X-ray image to begin fracture detection analysis.")

        # Show sample results
        st.header("📋 Sample Results")
        st.markdown("""
        **Example Output:**
        - Prediction: Fractured / Not Fractured
        - Confidence: 95.2%
        - Grad-CAM: Visual explanation of decision
        - Probability Chart: Detailed confidence scores
        """)

if __name__ == "__main__":
    main()
