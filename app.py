import os
from io import BytesIO

import numpy as np
import streamlit as st
import torch
from PIL import Image, UnidentifiedImageError
from torchvision import transforms

from models.model import FractureClassifier, combine_head_probabilities


st.set_page_config(page_title="FractureVision-AI", page_icon="🦴", layout="wide")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "models", "model.pth")
IMAGE_SIZE = 224
CLASS_NAMES = ["non-fracture", "simple", "comminuted", "spiral", "greenstick", "stress"]
MODEL_NAME = "efficientnet_b3"
DROPOUT = 0.3
NON_FRACTURE_INDEX = 0
GREENSTICK_INDEX = 4


def normalize_label_name(label_name):
    return str(label_name).strip().lower().replace("_", "-").replace(" ", "-")


def create_model():
    return FractureClassifier(
        model_name=MODEL_NAME,
        pretrained=False,
        num_classes=len(CLASS_NAMES),
        dropout=DROPOUT,
    )


@st.cache_resource
def load_model_cached(model_path):
    try:
        model = create_model()
        state = torch.load(model_path, map_location="cpu")
        state_dict = state.get("state_dict", state) if isinstance(state, dict) else state
        cleaned_state_dict = {}
        for key, value in state_dict.items():
            cleaned_state_dict[key.replace("module.", "", 1)] = value
        model.load_state_dict(cleaned_state_dict, strict=True)
        model.eval()
        return model
    except Exception as exc:
        raise RuntimeError(f"Failed to load model from '{model_path}': {exc}") from exc


def preprocess_image(image):
    transform = transforms.Compose(
        [
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    image = image.convert("RGB")
    return transform(image).unsqueeze(0)


def apply_probability_calibration(probs, binary_logits, temperature=1.5):
    calibrated_probs = torch.softmax(torch.log(probs.clamp_min(1e-8)) / temperature, dim=1)
    calibrated_probs[:, GREENSTICK_INDEX] = calibrated_probs[:, GREENSTICK_INDEX] * 1.2
    calibrated_probs = calibrated_probs / calibrated_probs.sum(dim=1, keepdim=True).clamp_min(1e-8)

    binary_fracture_prob = torch.sigmoid(binary_logits)
    binary_non_fracture_prob = 1.0 - binary_fracture_prob

    if binary_non_fracture_prob.item() > 0.6:
        override_probs = torch.zeros_like(calibrated_probs)
        override_probs[:, NON_FRACTURE_INDEX] = binary_non_fracture_prob
        remaining = (1.0 - binary_non_fracture_prob).clamp_min(0.0)
        fracture_mask = torch.ones(calibrated_probs.size(1), dtype=torch.bool, device=calibrated_probs.device)
        fracture_mask[NON_FRACTURE_INDEX] = False
        fracture_probs = calibrated_probs[:, fracture_mask]
        fracture_probs = fracture_probs / fracture_probs.sum(dim=1, keepdim=True).clamp_min(1e-8)
        override_probs[:, fracture_mask] = fracture_probs * remaining.unsqueeze(1)
        calibrated_probs = override_probs

    return calibrated_probs, binary_non_fracture_prob.item()


def run_tta_prediction(model, image_tensor):
    tta_inputs = [image_tensor, torch.flip(image_tensor, dims=[3])]
    multi_probs = []
    binary_logits = []

    with torch.no_grad():
        for tta_input in tta_inputs:
            outputs = model.forward_multitask(tta_input)
            probs = combine_head_probabilities(
                outputs["multi_logits"],
                outputs["binary_logits"],
                NON_FRACTURE_INDEX,
            )
            multi_probs.append(probs)
            binary_logits.append(outputs["binary_logits"])

    avg_probs = torch.mean(torch.stack(multi_probs, dim=0), dim=0)
    avg_binary_logits = torch.mean(torch.stack(binary_logits, dim=0), dim=0)
    return avg_probs, avg_binary_logits


def predict(image, model):
    image_tensor = preprocess_image(image)
    probs, binary_logits = run_tta_prediction(model, image_tensor)
    calibrated_probs, binary_non_fracture_prob = apply_probability_calibration(probs, binary_logits)

    prediction_index = int(torch.argmax(calibrated_probs, dim=1).item())
    return {
        "prediction_label": CLASS_NAMES[prediction_index],
        "confidence": float(calibrated_probs[0, prediction_index].item()),
        "probabilities": calibrated_probs[0].cpu().numpy(),
        "binary_non_fracture_prob": binary_non_fracture_prob,
    }


def safe_open_image(uploaded_file):
    try:
        image_bytes = uploaded_file.read()
        if not image_bytes:
            raise ValueError("Uploaded file is empty.")
        image = Image.open(BytesIO(image_bytes))
        image.load()
        return image
    except (UnidentifiedImageError, ValueError) as exc:
        raise ValueError("Please upload a valid X-ray image in JPG, JPEG, or PNG format.") from exc


def render_probability_chart(probabilities):
    chart_data = {
        "class": CLASS_NAMES,
        "probability": [float(x) for x in np.asarray(probabilities)],
    }
    st.bar_chart(chart_data, x="class", y="probability")


def main():
    st.title("🦴 FractureVision-AI")
    st.caption("AI-powered multi-class bone fracture classification system")

    model_path = os.path.join(BASE_DIR, "models", "model.pth")
    if not os.path.exists(model_path):
        st.error(f"Model file not found: {model_path}")
        st.stop()

    try:
        model = load_model_cached(model_path)
    except Exception as exc:
        st.error(f"Model loading failed: {exc}")
        st.stop()

    uploaded_file = st.file_uploader(
        "Upload an X-ray image",
        type=["jpg", "jpeg", "png"],
        help="Upload a clear X-ray image for fracture classification.",
    )

    if uploaded_file is None:
        st.info("Upload an X-ray image to run fracture prediction.")
        return

    try:
        image = safe_open_image(uploaded_file)
    except ValueError as exc:
        st.error(str(exc))
        return

    col1, col2 = st.columns([1, 1])
    with col1:
        st.subheader("Uploaded Image")
        st.image(image, use_container_width=True)

    with st.spinner("Running inference..."):
        try:
            prediction = predict(image, model)
        except Exception as exc:
            st.error(f"Prediction failed: {exc}")
            return

    predicted_label = prediction["prediction_label"]
    confidence = prediction["confidence"]

    with col2:
        st.subheader("Prediction Result")
        st.markdown(f"## **{predicted_label}**")
        st.metric("Confidence", f"{confidence:.2%}")
        st.metric("Binary non-fracture probability", f"{prediction['binary_non_fracture_prob']:.2%}")
        if normalize_label_name(predicted_label) == "non-fracture":
            st.success("The model predicts this X-ray as non-fracture.")
        else:
            st.warning("The model predicts a fracture pattern in this X-ray.")

    st.subheader("Probability Distribution")
    render_probability_chart(prediction["probabilities"])


if __name__ == "__main__":
    main()
