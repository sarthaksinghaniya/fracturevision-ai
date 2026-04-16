import os
from io import BytesIO

import numpy as np
import streamlit as st
import torch
from PIL import Image, UnidentifiedImageError
from torchvision import transforms

import json
import pandas as pd

from models.model import FractureClassifier, combine_head_probabilities
from src.active_learning import should_ask_feedback
from src.feedback import save_feedback
from src.gradcam import GradCAM, overlay_heatmap, resolve_target_layer


st.set_page_config(page_title="FractureVision-AI", page_icon="🦴", layout="wide")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "models", "model.pth")
IMAGE_SIZE = 224
CLASS_NAMES = ["non-fracture", "simple", "comminuted", "spiral", "greenstick", "stress"]
MODEL_NAME = "efficientnet_b3"
DROPOUT = 0.3
NON_FRACTURE_INDEX = 0
GREENSTICK_INDEX = 4

METRICS_DIR = os.path.join(BASE_DIR, "outputs", "metrics")
METRICS_SUMMARY_PATH = os.path.join(METRICS_DIR, "metrics_summary.json")
PER_CLASS_METRICS_PATH = os.path.join(METRICS_DIR, "per_class_metrics.csv")


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


@st.cache_resource
def get_gradcam(_model):
    target_layer = resolve_target_layer(_model)
    return GradCAM(_model, target_layer=target_layer)


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


def predict_from_tensor(image_tensor, model):
    probs, binary_logits = run_tta_prediction(model, image_tensor)
    calibrated_probs, binary_non_fracture_prob = apply_probability_calibration(probs, binary_logits)

    prediction_index = int(torch.argmax(calibrated_probs, dim=1).item())
    return {
        "prediction_index": prediction_index,
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


def get_confidence_style(confidence):
    if confidence > 0.8:
        return "High", "#1f9d55"
    if confidence >= 0.6:
        return "Medium", "#b58900"
    return "Low", "#d73a49"


def load_verified_metrics():
    if not os.path.exists(METRICS_SUMMARY_PATH):
        return None
    try:
        with open(METRICS_SUMMARY_PATH, "r", encoding="utf-8") as f:
            summary = json.load(f)
        if not isinstance(summary, dict) or summary.get("status") != "verified":
            return None
        per_class_df = None
        if os.path.exists(PER_CLASS_METRICS_PATH):
            per_class_df = pd.read_csv(PER_CLASS_METRICS_PATH)
        return {"summary": summary, "per_class": per_class_df}
    except Exception:
        return None


@st.cache_data(show_spinner=False)
def generate_gradcam_cached(image_bytes, class_idx, model_path):
    image = Image.open(BytesIO(image_bytes)).convert("RGB")
    gradcam_tensor = preprocess_image(image)
    model = load_model_cached(model_path)
    gradcam = get_gradcam(model)
    heatmap = gradcam.generate(gradcam_tensor, class_idx)
    return heatmap


def main():
    st.title("🦴 FractureVision-AI")
    st.caption("AI-powered multi-class bone fracture classification system")

    with st.expander("Verified Evaluation Metrics", expanded=False):
        verified = load_verified_metrics()
        if verified is None:
            st.info(
                "Verified metrics not available. Run `cd src && python evaluate.py` and then "
                "`python generate_verified_metrics.py` to generate traceable evaluation artifacts in `outputs/metrics/`."
            )
        else:
            overall = verified["summary"].get("overall_metrics", {})
            c1, c2, c3, c4, c5 = st.columns(5)
            c1.metric("Accuracy", f"{overall.get('accuracy', 0.0):.2%}")
            c2.metric("Macro Precision", f"{overall.get('macro_precision', 0.0):.2%}")
            c3.metric("Macro Recall", f"{overall.get('macro_recall', 0.0):.2%}")
            c4.metric("Macro F1", f"{overall.get('macro_f1', 0.0):.2%}")
            c5.metric("Weighted F1", f"{overall.get('weighted_f1', 0.0):.2%}")

            st.caption(
                "All displayed results are loaded from saved evaluation artifacts in `outputs/metrics/` and are not hardcoded."
            )
            if verified["per_class"] is not None and not verified["per_class"].empty:
                st.subheader("Per-class metrics")
                st.dataframe(verified["per_class"], use_container_width=True)
    feedback_threshold = st.sidebar.slider(
        "Feedback Confidence Threshold",
        min_value=0.50,
        max_value=0.95,
        value=0.75,
        step=0.01,
        help="If confidence is below this threshold, the app asks for corrective feedback.",
    )

    model_path = os.path.join(BASE_DIR, "models", "model.pth")
    if not os.path.exists(model_path):
        st.error(f"Model file not found: {model_path}")
        st.stop()

    try:
        model = load_model_cached(model_path)
    except Exception as exc:
        st.error(f"Model loading failed: {exc}")
        st.stop()
    gradcam = get_gradcam(model)

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
            image_tensor = preprocess_image(image)
            prediction = predict_from_tensor(image_tensor, model)
        except Exception as exc:
            st.error(f"Prediction failed: {exc}")
            return

    predicted_label = prediction["prediction_label"]
    confidence = prediction["confidence"]

    with col2:
        st.subheader("Prediction Result")
        st.markdown(f"## **{predicted_label}**")
        st.metric("Confidence", f"{confidence:.2%}")
        st.progress(float(max(0.0, min(1.0, confidence))))
        confidence_level, confidence_color = get_confidence_style(confidence)
        st.markdown(
            f"<span style='color:{confidence_color};font-weight:600;'>"
            f"Confidence Level: {confidence_level}"
            "</span>",
            unsafe_allow_html=True,
        )
        st.metric("Binary non-fracture probability", f"{prediction['binary_non_fracture_prob']:.2%}")
        if normalize_label_name(predicted_label) == "non-fracture":
            st.success("The model predicts this X-ray as non-fracture.")
        else:
            st.warning("The model predicts a fracture pattern in this X-ray.")

    st.subheader("Probability Distribution")
    render_probability_chart(prediction["probabilities"])

    show_gradcam = st.checkbox("Show Explainability (Grad-CAM)", value=False)
    st.session_state["gradcam_enabled"] = show_gradcam
    if show_gradcam:
        with st.spinner("Generating Grad-CAM..."):
            try:
                image_bytes = uploaded_file.getvalue()
                class_idx = prediction["prediction_index"]
                heatmap = generate_gradcam_cached(image_bytes, class_idx, model_path)
                original_image = Image.open(BytesIO(image_bytes)).convert("RGB")
                cam_image = overlay_heatmap(original_image, heatmap)
                st.subheader("Model Explainability (Grad-CAM)")
                cam_col1, cam_col2 = st.columns(2)
                cam_col1.image(original_image, caption="Original X-ray", use_container_width=True)
                cam_col2.image(cam_image, caption="Red regions = high model attention", use_container_width=True)
            except Exception:
                st.warning("Explainability unavailable for this input")

    st.subheader("Feedback")
    needs_feedback = should_ask_feedback(confidence, threshold=feedback_threshold)
    if needs_feedback:
        st.warning(
            f"Low-confidence prediction ({confidence:.2%}). "
            "Please review and submit the correct label."
        )

    default_index = CLASS_NAMES.index(predicted_label) if predicted_label in CLASS_NAMES else 0
    with st.form("feedback_form", clear_on_submit=False):
        correct_label = st.selectbox("Correct Label", options=CLASS_NAMES, index=default_index)
        submit_feedback = st.form_submit_button("Submit Feedback")

    if submit_feedback:
        try:
            image_ref = uploaded_file.name if uploaded_file is not None else "unknown"
            save_feedback(
                image_path=image_ref,
                predicted_label=predicted_label,
                correct_label=correct_label,
                confidence=confidence,
            )
            st.success("Feedback saved successfully.")
        except Exception as exc:
            st.error(f"Failed to save feedback: {exc}")


if __name__ == "__main__":
    main()
