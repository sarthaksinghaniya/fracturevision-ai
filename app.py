import os
import sys
from io import BytesIO
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
import torch
import yaml
from PIL import Image, UnidentifiedImageError
from torchvision import transforms

sys.path.insert(0, os.path.dirname(__file__))

from models.model import FractureClassifier, combine_head_probabilities
from src.pipeline_utils import load_config, normalize_label_name


st.set_page_config(
    page_title="FractureVision-AI",
    page_icon="🦴",
    layout="wide",
    initial_sidebar_state="expanded",
)


MODEL_CANDIDATES = [
    Path(os.environ.get("MODEL_PATH", "")) if os.environ.get("MODEL_PATH") else None,
    Path("outputs") / "models" / "best_model.pth",
    Path("outputs") / "models" / "final_model.pth",
]
ENSEMBLE_CANDIDATES = [
    Path(os.environ.get("ENSEMBLE_MODEL_PATH", "")) if os.environ.get("ENSEMBLE_MODEL_PATH") else None,
    Path("outputs") / "models" / "ensemble_model.pth",
    Path("outputs") / "models" / "secondary_model.pth",
    Path("outputs") / "models" / "best_model_b0.pth",
    Path("outputs") / "models" / "best_model_resnet50.pth",
]


def find_existing_path(candidates):
    for candidate in candidates:
        if candidate is not None and candidate.exists():
            return candidate
    return None


def load_model_metadata():
    metadata_path = Path("outputs") / "models" / "model_metadata.yaml"
    if not metadata_path.exists():
        return {}
    with open(metadata_path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def get_runtime_settings():
    config = load_config()
    metadata = load_model_metadata()
    class_names = metadata.get("class_names", config.get("class_names", ["non-fracture", "fracture"]))
    num_classes = metadata.get("num_classes", len(class_names))
    model_name = metadata.get("model_name", config.get("model", "efficientnet_b3"))
    non_fracture_index = next(
        (index for index, name in enumerate(class_names) if normalize_label_name(name) == "non-fracture"),
        0,
    )
    greenstick_index = next(
        (index for index, name in enumerate(class_names) if normalize_label_name(name) == "greenstick"),
        None,
    )
    return {
        "config": config,
        "metadata": metadata,
        "class_names": class_names,
        "num_classes": num_classes,
        "model_name": model_name,
        "non_fracture_index": non_fracture_index,
        "greenstick_index": greenstick_index,
    }


def create_model(model_name, num_classes, dropout):
    return FractureClassifier(
        model_name=model_name,
        pretrained=False,
        num_classes=num_classes,
        dropout=dropout,
    )


@st.cache_resource
def load_model(model_path_str, model_name, num_classes, dropout):
    model_path = Path(model_path_str)
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")

    device = torch.device("cpu")
    model = create_model(model_name, num_classes, dropout)
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()
    return model


def preprocess_image(image, image_size):
    transform = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    image = image.convert("RGB")
    tensor = transform(image)
    return tensor.unsqueeze(0)


def apply_probability_calibration(probs, binary_logits, non_fracture_index, greenstick_index, temperature=1.5):
    calibrated_probs = torch.softmax(torch.log(probs.clamp_min(1e-8)) / temperature, dim=1)

    if greenstick_index is not None:
        calibrated_probs[:, greenstick_index] = calibrated_probs[:, greenstick_index] * 1.2

    calibrated_probs = calibrated_probs / calibrated_probs.sum(dim=1, keepdim=True).clamp_min(1e-8)

    binary_fracture_prob = torch.sigmoid(binary_logits)
    binary_non_fracture_prob = 1.0 - binary_fracture_prob

    if binary_non_fracture_prob.item() > 0.6:
        override_probs = torch.zeros_like(calibrated_probs)
        override_probs[:, non_fracture_index] = binary_non_fracture_prob
        remaining = (1.0 - binary_non_fracture_prob).clamp_min(0.0)
        if calibrated_probs.size(1) > 1:
            fracture_mask = torch.ones(calibrated_probs.size(1), dtype=torch.bool, device=calibrated_probs.device)
            fracture_mask[non_fracture_index] = False
            fracture_probs = calibrated_probs[:, fracture_mask]
            fracture_probs = fracture_probs / fracture_probs.sum(dim=1, keepdim=True).clamp_min(1e-8)
            override_probs[:, fracture_mask] = fracture_probs * remaining.unsqueeze(1)
        calibrated_probs = override_probs

    return calibrated_probs, binary_non_fracture_prob.item()


def run_tta_prediction(model, image_tensor, runtime_settings):
    non_fracture_index = runtime_settings["non_fracture_index"]

    tta_inputs = [
        image_tensor,
        torch.flip(image_tensor, dims=[3]),
    ]

    multi_probs = []
    binary_logits = []

    with torch.no_grad():
        for tta_input in tta_inputs:
            outputs = model.forward_multitask(tta_input)
            probs = combine_head_probabilities(
                outputs["multi_logits"],
                outputs["binary_logits"],
                non_fracture_index,
            )
            multi_probs.append(probs)
            binary_logits.append(outputs["binary_logits"])

    avg_probs = torch.mean(torch.stack(multi_probs, dim=0), dim=0)
    avg_binary_logits = torch.mean(torch.stack(binary_logits, dim=0), dim=0)
    return avg_probs, avg_binary_logits


def predict(image, models, runtime_settings, use_ensemble=False):
    image_tensor = preprocess_image(image, runtime_settings["config"]["image_size"])

    collected_probs = []
    collected_binary_logits = []

    active_models = models if use_ensemble else models[:1]
    for model in active_models:
        probs, binary_logits = run_tta_prediction(model, image_tensor, runtime_settings)
        collected_probs.append(probs)
        collected_binary_logits.append(binary_logits)

    avg_probs = torch.mean(torch.stack(collected_probs, dim=0), dim=0)
    avg_binary_logits = torch.mean(torch.stack(collected_binary_logits, dim=0), dim=0)

    calibrated_probs, binary_non_fracture_prob = apply_probability_calibration(
        avg_probs,
        avg_binary_logits,
        runtime_settings["non_fracture_index"],
        runtime_settings["greenstick_index"],
        temperature=1.5,
    )

    prediction_index = int(torch.argmax(calibrated_probs, dim=1).item())
    confidence = float(calibrated_probs[0, prediction_index].item())

    return {
        "prediction_index": prediction_index,
        "prediction_label": runtime_settings["class_names"][prediction_index],
        "confidence": confidence,
        "probabilities": calibrated_probs[0].cpu().numpy(),
        "binary_non_fracture_prob": binary_non_fracture_prob,
        "image_tensor": image_tensor,
    }


def generate_gradcam(model, image_tensor, image, runtime_settings):
    try:
        from pytorch_grad_cam import GradCAM
        from pytorch_grad_cam.utils.image import show_cam_on_image
        from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
    except ImportError:
        return None, "Grad-CAM dependencies are not installed."

    try:
        image_np = np.array(image.convert("RGB").resize((runtime_settings["config"]["image_size"], runtime_settings["config"]["image_size"]))).astype(np.float32) / 255.0
        target_layer = model.backbone.conv_head
        cam = GradCAM(model=model, target_layers=[target_layer])

        with torch.no_grad():
            outputs = model.forward_multitask(image_tensor)
            probs = combine_head_probabilities(
                outputs["multi_logits"],
                outputs["binary_logits"],
                runtime_settings["non_fracture_index"],
            )
            pred_class = int(torch.argmax(probs, dim=1).item())

        grayscale_cam = cam(input_tensor=image_tensor, targets=[ClassifierOutputTarget(pred_class)])[0]
        cam_image = show_cam_on_image(image_np, grayscale_cam, use_rgb=True)
        return cam_image, None
    except Exception as exc:
        return None, str(exc)


def render_probability_chart(class_names, probabilities):
    chart_df = pd.DataFrame(
        {
            "Class": class_names,
            "Probability": probabilities,
        }
    ).set_index("Class")
    st.bar_chart(chart_df)


def load_available_models(runtime_settings):
    primary_path = find_existing_path(MODEL_CANDIDATES)
    if primary_path is None:
        raise FileNotFoundError("No trained model checkpoint found in outputs/models/")

    models = [
        load_model(
            str(primary_path),
            runtime_settings["model_name"],
            runtime_settings["num_classes"],
            runtime_settings["config"].get("classifier_dropout", 0.3),
        )
    ]

    secondary_path = find_existing_path(ENSEMBLE_CANDIDATES)
    ensemble_available = secondary_path is not None

    if ensemble_available:
        secondary_name = "efficientnet_b0" if "b0" in secondary_path.stem.lower() else runtime_settings["model_name"]
        try:
            secondary_model = load_model(
                str(secondary_path),
                secondary_name,
                runtime_settings["num_classes"],
                runtime_settings["config"].get("classifier_dropout", 0.3),
            )
            models.append(secondary_model)
        except Exception:
            ensemble_available = False

    return models, primary_path, ensemble_available


def safe_open_image(uploaded_file):
    try:
        image_bytes = uploaded_file.read()
        if not image_bytes:
            raise ValueError("Uploaded file is empty.")
        image = Image.open(BytesIO(image_bytes))
        image.load()
        return image
    except (UnidentifiedImageError, ValueError):
        raise ValueError("Please upload a valid X-ray image in JPG, JPEG, or PNG format.")


def main():
    st.title("🦴 FractureVision-AI")
    st.caption("AI-powered multi-class bone fracture classification system for clinical decision support")

    try:
        runtime_settings = get_runtime_settings()
        models, primary_path, ensemble_available = load_available_models(runtime_settings)
    except Exception as exc:
        st.error(f"Model initialization failed: {exc}")
        st.stop()

    with st.sidebar:
        st.header("Model Settings")
        st.write(f"Primary model: `{primary_path.name}`")
        use_ensemble = st.checkbox("Enable ensemble inference", value=False, disabled=not ensemble_available)
        show_gradcam = st.checkbox("Show Grad-CAM", value=True)
        st.header("Inference Pipeline")
        st.write("- Dual-head prediction")
        st.write("- Horizontal-flip TTA")
        st.write("- Temperature scaling (T=1.5)")
        st.write("- Greenstick probability boost")
        st.write("- Non-fracture override threshold: 0.6")

    uploaded_file = st.file_uploader(
        "Upload an X-ray image",
        type=["jpg", "jpeg", "png"],
        help="Upload a clear X-ray image for fracture classification.",
    )

    if uploaded_file is None:
        st.info("Upload an X-ray image to run multi-class fracture prediction.")
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
            prediction = predict(image, models, runtime_settings, use_ensemble=use_ensemble)
        except Exception as exc:
            st.error(f"Inference failed: {exc}")
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
    render_probability_chart(runtime_settings["class_names"], prediction["probabilities"])

    if show_gradcam:
        st.subheader("Grad-CAM Visualization")
        cam_image, gradcam_error = generate_gradcam(models[0], prediction["image_tensor"], image, runtime_settings)
        if cam_image is not None:
            st.image(cam_image, caption="Grad-CAM heatmap overlay", use_container_width=True)
        else:
            st.info(f"Grad-CAM unavailable: {gradcam_error}")


if __name__ == "__main__":
    main()
