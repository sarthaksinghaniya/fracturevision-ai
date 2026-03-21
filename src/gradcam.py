import argparse
import os
import sys

import cv2
import numpy as np
import torch
import yaml
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from models.model import FractureClassifier, combine_head_probabilities
from src.inference import preprocess_image
from src.pipeline_utils import load_config, normalize_label_name


def load_model_metadata():
    metadata_path = os.path.join("outputs", "models", "model_metadata.yaml")
    if not os.path.exists(metadata_path):
        return {}
    with open(metadata_path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def generate_gradcam_overlay(model, image_path, image_size, device, non_fracture_index):
    image_tensor = preprocess_image(image_path, image_size).to(device)
    original_image = cv2.imread(image_path)
    original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
    original_image = cv2.resize(original_image, (image_size, image_size)).astype(np.float32) / 255.0

    cam = GradCAM(model=model, target_layers=[model.backbone.conv_head])
    with torch.no_grad():
        outputs = model.forward_multitask(image_tensor)
        probs = combine_head_probabilities(outputs["multi_logits"], outputs["binary_logits"], non_fracture_index)
        pred_class = torch.argmax(probs, dim=1).item()

    grayscale_cam = cam(input_tensor=image_tensor, targets=[ClassifierOutputTarget(pred_class)])[0]
    return show_cam_on_image(original_image, grayscale_cam, use_rgb=True)


def main():
    parser = argparse.ArgumentParser(description="Generate Grad-CAM for one X-ray image")
    parser.add_argument("--image", required=True, type=str)
    args = parser.parse_args()

    config = load_config()
    metadata = load_model_metadata()
    num_classes = metadata.get("num_classes", len(metadata.get("class_names", config.get("class_names", []))) or 2)
    class_names = metadata.get("class_names", config.get("class_names", ["non-fracture", "fracture"]))
    non_fracture_index = next(
        (index for index, name in enumerate(class_names) if normalize_label_name(name) == "non-fracture"),
        0,
    )

    os.makedirs(os.path.join("outputs", "gradcam"), exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = FractureClassifier(
        model_name=config["model"],
        pretrained=False,
        num_classes=num_classes,
        dropout=config.get("classifier_dropout", 0.45),
    )
    model.load_state_dict(torch.load(os.path.join("outputs", "models", "best_model.pth"), map_location=device))
    model.to(device)
    model.eval()

    cam_image = generate_gradcam_overlay(model, args.image, config["image_size"], device, non_fracture_index)
    output_path = os.path.join(
        "outputs",
        "gradcam",
        f"gradcam_{os.path.splitext(os.path.basename(args.image))[0]}.png",
    )
    cv2.imwrite(output_path, cv2.cvtColor(cam_image, cv2.COLOR_RGB2BGR))
    print(f"Grad-CAM overlay saved to {output_path}")


if __name__ == "__main__":
    main()
