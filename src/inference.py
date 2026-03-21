import argparse
import csv
import os
import sys

import albumentations as A
import cv2
import torch
import yaml
from albumentations.pytorch import ToTensorV2

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from models.model import FractureClassifier, combine_head_probabilities
from src.pipeline_utils import load_config, normalize_label_name


def load_model_metadata():
    metadata_path = os.path.join("outputs", "models", "model_metadata.yaml")
    if not os.path.exists(metadata_path):
        return {}
    with open(metadata_path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def preprocess_image(image_path, image_size):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    transform = A.Compose(
        [
            A.Resize(image_size, image_size),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ]
    )
    return transform(image=image)["image"].unsqueeze(0)


def main():
    parser = argparse.ArgumentParser(description="Run inference on one X-ray image")
    parser.add_argument("--image", required=True, type=str)
    args = parser.parse_args()

    config = load_config()
    metadata = load_model_metadata()
    class_names = metadata.get("class_names", config.get("class_names", ["non-fracture", "fracture"]))
    num_classes = metadata.get("num_classes", len(class_names))
    non_fracture_index = next(
        (index for index, name in enumerate(class_names) if normalize_label_name(name) == "non-fracture"),
        0,
    )

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

    image_tensor = preprocess_image(args.image, config["image_size"]).to(device)
    with torch.no_grad():
        outputs = model.forward_multitask(image_tensor)
        probabilities = combine_head_probabilities(
            outputs["multi_logits"],
            outputs["binary_logits"],
            non_fracture_index,
        )[0]
        prediction = torch.argmax(probabilities).item()
        confidence = probabilities[prediction].item()

    predicted_class = class_names[prediction]
    os.makedirs(os.path.join("outputs", "predictions"), exist_ok=True)
    csv_path = os.path.join("outputs", "predictions", "sample_predictions.csv")
    if not os.path.exists(csv_path):
        with open(csv_path, "w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(["image_path", "prediction", "confidence"])
    with open(csv_path, "a", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow([args.image, predicted_class, confidence])

    print(f"Predicted class: {predicted_class}")
    print(f"Confidence: {confidence:.4f}")


if __name__ == "__main__":
    main()
