import os
import sys
import cv2
import torch
import yaml
import argparse
import numpy as np
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from models.model import FractureClassifier
from src.inference import preprocess_image  # Reuse the preprocess function

def generate_gradcam_overlay(model, image_path, config, device):
    # Preprocess image
    image_tensor = preprocess_image(image_path, config['image_size'])
    image_tensor = image_tensor.to(device)

    # Get original image for overlay
    original_image = cv2.imread(image_path)
    original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
    original_image = cv2.resize(original_image, (config['image_size'], config['image_size']))
    original_image = original_image.astype(np.float32) / 255.0

    # Get target layer for GradCAM
    target_layer = model.model.conv_head  # For EfficientNet-B0

    # Create GradCAM
    cam = GradCAM(model=model, target_layers=[target_layer])

    # Get prediction
    with torch.no_grad():
        output = model(image_tensor)
        pred_class = torch.argmax(output, dim=1).item()

    # Generate CAM
    targets = [ClassifierOutputTarget(pred_class)]
    grayscale_cam = cam(input_tensor=image_tensor, targets=targets)[0]

    # Overlay CAM on image
    cam_image = show_cam_on_image(original_image, grayscale_cam, use_rgb=True)

    return cam_image

def main():
    parser = argparse.ArgumentParser(description='Generate Grad-CAM heatmap for a single X-ray image')
    parser.add_argument('--image', type=str, required=True, help='Path to the X-ray image')
    args = parser.parse_args()

    # Load config
    with open('configs/config.yaml', 'r') as f:
        config = yaml.safe_load(f)

    # Convert config values to proper types
    config['batch_size'] = int(config['batch_size'])
    config['epochs'] = int(config['epochs'])
    config['lr'] = float(config['lr'])
    config['image_size'] = int(config['image_size'])

    # Create output directories
    os.makedirs('outputs/gradcam', exist_ok=True)

    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Initialize model and load weights
    model = FractureClassifier(model_name=config['model'], pretrained=False)
    model.load_state_dict(torch.load('outputs/models/best_model.pth', map_location=device))
    model.to(device)
    model.eval()

    # Generate GradCAM overlay
    cam_image = generate_gradcam_overlay(model, args.image, config, device)

    # Save image
    image_name = os.path.basename(args.image)
    output_path = f'outputs/gradcam/gradcam_{os.path.splitext(image_name)[0]}.png'
    cv2.imwrite(output_path, cv2.cvtColor(cam_image, cv2.COLOR_RGB2BGR))

    print(f"Grad-CAM overlay saved to {output_path}")

if __name__ == '__main__':
    main()
