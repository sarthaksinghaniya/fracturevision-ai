import os
import sys
import cv2
import torch
import yaml
import argparse
import csv
import albumentations as A
from albumentations.pytorch import ToTensorV2

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from models.model import FractureClassifier

def preprocess_image(image_path, image_size):
    # Load image
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Define transforms
    transform = A.Compose([
        A.Resize(image_size, image_size),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ])

    # Apply transforms
    transformed = transform(image=image)
    return transformed['image'].unsqueeze(0)  # Add batch dimension

def main():
    parser = argparse.ArgumentParser(description='Run inference on a single X-ray image')
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

    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Initialize model and load weights
    model = FractureClassifier(model_name=config['model'], pretrained=False)
    model.load_state_dict(torch.load('outputs/models/best_model.pth', map_location=device))
    model.to(device)
    model.eval()

    # Preprocess image
    image_tensor = preprocess_image(args.image, config['image_size'])
    image_tensor = image_tensor.to(device)

    # Predict
    with torch.no_grad():
        output = model(image_tensor)
        pred = torch.argmax(output, dim=1).item()
        prob = torch.softmax(output, dim=1)[0, pred].item()

    # Determine class
    if prob > 0.5:
        predicted_class = 'Fractured'
    else:
        predicted_class = 'Not Fractured'

    # Log prediction
    os.makedirs('outputs/predictions', exist_ok=True)
    csv_path = 'outputs/predictions/sample_predictions.csv'
    if not os.path.exists(csv_path):
        with open(csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['image_path', 'prediction', 'confidence'])
    with open(csv_path, 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([args.image, predicted_class, prob])

    # Print result
    print(f"Predicted class: {predicted_class}")
    print(f"Probability: {prob:.4f}")

if __name__ == '__main__':
    main()
