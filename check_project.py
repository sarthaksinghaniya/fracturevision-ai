import os
import sys
import torch
import torch.nn as nn
import yaml
import numpy as np

# Add current directory to path if needed
sys.path.insert(0, os.path.dirname(__file__))

from models.model import FractureClassifier
from src.data_loader import get_dataloaders

def check_project():
    print("Starting project check...")

    # 1. Check dataset loading
    print("1. Checking dataset loading...")
    try:
        with open('configs/config.yaml', 'r') as f:
            config = yaml.safe_load(f)

        # Convert config values to proper types
        config['batch_size'] = int(config['batch_size'])
        config['epochs'] = int(config['epochs'])
        config['lr'] = float(config['lr'])
        config['image_size'] = int(config['image_size'])

        train_loader, val_loader, test_loader = get_dataloaders(config)
        # Try to get one batch
        for images, labels in train_loader:
            print(f"   Dataset loaded successfully. Batch shape: {images.shape}")
            break
        print("   PASS")
    except Exception as e:
        print(f"   FAIL: {e}")
        return

    # 2. Check model forward pass
    print("2. Checking model forward pass...")
    try:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = FractureClassifier(model_name=config['model'], pretrained=False)
        model.to(device)
        model.eval()
        with torch.no_grad():
            dummy_input = torch.randn(1, 3, config['image_size'], config['image_size']).to(device)
            output = model(dummy_input)
            print(f"   Model output shape: {output.shape}")
        print("   PASS")
    except Exception as e:
        print(f"   FAIL: {e}")
        return

    # 3. Check single training step
    print("3. Checking single training step...")
    try:
        device = torch.device('cpu')  # Force CPU
        model.to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.0003)  # Hardcode lr
        criterion = nn.BCEWithLogitsLoss()
        # Use dummy data to avoid dataset issues
        dummy_input = torch.randn(config['batch_size'], 3, config['image_size'], config['image_size']).to(device)
        dummy_labels = torch.randint(0, 2, (config['batch_size'],)).float().to(device)
        optimizer.zero_grad()
        outputs = model(dummy_input)
        loss = criterion(outputs.squeeze(), dummy_labels)
        loss.backward()
        optimizer.step()
        print(f"   Training step completed. Loss: {loss.item():.4f}")
    except Exception as e:
        print(f"   FAIL: {e}")
        return

    # 4. Check inference
    print("4. Checking inference...")
    try:
        model.eval()
        with torch.no_grad():
            single_dummy = torch.randn(1, 3, config['image_size'], config['image_size']).to(device)
            output = model(single_dummy)
            prob = torch.sigmoid(output).item()
            pred = 'Fractured' if prob > 0.5 else 'Not Fractured'
            print(f"   Prediction: {pred}, Confidence: {prob:.4f}")
        print("   PASS")
    except Exception as e:
        print(f"   FAIL: {e}")
        return

    print("PROJECT CHECK PASSED")

if __name__ == '__main__':
    check_project()
