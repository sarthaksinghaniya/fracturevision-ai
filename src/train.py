import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import yaml
from tqdm import tqdm
import numpy as np
from torch.cuda.amp import autocast, GradScaler
import csv
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from models.model import FractureClassifier
from src.data_loader import get_dataloaders

class EarlyStopping:
    def __init__(self, patience=5, verbose=False, delta=0, path='outputs/models/best_model.pth'):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.path = path

    def __call__(self, val_loss, model):
        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}). Saving model ...')
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss

def train_epoch(model, dataloader, criterion, optimizer, device, scaler):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in tqdm(dataloader, desc='Training'):
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        if scaler:
            with autocast():
                outputs = model(images)
                loss = criterion(outputs, labels.long())
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(images)
            loss = criterion(outputs, labels.long())
            loss.backward()
            optimizer.step()

        running_loss += loss.item() * images.size(0)
        predicted = torch.argmax(outputs, dim=1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    epoch_loss = running_loss / len(dataloader.dataset)
    epoch_acc = correct / total
    return epoch_loss, epoch_acc

def validate_epoch(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in tqdm(dataloader, desc='Validating'):
            images, labels = images.to(device), labels.to(device)

            if device.type == 'cuda':
                with autocast():
                    outputs = model(images)
                    loss = criterion(outputs, labels.long())
            else:
                outputs = model(images)
                loss = criterion(outputs, labels.long())

            running_loss += loss.item() * images.size(0)
            predicted = torch.argmax(outputs, dim=1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    epoch_loss = running_loss / len(dataloader.dataset)
    epoch_acc = correct / total
    return epoch_loss, epoch_acc

def main():
    # Load config
    with open('configs/config.yaml', 'r') as f:
        config = yaml.safe_load(f)

    # Convert config values to proper types
    config['batch_size'] = int(config['batch_size'])
    config['epochs'] = int(config['epochs'])
    config['lr'] = float(config['lr'])
    config['image_size'] = int(config['image_size'])

    # Create output directories
    os.makedirs('outputs/models', exist_ok=True)
    os.makedirs('outputs/metrics', exist_ok=True)
    os.makedirs('outputs/logs', exist_ok=True)
    os.makedirs('outputs/plots', exist_ok=True)

    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # GradScaler for mixed precision
    scaler = GradScaler() if device.type == 'cuda' else None

    # Initialize logging
    metrics_file = 'outputs/metrics/training_metrics.csv'
    csv_writer = csv.writer(open(metrics_file, 'w', newline=''))
    csv_writer.writerow(['epoch', 'train_loss', 'val_loss', 'train_accuracy', 'val_accuracy', 'learning_rate'])

    log_file = 'outputs/logs/training_log.txt'

    # Lists for plotting
    epochs_list = []
    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []

    # Get dataloaders
    train_loader, val_loader, _ = get_dataloaders(config)

    # Initialize model
    model = FractureClassifier(model_name=config['model'], pretrained=True)
    model.to(device)

    # Optimizer and loss
    optimizer = optim.AdamW(model.parameters(), lr=config['lr'])
    criterion = nn.CrossEntropyLoss()

    # Scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=3, factor=0.5)

    # Early stopping
    early_stopping = EarlyStopping(patience=7, verbose=True)

    # Training loop
    num_epochs = config['epochs']
    for epoch in range(num_epochs):
        print(f'Epoch {epoch+1}/{num_epochs}')

        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device, scaler)
        val_loss, val_acc = validate_epoch(model, val_loader, criterion, device)

        print(f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}')
        print(f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}')

        # Scheduler step
        scheduler.step(val_loss)

        # Log metrics
        lr = optimizer.param_groups[0]['lr']
        epochs_list.append(epoch+1)
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)
        csv_writer.writerow([epoch+1, train_loss, val_loss, train_acc, val_acc, lr])
        with open(log_file, 'a') as f:
            f.write(f'Epoch {epoch+1}/{num_epochs}\n')
            f.write(f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}\n')
            f.write(f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}\n')
            f.write(f'Learning Rate: {lr}\n\n')

        # Early stopping
        early_stopping(val_loss, model)
        if early_stopping.early_stop:
            print("Early stopping")
            break

    # Save last model
    torch.save(model.state_dict(), 'outputs/models/last_model.pth')

    # Generate plots
    plt.figure()
    plt.plot(epochs_list, train_losses, label='Train Loss')
    plt.plot(epochs_list, val_losses, label='Val Loss')
    plt.title('Loss Curve')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig('outputs/plots/loss_curve.png')
    plt.close()

    plt.figure()
    plt.plot(epochs_list, train_accs, label='Train Accuracy')
    plt.plot(epochs_list, val_accs, label='Val Accuracy')
    plt.title('Accuracy Curve')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.savefig('outputs/plots/accuracy_curve.png')
    plt.close()

    print("Training completed")

if __name__ == '__main__':
    main()
