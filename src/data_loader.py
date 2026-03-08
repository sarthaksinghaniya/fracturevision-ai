import os
import cv2
import torch
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np

class FractureDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        self.image_paths = []
        self.labels = []

        # Check format
        if os.path.exists(os.path.join(data_dir, 'fractured')):  # Folder structure
            for label, class_name in enumerate(['not fractured', 'fractured']):
                class_dir = os.path.join(data_dir, class_name)
                if os.path.exists(class_dir):
                    for img_name in os.listdir(class_dir):
                        if img_name.endswith(('.png', '.jpg', '.jpeg')):
                            self.image_paths.append(os.path.join(class_dir, img_name))
                            self.labels.append(label)
        else:  # YOLO structure
            images_dir = os.path.join(data_dir, 'images')
            labels_dir = os.path.join(data_dir, 'labels')
            if os.path.exists(images_dir) and os.path.exists(labels_dir):
                for img_name in os.listdir(images_dir):
                    if img_name.endswith(('.png', '.jpg', '.jpeg')):
                        img_path = os.path.join(images_dir, img_name)
                        label_file = os.path.join(labels_dir, os.path.splitext(img_name)[0] + '.txt')
                        if os.path.exists(label_file):
                            with open(label_file, 'r') as f:
                                lines = f.readlines()
                                if lines:
                                    # Take the first detection's class
                                    class_id = int(lines[0].split()[0])
                                    # Assume class 4 'humerus' is not fractured, others are
                                    label = 0 if class_id == 4 else 1
                                    self.image_paths.append(img_path)
                                    self.labels.append(label)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]

        # Load image
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if self.transform:
            image = self.transform(image=image)['image']

        return image, torch.tensor(label, dtype=torch.float32)

def get_transforms(image_size, is_train=True):
    if is_train:
        return A.Compose([
            A.Resize(image_size, image_size),
            A.Rotate(limit=15, p=0.5),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.GaussianBlur(blur_limit=3, p=0.2),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ])
    else:
        return A.Compose([
            A.Resize(image_size, image_size),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ])

def get_dataloaders(config):
    # Use both datasets
    data_dirs = ["./datasets/BoneFractureYolo8/", "./datasets/bone fracture detection.v4-v4.yolov8/"]

    train_transforms = get_transforms(config['image_size'], is_train=True)
    val_test_transforms = get_transforms(config['image_size'], is_train=False)

    train_datasets = []
    val_datasets = []
    test_datasets = []

    for data_dir in data_dirs:
        train_dir = os.path.join(data_dir, 'train')
        val_dir = os.path.join(data_dir, 'valid')  # YOLO uses 'valid'
        test_dir = os.path.join(data_dir, 'test')

        if os.path.exists(train_dir):
            train_datasets.append(FractureDataset(train_dir, transform=train_transforms))
        if os.path.exists(val_dir):
            val_datasets.append(FractureDataset(val_dir, transform=val_test_transforms))
        if os.path.exists(test_dir):
            test_datasets.append(FractureDataset(test_dir, transform=val_test_transforms))

    # Concatenate if multiple
    from torch.utils.data import ConcatDataset
    train_dataset = ConcatDataset(train_datasets) if train_datasets else []
    val_dataset = ConcatDataset(val_datasets) if val_datasets else []
    test_dataset = ConcatDataset(test_datasets) if test_datasets else []

    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=False, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=config['batch_size'], shuffle=False, num_workers=4)

    return train_loader, val_loader, test_loader
