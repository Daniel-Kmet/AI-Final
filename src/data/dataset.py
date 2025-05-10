import torch
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler
from torchvision import transforms
import cv2
import numpy as np
from pathlib import Path
import logging
from PIL import Image
import random

logger = logging.getLogger(__name__)

class FaceDataset(Dataset):
    """Dataset for face authenticity detection."""
    def __init__(self, data_dir, transform=None, real_only=False):
        self.data_dir = Path(data_dir)
        self.transform = transform
        
        # Load all image paths and labels
        self.real_dir = self.data_dir / "real"
        self.fake_dir = self.data_dir / "fake"
        
        self.real_images = list(self.real_dir.glob("*.jpg")) + list(self.real_dir.glob("*.png"))
        self.labels = [1] * len(self.real_images)  # 1 for real
        self.image_paths = self.real_images
        
        if not real_only:
            self.fake_images = list(self.fake_dir.glob("*.jpg")) + list(self.fake_dir.glob("*.png"))
            self.image_paths.extend(self.fake_images)
            self.labels.extend([0] * len(self.fake_images))  # 0 for fake
        
        logger.info(f"Loaded {len(self.real_images)} real images and {len(self.fake_images if not real_only else [])} fake images")
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        # Load image using PIL (better for torchvision transforms)
        image = Image.open(str(image_path)).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
            
        label = self.labels[idx]
        return image, label

def create_data_loaders(data_dir, batch_size=32, img_size=224, val_split=0.2, test_split=0.1, real_only_train=False):
    """Create data loaders for training, validation, and testing."""
    # Define transforms
    train_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    val_test_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Create full dataset
    full_dataset = FaceDataset(data_dir, transform=None, real_only=False)
    
    # Create indices for train/val/test split
    dataset_size = len(full_dataset)
    indices = list(range(dataset_size))
    
    # Stratified sampling to maintain class balance
    real_indices = [i for i, label in enumerate(full_dataset.labels) if label == 1]
    fake_indices = [i for i, label in enumerate(full_dataset.labels) if label == 0]
    
    # Shuffle indices
    random.seed(42)
    random.shuffle(real_indices)
    random.shuffle(fake_indices)
    
    # Calculate split sizes
    n_real_test = int(len(real_indices) * test_split)
    n_real_val = int(len(real_indices) * val_split)
    n_fake_test = int(len(fake_indices) * test_split)
    n_fake_val = int(len(fake_indices) * val_split)
    
    # Split indices
    test_indices = real_indices[:n_real_test] + fake_indices[:n_fake_test]
    val_indices = real_indices[n_real_test:n_real_test + n_real_val] + fake_indices[n_fake_test:n_fake_test + n_fake_val]
    train_indices = real_indices[n_real_test + n_real_val:] + fake_indices[n_fake_test + n_fake_val:]
    
    # Create datasets with appropriate transforms
    train_dataset = FaceDataset(data_dir, transform=train_transform, real_only=real_only_train)
    if not real_only_train:
        train_dataset.image_paths = [full_dataset.image_paths[i] for i in train_indices]
        train_dataset.labels = [full_dataset.labels[i] for i in train_indices]
    
    val_dataset = FaceDataset(data_dir, transform=val_test_transform, real_only=False)
    val_dataset.image_paths = [full_dataset.image_paths[i] for i in val_indices]
    val_dataset.labels = [full_dataset.labels[i] for i in val_indices]
    
    test_dataset = FaceDataset(data_dir, transform=val_test_transform, real_only=False)
    test_dataset.image_paths = [full_dataset.image_paths[i] for i in test_indices]
    test_dataset.labels = [full_dataset.labels[i] for i in test_indices]
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    
    return train_loader, val_loader, test_loader 