import torch
import torchvision.transforms as T
import torchvision.transforms.functional as F
import random
import numpy as np

class RandomRotationWithResize:
    """Random rotation with automatic resizing to prevent black borders"""
    def __init__(self, degrees=10):
        self.degrees = degrees

    def __call__(self, img):
        angle = random.uniform(-self.degrees, self.degrees)
        return F.rotate(img, angle, expand=True)

class ColorJitter:
    """Custom color jitter that applies brightness, contrast, and saturation changes"""
    def __init__(self, brightness=0.2, contrast=0.2, saturation=0.2):
        self.brightness = brightness
        self.contrast = contrast
        self.saturation = saturation

    def __call__(self, img):
        # Brightness
        if random.random() < 0.5:
            brightness_factor = random.uniform(1 - self.brightness, 1 + self.brightness)
            img = F.adjust_brightness(img, brightness_factor)
        
        # Contrast
        if random.random() < 0.5:
            contrast_factor = random.uniform(1 - self.contrast, 1 + self.contrast)
            img = F.adjust_contrast(img, contrast_factor)
        
        # Saturation
        if random.random() < 0.5:
            saturation_factor = random.uniform(1 - self.saturation, 1 + self.saturation)
            img = F.adjust_saturation(img, saturation_factor)
        
        return img

class RandomCropResize:
    """Random crop and resize to simulate zoom in/out"""
    def __init__(self, scale=(0.8, 1.0)):
        self.scale = scale

    def __call__(self, img):
        # Get image dimensions
        h, w = img.shape[-2:]
        
        # Calculate crop size
        scale = random.uniform(*self.scale)
        crop_size = int(min(h, w) * scale)
        
        # Calculate crop coordinates
        top = random.randint(0, h - crop_size)
        left = random.randint(0, w - crop_size)
        
        # Crop and resize back to original size
        img = F.crop(img, top, left, crop_size, crop_size)
        img = F.resize(img, (h, w))
        
        return img

def get_training_transforms():
    """Get the complete set of training transforms"""
    return T.Compose([
        # Convert to tensor and normalize
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        
        # Random horizontal flip
        T.RandomHorizontalFlip(p=0.5),
        
        # Random rotation
        RandomRotationWithResize(degrees=10),
        
        # Color jitter
        ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        
        # Random crop and resize
        RandomCropResize(scale=(0.8, 1.0))
    ])

def get_validation_transforms():
    """Get transforms for validation (only normalization)"""
    return T.Compose([
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

# Example usage:
if __name__ == "__main__":
    # Test the transforms
    from PIL import Image
    import matplotlib.pyplot as plt
    
    # Load a sample image
    img = Image.open("data/processed/train_aligned/fake/fake_0001.jpg")
    
    # Apply training transforms
    train_transform = get_training_transforms()
    augmented_img = train_transform(img)
    
    # Convert back to PIL for visualization
    augmented_img = augmented_img.permute(1, 2, 0).numpy()
    augmented_img = np.clip(augmented_img, 0, 1)
    
    # Display original and augmented images
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(img)
    plt.title("Original")
    plt.subplot(1, 2, 2)
    plt.imshow(augmented_img)
    plt.title("Augmented")
    plt.show() 