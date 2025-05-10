import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
import os
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, classification_report, confusion_matrix, precision_recall_fscore_support, accuracy_score
import json
import logging
from tqdm import tqdm
import random
from PIL import Image
import torch_directml

from src.data.dataset import FaceDataset, create_data_loaders
from src.utils.device import get_device

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

device = get_device()
logger.info(f"Using device: {device}")

class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 3, kernel_size=2, stride=2),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded, encoded

def train_autoencoder(model, train_loader, val_loader, test_loader, mixed_dataset,
                     num_epochs=50, learning_rate=1e-3, img_size=128):
    """Train the convolutional autoencoder."""
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # Initialize variables to track best model
    best_val_loss = float('inf')
    best_model_path = "models/best_autoencoder.pth"
    os.makedirs("models", exist_ok=True)
    os.makedirs("results", exist_ok=True)
    
    # Create lists to store metrics
    train_losses = []
    val_losses = []
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        
        for images, _ in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Train]"):
            images = images.to(device)
            
            # Forward pass
            outputs, _ = model(images)
            loss = criterion(outputs, images)
            
            # Backward pass and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Update metrics
            train_loss += loss.item() * images.size(0)
        
        # Calculate average training loss
        train_loss = train_loss / len(train_loader.dataset)
        train_losses.append(train_loss)
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        
        with torch.no_grad():
            for images, _ in tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Val]"):
                images = images.to(device)
                outputs, _ = model(images)
                loss = criterion(outputs, images)
                val_loss += loss.item() * images.size(0)
        
        val_loss = val_loss / len(val_loader.sampler)
        val_losses.append(val_loss)
        
        logger.info(f"Epoch {epoch+1}/{num_epochs} - Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")
        
        # Save model if validation loss improves
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), best_model_path)
            logger.info(f"Saved new best model with Val Loss: {val_loss:.6f}")
        
        # Visualize reconstructions every 10 epochs
        if (epoch + 1) % 10 == 0 or epoch == 0:
            visualize_reconstructions(
                model, val_loader, n=8, 
                save_path=f"results/reconstructions_epoch_{epoch+1}.png"
            )
    
    # Plot training and validation losses
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training and Validation Losses')
    plt.savefig("results/autoencoder_training_loss.png")
    plt.close()
    
    # Load best model
    model.load_state_dict(torch.load(best_model_path))
    
    # Determine anomaly threshold on validation set
    threshold = determine_threshold(model, val_loader)
    
    # Evaluate on test set
    test_metrics = evaluate_autoencoder(model, test_loader, threshold)
    
    # Visualize reconstructions for the final model
    visualize_reconstructions(model, test_loader, n=8, save_path="results/final_reconstructions.png")
    
    # Visualize error histograms
    visualize_error_histograms(model, mixed_dataset, test_loader.sampler.indices)
    
    # Save threshold
    with open("models/anomaly_threshold.json", "w") as f:
        json.dump({"threshold": float(threshold)}, f)
    
    return best_model_path, threshold, test_metrics

def visualize_reconstructions(model, data_loader, n=8, save_path="results/reconstructions.png"):
    """Visualize original and reconstructed images."""
    model.eval()
    
    # Get a batch of images
    images, _ = next(iter(data_loader))
    images = images[:n].to(device)
    
    # Get reconstructions
    with torch.no_grad():
        reconstructed, _ = model(images)
    
    # Plot original and reconstructed images
    plt.figure(figsize=(2*n, 4))
    for i in range(n):
        # Original
        plt.subplot(2, n, i+1)
        plt.imshow(images[i].cpu().permute(1, 2, 0))
        plt.title("Original")
        plt.axis("off")
        
        # Reconstructed
        plt.subplot(2, n, i+n+1)
        plt.imshow(reconstructed[i].cpu().permute(1, 2, 0))
        plt.title("Reconstructed")
        plt.axis("off")
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    logger.info(f"Reconstructions saved to {save_path}")

def determine_threshold(model, val_loader):
    """Determine anomaly threshold using validation set."""
    model.eval()
    reconstruction_errors = []
    labels = []
    
    with torch.no_grad():
        for images, batch_labels in tqdm(val_loader, desc="Computing reconstruction errors"):
            images = images.to(device)
            reconstructed, _ = model(images)
            errors = ((reconstructed - images) ** 2).mean(dim=[1,2,3])
            reconstruction_errors.extend(errors.cpu().numpy())
            labels.extend(batch_labels.numpy())
    
    reconstruction_errors = np.array(reconstruction_errors)
    labels = np.array(labels)
    
    # Find threshold that maximizes F1 score
    best_f1 = 0
    best_threshold = 0
    
    for threshold in np.percentile(reconstruction_errors, np.linspace(0, 100, 100)):
        predictions = (reconstruction_errors > threshold).astype(int)
        f1 = precision_recall_fscore_support(labels, predictions, average='binary')[2]
        
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = threshold
    
    logger.info(f"Selected threshold: {best_threshold:.6f} (F1: {best_f1:.4f})")
    return best_threshold

def evaluate_autoencoder(model, test_loader, threshold):
    """Evaluate autoencoder on test set."""
    model.eval()
    reconstruction_errors = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc="Evaluating"):
            images = images.to(device)
            reconstructed, _ = model(images)
            errors = ((reconstructed - images) ** 2).mean(dim=[1,2,3])
            reconstruction_errors.extend(errors.cpu().numpy())
            all_labels.extend(labels.numpy())
    
    reconstruction_errors = np.array(reconstruction_errors)
    all_labels = np.array(all_labels)
    
    # Convert reconstruction errors to predictions
    predictions = (reconstruction_errors > threshold).astype(int)
    
    # Calculate metrics
    accuracy = accuracy_score(all_labels, predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(all_labels, predictions, average='binary')
    fpr, tpr, _ = roc_curve(all_labels, reconstruction_errors)
    roc_auc = auc(fpr, tpr)
    
    # Print classification report
    logger.info("\nClassification Report:")
    logger.info(classification_report(all_labels, predictions))
    
    # Plot ROC curve
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    plt.savefig("results/roc_curve.png")
    plt.close()
    
    # Plot confusion matrix
    cm = confusion_matrix(all_labels, predictions)
    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.colorbar()
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.savefig("results/confusion_matrix.png")
    plt.close()
    
    metrics = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'auc': roc_auc,
        'threshold': threshold
    }
    
    return metrics

def visualize_error_histograms(model, dataset, indices=None):
    """Visualize error histograms for real and fake images."""
    model.eval()
    real_errors = []
    fake_errors = []
    
    # If indices are not provided, use all samples
    if indices is None:
        indices = range(len(dataset))
    
    with torch.no_grad():
        for idx in indices:
            image, label = dataset[idx]
            image = image.unsqueeze(0).to(device)
            
            # Get reconstruction
            reconstruction = model(image)
            
            # Calculate reconstruction error
            error = torch.mean((image - reconstruction) ** 2).item()
            
            # Store error based on label
            if label == 0:  # Fake
                fake_errors.append(error)
            else:  # Real
                real_errors.append(error)
    
    # Create histograms
    plt.figure(figsize=(10, 6))
    plt.hist(real_errors, bins=50, alpha=0.5, label='Real Images', color='blue')
    plt.hist(fake_errors, bins=50, alpha=0.5, label='Fake Images', color='red')
    plt.xlabel('Reconstruction Error')
    plt.ylabel('Count')
    plt.title('Distribution of Reconstruction Errors')
    plt.legend()
    plt.savefig('results/error_histograms.png')
    plt.close()

def main(data_dir="data/processed", batch_size=32, num_epochs=50, img_size=128):
    """Main function to run the autoencoder training and evaluation."""
    # Create data loaders
    train_loader, val_loader, test_loader = create_data_loaders(
        data_dir, batch_size=batch_size, img_size=img_size, real_only_train=True
    )
    
    # Create model
    model = Autoencoder()
    model.to(device)
    logger.info(f"Created autoencoder model")
    
    # Train and evaluate model
    logger.info("Starting autoencoder training...")
    model_path, threshold, metrics = train_autoencoder(
        model, train_loader, val_loader, test_loader, None,
        num_epochs=num_epochs, img_size=img_size
    )
    
    logger.info(f"Training complete. Best model saved to {model_path}")
    logger.info(f"Final metrics: Accuracy={metrics['accuracy']:.4f}, AUC={metrics['auc']:.4f}")
    
    return model_path, threshold, metrics

if __name__ == "__main__":
    main() 