import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import models, transforms
import os
import numpy as np
from pathlib import Path
import cv2
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, classification_report, confusion_matrix
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

class CNNModel(nn.Module):
    def __init__(self):
        super(CNNModel, self).__init__()
        
        # Reduced number of filters and added batch normalization
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 28 * 28, 256),  # Reduced from 512 to 256
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 2),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

def train_model(model, train_loader, val_loader, num_epochs=10, 
                unfreeze_backbone_epoch=3, learning_rate=1e-3, backbone_lr=1e-4):
    """Train the CNN model."""
    # Move model to device
    model = model.to(device)
    
    # Loss function
    criterion = nn.CrossEntropyLoss()
    
    # Initially only train the classifier head
    optimizer = optim.Adam(model.classifier.parameters(), lr=learning_rate)
    
    # Initialize variables to track best model
    best_val_auc = 0.0
    best_model_path = "models/best_cnn_model.pth"
    os.makedirs("models", exist_ok=True)
    
    # Create results directory
    os.makedirs("results", exist_ok=True)
    
    # Create lists to store metrics
    train_losses = []
    train_accs = []
    val_losses = []
    val_accs = []
    val_aucs = []
    
    # Gradient accumulation steps
    accumulation_steps = 4  # Accumulate gradients for 4 steps
    
    for epoch in range(num_epochs):
        # Unfreeze backbone with reduced learning rate after specified epoch
        if epoch == unfreeze_backbone_epoch:
            logger.info("Unfreezing backbone layers with reduced learning rate")
            for param in model.parameters():
                param.requires_grad = True
            
            # Update optimizer with all parameters and reduced learning rate for backbone
            optimizer = optim.Adam([
                {'params': model.classifier.parameters(), 'lr': learning_rate},
                {'params': model.features.parameters(), 'lr': backbone_lr}
            ])
        
        # Training phase
        model.train()
        train_loss = 0.0
        correct = 0
        total = 0
        optimizer.zero_grad()  # Zero gradients at the start of epoch
        
        for i, (images, labels) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Train]")):
            # Move data to device
            images = images.to(device)
            labels = labels.to(device)
            
            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            # Scale loss by accumulation steps
            loss = loss / accumulation_steps
            
            # Backward pass
            loss.backward()
            
            # Update weights if we've accumulated enough gradients
            if (i + 1) % accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()
            
            # Update metrics
            train_loss += loss.item() * images.size(0) * accumulation_steps
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            # Clear cache periodically
            if i % 10 == 0:
                torch.cuda.empty_cache()
        
        # Calculate average metrics for the epoch
        train_loss = train_loss / len(train_loader.dataset)
        train_acc = correct / total
        train_losses.append(train_loss)
        train_accs.append(train_acc)
        
        logger.info(f"Epoch {epoch+1}/{num_epochs} - Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
        
        # Validation phase
        val_loss, val_acc, val_auc, _, _ = evaluate_model(model, val_loader, criterion)
        val_losses.append(val_loss)
        val_accs.append(val_acc)
        val_aucs.append(val_auc)
        
        logger.info(f"Epoch {epoch+1}/{num_epochs} - Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}, Val AUC: {val_auc:.4f}")
        
        # Save model if validation AUC improves
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            torch.save(model.state_dict(), best_model_path)
            logger.info(f"Saved new best model with Val AUC: {val_auc:.4f}")
        
        # Clear cache after each epoch
        torch.cuda.empty_cache()
    
    # Plot training and validation metrics
    plt.figure(figsize=(12, 8))
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(train_accs, label='Train Accuracy')
    plt.plot(val_accs, label='Val Accuracy')
    plt.plot(val_aucs, label='Val AUC')
    plt.xlabel('Epoch')
    plt.ylabel('Metric')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig("results/training_metrics.png")
    plt.close()
    
    # Save training history
    history = {
        'train_loss': train_losses,
        'train_acc': train_accs,
        'val_loss': val_losses,
        'val_acc': val_accs,
        'val_auc': val_aucs
    }
    
    with open("results/training_history.json", "w") as f:
        json.dump(history, f)
    
    return best_model_path

def evaluate_model(model, data_loader, criterion=None):
    """Evaluate the model on the given data loader."""
    model.eval()
    
    # Initialize variables
    if criterion is not None:
        running_loss = 0.0
    correct = 0
    total = 0
    
    # For computing ROC curve
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        for images, labels in data_loader:
            # Move data to device
            images = images.to(device)
            labels = labels.to(device)
            
            # Forward pass
            outputs = model(images)
            
            # Compute loss if criterion is provided
            if criterion is not None:
                loss = criterion(outputs, labels)
                running_loss += loss.item() * images.size(0)
            
            # Get predictions
            probs = torch.softmax(outputs, dim=1)
            _, predicted = torch.max(outputs, 1)
            
            # Update metrics
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            # Store labels and probabilities for ROC curve
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs[:, 1].cpu().numpy())  # Probability of being real (class 1)
    
    # Calculate accuracy
    accuracy = correct / total
    
    # Calculate AUC
    fpr, tpr, _ = roc_curve(all_labels, all_probs)
    roc_auc = auc(fpr, tpr)
    
    # Calculate loss if criterion is provided
    if criterion is not None:
        loss = running_loss / len(data_loader.dataset)
        return loss, accuracy, roc_auc, fpr, tpr
    else:
        return accuracy, roc_auc, fpr, tpr

def test_model(model_path, test_loader):
    """Test the model on the test set."""
    # Load best model
    model = CNNModel()
    model.load_state_dict(torch.load(model_path))
    model.to(device)
    model.eval()
    
    # Evaluate on test set
    accuracy, roc_auc, fpr, tpr = evaluate_model(model, test_loader)
    
    # Get predictions for detailed metrics
    all_labels = []
    all_preds = []
    
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
    
    # Calculate classification report and confusion matrix
    report = classification_report(all_labels, all_preds, target_names=['Fake', 'Real'], output_dict=True)
    conf_matrix = confusion_matrix(all_labels, all_preds)
    
    # Plot ROC curve
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {roc_auc:.4f})')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend(loc='lower right')
    plt.savefig("results/roc_curve.png")
    plt.close()
    
    # Plot confusion matrix
    plt.figure(figsize=(8, 6))
    plt.imshow(conf_matrix, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.colorbar()
    classes = ['Fake', 'Real']
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes)
    plt.yticks(tick_marks, classes)
    
    # Add text annotations
    thresh = conf_matrix.max() / 2.
    for i in range(conf_matrix.shape[0]):
        for j in range(conf_matrix.shape[1]):
            plt.text(j, i, format(conf_matrix[i, j], 'd'),
                    horizontalalignment="center",
                    color="white" if conf_matrix[i, j] > thresh else "black")
    
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.savefig("results/confusion_matrix.png")
    plt.close()
    
    # Save metrics to JSON
    metrics = {
        'accuracy': accuracy,
        'auc': roc_auc,
        'classification_report': report
    }
    
    with open("results/test_metrics.json", "w") as f:
        json.dump(metrics, f, indent=4)
    
    logger.info(f"Test Accuracy: {accuracy:.4f}")
    logger.info(f"Test AUC: {roc_auc:.4f}")
    logger.info(f"Test F1 (Real): {report['Real']['f1-score']:.4f}")
    logger.info(f"Test F1 (Fake): {report['Fake']['f1-score']:.4f}")
    
    return metrics

def main(data_dir="data/processed", batch_size=32, num_epochs=10):
    """Main function to run the CNN training and evaluation."""
    # Create data loaders
    train_loader, val_loader, test_loader = create_data_loaders(data_dir, batch_size)
    
    # Create model
    model = CNNModel()
    logger.info(f"Created model: {model.__class__.__name__}")
    
    # Train model
    logger.info("Starting model training...")
    best_model_path = train_model(model, train_loader, val_loader, num_epochs=num_epochs)
    
    # Test model
    logger.info("Evaluating model on test set...")
    test_metrics = test_model(best_model_path, test_loader)
    
    return test_metrics

if __name__ == "__main__":
    main() 