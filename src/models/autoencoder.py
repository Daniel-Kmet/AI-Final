import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
import logging
from tqdm import tqdm
import os
from pathlib import Path
import numpy as np

from src.utils.metrics.reporting import TrainingReporter
from src.utils.device import get_device
from src.data.dataset import create_data_loaders
from src.models.architectures.autoencoder_model import Autoencoder

logger = logging.getLogger(__name__)

def train_epoch(model, train_loader, criterion, optimizer, device, scaler):
    model.train()
    total_loss = 0
    reconstruction_errors = []
    all_features = []
    all_labels = []
    
    for images, labels in tqdm(train_loader, desc="Training"):
        images = images.to(device)
        
        optimizer.zero_grad()
        
        with autocast():
            reconstructed, encoded = model(images)
            loss = criterion(reconstructed, images)
        
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        total_loss += loss.item() * images.size(0)
        
        # Store reconstruction errors and features
        with torch.no_grad():
            errors = ((reconstructed - images) ** 2).mean(dim=[1,2,3])
            reconstruction_errors.extend(errors.cpu().numpy())
            all_features.extend(encoded.cpu().numpy())
            all_labels.extend(labels.numpy())
    
    avg_loss = total_loss / len(train_loader.dataset)
    return avg_loss, reconstruction_errors, np.array(all_features), np.array(all_labels)

def validate(model, val_loader, criterion, device):
    model.eval()
    total_loss = 0
    reconstruction_errors = []
    all_features = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in tqdm(val_loader, desc="Validating"):
            images = images.to(device)
            reconstructed, encoded = model(images)
            loss = criterion(reconstructed, images)
            
            total_loss += loss.item() * images.size(0)
            
            errors = ((reconstructed - images) ** 2).mean(dim=[1,2,3])
            reconstruction_errors.extend(errors.cpu().numpy())
            all_features.extend(encoded.cpu().numpy())
            all_labels.extend(labels.numpy())
    
    avg_loss = total_loss / len(val_loader.dataset)
    return avg_loss, reconstruction_errors, np.array(all_features), np.array(all_labels)

def main(data_dir="data/processed", batch_size=32, num_epochs=10, learning_rate=0.001):
    """Main function to run autoencoder training and evaluation."""
    
    # Initialize reporter
    reporter = TrainingReporter("autoencoder")
    
    # Log hyperparameters
    hyperparams = {
        "batch_size": batch_size,
        "num_epochs": num_epochs,
        "learning_rate": learning_rate,
        "optimizer": "Adam",
        "loss_function": "MSELoss",
        "architecture": {
            "encoder_layers": [
                {"filters": 32, "kernel_size": 3, "padding": 1},
                {"filters": 64, "kernel_size": 3, "padding": 1},
                {"filters": 128, "kernel_size": 3, "padding": 1}
            ],
            "decoder_layers": [
                {"filters": 64, "kernel_size": 2, "stride": 2},
                {"filters": 32, "kernel_size": 2, "stride": 2},
                {"filters": 3, "kernel_size": 2, "stride": 2}
            ]
        }
    }
    reporter.log_hyperparameters(hyperparams)
    
    # Set device
    device = get_device()
    logger.info(f"Using device: {device}")
    
    # Create data loaders
    train_loader, val_loader, test_loader = create_data_loaders(data_dir, batch_size)
    
    # Initialize model
    model = Autoencoder().to(device)
    
    # Loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scaler = GradScaler()
    
    # Training loop
    best_val_loss = float('inf')
    best_model_path = None
    
    for epoch in range(num_epochs):
        # Train
        train_loss, train_errors = train_epoch(
            model, train_loader, criterion, optimizer, device, scaler)
        
        # Validate
        val_loss, val_errors, val_features, val_labels = validate(
            model, val_loader, criterion, device)
        
        # Calculate accuracy using reconstruction error threshold
        threshold = np.percentile(val_errors, 95)  # Use 95th percentile as threshold
        train_acc = np.mean(train_errors < threshold)
        val_acc = np.mean(val_errors < threshold)
        
        # Log epoch metrics
        reporter.log_epoch(epoch + 1, train_loss, train_acc, val_loss, val_acc)
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            if best_model_path:
                os.remove(best_model_path)
            best_model_path = reporter.report_dir / f"autoencoder_model_best.pth"
            torch.save(model.state_dict(), best_model_path)
            logger.info("Saved best model")
    
    # Load best model for testing
    model.load_state_dict(torch.load(best_model_path))
    test_loss, test_errors, test_features, test_labels = validate(
        model, test_loader, criterion, device)
    
    # Calculate final metrics
    threshold = np.percentile(test_errors, 95)
    predictions = (test_errors < threshold).astype(int)
    
    # Generate classification report
    test_metrics, confusion_matrix, roc_curve_data = reporter.generate_classification_report(
        test_labels, predictions, 1 - test_errors/np.max(test_errors))
    
    # Calculate model size
    model_size = os.path.getsize(best_model_path) / (1024 * 1024)  # Convert to MB
    
    # Generate text report
    reporter.generate_text_report(test_metrics, confusion_matrix, roc_curve_data, model_size)
    
    # Generate final report
    final_report = reporter.generate_final_report(test_metrics, model_size)
    
    return final_report

if __name__ == "__main__":
    main() 