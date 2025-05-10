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
from src.utils.device import clear_memory, memory_status
from src.data.dataset import create_data_loaders
from src.models.architectures.cnn_model import CNNModel

logger = logging.getLogger(__name__)

def log_memory_usage():
    """Log current memory usage."""
    status = memory_status()
    if status:
        logger.info(f"Memory Usage: {status['usage_percent']:.1f}% "
                   f"(Used: {status['used_gb']:.1f}GB, "
                   f"Available: {status['available_gb']:.1f}GB)")

def train_epoch(model, train_loader, criterion, optimizer, device, scaler):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []
    
    for i, (images, labels) in enumerate(tqdm(train_loader, desc="Training")):
        images = images.to(device)
        labels = labels.to(device).float()
        
        optimizer.zero_grad()
        
        with autocast():
            outputs = model(images)
            loss = criterion(outputs, labels.unsqueeze(1))
        
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        total_loss += loss.item()
        predicted = (outputs > 0.5).float()
        total += labels.size(0)
        correct += (predicted.squeeze() == labels).sum().item()
        
        all_preds.extend(predicted.cpu().numpy().squeeze())
        all_labels.extend(labels.cpu().numpy())
        
        # Clear memory periodically
        if (i + 1) % 10 == 0:
            clear_memory()
            log_memory_usage()
    
    accuracy = correct / total
    avg_loss = total_loss / len(train_loader)
    return avg_loss, accuracy, np.array(all_preds), np.array(all_labels)

def validate(model, val_loader, criterion, device):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        for i, (images, labels) in enumerate(tqdm(val_loader, desc="Validating")):
            images = images.to(device)
            labels = labels.to(device).float()
            
            outputs = model(images)
            loss = criterion(outputs, labels.unsqueeze(1))
            
            total_loss += loss.item()
            predicted = (outputs > 0.5).float()
            total += labels.size(0)
            correct += (predicted.squeeze() == labels).sum().item()
            
            all_preds.extend(predicted.cpu().numpy().squeeze())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(outputs.cpu().numpy().squeeze())
            
            # Clear memory periodically
            if (i + 1) % 10 == 0:
                clear_memory()
                log_memory_usage()
    
    accuracy = correct / total
    avg_loss = total_loss / len(val_loader)
    return avg_loss, accuracy, np.array(all_preds), np.array(all_labels), np.array(all_probs)

def main(data_dir="data/processed", batch_size=32, num_epochs=10, learning_rate=0.001):
    """Main function to run CNN training and evaluation."""
    # Initialize reporter
    reporter = TrainingReporter("cnn")
    
    # Log hyperparameters
    hyperparams = {
        "batch_size": batch_size,
        "num_epochs": num_epochs,
        "learning_rate": learning_rate,
        "optimizer": "Adam",
        "loss_function": "BCELoss",
        "architecture": {
            "conv_layers": [
                {"filters": 32, "kernel_size": 3, "padding": 1},
                {"filters": 64, "kernel_size": 3, "padding": 1},
                {"filters": 128, "kernel_size": 3, "padding": 1}
            ],
            "fc_layers": [512, 1],
            "dropout_rate": 0.5
        }
    }
    reporter.log_hyperparameters(hyperparams)
    
    # Set device and log initial memory status
    device = get_device()
    log_memory_usage()
    
    # Create data loaders with smaller batch size if needed
    if memory_status() and memory_status()['usage_percent'] > 50:
        logger.warning("High memory usage detected, reducing batch size")
        batch_size = batch_size // 2
    
    train_loader, val_loader, test_loader = create_data_loaders(data_dir, batch_size)
    
    # Initialize model
    model = CNNModel().to(device)
    
    # Loss function and optimizer
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scaler = GradScaler()
    
    # Training loop
    best_val_accuracy = 0
    best_model_path = None
    
    for epoch in range(num_epochs):
        logger.info(f"\nEpoch {epoch+1}/{num_epochs}")
        log_memory_usage()
        
        # Train
        train_loss, train_acc, train_preds, train_labels = train_epoch(
            model, train_loader, criterion, optimizer, device, scaler)
        
        # Clear memory before validation
        clear_memory()
        
        # Validate
        val_loss, val_acc, val_preds, val_labels, val_probs = validate(
            model, val_loader, criterion, device)
        
        # Log epoch metrics
        reporter.log_epoch(epoch + 1, train_loss, train_acc, val_loss, val_acc)
        
        # Save best model
        if val_acc > best_val_accuracy:
            best_val_accuracy = val_acc
            if best_model_path:
                os.remove(best_model_path)
            best_model_path = reporter.report_dir / f"cnn_model_best.pth"
            torch.save(model.state_dict(), best_model_path)
            logger.info("Saved best model")
        
        # Clear memory after each epoch
        clear_memory()
        log_memory_usage()
    
    # Load best model for testing
    model.load_state_dict(torch.load(best_model_path))
    test_loss, test_acc, test_preds, test_labels, test_probs = validate(
        model, test_loader, criterion, device)
    
    # Generate classification report
    test_metrics, confusion_matrix, roc_curve_data = reporter.generate_classification_report(
        test_labels, test_preds, test_probs)
    
    # Calculate model size
    model_size = os.path.getsize(best_model_path) / (1024 * 1024)  # Convert to MB
    
    # Generate text report
    reporter.generate_text_report(test_metrics, confusion_matrix, roc_curve_data, model_size)
    
    # Generate final report
    final_report = reporter.generate_final_report(test_metrics, model_size)
    
    return final_report

if __name__ == "__main__":
    main() 