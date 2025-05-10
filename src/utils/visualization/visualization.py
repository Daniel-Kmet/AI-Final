import matplotlib.pyplot as plt
import numpy as np
import torch
import os
import logging
from sklearn.metrics import roc_curve, auc, confusion_matrix

logger = logging.getLogger(__name__)

def visualize_batch(dataloader, n=8):
    """Visualize a batch of images to verify data loading."""
    images, labels = next(iter(dataloader))
    images = images.cpu()
    
    # Denormalize images for visualization
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
    images = images * std + mean
    images = torch.clamp(images, 0, 1)
    
    plt.figure(figsize=(12, 8))
    for i in range(min(n, len(images))):
        plt.subplot(2, n//2, i+1)
        plt.imshow(images[i].permute(1, 2, 0))
        plt.title(f"Label: {'Real' if labels[i] == 1 else 'Fake'}")
        plt.axis("off")
    
    # Save the figure
    os.makedirs("results", exist_ok=True)
    plt.savefig("results/batch_visualization.png")
    plt.close()
    logger.info(f"Batch visualization saved to results/batch_visualization.png")

def plot_training_metrics(train_losses, val_losses, train_accs, val_accs, val_aucs):
    """Plot training and validation metrics."""
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

def plot_roc_curve(fpr, tpr, roc_auc):
    """Plot ROC curve."""
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {roc_auc:.4f})')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend(loc='lower right')
    plt.savefig("results/roc_curve.png")
    plt.close()

def plot_confusion_matrix(conf_matrix):
    """Plot confusion matrix."""
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