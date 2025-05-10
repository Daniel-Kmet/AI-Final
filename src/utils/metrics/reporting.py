import json
import os
from pathlib import Path
import logging
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
import pandas as pd
import seaborn as sns

logger = logging.getLogger(__name__)

class TrainingReporter:
    def __init__(self, model_name):
        self.model_name = model_name
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.report_dir = Path(f"results/{model_name}_{self.timestamp}")
        self.report_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize metrics storage
        self.train_losses = []
        self.val_losses = []
        self.train_accuracies = []
        self.val_accuracies = []
        self.best_val_accuracy = 0
        self.best_epoch = 0
        
        # Create log file
        self.log_file = self.report_dir / "training_log.txt"
        self.setup_file_handler()
        
    def setup_file_handler(self):
        """Setup a file handler for logging."""
        file_handler = logging.FileHandler(self.log_file)
        file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
        logger.addHandler(file_handler)
    
    def log_hyperparameters(self, hyperparams):
        """Log hyperparameters to a JSON file."""
        with open(self.report_dir / "hyperparameters.json", "w") as f:
            json.dump(hyperparams, f, indent=4)
        logger.info(f"Saved hyperparameters to {self.report_dir}/hyperparameters.json")
    
    def log_epoch(self, epoch, train_loss, train_acc, val_loss, val_acc):
        """Log metrics for a single epoch."""
        self.train_losses.append(train_loss)
        self.val_losses.append(val_loss)
        self.train_accuracies.append(train_acc)
        self.val_accuracies.append(val_acc)
        
        if val_acc > self.best_val_accuracy:
            self.best_val_accuracy = val_acc
            self.best_epoch = epoch
            
        logger.info(f"Epoch {epoch}:")
        logger.info(f"  Train Loss: {train_loss:.4f}, Train Accuracy: {train_acc:.4f}")
        logger.info(f"  Val Loss: {val_loss:.4f}, Val Accuracy: {val_acc:.4f}")
    
    def plot_training_curves(self):
        """Plot training and validation curves."""
        plt.figure(figsize=(12, 5))
        
        # Plot losses
        plt.subplot(1, 2, 1)
        plt.plot(self.train_losses, label='Train Loss')
        plt.plot(self.val_losses, label='Val Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training and Validation Loss')
        plt.legend()
        
        # Plot accuracies
        plt.subplot(1, 2, 2)
        plt.plot(self.train_accuracies, label='Train Accuracy')
        plt.plot(self.val_accuracies, label='Val Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.title('Training and Validation Accuracy')
        plt.legend()
        
        plt.tight_layout()
        plt.savefig(self.report_dir / "training_curves.png")
        plt.close()
        
    def plot_confusion_matrix(self, y_true, y_pred, classes=['Fake', 'Real']):
        """Plot confusion matrix."""
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=classes, yticklabels=classes)
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.savefig(self.report_dir / "confusion_matrix.png")
        plt.close()
        
        return cm
        
    def plot_roc_curve(self, y_true, y_prob):
        """Plot ROC curve."""
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        roc_auc = auc(fpr, tpr)
        
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, label=f'ROC curve (AUC = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC) Curve')
        plt.legend(loc="lower right")
        plt.savefig(self.report_dir / "roc_curve.png")
        plt.close()
        
        return roc_auc, fpr, tpr
    
    def generate_classification_report(self, y_true, y_pred, y_prob):
        """Generate and save detailed classification report."""
        # Calculate metrics
        report_dict = classification_report(y_true, y_pred, output_dict=True)
        roc_auc, fpr, tpr = self.plot_roc_curve(y_true, y_prob)
        
        # Add ROC AUC to report
        report_dict['roc_auc'] = roc_auc
        
        # Save report
        with open(self.report_dir / "classification_report.json", "w") as f:
            json.dump(report_dict, f, indent=4)
        
        # Create confusion matrix
        cm = self.plot_confusion_matrix(y_true, y_pred)
        
        return report_dict, cm, (fpr, tpr, roc_auc)
    
    def generate_text_report(self, test_metrics, confusion_matrix, roc_curve_data, model_size_mb=None):
        """Generate a detailed text report."""
        fpr, tpr, roc_auc = roc_curve_data
        
        report = f"""Model Training Report
==================
Model: {self.model_name}
Timestamp: {self.timestamp}

Training Summary
---------------
Best Validation Accuracy: {self.best_val_accuracy:.4f} (Epoch {self.best_epoch})
Final Training Loss: {self.train_losses[-1]:.4f}
Final Validation Loss: {self.val_losses[-1]:.4f}
Total Epochs: {len(self.train_losses)}

Test Results
-----------
Accuracy: {test_metrics['accuracy']:.4f}
ROC AUC: {test_metrics['auc']:.4f}

Detailed Classification Metrics:
{'-' * 30}
Precision:
  - Fake: {test_metrics['classification_report']['Fake']['precision']:.4f}
  - Real: {test_metrics['classification_report']['Real']['precision']:.4f}

Recall:
  - Fake: {test_metrics['classification_report']['Fake']['recall']:.4f}
  - Real: {test_metrics['classification_report']['Real']['recall']:.4f}

F1-Score:
  - Fake: {test_metrics['classification_report']['Fake']['f1-score']:.4f}
  - Real: {test_metrics['classification_report']['Real']['f1-score']:.4f}

Confusion Matrix:
{'-' * 30}
True Negatives (TN): {confusion_matrix[0][0]}
False Positives (FP): {confusion_matrix[0][1]}
False Negatives (FN): {confusion_matrix[1][0]}
True Positives (TP): {confusion_matrix[1][1]}

ROC Curve Analysis:
{'-' * 30}
Area Under Curve (AUC): {roc_auc:.4f}
"""
        
        if model_size_mb is not None:
            report += f"\nModel Size: {model_size_mb:.2f} MB\n"
        
        # Save text report
        with open(self.report_dir / "detailed_report.txt", "w") as f:
            f.write(report)
        
        return report
    
    def generate_final_report(self, test_metrics, model_size_mb=None):
        """Generate final training report with all metrics."""
        report = {
            "model_name": self.model_name,
            "timestamp": self.timestamp,
            "training_summary": {
                "best_validation_accuracy": float(self.best_val_accuracy),
                "best_epoch": self.best_epoch,
                "final_train_loss": float(self.train_losses[-1]),
                "final_val_loss": float(self.val_losses[-1]),
                "total_epochs": len(self.train_losses)
            },
            "test_metrics": test_metrics
        }
        
        if model_size_mb is not None:
            report["model_size_mb"] = model_size_mb
        
        # Save report
        with open(self.report_dir / "final_report.json", "w") as f:
            json.dump(report, f, indent=4)
        
        # Plot training curves
        self.plot_training_curves()
        
        # Log summary
        logger.info("\nTraining Summary:")
        logger.info(f"Best Validation Accuracy: {self.best_val_accuracy:.4f} (Epoch {self.best_epoch})")
        logger.info(f"Test Accuracy: {test_metrics['accuracy']:.4f}")
        if 'roc_auc' in test_metrics:
            logger.info(f"Test ROC AUC: {test_metrics['roc_auc']:.4f}")
        
        return report 