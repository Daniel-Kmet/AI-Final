import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import pandas as pd
import json
import platform
import psutil
import torch_directml
import torchvision
import sklearn
import cv2
from pathlib import Path
import logging
from torch.cuda.amp import GradScaler, autocast
import time
from PIL import Image
import torch.nn.functional as F
from sklearn.metrics import roc_curve, auc, confusion_matrix, classification_report
import matplotlib.gridspec as gridspec

logger = logging.getLogger(__name__)

class TrainingMetricsLogger:
    def __init__(self, model_name, save_dir="results"):
        self.model_name = model_name
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.save_dir = Path(save_dir) / f"{model_name}_{self.timestamp}"
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize metrics storage
        self.train_losses = []
        self.val_losses = []
        self.train_accs = []
        self.val_accs = []
        self.train_roc_aucs = []
        self.val_roc_aucs = []
        self.epoch_times = []
        self.best_val_metric = 0
        self.best_epoch = 0
        
        # Store hardware info
        self.hardware_info = self._get_hardware_info()
        self.framework_versions = self._get_framework_versions()
        
        # Create subdirectories
        (self.save_dir / "batch_viz").mkdir(exist_ok=True)
        (self.save_dir / "feature_viz").mkdir(exist_ok=True)
        (self.save_dir / "curves").mkdir(exist_ok=True)
        
    def _get_hardware_info(self):
        """Get system hardware information."""
        try:
            return {
                "cpu": platform.processor(),
                "ram_total": f"{psutil.virtual_memory().total / (1024**3):.1f}GB",
                "gpu": "AMD Radeon RX 6500 XT (DirectML)",
                "gpu_memory": f"{torch_directml.memory_stats()['total'] / (1024**3):.1f}GB"
            }
        except:
            return {"error": "Could not get complete hardware info"}
            
    def _get_framework_versions(self):
        """Get versions of key frameworks."""
        return {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "torchvision": torchvision.__version__,
            "sklearn": sklearn.__version__,
            "numpy": np.__version__,
            "opencv": cv2.__version__
        }
        
    def log_hyperparameters(self, params):
        """Log hyperparameters to JSON."""
        with open(self.save_dir / "hyperparameters.json", "w") as f:
            json.dump(params, f, indent=4)
            
    def log_epoch(self, epoch, train_loss, val_loss, train_acc, val_acc, 
                  train_roc_auc=None, val_roc_auc=None, epoch_time=None):
        """Log metrics for a single epoch."""
        self.train_losses.append(train_loss)
        self.val_losses.append(val_loss)
        self.train_accs.append(train_acc)
        self.val_accs.append(val_acc)
        if train_roc_auc is not None:
            self.train_roc_aucs.append(train_roc_auc)
        if val_roc_auc is not None:
            self.val_roc_aucs.append(val_roc_auc)
        if epoch_time is not None:
            self.epoch_times.append(epoch_time)
            
        metrics = {
            "epoch": epoch,
            "train_loss": f"{train_loss:.4f}",
            "val_loss": f"{val_loss:.4f}",
            "train_acc": f"{train_acc:.4f}",
            "val_acc": f"{val_acc:.4f}"
        }
        if train_roc_auc is not None:
            metrics["train_roc_auc"] = f"{train_roc_auc:.4f}"
        if val_roc_auc is not None:
            metrics["val_roc_auc"] = f"{val_roc_auc:.4f}"
        if epoch_time is not None:
            metrics["epoch_time"] = f"{epoch_time:.2f}s"
            
        # Log to file
        with open(self.save_dir / "epoch_logs.jsonl", "a") as f:
            f.write(json.dumps(metrics) + "\n")
            
    def plot_training_curves(self):
        """Plot comprehensive training curves."""
        fig = plt.figure(figsize=(15, 10))
        gs = gridspec.GridSpec(2, 2)
        
        # Loss curves
        ax1 = fig.add_subplot(gs[0, 0])
        ax1.plot(self.train_losses, label='Train Loss')
        ax1.plot(self.val_losses, label='Val Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.set_title('Training and Validation Loss')
        ax1.legend()
        
        # Accuracy curves
        ax2 = fig.add_subplot(gs[0, 1])
        ax2.plot(self.train_accs, label='Train Acc')
        ax2.plot(self.val_accs, label='Val Acc')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy')
        ax2.set_title('Training and Validation Accuracy')
        ax2.legend()
        
        # ROC-AUC curves if available
        if self.train_roc_aucs and self.val_roc_aucs:
            ax3 = fig.add_subplot(gs[1, 0])
            ax3.plot(self.train_roc_aucs, label='Train ROC-AUC')
            ax3.plot(self.val_roc_aucs, label='Val ROC-AUC')
            ax3.set_xlabel('Epoch')
            ax3.set_ylabel('ROC-AUC')
            ax3.set_title('Training and Validation ROC-AUC')
            ax3.legend()
        
        # Epoch time if available
        if self.epoch_times:
            ax4 = fig.add_subplot(gs[1, 1])
            ax4.plot(self.epoch_times, label='Epoch Time')
            ax4.set_xlabel('Epoch')
            ax4.set_ylabel('Time (s)')
            ax4.set_title('Epoch Training Time')
            ax4.legend()
            
        plt.tight_layout()
        plt.savefig(self.save_dir / "curves" / "training_curves.png")
        plt.close()
        
    def visualize_batch(self, images, labels, predictions=None, reconstructions=None):
        """Visualize a batch of images with optional predictions/reconstructions."""
        n = min(16, len(images))  # Show up to 16 images
        fig = plt.figure(figsize=(15, 10))
        
        for i in range(n):
            if reconstructions is not None:
                # Show original and reconstruction side by side
                plt.subplot(4, 8, 2*i + 1)
                plt.imshow(self._tensor_to_img(images[i]))
                plt.title(f"Original ({['Fake', 'Real'][labels[i]]})")
                plt.axis('off')
                
                plt.subplot(4, 8, 2*i + 2)
                plt.imshow(self._tensor_to_img(reconstructions[i]))
                plt.title("Reconstruction")
                plt.axis('off')
            else:
                # Show single image with prediction
                plt.subplot(4, 4, i + 1)
                plt.imshow(self._tensor_to_img(images[i]))
                title = f"True: {['Fake', 'Real'][labels[i]]}"
                if predictions is not None:
                    title += f"\nPred: {['Fake', 'Real'][predictions[i]]}"
                plt.title(title)
                plt.axis('off')
                
        plt.tight_layout()
        plt.savefig(self.save_dir / "batch_viz" / f"batch_{len(os.listdir(self.save_dir / 'batch_viz'))}.png")
        plt.close()
        
    def visualize_lbp_features(self, image, lbp_map, histogram):
        """Visualize LBP features for an image."""
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
        
        # Original image
        ax1.imshow(image)
        ax1.set_title("Original Image")
        ax1.axis('off')
        
        # LBP map
        ax2.imshow(lbp_map, cmap='gray')
        ax2.set_title("LBP Map")
        ax2.axis('off')
        
        # Histogram
        ax3.bar(range(len(histogram)), histogram)
        ax3.set_title("LBP Histogram")
        ax3.set_xlabel("LBP Value")
        ax3.set_ylabel("Frequency")
        
        plt.tight_layout()
        plt.savefig(self.save_dir / "feature_viz" / f"lbp_{len(os.listdir(self.save_dir / 'feature_viz'))}.png")
        plt.close()
        
    def plot_reconstruction_error_dist(self, real_errors, fake_errors):
        """Plot distribution of reconstruction errors."""
        plt.figure(figsize=(10, 6))
        plt.hist(real_errors, bins=50, alpha=0.5, label='Real', density=True)
        plt.hist(fake_errors, bins=50, alpha=0.5, label='Fake', density=True)
        plt.xlabel('Reconstruction Error')
        plt.ylabel('Density')
        plt.title('Distribution of Reconstruction Errors')
        plt.legend()
        plt.savefig(self.save_dir / "feature_viz" / "reconstruction_errors.png")
        plt.close()
        
    def plot_grad_cam(self, model, image, label, layer_name):
        """Generate and plot Grad-CAM visualization."""
        model.eval()
        img_tensor = torch.FloatTensor(image).unsqueeze(0)
        
        # Get the feature maps from the specified layer
        feature_maps = None
        def hook_fn(module, input, output):
            nonlocal feature_maps
            feature_maps = output
            
        # Register hook
        for name, module in model.named_modules():
            if name == layer_name:
                handle = module.register_forward_hook(hook_fn)
                break
                
        # Get model output and gradients
        output = model(img_tensor)
        output[:, label].backward()
        
        # Get gradients
        gradients = torch.autograd.grad(output[:, label], feature_maps)[0]
        
        # Calculate Grad-CAM
        pooled_gradients = torch.mean(gradients, dim=[2, 3])
        for i in range(feature_maps.size()[1]):
            feature_maps[:, i, :, :] *= pooled_gradients[0, i]
        
        heatmap = torch.mean(feature_maps, dim=1).squeeze()
        heatmap = F.relu(heatmap)
        heatmap /= torch.max(heatmap)
        
        # Convert to numpy and resize
        heatmap = heatmap.detach().numpy()
        heatmap = cv2.resize(heatmap, (image.shape[1], image.shape[0]))
        
        # Create visualization
        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1)
        plt.imshow(self._tensor_to_img(image))
        plt.title("Original Image")
        plt.axis('off')
        
        plt.subplot(1, 2, 2)
        plt.imshow(self._tensor_to_img(image))
        plt.imshow(heatmap, alpha=0.5, cmap='jet')
        plt.title("Grad-CAM")
        plt.axis('off')
        
        plt.tight_layout()
        plt.savefig(self.save_dir / "feature_viz" / f"gradcam_{len(os.listdir(self.save_dir / 'feature_viz'))}.png")
        plt.close()
        
        handle.remove()
        
    def _tensor_to_img(self, tensor):
        """Convert tensor to numpy image."""
        if torch.is_tensor(tensor):
            tensor = tensor.cpu().detach().numpy()
        if tensor.shape[0] == 3:  # CHW to HWC
            tensor = tensor.transpose(1, 2, 0)
        return tensor
        
    def generate_final_report(self, test_metrics, confusion_matrices, roc_curves):
        """Generate comprehensive final report."""
        report = {
            "model_name": self.model_name,
            "timestamp": self.timestamp,
            "hardware_info": self.hardware_info,
            "framework_versions": self.framework_versions,
            "training_summary": {
                "total_epochs": len(self.train_losses),
                "best_epoch": self.best_epoch,
                "best_val_metric": float(self.best_val_metric),
                "total_training_time": sum(self.epoch_times) if self.epoch_times else None,
                "avg_epoch_time": np.mean(self.epoch_times) if self.epoch_times else None
            },
            "test_metrics": test_metrics
        }
        
        # Save report
        with open(self.save_dir / "final_report.json", "w") as f:
            json.dump(report, f, indent=4)
            
        # Plot final ROC curves
        self.plot_final_roc_curves(roc_curves)
        
        # Plot confusion matrices
        self.plot_confusion_matrices(confusion_matrices)
        
        return report
        
    def plot_final_roc_curves(self, roc_curves):
        """Plot ROC curves for all models."""
        plt.figure(figsize=(10, 8))
        for name, (fpr, tpr, roc_auc) in roc_curves.items():
            plt.plot(fpr, tpr, label=f'{name} (AUC = {roc_auc:.3f})')
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curves')
        plt.legend(loc="lower right")
        plt.savefig(self.save_dir / "curves" / "final_roc_curves.png")
        plt.close()
        
    def plot_confusion_matrices(self, confusion_matrices):
        """Plot confusion matrices for all models."""
        n_models = len(confusion_matrices)
        fig, axes = plt.subplots(1, n_models, figsize=(5*n_models, 4))
        if n_models == 1:
            axes = [axes]
            
        for ax, (name, cm) in zip(axes, confusion_matrices.items()):
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                       xticklabels=['Fake', 'Real'],
                       yticklabels=['Fake', 'Real'])
            ax.set_title(f'{name} Confusion Matrix')
            ax.set_ylabel('True Label')
            ax.set_xlabel('Predicted Label')
            
        plt.tight_layout()
        plt.savefig(self.save_dir / "curves" / "confusion_matrices.png")
        plt.close() 