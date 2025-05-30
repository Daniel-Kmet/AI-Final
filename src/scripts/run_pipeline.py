import os
import logging
import json
from pathlib import Path
import torch
import joblib
import datetime
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import confusion_matrix, roc_curve, auc, roc_auc_score, classification_report
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

# Add src directory to Python path
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.utils.device import get_device
from src.models.architectures.cnn_model import main as train_cnn
from src.models.architectures.autoencoder_model import main as train_autoencoder
from src.models.training.train_svm_classifier import main as train_svm
from src.models.cnn import CNNModel
from src.models.autoencoder import Autoencoder
from src.data.dataset import FaceDataset

# Add safe globals for model loading
torch.serialization.add_safe_globals(['torch._utils._rebuild_device_tensor_from_numpy'])

logger = logging.getLogger(__name__)

class EnsembleModel(nn.Module):
    """Ensemble model combining CNN, Autoencoder, and SVM predictions."""
    def __init__(self, cnn_model, autoencoder, svm_model):
        super().__init__()
        self.cnn_model = cnn_model
        self.autoencoder = autoencoder
        self.svm_model = svm_model
        
        # Initialize weights for ensemble
        self.weights = nn.Parameter(torch.ones(3) / 3)  # Equal weights initially
    
    def forward(self, x):
        # Get predictions from each model
        cnn_pred = self.cnn_model(x)
        cnn_score = torch.sigmoid(cnn_pred)
        
        # Get reconstruction error from autoencoder
        recon = self.autoencoder(x)
        ae_error = torch.mean((x - recon) ** 2, dim=[1, 2, 3])
        ae_score = torch.sigmoid(-ae_error)  # Convert error to score
        
        # Get SVM predictions
        x_np = x.cpu().numpy()
        x_flat = x_np.reshape(x_np.shape[0], -1)
        svm_score = torch.tensor(self.svm_model.predict_proba(x_flat)[:, 1], device=x.device)
        
        # Combine predictions using learned weights
        weights = torch.softmax(self.weights, dim=0)  # Ensure weights sum to 1
        combined_score = (
            weights[0] * cnn_score +
            weights[1] * ae_score +
            weights[2] * svm_score
        )
        
        # Get final predictions
        predictions = (combined_score > 0.5).float()
        
        return predictions, combined_score

def setup_logging():
    """Set up logging configuration."""
    log_dir = "logs"
    os.makedirs(log_dir, exist_ok=True)
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(os.path.join(log_dir, 'pipeline.log')),
            logging.StreamHandler()
        ]
    )

def load_config():
    """Load configuration from config.json."""
    config_path = "config.json"
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            return json.load(f)
    else:
        # Default configuration
        return {
            "batch_size": 32,
            "num_epochs": 50,
            "learning_rate": 1e-3,
            "img_size": 128
        }

def create_directories():
    """Create necessary directories if they don't exist."""
    directories = [
        "results",
        "logs",
        "data/processed",
        "data/features",
        "models"
    ]
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)

def plot_confusion_matrix(y_true, y_pred, model_name, save_path):
    """Plot confusion matrix for a model."""
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f'Confusion Matrix - {model_name}')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig(save_path)
    plt.close()

def plot_roc_curve(y_true, y_score, model_name, save_path):
    """Plot ROC curve for a model."""
    fpr, tpr, _ = roc_curve(y_true, y_score)
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve - {model_name}')
    plt.legend(loc="lower right")
    plt.savefig(save_path)
    plt.close()

def plot_metrics_comparison(metrics_dict, save_path):
    """Plot comparison of metrics across models."""
    models = list(metrics_dict.keys())
    metrics = ['accuracy', 'precision', 'recall', 'f1-score']
    
    # Prepare data
    data = []
    for model in models:
        model_metrics = metrics_dict[model]
        # Extract metrics from classification report if available
        if 'classification_report' in model_metrics:
            report = model_metrics['classification_report']
            if isinstance(report, dict) and 'weighted avg' in report:
                data.append([
                    report['weighted avg'].get('precision', 0),
                    report['weighted avg'].get('recall', 0),
                    report['weighted avg'].get('f1-score', 0),
                    report.get('accuracy', 0)
                ])
            else:
                data.append([0, 0, 0, 0])
        else:
            data.append([0, 0, 0, 0])
    
    # Create plot
    plt.figure(figsize=(12, 6))
    x = np.arange(len(metrics))
    width = 0.8 / len(models)
    
    for i, model in enumerate(models):
        plt.bar(x + i * width, data[i], width, label=model)
    
    plt.xlabel('Metrics')
    plt.ylabel('Score')
    plt.title('Model Performance Comparison')
    plt.xticks(x + width * (len(models) - 1) / 2, metrics)
    plt.legend()
    plt.ylim(0, 1)
    plt.savefig(save_path)
    plt.close()

def generate_visualizations(report):
    """Generate visualizations for the final report."""
    # Create visualization directory
    vis_dir = Path("results/visualizations")
    vis_dir.mkdir(parents=True, exist_ok=True)
    
    # Extract metrics for comparison
    metrics_dict = {}
    for model_name, model_data in report["models"].items():
        if "metrics" in model_data and model_data["metrics"]:
            metrics_dict[model_name] = model_data["metrics"]
    
    # Generate comparison plot
    if metrics_dict:
        plot_metrics_comparison(metrics_dict, vis_dir / "model_comparison.png")
        logger.info("Generated model comparison plot")
    else:
        logger.warning("No metrics available for visualization")

def generate_final_report():
    """Generate final report with model metrics."""
    report = {
        "timestamp": str(datetime.datetime.now()),
        "models": {
            "cnn": {
                "path": "models/best_cnn_model.pth",
                "metrics": {}
            },
            "autoencoder": {
                "path": "models/best_autoencoder.pth",
                "threshold_path": "models/anomaly_threshold.json",
                "metrics": {}
            },
            "svm": {
                "path": "models/svm/svm_model.pkl",
                "metrics": {}
            },
            "ensemble": {
                "path": "models/ensemble_model.pth",
                "metrics": {}
            }
        }
    }
    
    # Load CNN metrics
    try:
        with open("results/test_metrics.json", "r") as f:
            report["models"]["cnn"]["metrics"] = json.load(f)
            logger.info("Loaded CNN metrics")
    except Exception as e:
        logger.warning(f"Could not load CNN metrics: {str(e)}")
    
    # Load Autoencoder metrics
    try:
        with open("results/autoencoder_metrics.json", "r") as f:
            report["models"]["autoencoder"]["metrics"] = json.load(f)
            logger.info("Loaded Autoencoder metrics")
    except Exception as e:
        logger.warning(f"Could not load Autoencoder metrics: {str(e)}")
    
    # Load SVM metrics
    try:
        with open("models/svm/test_metrics.json", "r") as f:
            report["models"]["svm"]["metrics"] = json.load(f)
            logger.info("Loaded SVM metrics")
    except Exception as e:
        logger.warning(f"Could not load SVM metrics: {str(e)}")
    
    # Load Ensemble metrics
    try:
        with open("results/ensemble_metrics.json", "r") as f:
            report["models"]["ensemble"]["metrics"] = json.load(f)
            logger.info("Loaded Ensemble metrics")
    except Exception as e:
        logger.warning(f"Could not load Ensemble metrics: {str(e)}")
    
    # Generate visualizations
    generate_visualizations(report)
    
    # Save report
    with open("results/final_report.json", "w") as f:
        json.dump(report, f, indent=4)
    
    logger.info("Final report and visualizations generated successfully!")

def load_model_weights(model, weights_path):
    """Safely load model weights."""
    try:
        # First try loading with weights_only=True
        state_dict = torch.load(weights_path, weights_only=True)
        model.load_state_dict(state_dict)
    except Exception as e:
        logger.warning(f"Failed to load weights with weights_only=True: {str(e)}")
        logger.info("Attempting to load with weights_only=False...")
        # If that fails, load with weights_only=False
        state_dict = torch.load(weights_path, weights_only=False)
        model.load_state_dict(state_dict)
    return model

def train_ensemble():
    """Train and evaluate the ensemble model."""
    device = get_device()
    
    # Load individual models
    cnn_model = CNNModel()
    cnn_model = load_model_weights(cnn_model, "models/best_cnn_model.pth")
    cnn_model.to(device)
    cnn_model.eval()
    
    autoencoder = Autoencoder()
    autoencoder = load_model_weights(autoencoder, "models/best_autoencoder.pth")
    autoencoder.to(device)
    autoencoder.eval()
    
    svm_model = joblib.load("models/svm/svm_model.pkl")
    
    # Create ensemble model
    ensemble = EnsembleModel(
        cnn_model=cnn_model,
        autoencoder=autoencoder,
        svm_model=svm_model
    )
    ensemble.to(device)
    
    # Load test data
    test_dataset = FaceDataset(
        root_dir="data/processed/test",
        transform=None  # Add appropriate transforms if needed
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=32,
        shuffle=False,
        num_workers=4
    )
    
    # Evaluate ensemble
    all_predictions = []
    all_labels = []
    all_scores = []
    
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            predictions, scores = ensemble(images)
            
            all_predictions.extend(predictions.cpu().numpy())
            all_labels.extend(labels.numpy())
            all_scores.extend(scores.cpu().numpy())
    
    # Calculate metrics
    auc_score = roc_auc_score(all_labels, all_scores)
    report = classification_report(all_labels, all_predictions, output_dict=True)
    
    # Save ensemble metrics
    metrics = {
        "auc": float(auc_score),
        "classification_report": report,
        "weights": ensemble.weights.tolist() if hasattr(ensemble, 'weights') else None
    }
    
    with open("results/ensemble_metrics.json", "w") as f:
        json.dump(metrics, f, indent=4)
    
    # Save ensemble model
    torch.save(ensemble.state_dict(), "models/ensemble_model.pth")
    
    return metrics

def main():
    # """Run the complete training and evaluation pipeline."""
    # # Create directories
    # create_directories()
    
    # # Set device
    # device = get_device()
    
    # # Train CNN model
    # logger.info("Training CNN model...")
    # train_cnn()
    
    # # Train autoencoder model
    # logger.info("Training autoencoder model...")
    # train_autoencoder()
    
    # # Train SVM model
    # logger.info("Training SVM model...")
    # train_svm()
    
    # # Train and evaluate ensemble
    # logger.info("Training and evaluating ensemble model...")

    ensemble_metrics = train_ensemble()
    
    # Print final results
    logger.info("\nFinal Results:")
    logger.info(f"Ensemble AUC: {ensemble_metrics['auc']:.4f}")
    logger.info(f"Ensemble weights: {ensemble_metrics['weights']}")
    
    # Save final results
    with open("results/final_metrics.json", "w") as f:
        json.dump(ensemble_metrics, f, indent=4)
    
    logger.info("Pipeline completed successfully!")


if __name__ == "__main__":
    main() 