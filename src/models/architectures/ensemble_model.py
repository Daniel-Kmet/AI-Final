import torch
import torch.nn as nn
import numpy as np
import json
import logging
from pathlib import Path
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, classification_report, confusion_matrix
from torchvision import transforms
from PIL import Image
import torch_directml
import joblib
from sklearn.preprocessing import StandardScaler
import torch.nn.functional as F

from src.models.architectures.cnn_model import CNNModel
from src.models.architectures.autoencoder_model import Autoencoder
from src.models.training.train_svm_classifier import SVMFeatureExtractor
from src.utils.device import get_device

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

device = get_device()

class EnsembleModel:
    def __init__(self, cnn_model_path, autoencoder_model_path, svm_model_path, autoencoder_threshold_path):
        """Initialize ensemble model with all three models."""
        self.device = get_device()
        
        # Load CNN model
        self.cnn_model = CNNModel().to(self.device)
        self.cnn_model.load_state_dict(torch.load(cnn_model_path, map_location=self.device))
        self.cnn_model.eval()
        
        # Load autoencoder model
        self.autoencoder_model = Autoencoder().to(self.device)
        self.autoencoder_model.load_state_dict(torch.load(autoencoder_model_path, map_location=self.device))
        self.autoencoder_model.eval()
        
        # Load autoencoder threshold
        with open(autoencoder_threshold_path, 'r') as f:
            self.autoencoder_threshold = float(json.load(f)['threshold'])
        
        # Load SVM model
        svm_data = joblib.load(svm_model_path)
        self.svm_model = svm_data.named_steps['svm']
        self.svm_scaler = svm_data.named_steps['scaler']
        
        # Initialize weights (can be optimized later)
        self.weights = {
            'cnn': 0.4,
            'autoencoder': 0.3,
            'svm': 0.3
        }
    
    def preprocess_image(self, image):
        """Preprocess image for model input."""
        # Convert to tensor and normalize
        image = torch.from_numpy(image).float()
        image = image.permute(2, 0, 1)  # Change from HWC to CHW
        image = image / 255.0  # Normalize to [0, 1]
        image = image.unsqueeze(0)  # Add batch dimension
        return image.to(self.device)
    
    def get_cnn_prediction(self, image):
        """Get prediction from CNN model."""
        with torch.no_grad():
            image = self.preprocess_image(image)
            output = self.cnn_model(image)
            probabilities = F.softmax(output, dim=1)
            return probabilities[0, 1].item()  # Probability of real class
    
    def get_autoencoder_prediction(self, image):
        """Get prediction from autoencoder model."""
        with torch.no_grad():
            image = self.preprocess_image(image)
            reconstructed, _ = self.autoencoder_model(image)  # Unpack the tuple
            mse = F.mse_loss(reconstructed, image)
            # Convert MSE to probability (lower MSE = higher probability of being real)
            probability = 1.0 / (1.0 + mse.item())
            return probability
    
    def get_svm_prediction(self, image):
        """Get prediction from SVM model."""
        # Extract features
        extractor = SVMFeatureExtractor(self.device)
        features = extractor.extract_lbp_features(image)
        
        if features is None:
            return 0.5  # Default to uncertain if feature extraction fails
        
        # Scale features and get prediction
        features_scaled = self.svm_scaler.transform(features.reshape(1, -1))
        probability = self.svm_model.predict_proba(features_scaled)[0, 1]
        return probability
    
    def get_ensemble_prediction(self, image):
        """Get weighted ensemble prediction."""
        cnn_prob = self.get_cnn_prediction(image)
        autoencoder_prob = self.get_autoencoder_prediction(image)
        svm_prob = self.get_svm_prediction(image)
        
        # Weighted average
        ensemble_prob = (
            self.weights['cnn'] * cnn_prob +
            self.weights['autoencoder'] * autoencoder_prob +
            self.weights['svm'] * svm_prob
        )
        
        return ensemble_prob
    
    def find_optimal_weights(self, val_loader):
        """Find optimal weights for ensemble using validation set."""
        best_auc = 0
        best_weights = self.weights.copy()
        
        # Grid search over weight combinations
        weight_step = 0.1
        for cnn_weight in np.arange(0.2, 0.7, weight_step):
            for ae_weight in np.arange(0.2, 0.7, weight_step):
                svm_weight = 1 - cnn_weight - ae_weight
                if svm_weight < 0.2:  # Ensure minimum weight for each model
                    continue
                
                self.weights = {
                    'cnn': cnn_weight,
                    'autoencoder': ae_weight,
                    'svm': svm_weight
                }
                
                # Evaluate current weights
                metrics = self.evaluate(val_loader)
                current_auc = metrics['auc']
                
                if current_auc > best_auc:
                    best_auc = current_auc
                    best_weights = self.weights.copy()
        
        self.weights = best_weights
        logger.info(f"Optimal weights found: {best_weights}")
        logger.info(f"Best validation AUC: {best_auc:.4f}")
        
        return best_weights
    
    def evaluate(self, data_loader):
        """Evaluate ensemble model on given data loader."""
        all_probs = []
        all_labels = []
        
        for images, labels in tqdm(data_loader, desc="Evaluating ensemble"):
            for img, label in zip(images, labels):
                # Convert to numpy
                img_np = img.numpy().transpose(1, 2, 0)
                
                # Get ensemble prediction
                prob = self.get_ensemble_prediction(img_np)
                
                all_probs.append(prob)
                all_labels.append(label.item())
        
        # Convert to numpy arrays
        all_probs = np.array(all_probs)
        all_labels = np.array(all_labels)
        
        # Calculate ROC curve
        fpr, tpr, _ = roc_curve(all_labels, all_probs)
        roc_auc = auc(fpr, tpr)
        
        # Calculate predictions and confusion matrix
        predictions = (all_probs > 0.5).astype(int)
        conf_matrix = confusion_matrix(all_labels, predictions)
        
        # Plot ROC curve
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {roc_auc:.4f})')
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Ensemble ROC Curve')
        plt.legend(loc='lower right')
        plt.savefig("results/ensemble_roc_curve.png")
        plt.close()
        
        # Plot confusion matrix
        plt.figure(figsize=(8, 6))
        plt.imshow(conf_matrix, interpolation='nearest', cmap=plt.cm.Blues)
        plt.title('Ensemble Confusion Matrix')
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
        plt.savefig("results/ensemble_confusion_matrix.png")
        plt.close()
        
        # Save metrics
        metrics = {
            'auc': float(roc_auc),
            'weights': self.weights,
            'confusion_matrix': conf_matrix.tolist()
        }
        
        with open("results/ensemble_metrics.json", "w") as f:
            json.dump(metrics, f, indent=4)
        
        logger.info(f"Ensemble AUC: {roc_auc:.4f}")
        logger.info(f"Ensemble weights: {self.weights}")
        
        return metrics

def main():
    """Main function to train and evaluate ensemble model."""
    # Create data loaders
    from src.data.dataset import create_data_loaders
    train_loader, val_loader, test_loader = create_data_loaders("data/processed")
    
    # Initialize ensemble model
    ensemble = EnsembleModel(
        cnn_model_path="models/cnn/best_model.pth",
        autoencoder_model_path="models/autoencoder/best_model.pth",
        svm_model_path="models/svm/best_model.pkl",
        autoencoder_threshold_path="models/autoencoder/threshold.json"
    )
    
    # Find optimal weights
    logger.info("Finding optimal weights...")
    optimal_weights = ensemble.find_optimal_weights(val_loader)
    
    # Evaluate on test set
    logger.info("Evaluating ensemble on test set...")
    test_metrics = ensemble.evaluate(test_loader)
    
    return test_metrics

if __name__ == "__main__":
    main() 