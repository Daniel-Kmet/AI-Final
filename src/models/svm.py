import torch
import numpy as np
import logging
from pathlib import Path
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
import cv2
import os
from tqdm import tqdm
import joblib

from src.utils.metrics.reporting import TrainingReporter
from src.utils.device import get_device
from src.data.dataset import create_data_loaders

logger = logging.getLogger(__name__)

class SVMFeatureExtractor:
    def __init__(self, device):
        self.device = device
        self.scaler = StandardScaler()
        self.svm = SVC(probability=True, kernel='rbf')
        
    def extract_lbp_features(self, img):
        """Extract Local Binary Pattern features from an image."""
        # Convert to numpy array and ensure correct data type
        if isinstance(img, torch.Tensor):
            img = img.cpu().numpy()
        
        # Convert to uint8 if needed
        if img.dtype != np.uint8:
            img = (img * 255).astype(np.uint8)
        
        # Ensure correct shape (H, W, C)
        if img.shape[0] == 3:  # If image is (C, H, W)
            img = img.transpose(1, 2, 0)
        
        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        
        # Calculate LBP
        radius = 3
        n_points = 24
        lbp = cv2.LBP_create(n_points, radius)
        lbp_image = lbp.compute(gray)
        
        # Calculate histogram
        n_bins = 256
        hist, _ = np.histogram(lbp_image.ravel(), bins=n_bins, range=(0, n_bins))
        hist = hist.astype(np.float32)
        hist = hist / (hist.sum() + 1e-7)  # Normalize
        
        return hist
    
    def extract_features_batch(self, dataloader):
        """Extract features from a batch of images."""
        features = []
        labels = []
        
        for images, batch_labels in tqdm(dataloader, desc="Extracting features"):
            images = images.to(self.device)
            
            for img, label in zip(images, batch_labels):
                # Extract LBP features
                lbp_features = self.extract_lbp_features(img)
                features.append(lbp_features)
                labels.append(label.item())
        
        return np.array(features), np.array(labels)

def main(data_dir="data/processed", batch_size=32):
    """Main function to run SVM training and evaluation."""
    # Initialize reporter
    reporter = TrainingReporter("svm")
    
    # Log hyperparameters
    hyperparams = {
        "batch_size": batch_size,
        "feature_extractor": {
            "type": "LBP",
            "radius": 3,
            "n_points": 24,
            "n_bins": 256
        },
        "svm": {
            "kernel": "rbf",
            "probability": True
        }
    }
    reporter.log_hyperparameters(hyperparams)
    
    # Set device
    device = get_device()
    logger.info(f"Using device: {device}")
    
    # Create data loaders
    train_loader, val_loader, test_loader = create_data_loaders(data_dir, batch_size)
    
    # Initialize feature extractor
    extractor = SVMFeatureExtractor(device)
    
    # Extract features
    logger.info("Extracting features from training set...")
    X_train, y_train = extractor.extract_features_batch(train_loader)
    
    logger.info("Extracting features from validation set...")
    X_val, y_val = extractor.extract_features_batch(val_loader)
    
    logger.info("Extracting features from test set...")
    X_test, y_test = extractor.extract_features_batch(test_loader)
    
    # Scale features
    X_train_scaled = extractor.scaler.fit_transform(X_train)
    X_val_scaled = extractor.scaler.transform(X_val)
    X_test_scaled = extractor.scaler.transform(X_test)
    
    # Train SVM
    logger.info("Training SVM...")
    extractor.svm.fit(X_train_scaled, y_train)
    
    # Calculate metrics for each epoch (we'll simulate epochs for consistent reporting)
    train_probs = extractor.svm.predict_proba(X_train_scaled)[:, 1]
    val_probs = extractor.svm.predict_proba(X_val_scaled)[:, 1]
    
    train_preds = extractor.svm.predict(X_train_scaled)
    val_preds = extractor.svm.predict(X_val_scaled)
    
    train_acc = np.mean(train_preds == y_train)
    val_acc = np.mean(val_preds == y_val)
    
    # Log metrics (as a single epoch since SVM trains in one shot)
    reporter.log_epoch(1, 0.0, train_acc, 0.0, val_acc)
    
    # Save model
    model_path = reporter.report_dir / "svm_model.joblib"
    scaler_path = reporter.report_dir / "svm_scaler.joblib"
    joblib.dump(extractor.svm, model_path)
    joblib.dump(extractor.scaler, scaler_path)
    
    # Generate test predictions
    test_preds = extractor.svm.predict(X_test_scaled)
    test_probs = extractor.svm.predict_proba(X_test_scaled)[:, 1]
    
    # Generate classification report
    test_metrics, confusion_matrix, roc_curve_data = reporter.generate_classification_report(
        y_test, test_preds, test_probs)
    
    # Calculate model size
    model_size = (os.path.getsize(model_path) + os.path.getsize(scaler_path)) / (1024 * 1024)  # Convert to MB
    
    # Generate text report
    reporter.generate_text_report(test_metrics, confusion_matrix, roc_curve_data, model_size)
    
    # Generate final report
    final_report = reporter.generate_final_report(test_metrics, model_size)
    
    return final_report

if __name__ == "__main__":
    main() 