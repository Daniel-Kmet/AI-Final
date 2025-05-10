import os
import numpy as np
import pandas as pd
import joblib
import logging
import json
from pathlib import Path
from time import time
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report
)
import matplotlib.pyplot as plt
from sklearn.metrics import RocCurveDisplay, precision_recall_curve
import cv2

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class SVMFeatureExtractor:
    """Class for extracting LBP features from images."""
    def __init__(self, device=None):
        self.device = device
        
    def extract_lbp_features(self, image):
        """Extract Local Binary Pattern (LBP) features from an image."""
        try:
            # Convert to grayscale if image is RGB
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            else:
                gray = image
                
            # Resize to a fixed size
            resized = cv2.resize(gray, (128, 128))
            
            # Calculate LBP features
            radius = 1
            n_points = 8 * radius
            lbp = self._local_binary_pattern(resized, n_points, radius)
            
            # Calculate histogram with 10 bins
            hist, _ = np.histogram(lbp.ravel(), bins=10, range=(0, 256))
            
            # Normalize histogram
            hist = hist.astype("float")
            hist /= (hist.sum() + 1e-7)
            
            return hist
            
        except Exception as e:
            logging.error(f"Error extracting LBP features: {str(e)}")
            return None
            
    def _local_binary_pattern(self, image, n_points, radius):
        """Calculate Local Binary Pattern."""
        rows = image.shape[0]
        cols = image.shape[1]
        output = np.zeros((rows, cols), dtype=np.uint8)
        
        for i in range(radius, rows - radius):
            for j in range(radius, cols - radius):
                center = image[i, j]
                pattern = 0
                
                for k in range(n_points):
                    # Get coordinates of the neighboring pixel
                    r = i + radius * np.cos(2.0 * np.pi * k / n_points)
                    c = j - radius * np.sin(2.0 * np.pi * k / n_points)
                    
                    # Get the pixel value using bilinear interpolation
                    r1, r2 = int(np.floor(r)), int(np.ceil(r))
                    c1, c2 = int(np.floor(c)), int(np.ceil(c))
                    
                    if r1 == r2 and c1 == c2:
                        value = image[r1, c1]
                    else:
                        value = (image[r1, c1] * (r2 - r) * (c2 - c) +
                                image[r1, c2] * (r2 - r) * (c - c1) +
                                image[r2, c1] * (r - r1) * (c2 - c) +
                                image[r2, c2] * (r - r1) * (c - c1))
                    
                    pattern |= (value > center) << k
                
                output[i, j] = pattern
                
        return output

def load_data():
    """Load LBP features and labels"""
    features_dir = Path("data/features")
    
    # Check if features exist
    if not features_dir.exists():
        raise FileNotFoundError("Features directory not found. Run 'extract_lbp_features.py' first.")
    
    # Load train, val, and test sets
    X_train = np.load(features_dir / "train_X.npy")
    y_train = np.load(features_dir / "train_y.npy")
    
    X_val = np.load(features_dir / "val_X.npy")
    y_val = np.load(features_dir / "val_y.npy")
    
    X_test = np.load(features_dir / "test_X.npy")
    y_test = np.load(features_dir / "test_y.npy")
    
    logging.info(f"Loaded data:")
    logging.info(f"  Train: {X_train.shape[0]} samples")
    logging.info(f"  Validation: {X_val.shape[0]} samples")
    logging.info(f"  Test: {X_test.shape[0]} samples")
    
    return X_train, y_train, X_val, y_val, X_test, y_test

def create_svm_pipeline():
    """Create an SVM pipeline with feature scaling"""
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('svm', SVC(probability=True, random_state=42))
    ])
    
    return pipeline

def tune_hyperparameters(pipeline, X_train, y_train):
    """Tune hyperparameters using grid search"""
    # Define parameter grid
    param_grid = {
        'svm__C': [0.1, 1, 10, 100],
        'svm__gamma': ['scale', 'auto', 0.1, 0.01, 0.001]
    }
    
    # Create grid search
    grid = GridSearchCV(
        pipeline,
        param_grid=param_grid,
        cv=5,
        scoring='roc_auc',
        verbose=1,
        n_jobs=-1
    )
    
    # Fit on training data
    logging.info("Starting grid search...")
    start_time = time()
    grid.fit(X_train, y_train)
    logging.info(f"Grid search completed in {time() - start_time:.2f} seconds")
    
    # Log best parameters
    logging.info(f"Best parameters: {grid.best_params_}")
    logging.info(f"Best CV score: {grid.best_score_:.4f}")
    
    return grid

def validate_model(model, X_val, y_val):
    """Validate model on validation set"""
    # Predict on validation set
    y_pred = model.predict(X_val)
    y_pred_proba = model.predict_proba(X_val)[:, 1]
    
    # Calculate metrics
    accuracy = accuracy_score(y_val, y_pred)
    precision = precision_score(y_val, y_pred)
    recall = recall_score(y_val, y_pred)
    f1 = f1_score(y_val, y_pred)
    roc_auc = roc_auc_score(y_val, y_pred_proba)
    
    logging.info("\nValidation Results:")
    logging.info(f"  Accuracy: {accuracy:.4f}")
    logging.info(f"  Precision: {precision:.4f}")
    logging.info(f"  Recall: {recall:.4f}")
    logging.info(f"  F1 Score: {f1:.4f}")
    logging.info(f"  ROC AUC: {roc_auc:.4f}")
    
    # Classification report
    report = classification_report(y_val, y_pred, target_names=['fake', 'real'])
    logging.info(f"\nClassification Report:\n{report}")
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'roc_auc': roc_auc,
        'y_pred': y_pred.tolist(),
        'y_pred_proba': y_pred_proba.tolist()
    }

def retrain_on_combined_data(best_params, X_train, y_train, X_val, y_val):
    """Retrain on combined train + validation data"""
    # Combine train and validation sets
    X_combined = np.vstack((X_train, X_val))
    y_combined = np.concatenate((y_train, y_val))
    
    logging.info(f"Retraining on combined data: {X_combined.shape[0]} samples")
    
    # Create pipeline with best parameters
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('svm', SVC(
            C=best_params['svm__C'],
            gamma=best_params['svm__gamma'],
            probability=True,
            random_state=42
        ))
    ])
    
    # Fit on combined data
    pipeline.fit(X_combined, y_combined)
    
    return pipeline

def evaluate_on_test(model, X_test, y_test, output_dir):
    """Evaluate model on test set"""
    # Predict on test set
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_pred_proba)
    conf_matrix = confusion_matrix(y_test, y_pred)
    
    logging.info("\nTest Results:")
    logging.info(f"  Accuracy: {accuracy:.4f}")
    logging.info(f"  Precision: {precision:.4f}")
    logging.info(f"  Recall: {recall:.4f}")
    logging.info(f"  F1 Score: {f1:.4f}")
    logging.info(f"  ROC AUC: {roc_auc:.4f}")
    
    # Classification report
    report = classification_report(y_test, y_pred, target_names=['fake', 'real'])
    logging.info(f"\nClassification Report:\n{report}")
    
    # Confusion matrix
    logging.info("\nConfusion Matrix:")
    logging.info(f"{conf_matrix}")
    
    # Create results dictionary
    test_results = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'roc_auc': roc_auc,
        'confusion_matrix': conf_matrix.tolist(),
        'y_pred': y_pred.tolist(),
        'y_pred_proba': y_pred_proba.tolist()
    }
    
    # Save test predictions
    test_df = pd.DataFrame({
        'true_label': y_test,
        'predicted_label': y_pred,
        'probability_real': y_pred_proba
    })
    test_df.to_csv(output_dir / "test_predictions.csv", index=False)
    
    # Save metrics
    with open(output_dir / "test_metrics.json", "w") as f:
        json.dump(test_results, f, indent=4)
    
    # Plot ROC curve
    plt.figure(figsize=(10, 8))
    RocCurveDisplay.from_predictions(
        y_test, 
        y_pred_proba,
        name="SVM Classifier",
        alpha=0.8,
        lw=2,
    )
    plt.title('ROC Curve')
    plt.savefig(output_dir / "roc_curve.png")
    
    # Plot precision-recall curve
    precision, recall, _ = precision_recall_curve(y_test, y_pred_proba)
    plt.figure(figsize=(10, 8))
    plt.plot(recall, precision, lw=2)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.savefig(output_dir / "precision_recall_curve.png")
    
    return test_results

def main():
    # Create output directory
    output_dir = Path("models/svm")
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # Configure file handler for logging
    file_handler = logging.FileHandler(output_dir / "training_log.txt")
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    logging.getLogger().addHandler(file_handler)
    
    # Step 1: Load the data
    logging.info("Loading data...")
    X_train, y_train, X_val, y_val, X_test, y_test = load_data()
    
    # Step 2: Create SVM pipeline (scaler is part of the pipeline)
    logging.info("Creating SVM pipeline...")
    pipeline = create_svm_pipeline()
    
    # Step 3: Tune hyperparameters
    logging.info("Tuning hyperparameters...")
    grid = tune_hyperparameters(pipeline, X_train, y_train)
    
    # Step 4: Validate the best model
    logging.info("Validating best model...")
    best_model = grid.best_estimator_
    val_metrics = validate_model(best_model, X_val, y_val)
    
    # Step 5: Retrain on combined train + validation data
    logging.info("Retraining on combined data...")
    final_model = retrain_on_combined_data(
        grid.best_params_, X_train, y_train, X_val, y_val
    )
    
    # Step 6: Evaluate on test set
    logging.info("Evaluating on test set...")
    test_metrics = evaluate_on_test(final_model, X_test, y_test, output_dir)
    
    # Step 7: Save the final model and scaler
    logging.info("Saving final model...")
    model_data = {
        'model': final_model.named_steps['svm'],
        'scaler': final_model.named_steps['scaler']
    }
    joblib.dump(model_data, output_dir / "svm_model.pkl")
    
    logging.info("Done!")

if __name__ == "__main__":
    main() 