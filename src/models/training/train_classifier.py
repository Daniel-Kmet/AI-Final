import os
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix
import joblib
import cv2
from pathlib import Path
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def extract_features(image_path):
    """Extract HOG features from an image."""
    try:
        # Read image
        img = cv2.imread(str(image_path))
        if img is None:
            raise ValueError(f"Could not read image: {image_path}")
        
        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Resize to a fixed size (e.g., 128x128)
        resized = cv2.resize(gray, (128, 128))
        
        # Calculate HOG features
        win_size = (128, 128)
        block_size = (16, 16)
        block_stride = (8, 8)
        cell_size = (8, 8)
        nbins = 9
        
        hog = cv2.HOGDescriptor(win_size, block_size, block_stride, cell_size, nbins)
        features = hog.compute(resized)
        
        return features.flatten()
    except Exception as e:
        logger.error(f"Error processing {image_path}: {str(e)}")
        return None

def load_dataset(data_dir):
    """Load and extract features from the dataset."""
    features = []
    labels = []
    
    # Process real images
    real_dir = Path(data_dir) / "real"
    for img_path in real_dir.glob("*.png"):
        feature = extract_features(img_path)
        if feature is not None:
            features.append(feature)
            labels.append(1)  # 1 for real
    
    for img_path in real_dir.glob("*.jpg"):
        feature = extract_features(img_path)
        if feature is not None:
            features.append(feature)
            labels.append(1)  # 1 for real
    
    # Process fake images
    fake_dir = Path(data_dir) / "fake"
    for img_path in fake_dir.glob("*.png"):
        feature = extract_features(img_path)
        if feature is not None:
            features.append(feature)
            labels.append(0)  # 0 for fake
    
    for img_path in fake_dir.glob("*.jpg"):
        feature = extract_features(img_path)
        if feature is not None:
            features.append(feature)
            labels.append(0)  # 0 for fake
    
    return np.array(features), np.array(labels)

def train_and_evaluate():
    """Main function to train and evaluate the classifier."""
    # Load dataset
    logger.info("Loading dataset...")
    X, y = load_dataset("data/processed")
    
    # Split into train and test sets (80-20 split)
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Feature scaling
    logger.info("Scaling features...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train SVM
    logger.info("Training SVM classifier...")
    svm = SVC(kernel='rbf', probability=True, random_state=42)
    svm.fit(X_train_scaled, y_train)
    
    # Evaluate
    logger.info("Evaluating model...")
    y_pred = svm.predict(X_test_scaled)
    
    # Print classification report
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=['Fake', 'Real']))
    
    # Print confusion matrix
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    
    # Save model and scaler
    logger.info("Saving model and scaler...")
    os.makedirs("models", exist_ok=True)
    joblib.dump(svm, "models/face_authenticity_svm.joblib")
    joblib.dump(scaler, "models/feature_scaler.joblib")
    
    logger.info("Training and evaluation complete!")

if __name__ == "__main__":
    train_and_evaluate() 