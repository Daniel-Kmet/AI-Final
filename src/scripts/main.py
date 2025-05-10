import argparse
import logging
import sys
from pathlib import Path
import os
import joblib

# Add src directory to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.models.training.train_classifier import extract_features, train_and_evaluate
from src.models.architectures.cnn_model import main as run_cnn, predict_image as predict_cnn
from src.models.architectures.autoencoder_model import main as run_autoencoder, predict_image as predict_autoencoder

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('face_authenticity.log')
    ]
)
logger = logging.getLogger(__name__)

def setup_project_structure():
    """Create necessary project directories if they don't exist."""
    directories = [
        "data/raw/real",
        "data/raw/fake",
        "data/processed/real",
        "data/processed/fake",
        "models",
        "results"
    ]
    
    for dir_path in directories:
        Path(dir_path).mkdir(parents=True, exist_ok=True)
        logger.info(f"Created directory: {dir_path}")

def validate_dataset():
    """Validate that the dataset exists and has the correct structure."""
    required_dirs = [
        "data/raw/real",
        "data/raw/fake",
        "data/processed/real",
        "data/processed/fake"
    ]
    
    for dir_path in required_dirs:
        path = Path(dir_path)
        if not path.exists():
            logger.error(f"Required directory not found: {dir_path}")
            return False
        
        # Check if directory contains images
        image_files = list(path.glob("*.jpg")) + list(path.glob("*.png"))
        if not image_files:
            logger.error(f"No image files found in {dir_path}")
            return False
        
        logger.info(f"Found {len(image_files)} images in {dir_path}")
    
    return True

def predict_image(image_path, model_type="svm"):
    """Predict whether a single image is real or fake."""
    try:
        if model_type == "svm":
            # Load the model and scaler
            model_path = Path("models/face_authenticity_svm.joblib")
            scaler_path = Path("models/feature_scaler.joblib")
            
            if not (model_path.exists() and scaler_path.exists()):
                raise FileNotFoundError("Model files not found. Please train the model first.")
            
            model = joblib.load(model_path)
            scaler = joblib.load(scaler_path)
            
            # Extract features
            features = extract_features(image_path)
            
            if features is None:
                raise ValueError("Could not extract features from the image")
            
            # Scale features
            features_scaled = scaler.transform(features.reshape(1, -1))
            
            # Make prediction
            prediction = model.predict(features_scaled)[0]
            probability = model.predict_proba(features_scaled)[0]
            
            result = {
                'prediction': 'Real' if prediction == 1 else 'Fake',
                'confidence': probability[prediction]
            }
        elif model_type == "cnn":
            result = predict_cnn(image_path)
        elif model_type == "autoencoder":
            result = predict_autoencoder(image_path)
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        return result
        
    except Exception as e:
        logger.error(f"Error during prediction: {str(e)}")
        return None

def main():
    parser = argparse.ArgumentParser(description='Face Authenticity Detection Project')
    parser.add_argument('action', choices=['setup', 'validate', 'train', 'train-cnn', 'train-autoencoder', 'predict'],
                      help='Action to perform')
    parser.add_argument('--image', type=str, help='Path to image for prediction')
    parser.add_argument('--model-type', type=str, choices=['svm', 'cnn', 'autoencoder'],
                      default='svm', help='Model type to use for prediction')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs for training')
    parser.add_argument('--img-size', type=int, default=128, help='Image size for model (square)')
    parser.add_argument('--bottleneck', type=int, default=128, help='Bottleneck dimension for autoencoder')
    
    args = parser.parse_args()
    
    try:
        if args.action == 'setup':
            logger.info("Setting up project structure...")
            setup_project_structure()
            logger.info("Project structure setup complete!")
            
        elif args.action == 'validate':
            logger.info("Validating dataset...")
            if validate_dataset():
                logger.info("Dataset validation successful!")
            else:
                logger.error("Dataset validation failed!")
            
        elif args.action == 'train':
            logger.info("Validating dataset before training...")
            if validate_dataset():
                logger.info("Starting SVM model training...")
                train_and_evaluate()
            else:
                logger.error("Training aborted due to dataset validation failure")
                
        elif args.action == 'train-cnn':
            logger.info("Validating dataset before training...")
            if validate_dataset():
                logger.info("Starting CNN model training...")
                run_cnn(batch_size=args.batch_size, num_epochs=args.epochs)
            else:
                logger.error("Training aborted due to dataset validation failure")
                
        elif args.action == 'train-autoencoder':
            logger.info("Validating dataset before training...")
            if validate_dataset():
                logger.info("Starting Autoencoder training...")
                run_autoencoder(
                    batch_size=args.batch_size,
                    num_epochs=args.epochs,
                    img_size=args.img_size,
                    bottleneck_dim=args.bottleneck
                )
            else:
                logger.error("Training aborted due to dataset validation failure")
                
        elif args.action == 'predict':
            if not args.image:
                parser.error("--image argument is required for predict action")
            
            logger.info(f"Predicting authenticity for image: {args.image}")
            result = predict_image(args.image, model_type=args.model_type)
            
            if result:
                print(f"\nPrediction: {result['prediction']}")
                print(f"Confidence: {result['confidence']:.2%}")
                if 'visualization' in result:
                    print(f"Visualization saved to: {result['visualization']}")
            else:
                logger.error("Prediction failed")
                
    except Exception as e:
        logger.error(f"An error occurred: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main() 