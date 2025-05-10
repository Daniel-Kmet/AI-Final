import argparse
import logging
import sys
from pathlib import Path
import os
import torch

# Add src directory to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

from src.models.architectures.cnn_model import main as run_cnn, CNNModel, get_device

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('face_authenticity_cnn.log')
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

def predict_image(image_path, model_path="models/best_cnn_model.pth"):
    """Predict whether a single image is real or fake using the CNN model."""
    try:
        from PIL import Image
        from torchvision import transforms
        import numpy as np
        
        # Check if model exists
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        # Set device with AMD GPU support
        device = get_device()
        
        # Load model
        model = CNNModel()
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.to(device)
        model.eval()
        
        # Load and preprocess image
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        img = Image.open(image_path).convert('RGB')
        img_tensor = transform(img).unsqueeze(0).to(device)
        
        # Make prediction
        with torch.no_grad():
            outputs = model(img_tensor)
            probs = torch.softmax(outputs, dim=1)
            _, predicted = torch.max(outputs, 1)
            
            prediction = predicted.item()
            confidence = probs[0][prediction].item()
            
        result = {
            'prediction': 'Real' if prediction == 1 else 'Fake',
            'confidence': confidence
        }
        
        return result
        
    except Exception as e:
        logger.error(f"Error during prediction: {str(e)}")
        return None

def main():
    parser = argparse.ArgumentParser(description='Face Authenticity CNN Classifier')
    parser.add_argument('action', choices=['setup', 'train', 'predict'],
                      help='Action to perform: setup project structure, train model, or predict on image')
    parser.add_argument('--image', type=str, help='Path to image for prediction')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs for training')
    
    args = parser.parse_args()
    
    try:
        if args.action == 'setup':
            logger.info("Setting up project structure...")
            setup_project_structure()
            logger.info("Project structure setup complete!")
            
        elif args.action == 'train':
            logger.info("Starting CNN model training...")
            run_cnn(batch_size=args.batch_size, num_epochs=args.epochs)
            logger.info("CNN model training complete!")
            
        elif args.action == 'predict':
            if not args.image:
                parser.error("--image argument is required for predict action")
            
            logger.info(f"Predicting authenticity for image: {args.image}")
            result = predict_image(args.image)
            
            if result:
                print(f"\nPrediction: {result['prediction']}")
                print(f"Confidence: {result['confidence']:.2%}")
            else:
                logger.error("Prediction failed")
                
    except Exception as e:
        logger.error(f"An error occurred: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main() 