import argparse
import logging
import sys
from pathlib import Path
import os

# Add src directory to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

from src.models.architectures.autoencoder_model import main as run_autoencoder, predict_image

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('face_authenticity_autoencoder.log')
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

def main():
    parser = argparse.ArgumentParser(description='Face Authenticity Convolutional Autoencoder')
    parser.add_argument('action', choices=['setup', 'train', 'predict'],
                      help='Action to perform: setup project structure, train model, or predict on image')
    parser.add_argument('--image', type=str, help='Path to image for prediction')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs for training')
    parser.add_argument('--img-size', type=int, default=128, help='Image size for model (square)')
    parser.add_argument('--bottleneck', type=int, default=128, help='Bottleneck dimension for autoencoder')
    
    args = parser.parse_args()
    
    try:
        if args.action == 'setup':
            logger.info("Setting up project structure...")
            setup_project_structure()
            logger.info("Project structure setup complete!")
            
        elif args.action == 'train':
            logger.info("Starting autoencoder model training...")
            run_autoencoder(
                batch_size=args.batch_size,
                num_epochs=args.epochs,
                img_size=args.img_size,
                bottleneck_dim=args.bottleneck
            )
            logger.info("Autoencoder training complete!")
            
        elif args.action == 'predict':
            if not args.image:
                parser.error("--image argument is required for predict action")
            
            logger.info(f"Predicting authenticity for image: {args.image}")
            result = predict_image(args.image, img_size=args.img_size)
            
            if result:
                print(f"\nPrediction: {result['prediction']}")
                print(f"Confidence: {result['confidence']:.2%}")
                print(f"Error: {result['error']:.6f} (Threshold: {result['threshold']:.6f})")
                print(f"Visualization saved to: {result['visualization']}")
            else:
                logger.error("Prediction failed")
                
    except Exception as e:
        logger.error(f"An error occurred: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main() 