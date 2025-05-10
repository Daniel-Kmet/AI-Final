import os
import numpy as np
from skimage.feature import local_binary_pattern
from PIL import Image
import pandas as pd
from tqdm import tqdm
import logging
import joblib
from pathlib import Path

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class LBPFeatureExtractor:
    def __init__(self, P=8, R=1, method='uniform'):
        """
        Initialize LBP feature extractor
        
        Parameters:
        -----------
        P : int
            Number of sampling points
        R : float
            Radius of circle (spatial resolution of the operator)
        method : str
            Method for LBP computation ('uniform', 'default', etc.)
        """
        self.P = P
        self.R = R
        self.method = method
        self.n_bins = P + 2 if method == 'uniform' else P * (P - 1) + 3
        
    def compute_lbp_histogram(self, image):
        """
        Compute LBP histogram for a single image
        
        Parameters:
        -----------
        image : numpy.ndarray
            Grayscale image
            
        Returns:
        --------
        numpy.ndarray
            Normalized LBP histogram
        """
        # Compute LBP map
        lbp_map = local_binary_pattern(
            image,
            P=self.P,
            R=self.R,
            method=self.method
        )
        
        # Compute histogram
        hist, _ = np.histogram(
            lbp_map.ravel(),
            bins=self.n_bins,
            range=(0, self.n_bins),
            density=True
        )
        
        return hist
    
    def process_image(self, image_path):
        """
        Process a single image: load, convert to grayscale, compute LBP histogram
        
        Parameters:
        -----------
        image_path : str
            Path to the image file
            
        Returns:
        --------
        numpy.ndarray or None
            LBP histogram if successful, None if failed
        """
        try:
            # Load and convert to grayscale
            image = np.array(Image.open(image_path).convert('L'))
            
            # Compute LBP histogram
            histogram = self.compute_lbp_histogram(image)
            
            return histogram
            
        except Exception as e:
            logging.error(f"Error processing {image_path}: {str(e)}")
            return None
    
    def process_dataset(self, manifest_path):
        """
        Process all images in a dataset split
        
        Parameters:
        -----------
        manifest_path : str
            Path to the CSV manifest file
            
        Returns:
        --------
        tuple
            (X, y) where X is the feature matrix and y is the label vector
        """
        # Load manifest
        df = pd.read_csv(manifest_path)
        
        # Initialize arrays
        X = []
        y = []
        
        # Process each image
        for _, row in tqdm(df.iterrows(), total=len(df), desc="Processing images"):
            histogram = self.process_image(row['image_path'])
            if histogram is not None:
                X.append(histogram)
                y.append(1 if row['class'] == 'real' else 0)
        
        return np.array(X), np.array(y)

def main():
    # Initialize feature extractor
    extractor = LBPFeatureExtractor(P=8, R=1, method='uniform')
    
    # Create output directory
    output_dir = Path("data/features")
    output_dir.mkdir(exist_ok=True)
    
    # Process each split
    splits = ['train', 'val', 'test']
    for split in splits:
        logging.info(f"\nProcessing {split} split...")
        
        # Load and process data
        manifest_path = f"data/processed/{split}_manifest.csv"
        X, y = extractor.process_dataset(manifest_path)
        
        # Save features and labels
        np.save(output_dir / f"{split}_X.npy", X)
        np.save(output_dir / f"{split}_y.npy", y)
        
        logging.info(f"Saved {len(X)} samples for {split} split")
        logging.info(f"Feature shape: {X.shape}")
        logging.info(f"Label distribution: {np.bincount(y)}")

if __name__ == "__main__":
    main() 