import os
import json
from pathlib import Path
import logging
from tqdm import tqdm
from PIL import Image
import pandas as pd

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def verify_folder_structure():
    """Verify that all required folders exist"""
    required_folders = [
        "data/processed/train/fake",
        "data/processed/train/real",
        "data/processed/val/fake",
        "data/processed/val/real",
        "data/processed/test/fake",
        "data/processed/test/real",
        "data/processed/train_aligned/fake",
        "data/processed/train_aligned/real",
        "data/processed/val_aligned/fake",
        "data/processed/val_aligned/real",
        "data/processed/test_aligned/fake",
        "data/processed/test_aligned/real"
    ]
    
    missing_folders = []
    for folder in required_folders:
        if not os.path.exists(folder):
            missing_folders.append(folder)
    
    if missing_folders:
        logging.error("Missing required folders:")
        for folder in missing_folders:
            logging.error(f"  - {folder}")
        return False
    
    logging.info("All required folders exist")
    return True

def count_images():
    """Count images in each split and class"""
    splits = ['train', 'val', 'test']
    classes = ['fake', 'real']
    versions = ['', '_aligned']  # Original and aligned versions
    
    counts = {}
    
    for split in splits:
        counts[split] = {}
        for version in versions:
            counts[split][version] = {}
            for class_name in classes:
                path = f"data/processed/{split}{version}/{class_name}"
                if os.path.exists(path):
                    image_files = [f for f in os.listdir(path) if f.endswith(('.jpg', '.png'))]
                    counts[split][version][class_name] = len(image_files)
                else:
                    counts[split][version][class_name] = 0
    
    # Print counts in a readable format
    logging.info("\nImage Counts:")
    for split in splits:
        logging.info(f"\n{split.upper()}:")
        for version in versions:
            version_name = "Original" if version == "" else "Aligned"
            logging.info(f"  {version_name}:")
            for class_name in classes:
                count = counts[split][version][class_name]
                logging.info(f"    {class_name}: {count}")
    
    return counts

def verify_image_quality():
    """Verify that all images can be opened and have correct dimensions"""
    splits = ['train', 'val', 'test']
    classes = ['fake', 'real']
    versions = ['', '_aligned']
    
    issues = []
    
    for split in splits:
        for version in versions:
            for class_name in classes:
                path = f"data/processed/{split}{version}/{class_name}"
                if not os.path.exists(path):
                    continue
                
                image_files = [f for f in os.listdir(path) if f.endswith(('.jpg', '.png'))]
                for img_file in tqdm(image_files, desc=f"Verifying {split}{version}/{class_name}"):
                    img_path = os.path.join(path, img_file)
                    try:
                        with Image.open(img_path) as img:
                            # Check if image can be opened
                            img.verify()
                            
                            # Check dimensions for aligned images
                            if version == '_aligned':
                                img = Image.open(img_path)
                                if img.size != (224, 224):
                                    issues.append(f"{img_path}: Incorrect size {img.size}, expected (224, 224)")
                    except Exception as e:
                        issues.append(f"{img_path}: {str(e)}")
    
    if issues:
        logging.warning("\nFound issues with images:")
        for issue in issues:
            logging.warning(f"  - {issue}")
    else:
        logging.info("\nAll images passed quality checks")
    
    return len(issues) == 0

def create_manifests():
    """Create CSV manifests for each split"""
    splits = ['train', 'val', 'test']
    classes = ['fake', 'real']
    
    for split in splits:
        manifest_data = []
        
        for class_name in classes:
            path = f"data/processed/{split}_aligned/{class_name}"
            if not os.path.exists(path):
                continue
            
            image_files = [f for f in os.listdir(path) if f.endswith(('.jpg', '.png'))]
            for img_file in image_files:
                img_path = os.path.join(path, img_file)
                manifest_data.append({
                    'image_path': img_path,
                    'class': class_name,
                    'split': split
                })
        
        # Create DataFrame and save to CSV
        df = pd.DataFrame(manifest_data)
        manifest_path = f"data/processed/{split}_manifest.csv"
        df.to_csv(manifest_path, index=False)
        logging.info(f"Created manifest: {manifest_path}")

def main():
    logging.info("Starting data verification...")
    
    # Verify folder structure
    if not verify_folder_structure():
        return
    
    # Count images
    counts = count_images()
    
    # Verify image quality
    verify_image_quality()
    
    # Create manifests
    create_manifests()
    
    logging.info("\nData verification complete!")

if __name__ == "__main__":
    main() 