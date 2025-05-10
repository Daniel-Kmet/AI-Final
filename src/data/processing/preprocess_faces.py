import os
import cv2
import numpy as np
from tqdm import tqdm
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# ImageNet normalization parameters
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406])
IMAGENET_STD = np.array([0.229, 0.224, 0.225])

def setup_detector():
    """Initialize OpenCV face detector"""
    # Load the pre-trained face detection model
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    return face_cascade

def normalize_image(img):
    """Normalize image using ImageNet statistics"""
    # Convert to float and scale to [0, 1]
    img = img.astype(np.float32) / 255.0
    
    # Convert BGR to RGB (OpenCV loads as BGR)
    img = img[:, :, ::-1]
    
    # Apply ImageNet normalization
    img = (img - IMAGENET_MEAN) / IMAGENET_STD
    
    return img

def process_image(detector, img_path, output_path):
    """Process a single image: detect face, align, normalize, and save"""
    try:
        # Load image
        img = cv2.imread(img_path)
        if img is None:
            logging.error(f"Could not read image: {img_path}")
            return False
            
        # Convert to grayscale for face detection
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Detect faces
        faces = detector.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30)
        )
        
        if len(faces) == 0:
            logging.warning(f"No face detected in {img_path}")
            return False
            
        # Get the first face
        x, y, w, h = faces[0]
        
        # Add margin around the face
        margin = 20
        x1 = max(0, x - margin)
        y1 = max(0, y - margin)
        x2 = min(img.shape[1], x + w + margin)
        y2 = min(img.shape[0], y + h + margin)
        
        # Crop the face region
        face_img = img[y1:y2, x1:x2]
        
        # Resize to standard size (224x224 for most pretrained models)
        face_img = cv2.resize(face_img, (224, 224))
        
        # Normalize the image
        normalized_img = normalize_image(face_img)
        
        # Save the processed image (convert back to uint8 for saving)
        normalized_img = ((normalized_img * IMAGENET_STD + IMAGENET_MEAN) * 255).astype(np.uint8)
        normalized_img = normalized_img[:, :, ::-1]  # Convert back to BGR for saving
        cv2.imwrite(output_path, normalized_img)
        return True
        
    except Exception as e:
        logging.error(f"Error processing {img_path}: {str(e)}")
        return False

def process_directory(detector, input_dir, output_dir, class_name):
    """Process all images in a directory"""
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Get all image files
    image_files = [f for f in os.listdir(input_dir) if f.endswith(('.jpg', '.png'))]
    
    # Process each image
    successful = 0
    failed = 0
    
    for img_file in tqdm(image_files, desc=f"Processing {class_name} images"):
        input_path = os.path.join(input_dir, img_file)
        output_path = os.path.join(output_dir, img_file)
        
        if process_image(detector, input_path, output_path):
            successful += 1
        else:
            failed += 1
    
    return successful, failed

def main():
    # Initialize OpenCV face detector
    logging.info("Initializing OpenCV face detector...")
    detector = setup_detector()
    
    # Process each dataset split
    splits = ['train', 'val', 'test']
    classes = ['fake', 'real']
    
    for split in splits:
        logging.info(f"\nProcessing {split} split...")
        for class_name in classes:
            input_dir = f"data/processed/{split}/{class_name}"
            output_dir = f"data/processed/{split}_aligned/{class_name}"
            
            logging.info(f"Processing {class_name} images in {split} split...")
            successful, failed = process_directory(detector, input_dir, output_dir, class_name)
            
            logging.info(f"Completed {class_name} in {split} split:")
            logging.info(f"Successfully processed: {successful} images")
            logging.info(f"Failed to process: {failed} images")

if __name__ == "__main__":
    main() 