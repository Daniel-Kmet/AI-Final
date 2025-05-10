# Face Authenticity Detection

This project implements a machine learning system for detecting fake (manipulated) and real face images using computer vision and machine learning techniques. It supports multiple approaches:
- Traditional feature-based (HOG + SVM)
- Deep learning (CNN)
- Anomaly detection (Convolutional Autoencoder)

## Project Structure

```
.
├── data/
│   ├── raw/           # Original images
│   │   ├── real/      # Real face images
│   │   └── fake/      # Fake face images
│   └── processed/     # Processed images
│       ├── real/      # Processed real images
│       └── fake/      # Processed fake images
├── models/            # Saved model files
├── results/           # Evaluation results
├── src/              # Source code
│   ├── main.py       # Main project script
│   ├── train_classifier.py  # SVM training pipeline
│   ├── cnn_model.py  # CNN implementation
│   ├── run_cnn.py    # CNN wrapper script
│   ├── autoencoder_model.py # Autoencoder implementation
│   └── run_autoencoder.py   # Autoencoder wrapper script
└── requirements.txt   # Project dependencies
```

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Set up project structure:
```bash
python src/main.py setup
```

## Usage

The project provides multiple functionalities:

### 1. Project Setup
Creates necessary directories for the project:
```bash
python src/main.py setup
```

### 2. Dataset Validation
Validates that your dataset is properly structured:
```bash
python src/main.py validate
```

### 3. Training Models

#### SVM Model (HOG Features)
Train a traditional machine learning model:
```bash
python src/main.py train
```

#### CNN Model (ResNet50)
Train a deep learning model:
```bash
python src/main.py train-cnn
```

You can adjust batch size and epochs:
```bash
python src/main.py train-cnn --batch-size 64 --epochs 20
```

#### Autoencoder Model
Train a convolutional autoencoder for anomaly detection:
```bash
python src/main.py train-autoencoder
```

You can adjust various parameters:
```bash
python src/main.py train-autoencoder --batch-size 32 --epochs 50 --img-size 128 --bottleneck 128
```

### 4. Making Predictions

Predict whether a single image is real or fake:

#### Using SVM model:
```bash
python src/main.py predict --image path/to/image.jpg --model svm
```

#### Using CNN model:
```bash
python src/main.py predict --image path/to/image.jpg --model cnn
```

#### Using Autoencoder model:
```bash
python src/main.py predict --image path/to/image.jpg --model autoencoder
```

## Features

### SVM Model
- HOG (Histogram of Oriented Gradients) feature extraction
- StandardScaler for feature normalization
- SVM classifier with RBF kernel

### CNN Model
- ResNet50 as backbone (pre-trained on ImageNet)
- Transfer learning with fine-tuning
- Data augmentation (random flips, rotations, color jitter)
- Model checkpointing (saves best model based on validation AUC)
- Comprehensive performance metrics (accuracy, AUC, F1)

### Autoencoder Model
- Convolutional autoencoder architecture
- Trains only on real faces as normal samples
- Anomaly detection approach using reconstruction error
- Automatic threshold determination (μ + 2σ of real samples)
- Visualizes reconstructions and error maps
- Provides detailed error histograms

## Model Performance

All models provide detailed evaluation metrics:
- Classification reports with precision, recall, and F1-score
- Confusion matrices
- ROC curves and AUC scores
- Prediction confidence scores

## Results Visualization

Models generate visualizations in the `results` directory:
- CNN: batch visualization, training metrics plots, ROC curve, confusion matrix
- Autoencoder: reconstructions, error maps, error histograms, training loss plots

## Logging

The project maintains log files that record all operations and any errors that occur during execution:
- `face_authenticity.log` - Main log file for SVM model
- `face_authenticity_cnn.log` - Log file for CNN model
- `face_authenticity_autoencoder.log` - Log file for Autoencoder model

