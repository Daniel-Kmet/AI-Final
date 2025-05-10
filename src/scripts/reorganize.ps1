# Move data processing files
Move-Item -Path "preprocess_faces.py" -Destination "src/data/processing/"
Move-Item -Path "augmentations.py" -Destination "src/data/processing/"
Move-Item -Path "extract_lbp_features.py" -Destination "src/data/processing/"
Move-Item -Path "verify_data.py" -Destination "src/data/processing/"
Move-Item -Path "split_data.ps1" -Destination "src/data/processing/"
Move-Item -Path "rename_files.ps1" -Destination "src/data/processing/"
Move-Item -Path "organize_data.ps1" -Destination "src/data/processing/"

# Move model architecture files
Move-Item -Path "src/cnn_model.py" -Destination "src/models/architectures/"
Move-Item -Path "src/autoencoder_model.py" -Destination "src/models/architectures/"
Move-Item -Path "src/ensemble_model.py" -Destination "src/models/architectures/"

# Move training files
Move-Item -Path "train_svm_classifier.py" -Destination "src/models/training/"
Move-Item -Path "src/run_cnn.py" -Destination "src/models/training/"
Move-Item -Path "src/run_autoencoder.py" -Destination "src/models/training/"
Move-Item -Path "src/train_classifier.py" -Destination "src/models/training/"

# Move utility files
Move-Item -Path "src/utils/reporting.py" -Destination "src/utils/metrics/"
Move-Item -Path "src/utils/training_metrics.py" -Destination "src/utils/metrics/"
Move-Item -Path "src/utils/visualization.py" -Destination "src/utils/visualization/"

# Move log files
Move-Item -Path "face_authenticity_cnn.log" -Destination "src/utils/logging/"
Move-Item -Path "face_authenticity.log" -Destination "src/utils/logging/"

# Move main scripts
Move-Item -Path "src/main.py" -Destination "src/scripts/"
Move-Item -Path "src/run_pipeline.py" -Destination "src/scripts/"
Move-Item -Path "src/test_directml.py" -Destination "src/scripts/"

# Create __init__.py files
New-Item -Path "src/data/__init__.py" -ItemType File
New-Item -Path "src/data/processing/__init__.py" -ItemType File
New-Item -Path "src/data/datasets/__init__.py" -ItemType File
New-Item -Path "src/models/__init__.py" -ItemType File
New-Item -Path "src/models/architectures/__init__.py" -ItemType File
New-Item -Path "src/models/training/__init__.py" -ItemType File
New-Item -Path "src/utils/__init__.py" -ItemType File
New-Item -Path "src/utils/visualization/__init__.py" -ItemType File
New-Item -Path "src/utils/metrics/__init__.py" -ItemType File
New-Item -Path "src/utils/logging/__init__.py" -ItemType File
New-Item -Path "src/config/__init__.py" -ItemType File
New-Item -Path "src/scripts/__init__.py" -ItemType File 