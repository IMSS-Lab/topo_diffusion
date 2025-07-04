#!/bin/bash

# Script to check and fix NumPy version compatibility issues
# and validate datasets before running the training script

echo "Checking NumPy version..."
NUMPY_VERSION=$(python -c "import numpy; print(numpy.__version__)")
echo "Current NumPy version: $NUMPY_VERSION"

if [[ $NUMPY_VERSION == 2.* ]]; then
    echo "WARNING: NumPy version 2.x detected. This may cause compatibility issues with PyTorch and torch_geometric."
    echo "Downgrading NumPy to version <2.0.0..."
    pip install 'numpy<2.0.0' --force-reinstall
    echo "NumPy downgraded successfully."
fi

echo "Checking for dataset files..."
TRAIN_DATASET="data/processed/train_dataset.json"
VAL_DATASET="data/processed/val_dataset.json"

if [ ! -f "$TRAIN_DATASET" ] || [ ! -f "$VAL_DATASET" ]; then
    echo "ERROR: Dataset files not found. Running data processing pipeline..."
    python src/download_jarvis.py
else
    # Check if datasets are empty
    TRAIN_SIZE=$(python -c "import json; f=open('$TRAIN_DATASET'); data=json.load(f); print(len(data)); f.close()")
    VAL_SIZE=$(python -c "import json; f=open('$VAL_DATASET'); data=json.load(f); print(len(data)); f.close()")
    
    echo "Training dataset size: $TRAIN_SIZE"
    echo "Validation dataset size: $VAL_SIZE"
    
    if [ "$TRAIN_SIZE" -eq 0 ] || [ "$VAL_SIZE" -eq 0 ]; then
        echo "ERROR: Empty datasets detected. Re-running data processing pipeline..."
        python src/download_jarvis.py
    else
        echo "Datasets validated successfully."
    fi
fi

echo "All checks passed. Ready to run training script."
echo "Run with: python src/train.py"
