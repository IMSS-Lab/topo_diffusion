# Running Instructions for Topological Materials Diffusion Project

This document provides detailed instructions for running the Topological Materials Diffusion project, which uses real JARVIS datasets to train a generative diffusion model for sustainable topological quantum materials.

## Environment Setup

1. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Install PyTorch and related packages:
   ```bash
   pip install torch
   pip install torch-geometric
   pip install torch-scatter
   ```

## Data Pipeline

The project uses real data from the JARVIS database. The data pipeline consists of the following steps:

1. **Download JARVIS datasets**:
   ```bash
   python src/download_jarvis.py
   ```
   This script will:
   - Download DFT and superconductor datasets from JARVIS
   - Process the data and extract topological materials
   - Create train/validation splits

2. **Verify downloaded data**:
   After running the download script, you should have the following files:
   - `data/raw/dft_3d.json`: Raw DFT data
   - `data/raw/supercon_3d.json`: Raw superconductor data
   - `data/processed/processed_dft.json`: Processed DFT data
   - `data/processed/topological_materials.json`: Extracted topological materials
   - `data/processed/unified_dataset.json`: Merged dataset
   - `data/processed/train_dataset.json`: Training split
   - `data/processed/val_dataset.json`: Validation split

## Model Training

1. **Train the model**:
   ```bash
   python src/train.py
   ```

2. **Training options**:
   The training script accepts several command-line arguments:
   ```bash
   python src/train.py --batch_size 32 --num_epochs 100 --learning_rate 1e-4
   ```

   Key arguments:
   - `--train_data`: Path to training data (default: `data/processed/train_dataset.json`)
   - `--val_data`: Path to validation data (default: `data/processed/val_dataset.json`)
   - `--batch_size`: Batch size (default: 32)
   - `--num_epochs`: Number of epochs (default: 100)
   - `--learning_rate`: Learning rate (default: 1e-4)
   - `--device`: Device to use (default: "cuda" if available, otherwise "cpu")

3. **Monitor training**:
   The training script will output metrics for each epoch, including:
   - Training loss (noise prediction and property prediction)
   - Validation loss (noise prediction and property prediction)

4. **Checkpoints**:
   The training script saves checkpoints to the `checkpoints` directory:
   - `best_model.pt`: Best model based on validation loss
   - `latest_model.pt`: Latest model

## Material Generation

1. **Generate materials**:
   ```bash
   python src/main.py generate --model-checkpoint checkpoints/best_model.pt
   ```

2. **Validate generated materials**:
   ```bash
   python src/main.py validate --input-dir generated_materials
   ```

## Troubleshooting

1. **Memory issues during training**:
   - Reduce batch size: `--batch_size 8`
   - Use CPU if GPU memory is limited: `--device cpu`

2. **JARVIS download issues**:
   - The script automatically falls back to using the JARVIS-Tools API if direct download fails
   - Ensure you have internet connectivity

3. **PyTorch installation issues**:
   - For CPU-only installation: `pip install torch --index-url https://download.pytorch.org/whl/cpu`
   - For specific CUDA versions, see PyTorch installation guide

## Project Structure

- `src/`: Source code
  - `data.py`: Data processing and dataset classes
  - `model.py`: Diffusion model architecture
  - `training.py`: Training utilities
  - `utils.py`: Utility functions
  - `main.py`: Command-line interface
  - `download_jarvis.py`: JARVIS data downloader
  - `train.py`: Training script
- `data/`: Data directory
  - `raw/`: Raw data
  - `processed/`: Processed data
- `config.yaml`: Model configuration
- `requirements.txt`: Dependencies
