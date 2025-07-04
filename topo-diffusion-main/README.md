# Generative Diffusion Model for Sustainable Topological Quantum Materials

This repository contains the implementation of a novel computational framework to discover sustainable topological quantum materials using deep generative models. The framework leverages a conditional diffusion model operating on crystal graphs, utilizing three key JARVIS datasets: JARVIS-DFT, JARVIS-TOPO, and JARVIS-ML.

## Project Overview

The model incorporates multi-objective conditioning on topological properties, stability, and sustainability metrics to generate novel materials with targeted characteristics. This approach addresses the urgent need for sustainable alternatives to rare-earth-based topological materials while advancing the field of AI-driven materials discovery.

### Key Features

- Crystal graph representation framework for structural and electronic features
- Conditional diffusion model for crystal structure generation with multi-objective optimization
- Integration with JARVIS datasets for comprehensive materials property information
- Validation pipeline for computational screening and property verification
- Open-source framework for sustainable materials discovery

## Installation

```bash
# Clone the repository
git clone https://github.com/brandonyee-cs/topo_diffusion.git
cd topo_diffusion_refactored

# Create and activate a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Project Structure

The project has been refactored to have a simpler, flatter structure:

```
topo_diffusion_refactored/
├── src/                # Core source code
│   ├── __init__.py     # Package initialization
│   ├── data.py         # Data processing and crystal graph representation
│   ├── model.py        # Diffusion model architecture
│   ├── training.py     # Training pipeline and validation tools
│   ├── utils.py        # Utility functions
│   └── main.py         # Command-line interface and main scripts
├── config.yaml         # Configuration file
└── requirements.txt    # Dependencies
```

## Usage

The project provides a command-line interface for all major operations:

### Data Processing

To download and process JARVIS datasets:

```bash
python -m src.main download --output-dir data/raw
```

Optional arguments:
- `--limit`: Limit the number of structures to download
- `--log-level`: Set logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)

### Training a Model

To train the diffusion model:

```bash
python -m src.main train --config config.yaml --data-dir data/processed --checkpoint-dir checkpoints
```

Optional arguments:
- `--use-wandb`: Enable Weights & Biases logging
- `--log-level`: Set logging level

### Generating Materials

To generate new materials using a trained model:

```bash
python -m src.main generate --model-checkpoint checkpoints/best_model.pt --num-samples 10 --output-dir generated_materials
```

### Validating Generated Materials

To validate generated materials:

```bash
python -m src.main validate --input-dir generated_materials --output-dir validation_results
```

Optional arguments:
- `--run-dft`: Enable DFT validation
- `--log-level`: Set logging level

## Development Workflow

1. **Extend the data processing module**:
   - Implement actual data downloading from JARVIS APIs in `src/data.py`
   - Complete the crystal graph conversion functionality

2. **Implement the diffusion model**:
   - Complete the forward pass implementation in `src/model.py`
   - Implement the noise schedule and diffusion process
   - Add the conditioning mechanisms

3. **Complete the training pipeline**:
   - Implement the actual training loop in `src/training.py`
   - Add monitoring and visualization

4. **Implement the validation pipeline**:
   - Add DFT validation interfaces in `src/training.py` (MaterialValidator class)
   - Implement property prediction

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Citation

If you use this code in your research, please cite:

```
@article{topo_diffusion2025,
  title={Generative Diffusion Model for Sustainable Topological Quantum Materials using JARVIS Datasets},
  author={Research Team},
  journal={TBD},
  year={2025}
}
```

## Acknowledgements

This project leverages the JARVIS framework developed by Choudhary, K., et al. (2020).
