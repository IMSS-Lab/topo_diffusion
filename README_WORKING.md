# Working Instructions for Topological Quantum Materials Generation

This document provides detailed instructions to set up and run the generative diffusion model for sustainable topological quantum materials.

## Prerequisites

Ensure that you have Python 3.11 installed on your system.

## Setup and Installation

1. **Install Python 3.11**
   - Download and install Python 3.11 from the [official Python website](https://www.python.org/downloads/).
   - Verify the installation by running `python --version` in your terminal.

2. **Create a Virtual Environment**
   - Use Python 3.11 to create a virtual environment to isolate your project dependencies.
   ```bash
   python -m venv venv
   ```

3. **Activate the Virtual Environment**
   - Activate the virtual environment to ensure all installations are contained within it.
   ```bash
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

4. **Install Required Dependencies**
   - Install all necessary Python packages using the provided `requirements.txt` file.
   ```bash
   pip install -r requirements.txt
   ```

5. **Install `torch-scatter`**
   - Run the following command to install `torch-scatter`:
   ```bash
   pip install torch-scatter
   ```
   - If the installation fails due to module identification issues, follow these steps to install from source:
     1. Clone the repository:
     ```bash
     git clone https://github.com/rusty1s/pytorch_scatter.git
     ```
     2. Navigate into the cloned directory and install:
     ```bash
     cd pytorch_scatter
     python setup.py install
     ```

6. **Run the Numpy Fix Script**
   - Execute the validation and fix script to ensure compatibility.
   ```bash
   sh validate_and_fix.sh
   ```

7. **Download Training Data**
   - Use the provided script to download necessary training datasets.
   ```bash
   python src/download_jarvis.py
   ```

8. **Run the Training Process**
   - Train the model using the training script.
   ```bash
   python src/train.py
   ```

9. **Generate Materials**
   - Generate new materials using the trained model.
   ```bash
   python src/main.py generate --model-checkpoint checkpoints/latest_model.pt
   ```

10. **Validate Generated Materials**
    - Validate the generated materials to ensure their quality and properties.
    ```bash
    python src/main.py validate --input-dir generated_materials
    ```

Follow these steps carefully to ensure a successful setup and execution of the generative diffusion model.

