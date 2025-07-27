
# Deconstructing Taxi Income: A Decoupled Framework for Spatio-Temporal Forecasting under Sparsity

This repository contains the official implementation for the KDD 2025 paper, "Deconstructing Taxi Income: A Decoupled Framework for Spatio-Temporal Forecasting under Sparsity".

## 1. Overview

This project introduces DHKI-Net, a novel deep learning architecture designed for spatio-temporal forecasting tasks, particularly in scenarios with sparse observational data. The core hypothesis is that by infusing a model with stable, long-term **prior knowledge** about the intrinsic properties of different regions, we can significantly improve prediction robustness and accuracy.

The model is applied to a taxi income prediction task, framed as a multi-task learning problem to predict future demand and income in different city zones.

## 2. Setup and Usage

This guide provides the steps to set up the environment and reproduce the main experimental results presented in the paper.

### Step 1: Environment Setup

It is recommended to use a Conda environment. You can set up the environment and install all necessary dependencies using the provided `requirements.txt` file.

```bash
# Create a new conda environment (optional, but recommended)
conda create -n dhki_net python=3.10
conda activate dhki_net

# Install all required packages
pip install -r requirements.txt
```

### Step 2: Data Preparation

Due to data privacy restrictions, we are unable to provide the raw dataset. 

### Step 3: Reproducing Experimental Results

Once the environment is set up, you can directly run the training scripts to reproduce the main results.

-   **Train the full DHKI-Net Model**:
    This command will train our proposed model using the default settings.
    ```bash
    python src/train_dhk.py --config config/main_config.yaml
    ```

-   **Train a Baseline Model (e.g., LSTM)**:
    This command will train the LSTM baseline model.
    ```bash
    python src/train_baseline.py --config config/main_config.yaml --baseline_model lstm
    ```

-   **Run Ablation Studies**:
    You can run the ablation studies described in the paper by using the command-line flags. For example, to train the model without the Contextual Stream:
    ```bash
    python src/train_dhk.py --config config/main_config.yaml --disable_contrastive_learning
    ```

All results, including model checkpoints, logs, and performance metrics, will be saved to the `results/` directory, which will be created automatically.

## 4. Code Structure

-   `config/`: Contains the main configuration file (`main_config.yaml`) with all hyperparameters.
-   `data/processed/`: Contains the pre-processed and anonymized data files.
-   `src/`: Contains the core source code.
    -   `models/`: Implementation of the DHKI-Net architecture and its components.
    -   `dataset/`: PyTorch `Dataset` and `DataLoader` classes.
    -   `baselines/`: Implementation of all baseline models.
    -   `utils/`: Utility functions for loss calculation, metrics, etc.
    -   `train_dhk.py`: Main script to train and evaluate the DHKI-Net.
    -   `train_baseline.py`: Main script to train and evaluate the baseline models.
-   `requirements.txt`: A list of all Python dependencies.
