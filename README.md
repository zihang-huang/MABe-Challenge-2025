# MABe 2025 Mouse Behavior Challenge

This repository contains code for the MABe 2025 challenge, focusing on multi-agent mouse behavior classification.

## Project Structure

- `MABe-Challenge.ipynb`: Main notebook for data exploration, preprocessing, and feature engineering.
- `TCN.ipynb`: Notebook for training the Temporal Convolutional Network (TCN) model.
- `data/`: Directory for raw and processed data.
- `outputs/`: Directory for model outputs and visualizations.

## Getting Started

1.  **Environment Setup:**
    Ensure you have Python 3.10+ and the required packages installed.
    ```bash
    pip install pandas numpy matplotlib seaborn scipy tqdm pyarrow torch scikit-learn
    ```

2.  **Data Preprocessing:**
    Run the `MABe-Challenge.ipynb` notebook to:
    - Load and explore the raw data.
    - Normalize spatial coordinates.
    - Resample to a common frame rate (30 Hz).
    - Extract common body parts.
    - Engineer features (velocity, acceleration, heading, inter-mouse distances).
    - Preprocess the entire training dataset and save to `data/processed/train`.

3.  **Model Training:**
    Run the `TCN.ipynb` notebook to train the TCN model.
    - Utilizes MPS acceleration on macOS (or CUDA/CPU).
    - Optimized with parallel data loading and increased batch size.
    - Saves the trained model to `tcn_mouse_behavior.pth`.
    - Plots training history to `training_history.png`.

## Model Details

The model is a Temporal Convolutional Network (TCN) designed for sequence modeling.
- **Input:** Sequence of engineered features (positions, velocities, etc.) for each mouse.
- **Architecture:** Stacked Temporal Blocks with dilated convolutions to capture long-range dependencies.
- **Output:** Multi-label classification for each time step (frame).

## Optimization

- **Data Loading:** Uses `num_workers=4` and `pin_memory=True` for efficient data transfer.
- **Batching:** `BATCH_SIZE=64` to maximize GPU utilization.
- **Hardware:** Automatically detects and uses `mps` (macOS), `cuda`, or `cpu`.