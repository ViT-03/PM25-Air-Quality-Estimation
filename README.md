# PM2.5 Air Quality Estimation

This project implements a machine learning pipeline to estimate PM2.5 levels using a multi-modal approach. It combines images (extracting visual features and applying Dark Channel Prior - DCP) with meteorological data to predict air quality.

## Repository Structure
Based on the project organization, the repository is divided into:
* **`data/`**: Contains the original dataset and processed data. Please read `data/README_DATA.md` for instructions on data configuration.
* **`src/`**: Contains scripts for data preprocessing and generating folds for cross-validation.
* **`experiments/`**: Contains the training and validation scripts for the various model versions.

## Execution Pipeline

### 1. Data Preparation
Navigate to the root directory and run the scripts located in `src/`:
1. Run `prepare_dataset.py`: This enriches the dataset by downloading historical weather data from the Open-Meteo API and calculates DCP (Dark Channel Prior) features, saving the "dehazed" images as well.
2. Run `create_kfold.py`: This creates a new `NewDataset` folder containing the enriched metadata mapped with assignments for a 5-Fold Stratified Cross-Validation.

### 2. Model Training
The project includes three different architectural iterations. Each features 4 ablation configurations (Base, Base+DCP, Base+Weather, Base+Weather+DCP):
* **V2 (`train_v2_kfold.py`)**: Baseline model based on EfficientNet-B0 with K-Fold training.
* **V3 (`train_v3_convnext.py`)**: Multi-task architecture based on ConvNeXt-Tiny for feature extraction, fused (Late Fusion) with an XGBoost model.
* **V4 (`train_v4_final.py`)**: An improvement over V3 that introduces hyperparameter optimization using Optuna (50 trials), CutMix data augmentation, and Asymmetric L1 Loss.

### 3. Evaluation and Statistics
* **`ablation.py`**: Analyzes the predictions and loss curves generated in the `NewResults/` folder, creating grouped bar charts, scatter plots, and error distributions, and saves them in `final_ablation_files/`.
* **`statistical_test.py`**: Verifies the statistical significance of the architectural results (both intra-version and inter-version) using paired t-tests and the Wilcoxon signed-rank test.