# Data Documentation (PM2.5 Dataset)

This folder is intended to store both the raw and processed data used for training the models.

## Initial Required Structure
For the `src/prepare_dataset.py` script to function correctly, you must place the raw data in a folder named `pm25dataset` with the following tree structure:

```text
pm25dataset/
├── train/
│   ├── images/              # Original training images
│   └── metadata_enriched.csv # (or equivalent base csv file)
└── test/
    ├── images/              # Original test images
    └── metadata_enriched.csv # (or equivalent base csv file)
```

## Data Generated from Preprocessing
After running `src/prepare_dataset.py`, the following folders and files will be automatically generated:
* **`cache/open_meteo_cache.sqlite`**: A local database cache to limit repeated API calls to Open-Meteo.
* **`pm25dataset/train/images_dehazed/`** and **`pm25dataset/test/images_dehazed/`**: Folders containing images that have had haze removed using the DCP algorithm.
* New CSV files featuring additional weather variables (e.g., `temperature_2m`, `is_rain`) and visual features (e.g., `dcp_dark_mean`, `dcp_haze_score`).

After running `src/create_kfold.py`, a new root directory named `NewDataset/` will be created. It will contain the unified images and the `metadata_enriched_folds.csv` file, fully prepared for model training.