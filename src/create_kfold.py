import pandas as pd
import numpy as np
import shutil
from pathlib import Path
from sklearn.model_selection import StratifiedKFold

from prepare_dataset import (
    build_enriched_metadata,
    TRAIN_CSV,
    TEST_CSV,
    TRAIN_IMG_DIR,
    TEST_IMG_DIR,
    TARGET_REG_COL_CANDIDATES
)

NEW_ROOT = Path("NewDataset")
NEW_IMG_DIR = NEW_ROOT / "images"
NEW_DCP_DIR = NEW_ROOT / "images_dehazed"
NEW_CSV_RAW = NEW_ROOT / "metadata_raw.csv"
NEW_CSV_ENRICHED = NEW_ROOT / "metadata_enriched.csv"
NEW_CSV_FOLDS = NEW_ROOT / "metadata_enriched_folds.csv"

AQI_BINS = [0, 51, 101, 151, 201, 301, float("inf")]

def aqi_to_class(value):
    for i in range(len(AQI_BINS) - 1):
        if AQI_BINS[i] <= value < AQI_BINS[i + 1]:
            return i
    return len(AQI_BINS) - 2


def setup_new_dataset():
    NEW_IMG_DIR.mkdir(parents=True, exist_ok=True)

    train_df = pd.read_csv(TRAIN_CSV)
    test_df = pd.read_csv(TEST_CSV)

    def copy_images(df, src_dir):
        img_col = next((c for c in ["image", "filename", "file_name", "img", "image_name", "path"] if c in df.columns),
                       None)
        for img in df[img_col]:
            src_path = Path(src_dir) / str(img)
            dst_path = NEW_IMG_DIR / str(img)
            if src_path.exists() and not dst_path.exists():
                shutil.copy(src_path, dst_path)

    copy_images(train_df, TRAIN_IMG_DIR)
    copy_images(test_df, TEST_IMG_DIR)

    merged_df = pd.concat([train_df, test_df], ignore_index=True)
    merged_df.to_csv(NEW_CSV_RAW, index=False)

def enrich_and_fold():
    enriched_df = build_enriched_metadata(
        input_csv=NEW_CSV_RAW,
        image_dir=NEW_IMG_DIR,
        output_csv=NEW_CSV_ENRICHED,
        save_dehazed=True,
        dehazed_dir=NEW_DCP_DIR,
        use_weather=True,
        use_dcp=True,
        overwrite=False,
        save_every=50
    )

    # K-FOLD STRATIFIED SPLIT
    target_col = next((c for c in TARGET_REG_COL_CANDIDATES if c in enriched_df.columns), None)

    y_classes = enriched_df[target_col].apply(aqi_to_class).values
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    enriched_df["fold"] = -1
    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(np.zeros(len(y_classes)), y_classes)):
        enriched_df.loc[val_idx, "fold"] = fold_idx

    enriched_df.to_csv(NEW_CSV_FOLDS, index=False)

if __name__ == "__main__":
    setup_new_dataset()
    enrich_and_fold()