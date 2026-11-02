import copy
import random
import os
import numpy as np
import pandas as pd
from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, r2_score

from pm2_5_dataset import PM25Dataset, train_tfms, test_tfms

BATCH_SIZE = 32
EPOCHS = 20
EPOCHS_HEAD = 8
EPOCHS_FULL = 12
LR_HEAD = 1e-3
LR_FULL_BACKBONE = 1e-5
LR_FULL_HEAD = 1e-4
LR = 1e-4
WEIGHT_DECAY = 1e-4
NUM_WORKERS = 4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
K_FOLDS = 5

AQI_BINS = [0, 51, 101, 151, 201, 301, float("inf")]

EXPERIMENTS_V2 = [
    {"name": "base", "use_weather": False, "use_dcp": False},
    {"name": "base_dcp", "use_weather": False, "use_dcp": True},
    {"name": "base_weather", "use_weather": True, "use_dcp": False},
    {"name": "base_weather_dcp", "use_weather": True, "use_dcp": True},
]

IMAGE_SIZE = 224

def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def aqi_to_class(value):
    for i in range(len(AQI_BINS) - 1):
        if AQI_BINS[i] <= value < AQI_BINS[i + 1]:
            return i
    return len(AQI_BINS) - 2

def regression_to_class(preds):
    return np.array([aqi_to_class(float(x)) for x in preds], dtype=np.int64)

class PM25Net(nn.Module):
    def __init__(self, extra_dim=0, dropout=0.0):
        super().__init__()
        weights = EfficientNet_B0_Weights.IMAGENET1K_V1
        self.backbone = efficientnet_b0(weights=weights)
        in_features = self.backbone.classifier[1].in_features
        self.backbone.classifier = nn.Identity()

        hidden = 256
        self.head = nn.Sequential(
            nn.Linear(in_features + extra_dim, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )

    def forward(self, x_img, x_extra=None):
        f = self.backbone(x_img)
        if x_extra is not None:
            f = torch.cat([f, x_extra], dim=1)
        out = self.head(f)
        return out.squeeze(1)

def compute_regression_metrics(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)

    y_true_cls = regression_to_class(y_true)
    y_pred_cls = regression_to_class(y_pred)

    return {
        "r2": r2,
        "mae": mae
    }

def train_one_epoch(model, loader, optimizer, criterion):
    model.train()
    total_loss = 0.0

    for x_img, x_extra, y in loader:
        x_img = x_img.to(DEVICE)
        y = y.to(DEVICE)
        if x_extra is not None:
            x_extra = x_extra.to(DEVICE)

        optimizer.zero_grad()
        out = model(x_img, x_extra)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * x_img.size(0)

    return total_loss / len(loader.dataset)

@torch.no_grad()
def evaluate(model, loader, criterion):
    model.eval()
    total_loss = 0.0
    ys, preds = [], []

    for x_img, x_extra, y in loader:
        x_img = x_img.to(DEVICE)
        y = y.to(DEVICE)
        if x_extra is not None:
            x_extra = x_extra.to(DEVICE)

        out = model(x_img, x_extra)
        loss = criterion(out, y)

        pred = out.detach().cpu().numpy()
        target = y.detach().cpu().numpy()

        total_loss += loss.item() * x_img.size(0)
        ys.append(target)
        preds.append(pred)

    ys = np.concatenate(ys)
    preds = np.concatenate(preds)
    avg_loss = total_loss / len(loader.dataset)

    metrics = compute_regression_metrics(ys, preds)

    return avg_loss, metrics, ys, preds

def save_loss_plot(history_df, out_path, metric_key="r2"):
    fig, ax1 = plt.subplots(figsize=(8, 5))

    for phase_name, g in history_df.groupby("phase"):
        ax1.plot(g["epoch"], g["train_loss"], marker="o", label=f"{phase_name}_train_loss")
        ax1.plot(g["epoch"], g["val_loss"], marker="s", linestyle="--", label=f"{phase_name}_val_loss")

    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()

def freeze_backbone(model):
    for p in model.backbone.parameters():
        p.requires_grad = False
    for p in model.head.parameters():
        p.requires_grad = True

def unfreeze_last_blocks(model, n_blocks=2):
    for p in model.backbone.parameters():
        p.requires_grad = False
    for block in model.backbone.features[-n_blocks:]:
        for p in block.parameters():
            p.requires_grad = True
    for p in model.head.parameters():
        p.requires_grad = True

def unfreeze_all(model):
    for p in model.parameters():
        p.requires_grad = True

def build_criterion():
    return nn.MSELoss(), "r2", True

def compute_extra_dim(use_weather, use_dcp):
    weather_dim = 18
    dcp_dim = 16

    extra_dim = 0
    if use_weather:
        extra_dim += weather_dim
    if use_dcp:
        extra_dim += dcp_dim
    return extra_dim

def run_phase(model, train_loader, val_loader, criterion, phase_name, epochs, optimizer, best_key, maximize,
              start_epoch=0):
    best_score = -float("inf") if maximize else float("inf")
    best_state = None
    history = []

    for local_epoch in range(1, epochs + 1):
        global_epoch = start_epoch + local_epoch

        train_loss = train_one_epoch(model, train_loader, optimizer, criterion)
        val_loss, val_metrics, _, _ = evaluate(model, val_loader, criterion)

        score = val_metrics[best_key]
        improved = score > best_score if maximize else score < best_score
        if improved:
            best_score = score
            best_state = copy.deepcopy(model.state_dict())

        row = {
            "phase": phase_name,
            "epoch": global_epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            **val_metrics,
        }
        history.append(row)

        print(f"[{phase_name}] Epoch {global_epoch:02d}")
        print(f" train_loss: {train_loss:.4f}")
        print(f" val_loss  : {val_loss:.4f}")
        print(f" val_metrics: {val_metrics}")
        print(f" best_{best_key}: {best_score:.4f}")
        print("-" * 60)

    if best_state is not None:
        model.load_state_dict(best_state)

    return model, history, best_score

def get_kfold_dataloaders(fold, full_csv, img_dir, use_weather, use_dcp):
    full_df = pd.read_csv(full_csv)
    train_df = full_df[full_df["fold"] != fold].reset_index(drop=True)
    val_df = full_df[full_df["fold"] == fold].reset_index(drop=True)

    train_csv = f"temp_train_fold{fold}.csv"
    val_csv = f"temp_val_fold{fold}.csv"
    train_df.to_csv(train_csv, index=False)
    val_df.to_csv(val_csv, index=False)

    train_ds = PM25Dataset(
        train_csv,
        img_dir,
        transform=train_tfms,
        use_weather=use_weather,
        use_dcp=use_dcp,
    )
    val_ds = PM25Dataset(
        val_csv,
        img_dir,
        transform=test_tfms,
        use_weather=use_weather,
        use_dcp=use_dcp,
    )

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)

    return train_loader, val_loader, train_csv, val_csv

def run_experiment_v2(cfg):
    print(f"\n{'=' * 60}")
    print(f"RUN K-FOLD: {cfg['name']}")
    print(f"{'=' * 60}")

    extra_dim = compute_extra_dim(cfg["use_weather"], cfg["use_dcp"])
    dropout = 0.1 if cfg["name"] == "base" else 0.3

    history_csv = f"history_v2_{cfg['name']}.csv"
    pred_csv = f"predictions_v2_{cfg['name']}.csv"

    oof_y_true = []
    oof_y_pred = []
    full_csv = Path("NewDataset") / "metadata_enriched_folds.csv"
    img_dir = Path("NewDataset") / "images"

    all_history = []

    for fold in range(K_FOLDS):
        print(f"\n--- FOLD {fold + 1}/{K_FOLDS} ---")
        train_loader, val_loader, train_csv, val_csv = get_kfold_dataloaders(
            fold, full_csv, img_dir, cfg["use_weather"], cfg["use_dcp"]
        )

        model = PM25Net(
            extra_dim=extra_dim,
            dropout=dropout,
        ).to(DEVICE)

        criterion, best_key, maximize = build_criterion()
        epoch_cursor = 0

        # Phase 1
        freeze_backbone(model)
        optimizer = torch.optim.AdamW(model.head.parameters(), lr=LR_HEAD, weight_decay=WEIGHT_DECAY)
        model, hist1, _ = run_phase(
            model, train_loader, val_loader, criterion, f"head_only_f{fold}", EPOCHS_HEAD, optimizer, best_key,
            maximize, start_epoch=epoch_cursor
        )
        all_history.extend(hist1)
        epoch_cursor += EPOCHS_HEAD

        # Phase 2
        unfreeze_all(model)
        optimizer = torch.optim.AdamW(
            [
                {"params": model.backbone.parameters(), "lr": LR_FULL_BACKBONE},
                {"params": model.head.parameters(), "lr": LR_FULL_HEAD},
            ],
            weight_decay=WEIGHT_DECAY
        )
        model, hist2, best_val_score = run_phase(
            model, train_loader, val_loader, criterion, f"full_ft_f{fold}", EPOCHS_FULL, optimizer, best_key, maximize,
            start_epoch=epoch_cursor
        )
        all_history.extend(hist2)

        # OOF Predictions
        _, _, y_true, y_pred = evaluate(model, val_loader, criterion)
        oof_y_true.append(y_true)
        oof_y_pred.append(y_pred)

        if os.path.exists(train_csv): os.remove(train_csv)
        if os.path.exists(val_csv): os.remove(val_csv)

    y_true_all = np.concatenate(oof_y_true)
    y_pred_all = np.concatenate(oof_y_pred)

    metrics = compute_regression_metrics(y_true_all, y_pred_all)

    pd.DataFrame(all_history).to_csv(history_csv, index=False)

    pred_df = pd.DataFrame({
        "y_true": y_true_all,
        "y_pred": y_pred_all,
        "error": y_pred_all - y_true_all,
    })
    pred_df.to_csv(pred_csv, index=False)

    return {
        "experiment": cfg["name"],
        "use_weather": cfg["use_weather"],
        "use_dcp": cfg["use_dcp"],
        "history_csv": history_csv,
        "prediction_csv": pred_csv,
        "r2": metrics["r2"],
        "mae": metrics["mae"]
    }

if __name__ == "__main__":
    torch.cuda.empty_cache()

    seed_everything(42)
    results = []
    for cfg in EXPERIMENTS_V2:
        res = run_experiment_v2(cfg)
        results.append(res)
        print(f"\nRESULTS {cfg['name']}: R2={res['r2']:.4f} | MAE={res['mae']:.4f}")

    df_results = pd.DataFrame(results)
    df_results.to_csv("ablation_v2_results.csv", index=False)