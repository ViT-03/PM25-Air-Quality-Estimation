import os
import copy
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import xgboost as xgb
from PIL import Image
from pathlib import Path
from torch.utils.data import DataLoader
from torchvision.models import convnext_tiny, ConvNeXt_Tiny_Weights
from torchvision import transforms
from prepare_dataset import PM25Dataset, TARGET_CLASS_COL_CANDIDATES
from train_v2_kfold import (
    NUM_WORKERS, DEVICE, EPOCHS_FULL, WEIGHT_DECAY,
    K_FOLDS, compute_regression_metrics, aqi_to_class, seed_everything)

BATCH_SIZE = 8
ACCUMULATION_STEPS = 4
LAMBDA_REG = 1.0
LAMBDA_CLS = 0.5

EXPERIMENTS_V3 = [
    {"name": "base", "use_weather": False, "use_dcp": False},
    {"name": "base_dcp", "use_weather": False, "use_dcp": True},
    {"name": "base_weather", "use_weather": True, "use_dcp": False},
    {"name": "base_dcp_weather", "use_weather": True, "use_dcp": True},
]

new_train_tfms = transforms.Compose([
    transforms.Resize((300, 300)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

new_val_tfms = transforms.Compose([
    transforms.Resize((300, 300)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

class PM25DatasetV2(PM25Dataset):
    def __init__(self, csv_path, image_dir, transform=None, use_weather=True, use_dcp=True):
        super().__init__(csv_path, image_dir, task="regression", transform=transform,
                         use_weather=use_weather, use_dcp=use_dcp, use_dehazed_images=False)

        self.class_col = None
        for c in TARGET_CLASS_COL_CANDIDATES:
            if c in self.df.columns:
                self.class_col = c
                break

        if "dehazed_image_path" not in self.df.columns:
            raise ValueError("'dehazed_image_path' missing.")

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        img_name = str(row[self.image_col])
        img_path = self.image_dir / img_name
        image_orig = Image.open(img_path).convert("RGB")

        dehazed_path = Path(str(row["dehazed_image_path"]).replace("\\", "/"))
        image_dehazed = Image.open(dehazed_path).convert("RGB")

        if self.transform:
            image_orig = self.transform(image_orig)
            image_dehazed = self.transform(image_dehazed)

        image_6c = torch.cat([image_orig, image_dehazed], dim=0)

        y_reg = torch.tensor(float(row[self.target_col]), dtype=torch.float32)
        if self.class_col is not None and pd.notna(row[self.class_col]):
            y_cls = int(row[self.class_col])
        else:
            y_cls = aqi_to_class(float(row[self.target_col]))
        y_cls = torch.tensor(y_cls, dtype=torch.long)

        extras = []
        if self.use_weather:
            extras.extend([float(row[col]) if pd.notna(row[col]) else 0.0 for col in self.weather_cols])
        if self.use_dcp:
            extras.extend([float(row[col]) if pd.notna(row[col]) else 0.0 for col in self.dcp_cols])

        x_extra = torch.tensor(extras, dtype=torch.float32) if extras else torch.empty(0)

        return image_6c, x_extra, y_reg, y_cls

def adapt_first_layer_for_6_channels(model):
    first_conv = model.features[0][0]
    new_conv = nn.Conv2d(6, first_conv.out_channels, kernel_size=first_conv.kernel_size,
                         stride=first_conv.stride)
    with torch.no_grad():
        new_conv.weight[:, :3] = first_conv.weight.clone() / 2.0
        new_conv.weight[:, 3:] = first_conv.weight.clone() / 2.0
    model.features[0][0] = new_conv

class PM25MultiTaskNetV2(nn.Module):
    def __init__(self, num_classes=6, extra_dim=0, dropout=0.3):
        super().__init__()
        self.backbone = convnext_tiny(weights=ConvNeXt_Tiny_Weights.IMAGENET1K_V1)
        in_features = self.backbone.classifier[2].in_features
        self.backbone.classifier = nn.Identity()

        adapt_first_layer_for_6_channels(self.backbone)

        self.extra_dim = extra_dim
        combined_dim = in_features
        if extra_dim > 0:
            self.extra_mlp = nn.Sequential(
                nn.BatchNorm1d(extra_dim),
                nn.Linear(extra_dim, 64),
                nn.ReLU(),
                nn.Linear(64, 64),
                nn.ReLU()
            )
            combined_dim = in_features + 64

        hidden = 256
        self.head_reg = nn.Sequential(
            nn.Linear(combined_dim, hidden),
            nn.BatchNorm1d(hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )
        self.head_cls = nn.Sequential(
            nn.Linear(combined_dim, hidden),
            nn.BatchNorm1d(hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, num_classes),
        )

    def extract_features(self, x_img, x_extra=None):
        f_img = self.backbone(x_img)
        if len(f_img.shape) > 2: f_img = torch.flatten(f_img, 1)
        if self.extra_dim > 0 and x_extra is not None and x_extra.numel() > 0:
            f_extra = self.extra_mlp(x_extra)
            return torch.cat([f_img, f_extra], dim=1)
        return f_img

    def forward(self, x_img, x_extra=None):
        f = self.extract_features(x_img, x_extra)
        out_reg = self.head_reg(f).squeeze(1)
        out_cls = self.head_cls(f)
        return out_reg, out_cls

def train_one_epoch_mt(model, loader, optimizer, criterion_reg, criterion_cls, scaler, accumulation_steps):
    model.train()
    total_loss = 0.0
    optimizer.zero_grad()

    for i, (x_img, x_extra, y_reg, y_cls) in enumerate(loader):
        x_img, y_reg, y_cls = x_img.to(DEVICE), y_reg.to(DEVICE), y_cls.to(DEVICE)
        x_extra = x_extra.to(DEVICE) if x_extra.numel() > 0 else None

        with torch.amp.autocast('cuda'):
            out_reg, out_cls = model(x_img, x_extra)
            loss_reg = criterion_reg(out_reg, y_reg)
            loss_cls = criterion_cls(out_cls, y_cls)
            loss = (LAMBDA_REG * loss_reg) + (LAMBDA_CLS * loss_cls)
            loss = loss / accumulation_steps

        scaler.scale(loss).backward()

        if (i + 1) % accumulation_steps == 0 or (i + 1) == len(loader):
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

        total_loss += (loss.item() * accumulation_steps) * x_img.size(0)

    return total_loss / len(loader.dataset)

@torch.no_grad()
def evaluate_mt(model, loader, criterion_reg, criterion_cls):
    model.eval()
    total_loss = 0.0
    ys_reg, preds_reg = [], []
    for x_img, x_extra, y_reg, y_cls in loader:
        x_img, y_reg, y_cls = x_img.to(DEVICE), y_reg.to(DEVICE), y_cls.to(DEVICE)
        x_extra = x_extra.to(DEVICE) if x_extra.numel() > 0 else None

        with torch.amp.autocast('cuda'):
            out_reg, out_cls = model(x_img, x_extra)
            loss_reg = criterion_reg(out_reg, y_reg)
            loss_cls = criterion_cls(out_cls, y_cls)
            loss = (LAMBDA_REG * loss_reg) + (LAMBDA_CLS * loss_cls)

        total_loss += loss.item() * x_img.size(0)

        ys_reg.append(y_reg.cpu().numpy())
        preds_reg.append(out_reg.cpu().numpy())

    ys_reg = np.concatenate(ys_reg)
    preds_reg = np.concatenate(preds_reg)
    metrics = compute_regression_metrics(ys_reg, preds_reg)
    return total_loss / len(loader.dataset), metrics, ys_reg, preds_reg

def extract_features_for_xgboost(model, loader):
    model.eval()
    features, targets = [], []
    with torch.no_grad():
        for x_img, x_extra, y_reg, _ in loader:
            x_img = x_img.to(DEVICE)
            x_extra = x_extra.to(DEVICE) if x_extra.numel() > 0 else None
            with torch.amp.autocast('cuda'):
                f = model.extract_features(x_img, x_extra)
            features.append(f.cpu().numpy())
            targets.append(y_reg.cpu().numpy())
    return np.concatenate(features), np.concatenate(targets)

def run_experiment_v3(cfg):
    print(f"\n{'=' * 60}")
    print(f"RUN CONVNEXT: {cfg['name']}")
    print(f"{'=' * 60}")

    extra_dim = 0
    if cfg["use_weather"]: extra_dim += 18
    if cfg["use_dcp"]: extra_dim += 16

    history_csv = f"history_v3_{cfg['name']}.csv"
    pred_csv = f"predictions_v3_{cfg['name']}.csv"

    full_csv = Path("NewDataset") / "metadata_enriched_folds.csv"
    img_dir = Path("NewDataset") / "images"

    oof_y_true_cnn, oof_y_pred_cnn = [], []
    oof_y_true_xgb, oof_y_pred_xgb = [], []
    all_history = []

    for fold in range(K_FOLDS):
        print(f"\n--- FOLD {fold + 1}/{K_FOLDS} ---")
        full_df = pd.read_csv(full_csv)
        train_csv, val_csv = f"temp_train_f{fold}.csv", f"temp_val_f{fold}.csv"
        full_df[full_df["fold"] != fold].to_csv(train_csv, index=False)
        full_df[full_df["fold"] == fold].to_csv(val_csv, index=False)

        train_ds = PM25DatasetV2(train_csv, img_dir, transform=new_train_tfms, use_weather=cfg["use_weather"],
                                 use_dcp=cfg["use_dcp"])
        val_ds = PM25DatasetV2(val_csv, img_dir, transform=new_val_tfms, use_weather=cfg["use_weather"],
                               use_dcp=cfg["use_dcp"])

        train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
        val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
        xgb_train_loader = DataLoader(
            PM25DatasetV2(train_csv, img_dir, transform=new_val_tfms, use_weather=cfg["use_weather"],
                          use_dcp=cfg["use_dcp"]), batch_size=BATCH_SIZE, shuffle=False)

        model = PM25MultiTaskNetV2(num_classes=6, extra_dim=extra_dim).to(DEVICE)
        scaler = torch.amp.GradScaler('cuda')

        criterion_reg = nn.SmoothL1Loss(beta=1.0)
        criterion_cls = nn.CrossEntropyLoss()

        param_groups = [
            {"params": model.backbone.parameters(), "lr": 1e-5},
            {"params": model.head_reg.parameters(), "lr": 1e-3},
            {"params": model.head_cls.parameters(), "lr": 1e-3}
        ]

        # Aggiungiamo i parametri extra solo se la rete MLP esiste
        if extra_dim > 0:
            param_groups.append({"params": model.extra_mlp.parameters(), "lr": 1e-3})

        optimizer = torch.optim.AdamW(param_groups, weight_decay=WEIGHT_DECAY)

        best_r2 = -float("inf")
        best_state = None

        # Training CNN
        for epoch in range(1, EPOCHS_FULL + 1):
            train_loss = train_one_epoch_mt(
                model, train_loader, optimizer, criterion_reg, criterion_cls, scaler, ACCUMULATION_STEPS
            )
            val_loss, metrics, y_t, y_p = evaluate_mt(model, val_loader, criterion_reg, criterion_cls)

            if metrics["r2"] > best_r2:
                best_r2 = metrics["r2"]
                best_state = copy.deepcopy(model.state_dict())
                best_y_t, best_y_p = y_t, y_p

            print(
                f"Epoch {epoch:02d} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val R2: {metrics['r2']:.4f}")

            all_history.append({
                "phase": f"fold_{fold}", "epoch": epoch, "train_loss": train_loss,
                "val_loss": val_loss, "r2": metrics["r2"], "mae": metrics["mae"]
            })

        model.load_state_dict(best_state)
        oof_y_true_cnn.append(best_y_t)
        oof_y_pred_cnn.append(best_y_p)

        # Late Fusion XGBoost
        print(" -> feature extraction for XGBoost...")
        X_train, y_train = extract_features_for_xgboost(model, xgb_train_loader)
        X_val, y_val = extract_features_for_xgboost(model, val_loader)

        print("  -> feature selection...")
        xgb_model = xgb.XGBRegressor(
            n_estimators=300, learning_rate=0.05, max_depth=5,
            subsample=0.8, colsample_bytree=0.8, random_state=42
        )
        xgb_model.fit(X_train, y_train)

        preds_xgb = xgb_model.predict(X_val)
        oof_y_true_xgb.append(y_val)
        oof_y_pred_xgb.append(preds_xgb)

        if os.path.exists(train_csv): os.remove(train_csv)
        if os.path.exists(val_csv): os.remove(val_csv)

    y_true_xgb = np.concatenate(oof_y_true_xgb)
    y_pred_xgb = np.concatenate(oof_y_pred_xgb)
    res_xgb = compute_regression_metrics(y_true_xgb, y_pred_xgb)

    pd.DataFrame(all_history).to_csv(history_csv, index=False)

    pred_df = pd.DataFrame({
        "y_true": y_true_xgb,
        "y_pred": y_pred_xgb,
        "error": y_pred_xgb - y_true_xgb
    })
    pred_df.to_csv(pred_csv, index=False)

    return {
        "experiment": cfg["name"],
        "use_weather": cfg["use_weather"],
        "use_dcp": cfg["use_dcp"],
        "history_csv": history_csv,
        "prediction_csv": pred_csv,
        "r2": res_xgb["r2"],
        "mae": res_xgb["mae"]
    }

if __name__ == "__main__":
    torch.cuda.empty_cache()

    seed_everything(42)
    results = []
    for cfg in EXPERIMENTS_V3:
        res = run_experiment_v3(cfg)
        results.append(res)
        print(f"\nRESULTS {cfg['name']}: R2={res['r2']:.4f} | MAE={res['mae']:.4f}")

    df_results = pd.DataFrame(results)
    df_results.to_csv("ablation_v3_results.csv", index=False)