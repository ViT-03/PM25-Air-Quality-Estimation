import os
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import xgboost as xgb
import optuna
from PIL import Image
from pathlib import Path
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision.models import convnext_tiny, ConvNeXt_Tiny_Weights
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from sklearn.metrics import r2_score

from prepare_dataset import PM25Dataset, TARGET_CLASS_COL_CANDIDATES
from train_v2_kfold import (
    NUM_WORKERS, DEVICE, WEIGHT_DECAY, K_FOLDS,
    compute_regression_metrics, aqi_to_class, seed_everything
)
from train_v3_convnext import new_train_tfms, new_val_tfms

# PARAMETERS
BATCH_SIZE = 8
ACCUMULATION_STEPS = 4
EPOCHS_FULL = 15
LAMBDA_REG = 1.0
LAMBDA_CLS = 0.5
CUTMIX_PROB = 0.5
OPTUNA_TRIALS = 50

EXPERIMENTS_V4 = [
    {"name": "base", "use_weather": False, "use_dcp": False},
    {"name": "base_dcp", "use_weather": False, "use_dcp": True},
    {"name": "base_weather", "use_weather": True, "use_dcp": False},
    {"name": "base_dcp_weather", "use_weather": True, "use_dcp": True},
]

class AsymmetricL1Loss(nn.Module):
    def __init__(self, alpha=1.5):
        super().__init__()
        self.alpha = alpha

    def forward(self, pred, target):
        diff = pred - target
        loss = torch.where(diff < 0, torch.abs(diff) * self.alpha, torch.abs(diff))
        return loss.mean()

class PM25DatasetV5(PM25Dataset):
    def __init__(self, csv_path, image_dir, transform=None, use_weather=True, use_dcp=False):
        super().__init__(csv_path, image_dir, task="regression", transform=transform,
                         use_weather=use_weather, use_dcp=use_dcp, use_dehazed_images=False)
        self.class_col = next((c for c in TARGET_CLASS_COL_CANDIDATES if c in self.df.columns), None)

        self.targets_cls = []
        for _, row in self.df.iterrows():
            if self.class_col and pd.notna(row[self.class_col]):
                self.targets_cls.append(int(row[self.class_col]))
            else:
                self.targets_cls.append(aqi_to_class(float(row[self.target_col])))

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = self.image_dir / str(row[self.image_col])
        img = Image.open(img_path).convert("RGB")
        if self.transform: img = self.transform(img)

        y_reg = torch.tensor(float(row[self.target_col]), dtype=torch.float32)
        y_cls = torch.tensor(self.targets_cls[idx], dtype=torch.long)

        extras = []
        if self.use_weather:
            extras.extend([float(row[c]) if pd.notna(row[c]) else 0.0 for c in self.weather_cols])
            t = float(row.get("temperature_2m", 0.0))
            rh = float(row.get("relative_humidity_2m", 50.0))
            app_t = float(row.get("apparent_temperature", t))
            dew_point = t - ((100 - rh) / 5.0)
            thermal_diff = app_t - t
            extras.extend([dew_point, thermal_diff])

        if self.use_dcp:
            extras.extend([float(row[c]) if pd.notna(row[c]) else 0.0 for c in self.dcp_cols])

        x_extra = torch.tensor(extras, dtype=torch.float32)
        return img, x_extra, y_reg, y_cls

def rand_bbox(size, lam):
    W, H = size[2], size[3]
    cut_rat = np.sqrt(1. - lam)
    cut_w, cut_h = int(W * cut_rat), int(H * cut_rat)
    cx, cy = np.random.randint(W), np.random.randint(H)
    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)
    return bbx1, bby1, bbx2, bby2

class ConvNeXtMultiTask(nn.Module):
    def __init__(self, extra_dim=20, dropout=0.3):
        super().__init__()
        self.backbone = convnext_tiny(weights=ConvNeXt_Tiny_Weights.IMAGENET1K_V1)
        in_features = self.backbone.classifier[2].in_features
        self.backbone.classifier = nn.Identity()

        self.extra_dim = extra_dim
        if extra_dim > 0:
            self.extra_mlp = nn.Sequential(
                nn.BatchNorm1d(extra_dim), nn.Linear(extra_dim, 64), nn.ReLU(),
                nn.Linear(64, 64), nn.ReLU()
            )
            combined_dim = in_features + 64
        else:
            combined_dim = in_features

        self.head_reg = nn.Sequential(
            nn.Linear(combined_dim, 256), nn.BatchNorm1d(256), nn.ReLU(),
            nn.Dropout(dropout), nn.Linear(256, 1)
        )
        self.head_cls = nn.Sequential(
            nn.Linear(combined_dim, 256), nn.BatchNorm1d(256), nn.ReLU(),
            nn.Dropout(dropout), nn.Linear(256, 6)
        )

    def forward(self, x_img, x_extra=None, return_features=False):
        f_img = self.backbone(x_img)
        if len(f_img.shape) > 2: f_img = torch.flatten(f_img, 1)

        f_extra = self.extra_mlp(
            x_extra) if self.extra_dim > 0 and x_extra is not None and x_extra.numel() > 0 else None
        f_combined = torch.cat([f_img, f_extra], dim=1) if f_extra is not None else f_img

        out_cls = self.head_cls(f_combined)

        if return_features:
            probs = F.softmax(out_cls, dim=1)
            return f_img, probs

        out_reg = self.head_reg(f_combined).squeeze(1)
        return out_reg, out_cls

def train_epoch(model, loader, optimizer, crit_reg, crit_cls, scaler, acc_steps):
    model.train()
    total_loss = 0.0
    optimizer.zero_grad()

    for i, (x_img, x_extra, y_reg, y_cls) in enumerate(loader):
        x_img, y_reg, y_cls = x_img.to(DEVICE), y_reg.to(DEVICE), y_cls.to(DEVICE)
        x_extra = x_extra.to(DEVICE)

        r = np.random.rand(1)
        use_cutmix = r < CUTMIX_PROB
        if use_cutmix:
            lam = np.random.beta(1.0, 1.0)
            rand_index = torch.randperm(x_img.size()[0]).cuda()
            y_reg_a, y_reg_b = y_reg, y_reg[rand_index]
            y_cls_a, y_cls_b = y_cls, y_cls[rand_index]
            bbx1, bby1, bbx2, bby2 = rand_bbox(x_img.size(), lam)
            x_img[:, :, bbx1:bbx2, bby1:bby2] = x_img[rand_index, :, bbx1:bbx2, bby1:bby2]
            lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (x_img.size()[-1] * x_img.size()[-2]))

        with torch.amp.autocast('cuda'):
            out_reg, out_cls = model(x_img, x_extra)
            if use_cutmix:
                loss_reg = crit_reg(out_reg, y_reg_a) * lam + crit_reg(out_reg, y_reg_b) * (1. - lam)
                loss_cls = crit_cls(out_cls, y_cls_a) * lam + crit_cls(out_cls, y_cls_b) * (1. - lam)
            else:
                loss_reg = crit_reg(out_reg, y_reg)
                loss_cls = crit_cls(out_cls, y_cls)

            loss = ((LAMBDA_REG * loss_reg) + (LAMBDA_CLS * loss_cls)) / acc_steps

        scaler.scale(loss).backward()

        if (i + 1) % acc_steps == 0 or (i + 1) == len(loader):
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

        total_loss += (loss.item() * acc_steps) * x_img.size(0)
    return total_loss / len(loader.dataset)

@torch.no_grad()
def evaluate_epoch_mt(model, loader, crit_reg, crit_cls):
    model.eval()
    total_loss = 0.0
    ys_reg, preds_reg = [], []
    for x_img, x_extra, y_reg, y_cls in loader:
        x_img, y_reg, y_cls = x_img.to(DEVICE), y_reg.to(DEVICE), y_cls.to(DEVICE)
        x_extra = x_extra.to(DEVICE)
        with torch.amp.autocast('cuda'):
            out_reg, out_cls = model(x_img, x_extra)
            loss_reg = crit_reg(out_reg, y_reg)
            loss_cls = crit_cls(out_cls, y_cls)
            loss = (LAMBDA_REG * loss_reg) + (LAMBDA_CLS * loss_cls)

        total_loss += loss.item() * x_img.size(0)
        ys_reg.append(y_reg.cpu().numpy())
        preds_reg.append(out_reg.cpu().numpy())

    ys_reg = np.concatenate(ys_reg)
    preds_reg = np.concatenate(preds_reg)
    metrics = compute_regression_metrics(ys_reg, preds_reg)
    return total_loss / len(loader.dataset), metrics

@torch.no_grad()
def extract_xgb_features(model, loader, apply_tta=False):
    model.eval()
    features, targets = [], []
    for x_img, x_extra, y_reg, _ in loader:
        x_img, x_extra = x_img.to(DEVICE), x_extra.to(DEVICE)
        x_extra_np = x_extra.cpu().numpy()

        with torch.amp.autocast('cuda'):
            f_cnn_base, probs_base = model(x_img, x_extra, return_features=True)
            f_cnn, probs = f_cnn_base.cpu().numpy(), probs_base.cpu().numpy()

            if apply_tta:
                f_cnn_flip, probs_flip = model(torch.flip(x_img, dims=[3]), x_extra, return_features=True)
                f_cnn_dark, probs_dark = model(x_img * 0.8, x_extra, return_features=True)
                f_cnn = (f_cnn + f_cnn_flip.cpu().numpy() + f_cnn_dark.cpu().numpy()) / 3.0
                probs = (probs + probs_flip.cpu().numpy() + probs_dark.cpu().numpy()) / 3.0

        if x_extra_np.shape[1] > 0:
            features.append(np.hstack([f_cnn, probs, x_extra_np]))
        else:
            features.append(np.hstack([f_cnn, probs]))
        targets.append(y_reg.numpy())

    return np.concatenate(features), np.concatenate(targets)

def run_experiment_v4(cfg):
    print(f"\n{'=' * 60}")
    print(f"RUN CONVNEXT V4: {cfg['name']}")
    print(f"{'=' * 60}")

    extra_dim = 0
    if cfg["use_weather"]: extra_dim += 20
    if cfg["use_dcp"]: extra_dim += 16

    history_csv = f"history_v4_{cfg['name']}.csv"
    pred_csv = f"predictions_v4_{cfg['name']}.csv"
    full_csv = Path("NewDataset") / "metadata_enriched_folds.csv"
    img_dir = Path("NewDataset") / "images"
    os.makedirs("ensemble_models", exist_ok=True)

    oof_y_true, oof_y_pred_xgb = [], []
    all_history = []

    for fold in range(K_FOLDS):
        print(f"\n--- FOLD {fold + 1}/{K_FOLDS} ---")
        full_df = pd.read_csv(full_csv)
        train_df = full_df[full_df["fold"] != fold]
        val_df = full_df[full_df["fold"] == fold]

        train_csv, val_csv = f"temp_train_f{fold}.csv", f"temp_val_f{fold}.csv"
        train_df.to_csv(train_csv, index=False)
        val_df.to_csv(val_csv, index=False)

        train_ds = PM25DatasetV5(train_csv, img_dir, new_train_tfms, cfg["use_weather"], cfg["use_dcp"])
        val_ds = PM25DatasetV5(val_csv, img_dir, new_val_tfms, cfg["use_weather"], cfg["use_dcp"])

        class_counts = np.bincount(train_ds.targets_cls)
        weights = 1.0 / np.maximum(class_counts, 1)
        sample_weights = np.array([weights[t] for t in train_ds.targets_cls])
        sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(sample_weights), replacement=True)

        train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, sampler=sampler, num_workers=NUM_WORKERS)
        val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
        xgb_train_loader = DataLoader(PM25DatasetV5(train_csv, img_dir, new_val_tfms, cfg["use_weather"], cfg["use_dcp"]),
                                      batch_size=BATCH_SIZE, shuffle=False)

        model = ConvNeXtMultiTask(extra_dim=extra_dim).to(DEVICE)
        scaler = torch.amp.GradScaler('cuda')

        crit_reg = AsymmetricL1Loss(alpha=1.5)
        crit_cls = nn.CrossEntropyLoss()

        param_groups = [
            {"params": model.backbone.parameters(), "lr": 2e-5},
            {"params": model.head_reg.parameters(), "lr": 1e-3},
            {"params": model.head_cls.parameters(), "lr": 1e-3}
        ]

        if extra_dim > 0:
            param_groups.append({"params": model.extra_mlp.parameters(), "lr": 1e-3})

        optimizer = torch.optim.AdamW(param_groups, weight_decay=WEIGHT_DECAY)

        scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=5, T_mult=1, eta_min=1e-6)

        best_cnn_state = None
        best_cnn_loss = float('inf')

        for epoch in range(1, EPOCHS_FULL + 1):
            train_loss = train_epoch(model, train_loader, optimizer, crit_reg, crit_cls, scaler, ACCUMULATION_STEPS)
            val_loss, val_metrics = evaluate_epoch_mt(model, val_loader, crit_reg, crit_cls)
            scheduler.step()

            if val_loss < best_cnn_loss:
                best_cnn_loss = val_loss
                best_cnn_state = copy.deepcopy(model.state_dict())

            print(f"Epoch {epoch:02d} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

            all_history.append({
                "phase": f"fold_{fold}", "epoch": epoch,
                "train_loss": train_loss, "val_loss": val_loss,
                "r2": val_metrics["r2"], "mae": val_metrics["mae"]
            })

        model.load_state_dict(best_cnn_state)
        torch.save(model.state_dict(), f"ensemble_models/cnn_fold_{fold}_{cfg['name']}.pth")

        print(" -> feature extraction for XGBoost...")
        X_train, y_train = extract_xgb_features(model, xgb_train_loader, apply_tta=False)
        X_val, y_val = extract_xgb_features(model, val_loader, apply_tta=True)

        cnn_and_probs_dim = 768 + 6

        print("  -> feature selection...")
        temp_xgb = xgb.XGBRegressor(n_estimators=100, random_state=42).fit(X_train, y_train)

        if extra_dim > 0:
            tabular_importances = temp_xgb.feature_importances_[cnn_and_probs_dim:]
            keep_tabular_idx = np.where(tabular_importances > 0.005)[0]

            if len(keep_tabular_idx) < extra_dim:
                X_train = np.hstack(
                    [X_train[:, :cnn_and_probs_dim], X_train[:, cnn_and_probs_dim:][:, keep_tabular_idx]])
                X_val = np.hstack([X_val[:, :cnn_and_probs_dim], X_val[:, cnn_and_probs_dim:][:, keep_tabular_idx]])
                np.save(f"ensemble_models/xgb_features_idx_f{fold}_{cfg['name']}.npy", keep_tabular_idx)

        print("  -> hyperparameters XGBoost with Optuna (50 Trials)...")

        def objective(trial):
            param = {
                'n_estimators': trial.suggest_int('n_estimators', 150, 500),
                'max_depth': trial.suggest_int('max_depth', 3, 8),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1),
                'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                'reg_alpha': trial.suggest_float('reg_alpha', 1e-3, 10.0, log=True),
                'reg_lambda': trial.suggest_float('reg_lambda', 1e-3, 10.0, log=True),
                'gamma': trial.suggest_float('gamma', 0.0, 5.0),
                'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
                'random_state': 42,
                'n_jobs': -1
            }
            xgb_temp = xgb.XGBRegressor(**param)
            xgb_temp.fit(X_train, y_train)
            return r2_score(y_val, xgb_temp.predict(X_val))

        optuna.logging.set_verbosity(optuna.logging.WARNING)
        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=OPTUNA_TRIALS)

        best_xgb = xgb.XGBRegressor(**study.best_params, random_state=42, n_jobs=-1)
        best_xgb.fit(X_train, y_train)
        best_xgb.save_model(f"ensemble_models/xgb_fold_{fold}_{cfg['name']}.json")

        preds_xgb = best_xgb.predict(X_val)

        oof_y_true.append(y_val)
        oof_y_pred_xgb.append(preds_xgb)

        if os.path.exists(train_csv): os.remove(train_csv)
        if os.path.exists(val_csv): os.remove(val_csv)

    y_true_all = np.concatenate(oof_y_true)
    y_pred_all = np.concatenate(oof_y_pred_xgb)
    res_xgb = compute_regression_metrics(y_true_all, y_pred_all)

    pd.DataFrame(all_history).to_csv(history_csv, index=False)

    pred_df = pd.DataFrame({
        "y_true": y_true_all,
        "y_pred": y_pred_all,
        "error": y_pred_all - y_true_all
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
    for cfg in EXPERIMENTS_V4:
        res = run_experiment_v4(cfg)
        results.append(res)
        print(f"\nRESULTS {cfg['name']}: R2={res['r2']:.4f} | MAE={res['mae']:.4f}")

    df_results = pd.DataFrame(results)
    df_results.to_csv("ablation_v4_results.csv", index=False)