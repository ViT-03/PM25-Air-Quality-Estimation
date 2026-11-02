import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_absolute_error

# =========================================================
# CONFIGURAZIONE PERCORSI E METRICHE V1 FISSE
# =========================================================
NEW_RESULTS_DIR = "NewResults"
OUT_DIR = "final_ablation_files"
os.makedirs(OUT_DIR, exist_ok=True)

# Valori fissi per V1 (usati solo nel Bar Chart generale di confronto)
V1_R2 = 0.550
V1_MAE = 36.60

CONFIG_FILES = {
    "V2": {
        "Base": ("V2/predictions_v2_base.csv", "V2/history_v2_base.csv"),
        "Base + DCP": ("V2/predictions_v2_base_dcp.csv", "V2/history_v2_base_dcp.csv"),
        "Base + Weather": ("V2/predictions_v2_base_weather.csv", "V2/history_v2_base_weather.csv"),
        "Base + Weather + DCP": ("V2/predictions_v2_base_weather_dcp.csv", "V2/history_v2_base_weather_dcp.csv"),
    },
    "V3": {
        "Base": ("V3/predictions_v3_base.csv", "V3/history_v3_base.csv"),
        "Base + DCP": ("V3/predictions_v3_base_dcp.csv", "V3/history_v3_base_dcp.csv"),
        "Base + Weather": ("V3/predictions_v3_base_weather.csv", "V3/history_v3_base_weather.csv"),
        "Base + Weather + DCP": ("V3/predictions_v3_base_dcp_weather.csv", "V3/history_v3_base_dcp_weather.csv"),
    },
    "V4": {
        "Base": ("V4/predictions_v4_base.csv", "V4/history_v4_base.csv"),
        "Base + DCP": ("V4/predictions_v4_base_dcp.csv", "V4/history_v4_base_dcp.csv"),
        "Base + Weather": ("V4/predictions_v4_base_weather.csv", "V4/history_v4_base_weather.csv"),
        "Base + Weather + DCP": ("V4/predictions_v4_base_dcp_weather.csv", "V4/history_v4_base_dcp_weather.csv"),
    }
}


# =========================================================
# FUNZIONI DI PLOTTING
# =========================================================
def create_bar_chart(labels, r2_scores, mae_scores, title, filename):
    x = np.arange(len(labels))
    width = 0.35

    fig, ax1 = plt.subplots(figsize=(12, 6))

    rects1 = ax1.bar(x - width / 2, r2_scores, width, label='R²', color='royalblue')
    ax1.set_ylabel('R² Score', color='royalblue', fontweight='bold', fontsize=12)
    ax1.tick_params(axis='y', labelcolor='royalblue')
    ax1.set_ylim(0, 1.0)

    ax2 = ax1.twinx()
    rects2 = ax2.bar(x + width / 2, mae_scores, width, label='MAE', color='darkorange')
    ax2.set_ylabel('Mean Absolute Error', color='darkorange', fontweight='bold', fontsize=12)
    ax2.tick_params(axis='y', labelcolor='darkorange')
    ax2.set_ylim(0, max(mae_scores) * 1.3)

    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, fontweight='bold', fontsize=10)
    plt.title(title, fontweight='bold', fontsize=14, pad=20)

    for rect in rects1:
        height = rect.get_height()
        ax1.annotate(f'{height:.3f}', xy=(rect.get_x() + rect.get_width() / 2, height),
                     xytext=(0, 5), textcoords="offset points", ha='center', va='bottom', fontweight='bold')

    for rect in rects2:
        height = rect.get_height()
        ax2.annotate(f'{height:.2f}', xy=(rect.get_x() + rect.get_width() / 2, height),
                     xytext=(0, 5), textcoords="offset points", ha='center', va='bottom', fontweight='bold')

    lines_1, labels_1 = ax1.get_legend_handles_labels()
    lines_2, labels_2 = ax2.get_legend_handles_labels()
    ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc='upper left', framealpha=0.9)

    fig.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[OK] Saved: {filename}")


def create_grouped_ablation_chart(summary_df, filename):
    configs = ["Base", "Base + DCP", "Base + Weather", "Base + Weather + DCP"]
    versions = ["V2", "V3", "V4"]

    x = np.arange(len(configs))
    width = 0.25

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    colors = {"V2": "#6baed6", "V3": "#2171b5", "V4": "#08306b"}

    for idx, ver in enumerate(versions):
        sub = summary_df[summary_df["Version"] == ver].set_index("Configuration").reindex(configs)
        offset = (idx - 1) * width

        b1 = ax1.bar(x + offset, sub["R2"], width, label=ver, color=colors[ver])
        b2 = ax2.bar(x + offset, sub["MAE"], width, label=ver, color=colors[ver])

        for rect in b1:
            h = rect.get_height()
            if not np.isnan(h):
                ax1.annotate(f'{h:.2f}', xy=(rect.get_x() + rect.get_width() / 2, h),
                             xytext=(0, 3), textcoords="offset points", ha='center', fontsize=8, fontweight='bold')
        for rect in b2:
            h = rect.get_height()
            if not np.isnan(h):
                ax2.annotate(f'{h:.1f}', xy=(rect.get_x() + rect.get_width() / 2, h),
                             xytext=(0, 3), textcoords="offset points", ha='center', fontsize=8, fontweight='bold')

    ax1.set_title("Ablation Study (V2, V3, V4): R² Score", fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(configs, rotation=15, fontweight='bold')
    ax1.set_ylim(0, 1.0)
    ax1.set_ylabel("R² Score", fontweight='bold')
    ax1.grid(True, axis='y', alpha=0.3)
    ax1.legend()

    ax2.set_title("Ablation Study (V2, V3, V4): MAE", fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(configs, rotation=15, fontweight='bold')
    ax2.set_ylim(0, summary_df["MAE"].max() * 1.25)
    ax2.set_ylabel("MAE", fontweight='bold')
    ax2.grid(True, axis='y', alpha=0.3)
    ax2.legend()

    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[OK] Saved Grouped Ablation Chart: {filename}")


def save_scatter_plot(pred_df, out_path, title):
    y_true = pred_df["y_true"].values
    y_pred = pred_df["y_pred"].values

    plt.figure(figsize=(6, 6))
    plt.scatter(y_true, y_pred, alpha=0.5, color="indigo")

    mn = min(y_true.min(), y_pred.min())
    mx = max(y_true.max(), y_pred.max())
    plt.plot([mn, mx], [mn, mx], color="red", linestyle="--", linewidth=2)

    plt.xlabel("Real Target (PM2.5)", fontweight='bold')
    plt.ylabel("Model Prediction", fontweight='bold')
    plt.title(title, fontweight='bold', pad=15)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()


def save_error_hist(pred_df, out_path, title):
    if "error" not in pred_df.columns:
        pred_df["error"] = pred_df["y_pred"] - pred_df["y_true"]

    err = pred_df["error"].dropna().values

    plt.figure(figsize=(7, 5))
    plt.hist(err, bins=40, alpha=0.8, color="teal", edgecolor="black", linewidth=0.5)
    plt.axvline(0.0, color="red", linestyle="--", linewidth=2)

    plt.xlabel("Prediction Error (Pred - True)", fontweight='bold')
    plt.ylabel("Frequency", fontweight='bold')
    plt.title(title, fontweight='bold', pad=15)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()


def save_loss_plot(history_df, out_path, title):
    plt.figure(figsize=(8, 5))

    if "phase" in history_df.columns:
        for phase_name, g in history_df.groupby("phase"):
            plt.plot(g["epoch"], g["train_loss"], marker="o", label=f"{phase_name} Train Loss")
            if "val_loss" in g.columns and g["val_loss"].notna().any():
                plt.plot(g["epoch"], g["val_loss"], marker="s", linestyle="--", label=f"{phase_name} Val Loss")
    else:
        plt.plot(history_df["epoch"], history_df["train_loss"], marker="o", label="Train Loss", color="blue")
        if "val_loss" in history_df.columns and history_df["val_loss"].notna().any():
            plt.plot(history_df["epoch"], history_df["val_loss"], marker="s", linestyle="--", label="Val Loss",
                     color="orange")

    plt.xlabel("Epoch", fontweight='bold')
    plt.ylabel("Loss", fontweight='bold')
    plt.title(title, fontweight='bold', pad=15)
    plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', fontsize='small', ncol=1)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()


# =========================================================
# MAIN
# =========================================================
def main():
    summary_rows = []
    best_per_version = {}

    # 1. Scansiona tutte le configurazioni di V2, V3, V4 e trova il "Best" per ciascuna
    for version, configs in CONFIG_FILES.items():
        best_r2 = -float("inf")
        for cfg_name, (pred_rel, hist_rel) in configs.items():
            pred_path = os.path.join(NEW_RESULTS_DIR, pred_rel)
            hist_path = os.path.join(NEW_RESULTS_DIR, hist_rel)

            if os.path.exists(pred_path):
                df_p = pd.read_csv(pred_path)
                r2 = r2_score(df_p["y_true"], df_p["y_pred"])
                mae = mean_absolute_error(df_p["y_true"], df_p["y_pred"])

                summary_rows.append({
                    "Version": version,
                    "Configuration": cfg_name,
                    "R2": r2,
                    "MAE": mae,
                    "Pred_File": pred_path,
                    "Hist_File": hist_path
                })

                if r2 > best_r2:
                    best_r2 = r2
                    best_per_version[version] = {
                        "config": cfg_name,
                        "r2": r2,
                        "mae": mae,
                        "pred_path": pred_path,
                        "hist_path": hist_path
                    }
            else:
                print(f"[WARNING] Missing file: {pred_path}")

    # 2. Salva Tabella Completa 3x4 in CSV
    summary_df = pd.DataFrame(summary_rows)
    summary_csv_path = os.path.join(OUT_DIR, "full_ablation_matrix_v2_v3_v4.csv")
    summary_df.to_csv(summary_csv_path, index=False)
    print("\n=== FULL ABLATION TABLE (V2, V3, V4) ===")
    print(summary_df[["Version", "Configuration", "R2", "MAE"]].to_string(index=False))

    # 3. Grafico a barre raggruppato (tutte le 12 configurazioni in un'unica figura)
    if not summary_df.empty:
        create_grouped_ablation_chart(
            summary_df,
            filename=os.path.join(OUT_DIR, "ablation_all_configs_grouped.png")
        )

    # 4. Grafico "Best-of-Each-Iteration" (V1 statico + Best V2 + Best V3 + Best V4)
    labels_apples = [
        "V1 Baseline EffNet \n",
        f"V2 EffNet KFold\n({best_per_version['V2']['config']})",
        f"V3 ConvNeXt+XGB\n({best_per_version['V3']['config']})",
        f"V4 ConvNeXt Improved\n({best_per_version['V4']['config']})"
    ]
    r2_apples = [V1_R2, best_per_version["V2"]["r2"], best_per_version["V3"]["r2"], best_per_version["V4"]["r2"]]
    mae_apples = [V1_MAE, best_per_version["V2"]["mae"], best_per_version["V3"]["mae"], best_per_version["V4"]["mae"]]

    create_bar_chart(
        labels=labels_apples,
        r2_scores=r2_apples,
        mae_scores=mae_apples,
        title="Best-of-Each-Iteration Comparison",
        filename=os.path.join(OUT_DIR, "comparison_apples_to_apples.png")
    )

    # 5. Genera SOLO I 3 GRAFICI DEI MIGLIORI (Best V2, Best V3, Best V4) per Scatter, Error Hist e Loss
    models_to_plot = [
        ("1_V2_Best", f"1. V2 EffNet KFold ({best_per_version['V2']['config']})", best_per_version["V2"]),
        ("2_V3_Best", f"2. V3 ConvNeXt+XGB ({best_per_version['V3']['config']})", best_per_version["V3"]),
        ("3_V4_Best", f"3. V4 ConvNeXt Improved ({best_per_version['V4']['config']})", best_per_version["V4"]),
    ]

    for tag, title_prefix, info in models_to_plot:
        p_path = info["pred_path"]
        h_path = info["hist_path"]

        if os.path.exists(p_path):
            df_pred = pd.read_csv(p_path)
            save_scatter_plot(df_pred, os.path.join(OUT_DIR, f"scatter_{tag}.png"), f"{title_prefix} - Scatter Plot")
            save_error_hist(df_pred, os.path.join(OUT_DIR, f"error_hist_{tag}.png"), f"{title_prefix} - Error Distribution")
            print(f"[OK] Saved Scatter & Error Hist for {tag}")

        if os.path.exists(h_path):
            df_hist = pd.read_csv(h_path)
            save_loss_plot(df_hist, os.path.join(OUT_DIR, f"loss_{tag}.png"), f"{title_prefix} - Learning Curve")
            print(f"[OK] Saved Learning Curve for {tag}")


if __name__ == "__main__":
    main()