import os
import pandas as pd
import numpy as np
from scipy.stats import ttest_rel, wilcoxon
from sklearn.metrics import r2_score

NEW_RESULTS_DIR = "NewResults"
FOLDS_CSV_PATH = os.path.join("NewDataset", "metadata_enriched_folds.csv")
OUT_DIR = "final_ablation_files"
os.makedirs(OUT_DIR, exist_ok=True)

CONFIG_FILES = {
    "V2": {
        "Base": "V2/predictions_v2_base.csv",
        "Base+DCP": "V2/predictions_v2_base_dcp.csv",
        "Base+Weather": "V2/predictions_v2_base_weather.csv",
        "Base+Weather+DCP": "V2/predictions_v2_base_weather_dcp.csv",
    },
    "V3": {
        "Base": "V3/predictions_v3_base.csv",
        "Base+DCP": "V3/predictions_v3_base_dcp.csv",
        "Base+Weather": "V3/predictions_v3_base_weather.csv",
        "Base+Weather+DCP": "V3/predictions_v3_base_dcp_weather.csv",
    },
    "V4": {
        "Base": "V4/predictions_v4_base.csv",
        "Base+DCP": "V4/predictions_v4_base_dcp.csv",
        "Base+Weather": "V4/predictions_v4_base_weather.csv",
        "Base+Weather+DCP": "V4/predictions_v4_base_dcp_weather.csv",
    }
}


def format_p(p_val):
    return "< 1e-300" if p_val == 0.0 else f"{p_val:.4e}"


def format_sig(p_value):
    if p_value < 0.001:
        return "*** (p < 0.001)"
    elif p_value < 0.01:
        return "** (p < 0.01)"
    elif p_value < 0.05:
        return "* (p < 0.05)"
    return "ns (Not Significant)"


def verify_strict_alignment(loaded_dfs, sample_idx=123):
    """
    Verifica rigorosa che tutti i CSV contengano esattamente gli stessi campioni
    nello stesso identico ordine (e li confronta con metadata_enriched_folds.csv se presente).
    """
    print("=========================================================")
    print(" 1. VERIFICA RIGOROSA ALLINEAMENTO CAMPIONI (SAMPLE ORDER)")
    print("=========================================================")

    names = list(loaded_dfs.keys())
    ref_name = names[0]
    ref_y = loaded_dfs[ref_name]["y_true"].values

    # Check opzionale contro il dataset master con i fold
    if os.path.exists(FOLDS_CSV_PATH):
        folds_df = pd.read_csv(FOLDS_CSV_PATH)
        target_col = next(c for c in ["aqi", "pm25", "pm2.5", "pm25_aqi", "label", "target"] if c in folds_df.columns)
        oof_master_y = np.concatenate([
            folds_df[folds_df["fold"] == f][target_col].values for f in range(5)
        ])
        assert np.allclose(ref_y, oof_master_y, atol=1e-4), "ERRORE: Disallineamento con metadata_enriched_folds.csv!"
        print(f"[OK] Allineamento verificato al 100% con '{FOLDS_CSV_PATH}' ({len(ref_y)} campioni).")

    for name in names[1:]:
        curr_y = loaded_dfs[name]["y_true"].values
        if len(ref_y) != len(curr_y):
            raise AssertionError(f"ERRORE LUNGHEZZA: {ref_name} ({len(ref_y)}) != {name} ({len(curr_y)})")

        max_diff = np.max(np.abs(ref_y - curr_y))
        if not np.allclose(ref_y, curr_y, atol=1e-4):
            bad_idx = np.where(np.abs(ref_y - curr_y) > 1e-4)[0][0]
            raise AssertionError(
                f"ERRORE ALLINEAMENTO tra {ref_name} e {name} alla riga {bad_idx}: "
                f"{ref_y[bad_idx]} != {curr_y[bad_idx]}"
            )
        print(f"[OK] {ref_name:<22} == {name:<22} | N={len(curr_y)} | Max diff y_true = {max_diff:.6f}")

    print(f"\n--- CONTROLLO PUNTUALE SUL SAMPLE #{sample_idx} ---")
    for name in names:
        row = loaded_dfs[name].iloc[sample_idx]
        print(f"  {name:<22} -> y_true: {row['y_true']:6.1f} | y_pred: {row['y_pred']:7.3f}")
    print("=========================================================\n")


def main():
    loaded_dfs = {}
    all_errors = {"V2": {}, "V3": {}, "V4": {}}
    best_models = {}

    # 1. Carica tutti i CSV
    for ver, cfg_dict in CONFIG_FILES.items():
        best_r2 = -float("inf")
        for cfg_name, rel_path in cfg_dict.items():
            full_path = os.path.join(NEW_RESULTS_DIR, rel_path)
            if os.path.exists(full_path):
                df = pd.read_csv(full_path)
                key_name = f"{ver}_{cfg_name}"
                loaded_dfs[key_name] = df

                abs_err = np.abs(df["y_pred"].values - df["y_true"].values)
                r2 = r2_score(df["y_true"], df["y_pred"])
                all_errors[ver][cfg_name] = abs_err

                if r2 > best_r2:
                    best_r2 = r2
                    best_models[ver] = (f"{ver}_Best ({cfg_name})", abs_err)

    # 2. Esegui la verifica bloccante sull'ordine dei campioni
    if loaded_dfs:
        verify_strict_alignment(loaded_dfs, sample_idx=123)

    # 3. INTER-VERSION TESTS (Paired t-test + Wilcoxon Signed-Rank Test)
    inter_candidates = [best_models[ver] for ver in ["V2", "V3", "V4"] if ver in best_models]
    inter_results = []

    print("=== INTER-VERSION TESTS (PAIRED T-TEST + WILCOXON) ===")
    for i in range(len(inter_candidates)):
        for j in range(i + 1, len(inter_candidates)):
            name_a, err_a = inter_candidates[i]
            name_b, err_b = inter_candidates[j]

            t_stat, t_p = ttest_rel(err_a, err_b)
            w_stat, w_p = wilcoxon(err_a, err_b)

            inter_results.append({
                "Model_A": name_a,
                "Model_B": name_b,
                "MAE_A": np.mean(err_a),
                "MAE_B": np.mean(err_b),
                "Median_AE_A": np.median(err_a),
                "Median_AE_B": np.median(err_b),
                "T-Statistic": t_stat,
                "T-Test_P-Value": format_p(t_p),
                "Wilcoxon_Stat": w_stat,
                "Wilcoxon_P-Value": format_p(w_p),
                "Significance": format_sig(max(t_p, w_p))
            })
            print(f"{name_a} vs {name_b}:")
            print(f"  -> Paired t-test : T={t_stat:.3f}, p={format_p(t_p)} [{format_sig(t_p)}]")
            print(f"  -> Wilcoxon test : W={w_stat:.1f}, p={format_p(w_p)} [{format_sig(w_p)}]")

    pd.DataFrame(inter_results).to_csv(os.path.join(OUT_DIR, "statistical_significance_inter_version.csv"), index=False)

    # 4. INTRA-VERSION TESTS (Paired t-test + Wilcoxon Signed-Rank Test)
    intra_results = []
    print("\n=== INTRA-VERSION FEATURE ABLATION (PAIRED T-TEST + WILCOXON) ===")
    for ver in ["V2", "V3", "V4"]:
        cfgs = list(all_errors[ver].keys())
        for i in range(len(cfgs)):
            for j in range(i + 1, len(cfgs)):
                c_a, c_b = cfgs[i], cfgs[j]
                err_a, err_b = all_errors[ver][c_a], all_errors[ver][c_b]

                t_stat, t_p = ttest_rel(err_a, err_b)
                w_stat, w_p = wilcoxon(err_a, err_b)

                intra_results.append({
                    "Version": ver,
                    "Config_A": c_a,
                    "Config_B": c_b,
                    "MAE_A": np.mean(err_a),
                    "MAE_B": np.mean(err_b),
                    "Median_AE_A": np.median(err_a),
                    "Median_AE_B": np.median(err_b),
                    "T-Statistic": t_stat,
                    "T-Test_P-Value": format_p(t_p),
                    "T-Test_Sig": format_sig(t_p),
                    "Wilcoxon_Stat": w_stat,
                    "Wilcoxon_P-Value": format_p(w_p),
                    "Wilcoxon_Sig": format_sig(w_p)
                })
                print(
                    f"[{ver}] {c_a} vs {c_b} -> T-test p={format_p(t_p)} ({format_sig(t_p)}) | Wilcoxon p={format_p(w_p)} ({format_sig(w_p)})")

    pd.DataFrame(intra_results).to_csv(os.path.join(OUT_DIR, "statistical_significance_intra_versions.csv"),
                                       index=False)
    print(f"\n[OK] Risultati salvati in '{OUT_DIR}/'")


if __name__ == "__main__":
    main()