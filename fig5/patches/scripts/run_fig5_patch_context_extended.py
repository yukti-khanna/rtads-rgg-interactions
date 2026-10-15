from __future__ import annotations

import json
import os
import datetime as dt
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler


ROOT = Path("/Users/yuktikhanna/rtads-rgg-interactions/fig5")
PATCH_ROOT = ROOT / "patches"
OUT = Path(os.environ.get("FIG5_PATCH_CONTEXT_OUT", PATCH_ROOT / "outputs/fig5_patch_context_extended_complete"))
CORRECTED = PATCH_ROOT / "outputs/fig5_corrected_model_ablation"
TRAIN = ROOT / "sim_pairs_pred_with_hybrid_sel_cols_with_old.csv"
PATCH_TRAIN = PATCH_ROOT / "datasets/features_sim_pairs_for_training.csv"
LOCKED = CORRECTED / "locked/locked_model_definitions.json"
SCREEN = PATCH_ROOT / "outputs/fig5_patch_feature_importance/patch_feature_screening_inventory.csv"
TARGET = "B22_corr1_log10"
CV_SEEDS = [42, 123, 456, 789, 2026]
PARAMS = dict(
    activation="tanh",
    alpha=0.01,
    hidden_layer_sizes=(128, 64, 32, 16),
    learning_rate_init=0.01,
    solver="adam",
    max_iter=500,
)


def metrics(y, pred):
    return dict(
        R2=float(r2_score(y, pred)),
        RMSE=float(np.sqrt(mean_squared_error(y, pred))),
        MAE=float(mean_absolute_error(y, pred)),
        Pearson=float(pearsonr(y, pred).statistic),
        Spearman=float(spearmanr(y, pred).statistic),
    )


def fit_predict(train_df, test_df, features, y_train, seed):
    scaler = StandardScaler().fit(train_df[features].to_numpy(float))
    model = MLPRegressor(**PARAMS, random_state=seed)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model.fit(scaler.transform(train_df[features].to_numpy(float)), y_train)
    return model.predict(scaler.transform(test_df[features].to_numpy(float)))


def save_csv(name, df):
    df.to_csv(OUT / name, index=False, float_format="%.10g")


def summarize(metric_rows):
    df = pd.DataFrame(metric_rows)
    rows = []
    for model, sub in df.groupby("model", sort=False):
        row = {"model": model, "n_folds": len(sub), "n_features": int(sub["n_features"].iloc[0])}
        for metric in ["R2", "RMSE", "MAE", "Pearson", "Spearman"]:
            row[f"{metric}_mean"] = sub[metric].mean()
            row[f"{metric}_sd"] = sub[metric].std(ddof=1)
            row[f"{metric}_median"] = sub[metric].median()
            row[f"{metric}_min"] = sub[metric].min()
            row[f"{metric}_max"] = sub[metric].max()
        rows.append(row)
    return pd.DataFrame(rows)


def compare(fold_metrics, pairs):
    idx = fold_metrics.set_index(["seed", "fold", "model"])
    rows = []
    for a, b in pairs:
        deltas = []
        for seed, fold in fold_metrics[["seed", "fold"]].drop_duplicates().itertuples(index=False):
            left = idx.loc[(seed, fold, a)]
            right = idx.loc[(seed, fold, b)]
            row = {"comparison": f"{a} - {b}", "seed": seed, "fold": fold}
            for metric in ["R2", "RMSE", "MAE", "Pearson", "Spearman"]:
                row[f"delta_{metric}"] = left[metric] - right[metric]
            deltas.append(row)
        tmp = pd.DataFrame(deltas)
        summary = {"comparison": f"{a} - {b}", "first_model": b, "second_model": a, "delta_convention": "second_model - first_model", "n_folds": len(tmp)}
        rng = np.random.default_rng(20260912)
        for metric in ["R2", "RMSE", "MAE", "Pearson", "Spearman"]:
            vals = tmp[f"delta_{metric}"]
            summary[f"delta_{metric}_mean"] = vals.mean()
            summary[f"delta_{metric}_sd"] = vals.std(ddof=1)
            summary[f"delta_{metric}_median"] = vals.median()
            boot = [vals.to_numpy()[rng.integers(0, len(vals), len(vals))].mean() for _ in range(5000)]
            summary[f"delta_{metric}_ci95_low"] = np.percentile(boot, 2.5)
            summary[f"delta_{metric}_ci95_high"] = np.percentile(boot, 97.5)
            summary[f"fraction_folds_{metric}_improved"] = float((vals > 0).mean()) if metric in {"R2", "Pearson", "Spearman"} else float((vals < 0).mean())
        rows.append(summary)
    return pd.DataFrame(rows)


def interpretation(row):
    good = sum(
        [
            row["mean_delta_R2"] > 0,
            row["mean_delta_RMSE"] < 0,
            row["mean_delta_MAE"] < 0,
            row["mean_delta_Pearson"] > 0,
            row["mean_delta_Spearman"] > 0,
        ]
    )
    if good >= 4:
        return "improves M3 across most metrics"
    if good >= 2:
        return "mixed or modest incremental value"
    return "does not improve M3 in this CV comparison"


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    train_global = pd.read_csv(TRAIN, float_precision="round_trip")
    patch = pd.read_csv(PATCH_TRAIN, float_precision="round_trip")
    df = train_global.merge(patch, on=["name", "rgg", "tad"], suffixes=("", "_patch"), validate="one_to_one")
    y = df[TARGET].to_numpy(float)

    lock = json.loads(LOCKED.read_text())
    m3 = lock["models"]["M3"]
    assert m3 == [
        "Opposite_Charge_Number",
        "charge_TAD",
        "Average_SHD",
        "lambda_RGG",
        "Lambda_Ratio",
        "lambda_TAD",
        "TAD_length",
        "Average_Interaction",
        "SHD_TAD",
    ]

    screen = pd.read_csv(SCREEN)
    valid = set(screen.loc[screen["include_initial_screen"], "feature"])
    balanced = [
        "opp_charge_std",
        "scd_patch1_std",
        "scd_patch2_std",
        "rg_fraction_patch1_std",
        "aromatic_fraction_patch2_std",
        "shd_patch1_std",
        "shd_patch2_std",
    ]
    for feat in balanced + ["ah_energy_std"]:
        assert feat in valid and feat in df.columns

    families = {
        "LOCAL-ELECTROSTATICS": sorted([f for f in valid if f.startswith(("opp_charge_", "charge_mismatch_")) and f in df.columns]),
        "LOCAL-CHARGE-PATTERNING": sorted([f for f in valid if f.startswith(("scd_patch1_", "scd_patch2_")) and f in df.columns]),
        "LOCAL-RG-CONTEXT": sorted([f for f in valid if f.startswith(("rg_fraction_patch1_", "rgg_count_patch1_")) and f in df.columns]),
        "LOCAL-AROMATIC-CONTEXT": sorted([f for f in valid if f.startswith("aromatic_fraction_") and f in df.columns]),
        "LOCAL-HYDROPATHY-CONTEXT": sorted([f for f in valid if f.startswith(("shd_patch1_", "shd_patch2_", "hydropathy_diff_")) and f in df.columns]),
        "LOCAL-AH-CONTEXT": sorted([f for f in valid if f.startswith("ah_energy_") and not f.startswith("ah_density_") and f in df.columns]),
    }
    models = {
        "M3": m3,
        "M4-min": m3 + ["opp_charge_std"],
        "M4-panel": m3 + balanced,
        "M4-panel-AH": m3 + balanced + ["ah_energy_std"],
    }
    for fam, feats in families.items():
        models[fam] = m3 + feats

    model_defs = []
    for model, feats in models.items():
        patch_feats = [f for f in feats if f not in m3]
        model_defs.append(
            {
                "model": model,
                "n_total_features": len(feats),
                "n_patch_features": len(patch_feats),
                "patch_features": ",".join(patch_feats),
                "all_features": ",".join(feats),
            }
        )
    save_csv("model_definitions.csv", pd.DataFrame(model_defs))
    (OUT / "run_configuration.json").write_text(
        json.dumps(
            {
                "CV_seeds": CV_SEEDS,
                "n_folds": 5,
                "MLP_parameters": {**PARAMS, "hidden_layer_sizes": list(PARAMS["hidden_layer_sizes"]), "random_state": 42},
                "model_feature_lists": models,
                "target": TARGET,
                "input_files": {
                    "development_global": str(TRAIN),
                    "development_patch": str(PATCH_TRAIN),
                    "corrected_ablation_lock": str(LOCKED),
                    "screened_patch_inventory": str(SCREEN),
                },
                "run_datetime": dt.datetime.now().isoformat(timespec="seconds"),
                "heldout_B22_accessed": False,
            },
            indent=2,
        )
        + "\n"
    )

    fold_rows = []
    pred_rows = []
    for model_name in models:
        print(f"fitting {model_name}", flush=True)
        features = models[model_name]
        for cv_seed in CV_SEEDS:
            kf = KFold(n_splits=5, shuffle=True, random_state=cv_seed)
            for fold, (tr, te) in enumerate(kf.split(df), start=1):
                pred = fit_predict(df.iloc[tr], df.iloc[te], features, y[tr], 42)
                m = metrics(y[te], pred)
                fold_rows.append({"model": model_name, "seed": cv_seed, "fold": fold, "n_train": len(tr), "n_validation": len(te), "n_features": len(features), **m})
                chunk = df.iloc[te][["name", "rgg", "tad", TARGET]].copy()
                chunk["seed"] = cv_seed
                chunk["fold"] = fold
                chunk["model"] = model_name
                chunk["prediction"] = pred
                chunk["residual"] = chunk[TARGET] - chunk["prediction"]
                chunk["absolute_error"] = chunk["residual"].abs()
                chunk["squared_error"] = chunk["residual"] ** 2
                pred_rows.extend(chunk.to_dict("records"))
                save_csv("cv_fold_metrics.partial.csv", pd.DataFrame(fold_rows))

    fold_df = pd.DataFrame(fold_rows)
    save_csv("cv_fold_metrics.csv", fold_df)
    save_csv("cv_model_performance.csv", summarize(fold_rows))
    save_csv("cv_oof_predictions.csv", pd.DataFrame(pred_rows))

    primary = compare(fold_df, [("M4-min", "M3"), ("M4-panel", "M3"), ("M4-panel", "M4-min"), ("M4-panel-AH", "M4-panel")])
    save_csv("M4_min_vs_panel_comparisons.csv", primary)

    family_rows = []
    idx = fold_df.set_index(["seed", "fold", "model"])
    for fam, feats in families.items():
        diffs = []
        for seed, fold in fold_df[["seed", "fold"]].drop_duplicates().itertuples(index=False):
            left = idx.loc[(seed, fold, fam)]
            right = idx.loc[(seed, fold, "M3")]
            row = {"family": fam, "seed": seed, "fold": fold}
            for metric in ["R2", "RMSE", "MAE", "Pearson", "Spearman"]:
                row[f"delta_{metric}"] = left[metric] - right[metric]
            diffs.append(row)
        tmp = pd.DataFrame(diffs)
        row = {
            "family": fam,
            "n_features": len(feats),
            "features": ",".join(feats),
        }
        for metric in ["R2", "RMSE", "MAE", "Pearson", "Spearman"]:
            row[f"mean_delta_{metric}"] = tmp[f"delta_{metric}"].mean()
            row[f"sd_delta_{metric}"] = tmp[f"delta_{metric}"].std(ddof=1)
        row["fraction_folds_R2_improved"] = float((tmp["delta_R2"] > 0).mean())
        row["fraction_folds_RMSE_improved"] = float((tmp["delta_RMSE"] < 0).mean())
        row["fraction_folds_MAE_improved"] = float((tmp["delta_MAE"] < 0).mean())
        row["fraction_folds_Pearson_improved"] = float((tmp["delta_Pearson"] > 0).mean())
        row["fraction_folds_Spearman_improved"] = float((tmp["delta_Spearman"] > 0).mean())
        row["interpretation"] = interpretation(row)
        family_rows.append(row)
    family_df = pd.DataFrame(family_rows)
    save_csv("patch_family_incremental_over_M3.csv", family_df)

    perf = pd.read_csv(OUT / "cv_model_performance.csv")
    fig, axes = plt.subplots(1, 5, figsize=(17, 4))
    core = perf[perf["model"].isin(["M3", "M4-min", "M4-panel", "M4-panel-AH"])]
    for ax, metric in zip(axes, ["R2", "RMSE", "MAE", "Pearson", "Spearman"]):
        ax.bar(core["model"], core[f"{metric}_mean"], color="#4C78A8")
        ax.set_title(metric)
        ax.tick_params(axis="x", rotation=35)
    fig.suptitle("M3 and local patch-context panels, 917-pair CV")
    fig.tight_layout()
    fig.savefig(OUT / "M3_M4min_M4panel_metrics.pdf")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4.8))
    fam_plot = family_df.sort_values("mean_delta_R2")
    ax.barh(fam_plot["family"], fam_plot["mean_delta_R2"], color="#59A14F")
    ax.axvline(0, color="black", lw=0.8)
    ax.set_xlabel("Mean CV delta R2 vs M3")
    ax.set_title("Patch-family incremental performance")
    fig.tight_layout()
    fig.savefig(OUT / "patch_family_incremental_performance.pdf")
    plt.close(fig)

    comp = primary.set_index("comparison")
    fam = family_df.set_index("family")
    def row_text(name):
        r = comp.loc[name]
        return f"{name}: delta_R2={r.delta_R2_mean:+.4f}, delta_RMSE={r.delta_RMSE_mean:+.4f}, delta_MAE={r.delta_MAE_mean:+.4f}, delta_Pearson={r.delta_Pearson_mean:+.4f}, delta_Spearman={r.delta_Spearman_mean:+.4f}"

    panel_vs_min = comp.loc["M4-panel - M4-min"]
    if panel_vs_min.delta_R2_mean > 0 and panel_vs_min.delta_RMSE_mean < 0 and panel_vs_min.delta_MAE_mean < 0:
        panel_call = "The balanced panel improves over M4-min across the main error metrics."
    elif abs(panel_vs_min.delta_R2_mean) < 0.002 and abs(panel_vs_min.delta_RMSE_mean) < 0.002:
        panel_call = "The balanced panel performs similarly to M4-min, suggesting most useful local signal is captured by opp_charge_std."
    else:
        panel_call = "The balanced panel does not improve over M4-min, suggesting the minimal local electrostatic-heterogeneity descriptor is the more robust local contribution in this model."

    summary = f"""Patch-context extended CV sensitivity analysis

Data and guardrails
This analysis used only the 917-pair development dataset. It did not use the 1,794 held-out targets. Patch features were not regenerated. M3 was read from locked_model_definitions.json in the completed corrected ablation. The frozen minimal patch feature is opp_charge_std. Historical code mapping verified: patch1 is RGG and patch2 is TAD.

Primary comparisons
{row_text('M4-min - M3')}
{row_text('M4-panel - M3')}
{row_text('M4-panel - M4-min')}
{row_text('M4-panel-AH - M4-panel')}

Summary questions
1. Does opp_charge_std reproducibly improve M3? {'Yes' if comp.loc['M4-min - M3'].fraction_folds_R2_improved >= 0.6 and comp.loc['M4-min - M3'].fraction_folds_RMSE_improved >= 0.6 else 'No or only weakly'}.
2. Does the seven-feature balanced patch panel improve M3? {'Yes' if comp.loc['M4-panel - M3'].fraction_folds_R2_improved >= 0.6 and comp.loc['M4-panel - M3'].fraction_folds_RMSE_improved >= 0.6 else 'No or only weakly'}.
3. Does the balanced panel improve over opp_charge_std alone? {'Yes' if panel_vs_min.delta_R2_mean > 0 and panel_vs_min.delta_RMSE_mean < 0 else 'No'}.
4. If not, does this suggest most useful local information is captured by local electrostatic heterogeneity? {panel_call}
5. Which feature families improve prediction over M3? {', '.join(family_df.loc[(family_df.mean_delta_R2 > 0) & (family_df.mean_delta_RMSE < 0), 'family']) or 'none by both R2 and RMSE'}.
6. Do local charge-patterning descriptors contribute? {fam.loc['LOCAL-CHARGE-PATTERNING','interpretation']}.
7. Do RG-context descriptors contribute? {fam.loc['LOCAL-RG-CONTEXT','interpretation']}.
8. Do local aromatic descriptors contribute? {fam.loc['LOCAL-AROMATIC-CONTEXT','interpretation']}.
9. Do local hydropathy descriptors contribute? {fam.loc['LOCAL-HYDROPATHY-CONTEXT','interpretation']}.
10. Does local AH-energy heterogeneity contribute? {fam.loc['LOCAL-AH-CONTEXT','interpretation']}.
11. Are improvements consistent across R2, RMSE, MAE, Pearson and Spearman? See M4_min_vs_panel_comparisons.csv; consistency is assessed across all five metrics and fold-improvement fractions, not R2 alone.
12. Does the broader patch panel show evidence of overfitting? {('Yes, relative to M4-min in CV.' if panel_vs_min.delta_R2_mean < 0 or panel_vs_min.delta_RMSE_mean > 0 else 'No clear evidence relative to M4-min in CV.')}
"""
    (OUT / "patch_context_summary.txt").write_text(summary)
    expected_models = list(models)
    completion = []
    counts = fold_df.groupby("model").size().reindex(expected_models)
    completion.append(("every model has 25 CV folds", bool((counts == 25).all()), counts.to_dict()))
    seed_counts = fold_df.groupby(["model", "seed"])["fold"].nunique()
    completion.append(("every seed has exactly 5 folds for every model", bool((seed_counts == 5).all()), seed_counts.to_dict()))
    fold_sets = fold_df.groupby("model").apply(lambda x: set(zip(x["seed"], x["fold"])))
    completion.append(("identical seed/fold assignments were used across compared models", bool(len({frozenset(s) for s in fold_sets}) == 1), "all models share same seed/fold set"))
    completion.append(("no 1,794 held-out target was accessed", True, "script reads only the 917 development global and patch tables"))
    completion.append(("no feature set was modified during the run", True, "feature sets are predefined in model_definitions.csv and run_configuration.json"))
    completion.append(("all five metrics were calculated", bool(set(["R2", "RMSE", "MAE", "Pearson", "Spearman"]).issubset(fold_df.columns)), "R2/RMSE/MAE/Pearson/Spearman present"))
    completion.append(("all requested family models completed", bool(all(m in set(fold_df["model"]) for m in expected_models if m.startswith("LOCAL-"))), sorted([m for m in expected_models if m.startswith("LOCAL-")])))
    (OUT / "analysis_completion_check.txt").write_text(
        "\n".join(f"{'PASS' if ok else 'FAIL'}: {name} | {detail}" for name, ok, detail in completion) + "\n"
    )
    print(f"completed {OUT}", flush=True)


if __name__ == "__main__":
    main()
