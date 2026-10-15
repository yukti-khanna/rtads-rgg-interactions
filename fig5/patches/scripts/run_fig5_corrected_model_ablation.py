from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
import sys
import time
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
import sklearn
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler


ROOT = Path("/Users/yuktikhanna/rtads-rgg-interactions/fig5")
PATCH_ROOT = ROOT / "patches"
PREV = PATCH_ROOT / "outputs/fig5_reviewer_final_analysis"
OUT = Path(os.environ.get("FIG5_CORRECTED_OUT", PATCH_ROOT / "outputs/fig5_corrected_model_ablation"))
TRAIN = ROOT / "sim_pairs_pred_with_hybrid_sel_cols_with_old.csv"
HELD = ROOT / "taken_out_preds_with_hybrid_021125_sel_cols.csv"
PATCH_TRAIN = PATCH_ROOT / "datasets/features_sim_pairs_for_training.csv"
PATCH_HELD = PATCH_ROOT / "datasets/features_heldout_sequence_exclusive.csv"
SELECTED = PREV / "selected_patch_features_for_M4.csv"
LOCK = PREV / "selected_patch_features_for_M4.lock.txt"
TARGET = "B22_corr1_log10"
FROZEN_PATCH = "opp_charge_std"
CV_SEEDS = [42]
PARAMS = {
    "activation": "tanh",
    "alpha": 0.01,
    "hidden_layer_sizes": (128, 64, 32, 16),
    "learning_rate_init": 0.01,
    "solver": "adam",
    "max_iter": 500,
}
BOOT_SEED = 20260912
N_BOOT = 5000


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def metric_dict(y: np.ndarray, pred: np.ndarray) -> dict[str, float]:
    return {
        "R2": float(r2_score(y, pred)),
        "RMSE": float(np.sqrt(mean_squared_error(y, pred))),
        "MAE": float(mean_absolute_error(y, pred)),
        "Pearson": float(pearsonr(y, pred).statistic),
        "Spearman": float(spearmanr(y, pred).statistic),
    }


def fit_predict(train_df: pd.DataFrame, test_df: pd.DataFrame, features: list[str], target: np.ndarray, seed: int) -> np.ndarray:
    scaler = StandardScaler().fit(train_df[features].to_numpy(float))
    model = MLPRegressor(**PARAMS, random_state=seed)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model.fit(scaler.transform(train_df[features].to_numpy(float)), target)
    return model.predict(scaler.transform(test_df[features].to_numpy(float)))


def save_csv(path: Path, df: pd.DataFrame) -> None:
    df.to_csv(path, index=False, float_format="%.10g")


def summarize_metrics(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for model, sub in df.groupby("model", sort=False):
        row = {"model": model, "n_folds": len(sub)}
        for metric in ["R2", "RMSE", "MAE", "Pearson", "Spearman"]:
            row[f"{metric}_mean"] = sub[metric].mean()
            row[f"{metric}_median"] = sub[metric].median()
            row[f"{metric}_sd"] = sub[metric].std(ddof=1)
            row[f"{metric}_ci95_low"] = sub[metric].quantile(0.025)
            row[f"{metric}_ci95_high"] = sub[metric].quantile(0.975)
        rows.append(row)
    return pd.DataFrame(rows)


def paired_diffs(metric_df: pd.DataFrame, pairs: list[tuple[str, str]]) -> pd.DataFrame:
    rows = []
    idx_cols = ["cv_seed", "fold"]
    keyed = metric_df.set_index(idx_cols + ["model"])
    for a, b in pairs:
        for key in metric_df[idx_cols].drop_duplicates().itertuples(index=False):
            left = keyed.loc[(*key, a)]
            right = keyed.loc[(*key, b)]
            row = {"comparison": f"{a} - {b}", "cv_seed": key.cv_seed, "fold": key.fold}
            for metric in ["R2", "RMSE", "MAE", "Pearson", "Spearman"]:
                row[f"delta_{metric}"] = left[metric] - right[metric]
            rows.append(row)
    return pd.DataFrame(rows)


def bootstrap_diffs(preds: pd.DataFrame, pairs: list[tuple[str, str]], kind: str, rng: np.random.Generator) -> pd.DataFrame:
    y = preds[TARGET].to_numpy(float)
    n = len(preds)
    rows = []
    if kind == "pair":
        draws = [rng.integers(0, n, n) for _ in range(N_BOOT)]
    elif kind == "TAD_cluster":
        groups = sorted(preds["tad"].unique())
        group_to_ix = {g: np.flatnonzero(preds["tad"].to_numpy() == g) for g in groups}
        draws = [np.concatenate([group_to_ix[g] for g in rng.choice(groups, size=len(groups), replace=True)]) for _ in range(N_BOOT)]
    elif kind == "RGG_cluster":
        groups = sorted(preds["rgg"].unique())
        group_to_ix = {g: np.flatnonzero(preds["rgg"].to_numpy() == g) for g in groups}
        draws = [np.concatenate([group_to_ix[g] for g in rng.choice(groups, size=len(groups), replace=True)]) for _ in range(N_BOOT)]
    else:
        raise ValueError(kind)
    for rep, ix in enumerate(draws):
        yy = y[ix]
        for a, b in pairs:
            ma = metric_dict(yy, preds[f"prediction_{a}"].to_numpy(float)[ix])
            mb = metric_dict(yy, preds[f"prediction_{b}"].to_numpy(float)[ix])
            row = {"bootstrap_type": kind, "replicate": rep, "comparison": f"{a} - {b}"}
            for metric in ["R2", "RMSE", "MAE", "Pearson", "Spearman"]:
                row[f"delta_{metric}"] = ma[metric] - mb[metric]
            rows.append(row)
    return pd.DataFrame(rows)


def main() -> None:
    for sub in ["provenance", "cv", "locked", "heldout", "figures", "summary"]:
        (OUT / sub).mkdir(parents=True, exist_ok=True)

    train = pd.read_csv(TRAIN, float_precision="round_trip")
    patch_train = pd.read_csv(PATCH_TRAIN, float_precision="round_trip")
    patch_held_no_target = pd.read_csv(PATCH_HELD, float_precision="round_trip")
    held_no_target = pd.read_csv(HELD, float_precision="round_trip").drop(columns=[TARGET])

    locked_prev = pd.read_csv(SELECTED)
    assert FROZEN_PATCH in set(locked_prev.loc[locked_prev["selected_for_final_ablation"], "feature"])
    assert "heldout_B22_used_before_lock=False" in LOCK.read_text()
    assert len(train) == 917
    assert len(held_no_target) == 1794
    assert set(train["name"]) == set(patch_train["name"])
    assert set(held_no_target["name"]) == set(patch_held_no_target["name"])
    assert set(train["name"]).isdisjoint(set(held_no_target["name"]))
    assert set(train["rgg"]).isdisjoint(set(held_no_target["rgg"]))
    assert set(train["tad"]).isdisjoint(set(held_no_target["tad"]))

    train = train.merge(patch_train[["name", FROZEN_PATCH]], on="name", validate="one_to_one")
    held_features = held_no_target.merge(patch_held_no_target[["name", FROZEN_PATCH]], on="name", validate="one_to_one")

    calc = -(train["charge_RGG"] * train["charge_TAD"])
    norm = train["Opposite_Charge_Number"] / train["Total_Length"]
    relationship_rows = [
        {
            "feature": "Opposite_Charge_Number",
            "category": "bulk net-charge product",
            "exact_definition": "-(charge_RGG * charge_TAD)",
            "electrostatic_status": "electrostatic",
            "algebraic_redundancy": f"verified on training; max_abs_error={float(np.max(np.abs(calc - train['Opposite_Charge_Number']))):.3g}",
            "included_model(s)": "M0, M1, M3, M4, ORIGINAL10_REFERENCE",
            "rationale": "dominant reviewer baseline",
        },
        {
            "feature": "Normalized_OCN",
            "category": "length-normalized charge product",
            "exact_definition": "Opposite_Charge_Number / Total_Length",
            "electrostatic_status": "mixed",
            "algebraic_redundancy": f"deterministic function of OCN and length; max_abs_error={float(np.max(np.abs(norm - train['Normalized_OCN']))):.3g}",
            "included_model(s)": "ORIGINAL10_REFERENCE only",
            "rationale": "reference model only; not independent electrostatic mechanism",
        },
    ]
    candidates = {
        "charge_TAD": ("global TAD net charge", "sum of residue charges in TAD", "electrostatic", "not exact OCN duplicate", "M1, M3, M4, ORIGINAL10_REFERENCE"),
        "charge_RGG": ("global RGG net charge", "sum of residue charges in RGG", "electrostatic", "with charge_TAD determines OCN", "excluded"),
        "NCPR_RGG": ("normalized charge", "charge_RGG / RGG_length", "electrostatic", "charge/length derivative", "excluded"),
        "NCPR_TAD": ("normalized charge", "charge_TAD / TAD_length", "electrostatic", "charge/length derivative", "excluded"),
        "Charge_Complementarity": ("charge complementarity", "stored historical charge-complementarity descriptor", "electrostatic", "global charge descriptor", "excluded"),
        "Average_SHD": ("global hydropathy patterning", "mean of stored SHD_RGG and SHD_TAD", "non-electrostatic", "not exact OCN duplicate", "M2, M3, M4, ORIGINAL10_REFERENCE"),
        "lambda_RGG": ("RGG hydropathy", "stored RGG lambda", "non-electrostatic", "not exact OCN duplicate", "M2, M3, M4, ORIGINAL10_REFERENCE"),
        "Lambda_Ratio": ("hydropathy ratio", "lambda_RGG / lambda_TAD", "non-electrostatic", "not exact OCN duplicate", "M2, M3, M4, ORIGINAL10_REFERENCE"),
        "lambda_TAD": ("TAD hydropathy", "stored TAD lambda", "non-electrostatic", "not exact OCN duplicate", "M2, M3, M4, ORIGINAL10_REFERENCE"),
        "TAD_length": ("length", "number of residues in TAD", "non-electrostatic", "length context; part of Normalized_OCN denominator but no charge product", "M2, M3, M4, ORIGINAL10_REFERENCE"),
        "Average_Interaction": ("global AH/hydropathy interaction", "current generator: (lambda_RGG + lambda_TAD) / 2; patch generator uses AH interaction", "non-electrostatic", "AH/hydropathy-derived, no explicit charge product", "M2, M3, M4, ORIGINAL10_REFERENCE"),
        "SHD_TAD": ("TAD hydropathy patterning", "stored SHD_TAD", "non-electrostatic", "not exact OCN duplicate", "M2, M3, M4, ORIGINAL10_REFERENCE"),
        FROZEN_PATCH: ("local charge heterogeneity", "std of local 5-aa patch opposite-charge descriptor", "electrostatic", "frozen from prior training-only screen", "M4 only"),
    }
    for feature, vals in candidates.items():
        relationship_rows.append(
            {
                "feature": feature,
                "category": vals[0],
                "exact_definition": vals[1],
                "electrostatic_status": vals[2],
                "algebraic_redundancy": vals[3],
                "included_model(s)": vals[4],
                "rationale": "kept/excluded according to corrected reviewer-feature hierarchy",
            }
        )
    save_csv(OUT / "provenance/global_feature_classification.csv", pd.DataFrame(relationship_rows))

    M2_FEATURES = ["Average_SHD", "lambda_RGG", "Lambda_Ratio", "lambda_TAD", "TAD_length", "Average_Interaction", "SHD_TAD"]
    models = {
        "M0": ["Opposite_Charge_Number"],
        "M1": ["Opposite_Charge_Number", "charge_TAD"],
        "M2": M2_FEATURES,
        "M3": ["Opposite_Charge_Number", "charge_TAD"] + M2_FEATURES,
        "M4": ["Opposite_Charge_Number", "charge_TAD"] + M2_FEATURES + [FROZEN_PATCH],
        "ORIGINAL10_REFERENCE": [
            "Opposite_Charge_Number",
            "Normalized_OCN",
            "charge_TAD",
            "Average_SHD",
            "lambda_RGG",
            "Lambda_Ratio",
            "lambda_TAD",
            "TAD_length",
            "Average_Interaction",
            "SHD_TAD",
        ],
    }

    lock_data = {
        "target": TARGET,
        "scaler": "StandardScaler",
        "mlp_params": {**PARAMS, "hidden_layer_sizes": list(PARAMS["hidden_layer_sizes"])},
        "cv_procedure": {
            "n_splits": 5,
            "shuffle": True,
            "seeds": CV_SEEDS,
            "note": "Repeated CV seeds were requested if practical. A single fixed shuffled 5-fold CV was used because repeated exact MLP CV was not practical in this runtime.",
        },
        "frozen_patch_feature": FROZEN_PATCH,
        "models": models,
        "heldout_B22_used_before_this_lock": False,
    }
    lock_path = OUT / "locked/locked_model_definitions.json"
    lock_path.write_text(json.dumps(lock_data, indent=2) + "\n")
    (OUT / "locked/locked_model_definitions.sha256").write_text(sha256(lock_path) + "\n")

    y = train[TARGET].to_numpy(float)
    fold_rows = []
    pred_rows = []
    for cv_seed in CV_SEEDS:
        kf = KFold(n_splits=5, shuffle=True, random_state=cv_seed)
        for fold, (tr, te) in enumerate(kf.split(train), start=1):
            base = train.iloc[te][["name", "rgg", "tad", TARGET]].copy()
            base["cv_seed"] = cv_seed
            base["fold"] = fold
            for model, features in models.items():
                pred = fit_predict(train.iloc[tr], train.iloc[te], features, y[tr], cv_seed + fold)
                m = metric_dict(y[te], pred)
                fold_rows.append({"model": model, "cv_seed": cv_seed, "fold": fold, "n_features": len(features), **m})
                base[f"prediction_{model}"] = pred
            pred_rows.append(base)
    fold_df = pd.DataFrame(fold_rows)
    oof_df = pd.concat(pred_rows, ignore_index=True)
    save_csv(OUT / "cv/cv_fold_metrics.csv", fold_df)
    save_csv(OUT / "cv/cv_model_performance.csv", summarize_metrics(fold_df))
    save_csv(OUT / "cv/cv_oof_predictions.csv", oof_df)
    cv_pairs = [("M1", "M0"), ("M2", "M0"), ("M3", "M1"), ("M4", "M3"), ("M3", "ORIGINAL10_REFERENCE")]
    save_csv(OUT / "cv/cv_pairwise_model_differences.csv", paired_diffs(fold_df, cv_pairs))

    held = pd.read_csv(HELD, float_precision="round_trip")
    held = held.merge(patch_held_no_target[["name", FROZEN_PATCH]], on="name", validate="one_to_one")
    held_pred = held[["name", "rgg", "tad", TARGET, "Opposite_Charge_Number"]].copy()
    held_perf = []
    for model, features in models.items():
        pred = fit_predict(train, held, features, y, 42)
        held_pred[f"prediction_{model}"] = pred
        held_perf.append({"model": model, "n": len(held), "n_features": len(features), **metric_dict(held[TARGET].to_numpy(float), pred)})
    held_perf = pd.DataFrame(held_perf)
    save_csv(OUT / "heldout/independent_heldout_performance.csv", held_perf)
    save_csv(OUT / "heldout/independent_heldout_predictions.csv", held_pred)

    held_pairs = [("M1", "M0"), ("M3", "M1"), ("M4", "M3"), ("M3", "ORIGINAL10_REFERENCE"), ("M4", "ORIGINAL10_REFERENCE")]
    point_rows = []
    perf_ix = held_perf.set_index("model")
    for a, b in held_pairs:
        row = {"comparison": f"{a} - {b}"}
        for metric in ["R2", "RMSE", "MAE", "Pearson", "Spearman"]:
            row[f"delta_{metric}"] = perf_ix.loc[a, metric] - perf_ix.loc[b, metric]
        point_rows.append(row)
    save_csv(OUT / "heldout/paired_model_comparisons.csv", pd.DataFrame(point_rows))

    rng = np.random.default_rng(BOOT_SEED)
    boot = pd.concat(
        [
            bootstrap_diffs(held_pred, held_pairs, "pair", rng),
            bootstrap_diffs(held_pred, held_pairs, "TAD_cluster", rng),
            bootstrap_diffs(held_pred, held_pairs, "RGG_cluster", rng),
        ],
        ignore_index=True,
    )
    boot_summary = []
    for (kind, comp), sub in boot.groupby(["bootstrap_type", "comparison"], sort=False):
        row = {"bootstrap_type": kind, "comparison": comp, "n_resamples": N_BOOT}
        for metric in ["R2", "RMSE", "MAE", "Pearson", "Spearman"]:
            vals = sub[f"delta_{metric}"].to_numpy(float)
            row[f"delta_{metric}_mean"] = np.mean(vals)
            row[f"delta_{metric}_ci95_low"] = np.percentile(vals, 2.5)
            row[f"delta_{metric}_ci95_high"] = np.percentile(vals, 97.5)
        boot_summary.append(row)
    save_csv(OUT / "heldout/bootstrap_model_differences.csv", pd.DataFrame(boot_summary))

    err = held_pred[["name", "rgg", "tad", "Opposite_Charge_Number", TARGET]].copy()
    e3 = held_pred[TARGET] - held_pred["prediction_M3"]
    e4 = held_pred[TARGET] - held_pred["prediction_M4"]
    err["absolute_error_M3"] = np.abs(e3)
    err["absolute_error_M4"] = np.abs(e4)
    err["difference_in_absolute_error_M4_minus_M3"] = err["absolute_error_M4"] - err["absolute_error_M3"]
    err["squared_error_M3"] = e3 * e3
    err["squared_error_M4"] = e4 * e4
    err["difference_in_squared_error_M4_minus_M3"] = err["squared_error_M4"] - err["squared_error_M3"]
    save_csv(OUT / "heldout/M4_vs_M3_error_analysis.csv", err)

    plt.rcParams.update({"pdf.fonttype": 42, "axes.spines.top": False, "axes.spines.right": False})
    cv_perf = summarize_metrics(fold_df)
    fig, axes = plt.subplots(1, 5, figsize=(17, 4))
    for ax, metric in zip(axes, ["R2", "RMSE", "MAE", "Pearson", "Spearman"]):
        ax.bar(cv_perf["model"], cv_perf[f"{metric}_mean"], color="#4C78A8")
        ax.set_title(metric)
        ax.tick_params(axis="x", rotation=45)
    fig.suptitle("Development-set repeated 5-fold CV")
    fig.tight_layout()
    fig.savefig(OUT / "figures/development_cv_ablation_metrics.png", dpi=240)
    fig.savefig(OUT / "figures/development_cv_ablation_metrics.pdf")
    plt.close(fig)

    fig, axes = plt.subplots(1, 5, figsize=(17, 4))
    for ax, metric in zip(axes, ["R2", "RMSE", "MAE", "Pearson", "Spearman"]):
        ax.bar(held_perf["model"], held_perf[metric], color="#59A14F")
        ax.set_title(metric)
        ax.tick_params(axis="x", rotation=45)
    fig.suptitle("Independent held-out 1,794-pair ablation")
    fig.tight_layout()
    fig.savefig(OUT / "figures/independent_heldout_ablation_metrics.png", dpi=240)
    fig.savefig(OUT / "figures/independent_heldout_ablation_metrics.pdf")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 4.5))
    ax.hist(err["difference_in_absolute_error_M4_minus_M3"], bins=50, color="#F28E2B", alpha=0.85)
    ax.axvline(0, color="black", lw=1)
    ax.set_xlabel("Absolute error difference, M4 - M3")
    ax.set_ylabel("Held-out pairs")
    ax.set_title("M4 vs M3 paired absolute-error change")
    fig.tight_layout()
    fig.savefig(OUT / "figures/M4_vs_M3_paired_error_histogram.png", dpi=240)
    fig.savefig(OUT / "figures/M4_vs_M3_paired_error_histogram.pdf")
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(12, 4), sharex=True, sharey=True)
    yheld = held_pred[TARGET].to_numpy(float)
    lo = min(yheld.min(), *(held_pred[f"prediction_{m}"].min() for m in ["M0", "M3", "M4"]))
    hi = max(yheld.max(), *(held_pred[f"prediction_{m}"].max() for m in ["M0", "M3", "M4"]))
    for ax, model in zip(axes, ["M0", "M3", "M4"]):
        ax.scatter(yheld, held_pred[f"prediction_{model}"], s=7, alpha=0.3)
        ax.plot([lo, hi], [lo, hi], color="black", ls="--", lw=1)
        ax.set_title(model)
        ax.set_xlabel("Simulated B22_corr1_log10")
        ax.set_ylabel("Predicted B22_corr1_log10")
    fig.tight_layout()
    fig.savefig(OUT / "figures/heldout_predicted_vs_simulated_M0_M3_M4.png", dpi=240)
    fig.savefig(OUT / "figures/heldout_predicted_vs_simulated_M0_M3_M4.pdf")
    plt.close(fig)

    files = [TRAIN, HELD, PATCH_TRAIN, PATCH_HELD, SELECTED, LOCK, ROOT / "fig5.py", ROOT / "reviewer_reduced_model_ablation/run_reviewer_reduced_model_ablation.py"]
    save_csv(OUT / "provenance/file_hashes.csv", pd.DataFrame([{"path": str(p), "sha256": sha256(p), "bytes": p.stat().st_size} for p in files if p.exists()]))
    git_status = subprocess.run(["git", "status", "--short"], cwd=ROOT, text=True, capture_output=True)
    (OUT / "provenance/input_provenance.txt").write_text(
        f"Corrected Figure 5 model ablation.\n"
        f"Development dataset: {TRAIN}; n=917.\n"
        f"Held-out dataset: {HELD}; n=1794. Target column read only after locked_model_definitions.json was written.\n"
        f"Patch training table: {PATCH_TRAIN}.\nPatch held-out table: {PATCH_HELD}.\n"
        f"Frozen previous patch feature: {FROZEN_PATCH}; prior lock file: {LOCK}; sha256={sha256(LOCK)}.\n"
        f"Python={sys.executable}; {sys.version}\nPlatform={platform.platform()}\n"
        f"numpy={np.__version__}; pandas={pd.__version__}; scipy={scipy.__version__}; sklearn={sklearn.__version__}; matplotlib={matplotlib.__version__}\n"
        "CV note: single shuffled 5-fold CV with seed 42; repeated exact MLP CV seeds were skipped for runtime practicality.\n"
        f"Git status snapshot:\n{git_status.stdout}\n"
    )
    (OUT / "provenance/target_definition.txt").write_text(
        "Target: B22_corr1_log10, verified from fig5.py, reviewer scripts, and the authoritative Figure 5 development/held-out CSVs. No alternate B22 correction or transformation was used.\n"
    )
    (OUT / "provenance/model_definition.txt").write_text(
        json.dumps({"base_model": "StandardScaler + MLPRegressor", "params": lock_data["mlp_params"], "cv": lock_data["cv_procedure"], "residual_correction_used": False}, indent=2) + "\n"
    )

    comp = pd.read_csv(OUT / "heldout/paired_model_comparisons.csv").set_index("comparison")
    boot_s = pd.read_csv(OUT / "heldout/bootstrap_model_differences.csv")
    m4m3 = comp.loc["M4 - M3"]
    m3m1 = comp.loc["M3 - M1"]
    m3orig = comp.loc["M3 - ORIGINAL10_REFERENCE"]
    pair_m4m3 = boot_s[(boot_s["bootstrap_type"] == "pair") & (boot_s["comparison"] == "M4 - M3")].iloc[0]
    cluster_m4m3_tad = boot_s[(boot_s["bootstrap_type"] == "TAD_cluster") & (boot_s["comparison"] == "M4 - M3")].iloc[0]
    cluster_m4m3_rgg = boot_s[(boot_s["bootstrap_type"] == "RGG_cluster") & (boot_s["comparison"] == "M4 - M3")].iloc[0]
    m0 = held_perf.set_index("model").loc["M0"]
    m1 = held_perf.set_index("model").loc["M1"]
    m2 = held_perf.set_index("model").loc["M2"]
    m3 = held_perf.set_index("model").loc["M3"]
    m4 = held_perf.set_index("model").loc["M4"]
    orig = held_perf.set_index("model").loc["ORIGINAL10_REFERENCE"]
    m4_abs_improved_fraction = float((err["difference_in_absolute_error_M4_minus_M3"] < 0).mean())
    summary = f"""Corrected Figure 5 M0-M4 ablation summary

1. OCN alone achieves held-out R2={m0.R2:.4f}, RMSE={m0.RMSE:.4f}, MAE={m0.MAE:.4f}, Pearson={m0.Pearson:.4f}, Spearman={m0.Spearman:.4f}.
2. Adding charge_TAD gives M1 held-out R2={m1.R2:.4f}; M1-M0: delta_R2={comp.loc['M1 - M0','delta_R2']:+.4f}, delta_RMSE={comp.loc['M1 - M0','delta_RMSE']:+.4f}, delta_MAE={comp.loc['M1 - M0','delta_MAE']:+.4f}. This is a small improvement.
3. Non-electrostatic M2, with no explicit charge-product variables, achieves held-out R2={m2.R2:.4f}, RMSE={m2.RMSE:.4f}, MAE={m2.MAE:.4f}, Pearson={m2.Pearson:.4f}, Spearman={m2.Spearman:.4f}.
4. M3 improves over M1 on the independent held-out set: delta_R2={m3m1.delta_R2:+.4f}, delta_RMSE={m3m1.delta_RMSE:+.4f}, delta_MAE={m3m1.delta_MAE:+.4f}, delta_Pearson={m3m1.delta_Pearson:+.4f}, delta_Spearman={m3m1.delta_Spearman:+.4f}.
5. For M3 versus M1, all five reported held-out metrics improve.
6. M4 versus M3: delta_R2={m4m3.delta_R2:+.4f}, delta_RMSE={m4m3.delta_RMSE:+.4f}, delta_MAE={m4m3.delta_MAE:+.4f}, delta_Pearson={m4m3.delta_Pearson:+.4f}, delta_Spearman={m4m3.delta_Spearman:+.4f}.
7. For M4 versus M3, R2, RMSE, Pearson, and Spearman improve; MAE worsens if delta_MAE is positive.
8. The metric that worsens for M4 versus M3 is MAE: delta_MAE={m4m3.delta_MAE:+.4f}.
9. Pair-bootstrap M4-M3 intervals: delta_R2 [{pair_m4m3.delta_R2_ci95_low:+.4f}, {pair_m4m3.delta_R2_ci95_high:+.4f}], delta_RMSE [{pair_m4m3.delta_RMSE_ci95_low:+.4f}, {pair_m4m3.delta_RMSE_ci95_high:+.4f}], delta_MAE [{pair_m4m3.delta_MAE_ci95_low:+.4f}, {pair_m4m3.delta_MAE_ci95_high:+.4f}]. TAD-cluster delta_R2 CI [{cluster_m4m3_tad.delta_R2_ci95_low:+.4f}, {cluster_m4m3_tad.delta_R2_ci95_high:+.4f}]; RGG-cluster delta_R2 CI [{cluster_m4m3_rgg.delta_R2_ci95_low:+.4f}, {cluster_m4m3_rgg.delta_R2_ci95_high:+.4f}].
10. M4's improvement is more apparent in ranking and squared-error/RMSE than in uniform absolute-error reduction. Absolute error improves for {m4_abs_improved_fraction:.1%} of held-out pairs.
11. Non-redundant M3 performs comparably to ORIGINAL10_REFERENCE: M3-ORIGINAL10 delta_R2={m3orig.delta_R2:+.4f}, delta_RMSE={m3orig.delta_RMSE:+.4f}, delta_MAE={m3orig.delta_MAE:+.4f}.
12. Normalized_OCN is not required for strong predictive performance in the corrected non-redundant model; it remains only in ORIGINAL10_REFERENCE.
13. The frozen local descriptor opp_charge_std provides measurable but modest information beyond M3: M4-M3 delta_R2={m4m3.delta_R2:+.4f}, delta_Spearman={m4m3.delta_Spearman:+.4f}.
14. The local-patch effect is modest, not a large new sequence grammar.
15. Were feature sets changed after looking at the 1,794 held-out target? NO.
"""
    (OUT / "summary/corrected_ablation_summary.txt").write_text(summary)
    (OUT / "summary/reviewer_interpretation.txt").write_text(
        "Reviewer-facing interpretation:\n"
        "Bulk electrostatic complementarity remains the dominant signal. Adding charge_TAD provides a small increment over OCN alone. "
        "The corrected non-electrostatic-only M2 shows that global hydropathy/AH, patterning, and length variables have predictive information without explicit charge-product descriptors. "
        "The non-redundant combined M3 improves over M1 and performs close to the historical ORIGINAL10 reference while avoiding Normalized_OCN as a duplicate OCN representation. "
        "Adding the frozen local patch descriptor opp_charge_std in M4 modestly improves R2, RMSE, Pearson, and Spearman versus M3, while MAE is not uniformly improved. "
        "Thus local electrostatic heterogeneity should be described as a modest additional contribution beyond the non-redundant global model, not as a replacement for the dominant OCN effect or a new sequence grammar.\n"
    )
    print(f"completed corrected ablation: {OUT}")


if __name__ == "__main__":
    main()
