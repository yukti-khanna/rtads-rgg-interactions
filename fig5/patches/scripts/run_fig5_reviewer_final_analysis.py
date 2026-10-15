from __future__ import annotations

import hashlib
import json
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
OUT = PATCH_ROOT / "outputs/fig5_reviewer_final_analysis"
TRAIN = ROOT / "sim_pairs_pred_with_hybrid_sel_cols_with_old.csv"
HELD = ROOT / "taken_out_preds_with_hybrid_021125_sel_cols.csv"
PATCH_TRAIN = PATCH_ROOT / "datasets/features_sim_pairs_for_training.csv"
PATCH_HELD = PATCH_ROOT / "datasets/features_heldout_sequence_exclusive.csv"
TARGET = "B22_corr1_log10"
CV_SEED = 42
MODEL_SEED = 42
BOOTSTRAP_SEED = 20260912
N_BOOT = 1000
PARAMS = dict(
    activation="tanh",
    alpha=0.01,
    hidden_layer_sizes=(128, 64, 32, 16),
    learning_rate_init=0.01,
    max_iter=500,
    solver="adam",
)
GLOBAL10_REFERENCE = [
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
]
GLOBAL_MODELS = {
    "M0": ["Opposite_Charge_Number"],
    "M1": ["Opposite_Charge_Number", "charge_TAD"],
    "M2": [
        "Opposite_Charge_Number",
        "charge_TAD",
        "SCD_RGG",
        "SCD_TAD",
        "Average_SHD",
        "lambda_RGG",
        "lambda_TAD",
        "TAD_length",
        "RGG_length",
    ],
    "GLOBAL10_REFERENCE": GLOBAL10_REFERENCE,
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save_csv(name: str, df: pd.DataFrame) -> None:
    df.to_csv(OUT / name, index=False, float_format="%.10g")


def save_text(name: str, text: str) -> None:
    (OUT / name).write_text(text)


def metrics(y: np.ndarray, pred: np.ndarray) -> dict[str, float]:
    return {
        "n": int(len(y)),
        "R2": float(r2_score(y, pred)),
        "RMSE": float(np.sqrt(mean_squared_error(y, pred))),
        "MAE": float(mean_absolute_error(y, pred)),
        "Pearson": float(pearsonr(y, pred).statistic),
        "Spearman": float(spearmanr(y, pred).statistic),
    }


def mean_ci(values: list[float] | np.ndarray) -> tuple[float, float, float, float, float]:
    arr = np.asarray(values, dtype=float)
    return (
        float(np.mean(arr)),
        float(np.median(arr)),
        float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0,
        float(np.percentile(arr, 2.5)),
        float(np.percentile(arr, 97.5)),
    )


def train_predict(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    features: list[str],
    y_train: np.ndarray,
    seed: int,
) -> np.ndarray:
    scaler = StandardScaler().fit(train_df[features].to_numpy(float))
    model = MLPRegressor(**PARAMS, random_state=seed)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model.fit(scaler.transform(train_df[features].to_numpy(float)), y_train)
    return model.predict(scaler.transform(test_df[features].to_numpy(float)))


def cv_predictions(df: pd.DataFrame, features: list[str], y: np.ndarray) -> tuple[pd.DataFrame, list[dict]]:
    rows = []
    pred_rows = []
    kf = KFold(n_splits=5, shuffle=True, random_state=CV_SEED)
    for fold, (tr, te) in enumerate(kf.split(df), start=1):
        pred = train_predict(df.iloc[tr], df.iloc[te], features, y[tr], MODEL_SEED + fold)
        m = metrics(y[te], pred)
        rows.append({"fold": fold, "n_features": len(features), **m})
        tmp = df.iloc[te][["name", "rgg", "tad"]].copy()
        tmp["fold"] = fold
        tmp["observed"] = y[te]
        tmp["predicted"] = pred
        pred_rows.append(tmp)
    return pd.concat(pred_rows, ignore_index=True), rows


def family_for(feature: str) -> str:
    mapping = [
        ("opp_charge", "local electrostatics"),
        ("charge_mismatch", "local electrostatics"),
        ("scd_patch1", "RGG charge patterning"),
        ("scd_patch2", "TAD charge patterning"),
        ("shd_patch1", "RGG hydropathy patterning"),
        ("shd_patch2", "TAD hydropathy patterning"),
        ("rg_fraction_patch1", "RG localization / clustering"),
        ("rgg_count_patch1", "RG localization / clustering"),
        ("aromatic_fraction_patch1", "RGG aromatic localization"),
        ("aromatic_fraction_patch2", "TAD aromatic localization"),
        ("ah_energy", "local AH interaction landscape"),
        ("hydropathy_diff", "hydropathy mismatch"),
        ("entropy_patch1", "RGG sequence complexity"),
        ("entropy_patch2", "TAD sequence complexity"),
    ]
    for prefix, fam in mapping:
        if feature.startswith(prefix):
            return fam
    return "other patch descriptor"


def screen_patch_features(patch: pd.DataFrame) -> tuple[list[str], pd.DataFrame, pd.DataFrame]:
    base_exclude = {"name", "rgg", "tad"}
    raw = [
        c
        for c in patch.columns
        if c not in base_exclude
        and (
            "patch" in c
            or c.startswith(
                (
                    "ah_energy_",
                    "ah_density_",
                    "opp_charge_",
                    "hydropathy_diff_",
                    "count_strong",
                    "corr_oppcharge",
                    "top_patch_gap",
                )
            )
        )
    ]
    rows = []
    seen_signature: dict[tuple, str] = {}
    included = []
    for col in raw:
        x = patch[col]
        reason = ""
        dep = ""
        caution = ""
        if x.nunique(dropna=False) <= 1 or float(x.var(ddof=0)) < 1e-12:
            reason = "constant or near-zero variance on 917 training rows"
        elif col.startswith("ah_density_"):
            reason = "excluded deterministic AH density rescaling"
            dep = "ah_density = ah_energy / 25 for fixed 5x5 windows"
        elif col.startswith(("ah_energy_top5", "ah_energy_top10")):
            reason = "excluded from primary screen"
            caution = "historical top5/top10 are numerically largest energies, not most-negative attraction tail"
        elif col == "count_strong_patches":
            reason = "excluded raw opportunity-dependent count"
            dep = "depends on number of patch-patch combinations and sequence length"
        elif col.startswith("kappa_patch"):
            reason = "excluded non-standard custom kappa-like descriptor"
            caution = "definition is not sufficiently interpretable for reviewer-facing local panel"
        else:
            signature = tuple(pd.util.hash_pandas_object(x, index=False).to_numpy())
            if signature in seen_signature:
                reason = f"excluded exact duplicate of {seen_signature[signature]}"
                dep = f"exact duplicate of {seen_signature[signature]}"
            else:
                seen_signature[signature] = col
        include = reason == ""
        if include:
            included.append(col)
        rows.append(
            {
                "feature": col,
                "included": include,
                "reason": "included" if include else reason,
                "feature_family": family_for(col),
                "historical_definition": "stored get_features.py 5-aa overlapping-window patch descriptor",
                "known_dependency": dep,
                "caution": caution,
            }
        )
    inventory = pd.DataFrame(rows)
    groups = inventory.loc[inventory["included"], ["feature", "feature_family"]].copy()
    return included, inventory, groups


def aggregate_delta(rows: list[dict], keys: list[str]) -> pd.DataFrame:
    out = []
    for key_values, sub in pd.DataFrame(rows).groupby(keys, sort=False):
        if not isinstance(key_values, tuple):
            key_values = (key_values,)
        row = dict(zip(keys, key_values))
        for metric in ["R2", "RMSE", "MAE", "Pearson", "Spearman"]:
            vals = sub[f"delta_{metric}"].to_numpy(float)
            mean, med, sd, lo, hi = mean_ci(vals)
            row[f"delta_{metric}_mean"] = mean
            row[f"delta_{metric}_median"] = med
            row[f"delta_{metric}_sd"] = sd
            row[f"delta_{metric}_ci95_low"] = lo
            row[f"delta_{metric}_ci95_high"] = hi
            if metric in {"R2", "Pearson", "Spearman"}:
                row[f"{metric}_improved_fraction"] = float(np.mean(vals > 0))
            else:
                row[f"{metric}_improved_fraction"] = float(np.mean(vals < 0))
        out.append(row)
    return pd.DataFrame(out)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    train = pd.read_csv(TRAIN, float_precision="round_trip")
    held = pd.read_csv(HELD, float_precision="round_trip")
    patch_train = pd.read_csv(PATCH_TRAIN, float_precision="round_trip")
    patch_held = pd.read_csv(PATCH_HELD, float_precision="round_trip")
    assert train.shape[0] == 917
    assert held.shape[0] == 1794
    assert patch_train["name"].is_unique and patch_held["name"].is_unique
    assert set(train["name"]) == set(patch_train["name"])
    assert set(held["name"]) == set(patch_held["name"])
    assert set(train["name"]).isdisjoint(set(held["name"]))
    assert set(train["rgg"]).isdisjoint(set(held["rgg"]))
    assert set(train["tad"]).isdisjoint(set(held["tad"]))
    assert held["rgg"].nunique() == 78 and held["tad"].nunique() == 23

    for df in (train, held):
        df["mean_lambda_RGG"] = df["lambda_RGG"] / df["RGG_length"]
        df["mean_lambda_TAD"] = df["lambda_TAD"] / df["TAD_length"]
    merged = train.merge(patch_train, on=["name", "rgg", "tad"], suffixes=("", "_patch"), validate="one_to_one")
    merged_held = held.merge(patch_held, on=["name", "rgg", "tad"], suffixes=("", "_patch"), validate="one_to_one")

    input_files = [TRAIN, HELD, PATCH_TRAIN, PATCH_HELD, PATCH_ROOT / "scripts/get_features.py"]
    save_text(
        "project_structure.txt",
        "\n".join(
            [
                "Relevant directories:",
                str(ROOT / "scripts"),
                str(ROOT / "outputs"),
                str(PATCH_ROOT / "scripts"),
                str(PATCH_ROOT / "features"),
                str(PATCH_ROOT / "datasets"),
                str(ROOT / "reviewer_reduced_model_ablation"),
                str(PATCH_ROOT / "outputs/fig5_patch_feature_importance"),
                "",
                "Authoritative inputs and SHA256:",
                *[f"{p}\t{sha256(p)}" for p in input_files if p.exists()],
                "",
                "Current Figure 5/reviewer model scripts inspected:",
                str(ROOT / "fig5.py"),
                str(ROOT / "reviewer_reduced_model_ablation/run_reviewer_reduced_model_ablation.py"),
                str(Path("/Users/yuktikhanna/idr-interaction-predict/scripts/mlp_with_plateau_residual_calibrated.py")),
                str(Path("/Users/yuktikhanna/idr-interaction-predict/scripts/mlp_feature_pruning.py")),
            ]
        )
        + "\n",
    )
    save_text(
        "target_definition.txt",
        "Target verified from fig5.py and reviewer/model scripts: B22_corr1_log10. "
        "It is the corrected simulated B22 value after log10 transformation used by the current Figure 5 analysis. "
        "No alternate B22 column is substituted; larger values follow the stored Figure 5 sign convention.\n",
    )
    save_text(
        "model_definition.txt",
        json.dumps(
            {
                "scaler": "StandardScaler fit inside each training fold or on 917 rows for frozen held-out evaluation",
                "estimator": "MLPRegressor",
                "params": PARAMS,
                "cv": {"n_splits": 5, "shuffle": True, "random_state": CV_SEED},
                "main_seed": MODEL_SEED,
                "note": "Main ablations do not use the high-OCN residual correction.",
            },
            indent=2,
        )
        + "\n",
    )

    relationships = []
    for label, df in [("training_917", train), ("heldout_1794", held)]:
        calc_ocn = -(df["charge_RGG"] * df["charge_TAD"])
        calc_norm = df["Opposite_Charge_Number"] / df["Total_Length"]
        relationships.append(
            {
                "dataset": label,
                "relationship": "Opposite_Charge_Number = -(charge_RGG * charge_TAD)",
                "max_abs_error": float(np.max(np.abs(calc_ocn - df["Opposite_Charge_Number"]))),
            }
        )
        relationships.append(
            {
                "dataset": label,
                "relationship": "Normalized_OCN = Opposite_Charge_Number / Total_Length",
                "max_abs_error": float(np.max(np.abs(calc_norm - df["Normalized_OCN"]))),
            }
        )
    save_csv("global_deterministic_relationships.csv", pd.DataFrame(relationships))

    included, inventory, groups = screen_patch_features(merged)
    save_csv("patch_feature_screening_inventory.csv", inventory)
    save_csv("patch_feature_groups.csv", groups)
    families = {fam: list(sub["feature"]) for fam, sub in groups.groupby("feature_family", sort=False)}
    y = merged[TARGET].to_numpy(float)

    uni_rows = []
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    for c in included:
        x = merged[c].to_numpy(float)
        pred = np.polyval(np.polyfit(x, y, 1), x)
        pear = pearsonr(x, y).statistic
        spear = spearmanr(x, y).statistic
        pboots = []
        sboots = []
        for _ in range(300):
            idx = rng.integers(0, len(y), len(y))
            pboots.append(pearsonr(x[idx], y[idx]).statistic)
            sboots.append(spearmanr(x[idx], y[idx]).statistic)
        uni_rows.append(
            {
                "feature": c,
                "Pearson_r": pear,
                "Pearson_ci95_low": np.percentile(pboots, 2.5),
                "Pearson_ci95_high": np.percentile(pboots, 97.5),
                "Spearman_rho": spear,
                "Spearman_ci95_low": np.percentile(sboots, 2.5),
                "Spearman_ci95_high": np.percentile(sboots, 97.5),
                "univariate_R2": pear * pear,
                "RMSE": np.sqrt(mean_squared_error(y, pred)),
                "MAE": mean_absolute_error(y, pred),
                "variance": np.var(x),
                "unique_values": pd.Series(x).nunique(),
                "missing_fraction": merged[c].isna().mean(),
            }
        )
    univariate = pd.DataFrame(uni_rows).merge(groups, on="feature", how="left")
    save_csv("patch_univariate_b22_associations.csv", univariate)

    oof, fold_metrics = cv_predictions(merged, included, y)
    save_csv("patch_model_oof_predictions.csv", oof)
    save_csv("patch_model_cv_fold_metrics.csv", pd.DataFrame(fold_metrics))

    kf = KFold(n_splits=5, shuffle=True, random_state=CV_SEED)
    individual_rows = []
    grouped_rows = []
    ablation_rows = []
    for fold, (tr, te) in enumerate(kf.split(merged), start=1):
        train_fold = merged.iloc[tr]
        test_fold = merged.iloc[te]
        ytr = y[tr]
        yte = y[te]
        scaler = StandardScaler().fit(train_fold[included].to_numpy(float))
        model = MLPRegressor(**PARAMS, random_state=MODEL_SEED + fold)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model.fit(scaler.transform(train_fold[included].to_numpy(float)), ytr)
        pred = model.predict(scaler.transform(test_fold[included].to_numpy(float)))
        base_metrics = metrics(yte, pred)
        rng = np.random.default_rng(5000 + fold)
        for feature in included:
            for repeat in range(5):
                shuffled = test_fold[included].copy()
                shuffled[feature] = shuffled[feature].to_numpy()[rng.permutation(len(shuffled))]
                perm_metrics = metrics(yte, model.predict(scaler.transform(shuffled.to_numpy(float))))
                individual_rows.append(
                    {"feature": feature, "fold": fold, "repeat": repeat, **{f"delta_{k}": base_metrics[k] - perm_metrics[k] for k in ["R2", "Pearson", "Spearman"]}, **{f"delta_{k}": perm_metrics[k] - base_metrics[k] for k in ["RMSE", "MAE"]}}
                )
        for fam, cols in families.items():
            for repeat in range(5):
                shuffled = test_fold[included].copy()
                order = rng.permutation(len(shuffled))
                for col in cols:
                    shuffled[col] = shuffled[col].to_numpy()[order]
                perm_metrics = metrics(yte, model.predict(scaler.transform(shuffled.to_numpy(float))))
                grouped_rows.append(
                    {"feature_family": fam, "fold": fold, "repeat": repeat, **{f"delta_{k}": base_metrics[k] - perm_metrics[k] for k in ["R2", "Pearson", "Spearman"]}, **{f"delta_{k}": perm_metrics[k] - base_metrics[k] for k in ["RMSE", "MAE"]}}
                )
        for fam, cols in families.items():
            keep = [c for c in included if c not in cols]
            pred_drop = train_predict(train_fold, test_fold, keep, ytr, MODEL_SEED + fold)
            drop_metrics = metrics(yte, pred_drop)
            ablation_rows.append(
                {"feature_family": fam, "fold": fold, **{f"delta_{k}": base_metrics[k] - drop_metrics[k] for k in ["R2", "Pearson", "Spearman"]}, **{f"delta_{k}": drop_metrics[k] - base_metrics[k] for k in ["RMSE", "MAE"]}}
            )
    perm = aggregate_delta(individual_rows, ["feature"]).merge(groups, on="feature", how="left")
    perm["rank_stability"] = pd.DataFrame(individual_rows).groupby("feature")["delta_R2"].rank(ascending=False).groupby(pd.DataFrame(individual_rows)["feature"]).std().reindex(perm["feature"]).to_numpy()
    grouped_perm = aggregate_delta(grouped_rows, ["feature_family"])
    group_abl = aggregate_delta(ablation_rows, ["feature_family"])
    save_csv("patch_permutation_importance_cv.csv", perm)
    save_csv("patch_grouped_permutation_importance_cv.csv", grouped_perm)
    save_csv("patch_group_ablation_cv.csv", group_abl)
    save_csv("patch_group_ablation_cv_fold_metrics.csv", pd.DataFrame(ablation_rows))

    def incremental_screen(base_features: list[str], out_name: str) -> pd.DataFrame:
        base_preds, base_folds = cv_predictions(merged, base_features, y)
        base_by_fold = {m["fold"]: m for m in base_folds}
        rows = []
        for feature in included:
            _, add_folds = cv_predictions(merged, base_features + [feature], y)
            for m in add_folds:
                b = base_by_fold[m["fold"]]
                rows.append(
                    {
                        "feature": feature,
                        "fold": m["fold"],
                        **{f"delta_{k}": m[k] - b[k] for k in ["R2", "Pearson", "Spearman"]},
                        **{f"delta_{k}": m[k] - b[k] for k in ["RMSE", "MAE"]},
                    }
                )
        res = aggregate_delta(rows, ["feature"]).merge(groups, on="feature", how="left")
        save_csv(out_name, res)
        return res

    inc_ocn = incremental_screen(["Opposite_Charge_Number"], "patch_incremental_over_OCN_cv.csv")
    inc_g10 = incremental_screen(GLOBAL10_REFERENCE, "patch_incremental_over_current_global10_cv.csv")

    shortlist = (
        inc_g10.sort_values(
            ["R2_improved_fraction", "RMSE_improved_fraction", "delta_R2_mean", "delta_RMSE_mean"],
            ascending=[False, False, False, True],
        )
        .head(8)["feature"]
        .tolist()
    )
    combo_rows = []
    selected_combo = []
    best_score = None
    for feature in shortlist:
        trial = selected_combo + [feature]
        _, folds = cv_predictions(merged, GLOBAL10_REFERENCE + trial, y)
        vals = pd.DataFrame(folds)
        row = {"panel": "+".join(trial), "n_patch_features": len(trial), "features": ",".join(trial)}
        for metric in ["R2", "RMSE", "MAE", "Pearson", "Spearman"]:
            row[f"{metric}_mean"] = vals[metric].mean()
            row[f"{metric}_sd"] = vals[metric].std(ddof=1)
        combo_rows.append(row)
        score = (row["R2_mean"], -row["RMSE_mean"], -row["MAE_mean"], row["Pearson_mean"], row["Spearman_mean"])
        if best_score is None or score > best_score:
            selected_combo = trial
            best_score = score
        else:
            break
    combos = pd.DataFrame(combo_rows)
    save_csv("top_patch_combinations_over_global10_cv.csv", combos)

    global_cols = [c for c in train.columns if np.issubdtype(train[c].dtype, np.number)]
    global_corr = []
    for f in included:
        corrs = [abs(pearsonr(merged[f], merged[g]).statistic) for g in global_cols if g != TARGET and np.std(merged[g]) > 0]
        global_corr.append({"feature": f, "strongest_global_correlation": max(corrs) if corrs else np.nan})
    selected = (
        groups.merge(univariate[["feature", "Pearson_r", "Spearman_rho"]], on="feature", how="left")
        .merge(perm[["feature", "delta_R2_mean"]].rename(columns={"delta_R2_mean": "patch_only_permutation_importance"}), on="feature", how="left")
        .merge(inc_ocn[["feature", "delta_R2_mean", "delta_RMSE_mean", "delta_MAE_mean", "delta_Pearson_mean", "delta_Spearman_mean"]].rename(columns=lambda c: c.replace("delta_", "incremental_over_OCN_") if c.startswith("delta_") else c), on="feature", how="left")
        .merge(inc_g10[["feature", "delta_R2_mean", "delta_RMSE_mean", "delta_MAE_mean", "delta_Pearson_mean", "delta_Spearman_mean"]].rename(columns=lambda c: c.replace("delta_", "incremental_over_GLOBAL10_") if c.startswith("delta_") else c), on="feature", how="left")
        .merge(pd.DataFrame(global_corr), on="feature", how="left")
    )
    selected["historical_definition"] = "stored get_features.py 5-aa overlapping-window patch descriptor"
    selected["grouped_family_importance"] = selected["feature_family"].map(grouped_perm.set_index("feature_family")["delta_R2_mean"])
    selected["drop_group_effect"] = selected["feature_family"].map(group_abl.set_index("feature_family")["delta_R2_mean"])
    selected["selected_for_final_ablation"] = selected["feature"].isin(selected_combo)
    selected["rationale"] = np.where(
        selected["selected_for_final_ablation"],
        "selected by training-only nested GLOBAL10 incremental screen before held-out B22 access",
        "not selected by training-only multi-metric incremental screen",
    )
    selected = selected.rename(columns={"Pearson_r": "Pearson_B22_training", "Spearman_rho": "Spearman_B22_training"})
    ordered_cols = [
        "feature",
        "feature_family",
        "historical_definition",
        "Pearson_B22_training",
        "Spearman_B22_training",
        "patch_only_permutation_importance",
        "grouped_family_importance",
        "drop_group_effect",
        "incremental_over_OCN_R2_mean",
        "incremental_over_OCN_RMSE_mean",
        "incremental_over_OCN_MAE_mean",
        "incremental_over_OCN_Pearson_mean",
        "incremental_over_OCN_Spearman_mean",
        "incremental_over_GLOBAL10_R2_mean",
        "incremental_over_GLOBAL10_RMSE_mean",
        "incremental_over_GLOBAL10_MAE_mean",
        "incremental_over_GLOBAL10_Pearson_mean",
        "incremental_over_GLOBAL10_Spearman_mean",
        "strongest_global_correlation",
        "selected_for_final_ablation",
        "rationale",
    ]
    save_csv("selected_patch_features_for_M4.csv", selected[ordered_cols])
    lock_path = OUT / "selected_patch_features_for_M4.csv"
    save_text("selected_patch_features_for_M4.lock.txt", f"timestamp_unix={time.time()}\nsha256={sha256(lock_path)}\nheldout_B22_used_before_lock=False\n")

    final_models = dict(GLOBAL_MODELS)
    final_models["M4"] = GLOBAL_MODELS["M2"] + selected_combo
    model_rows = []
    train_metrics = []
    held_metrics = []
    pred_train = train[["name", "rgg", "tad", TARGET]].copy()
    pred_held = held[["name", "rgg", "tad", TARGET]].copy()
    for name, feats in final_models.items():
        model_rows.extend({"model": name, "feature": f, "feature_order": i + 1} for i, f in enumerate(feats))
        p_train = train_predict(merged, merged, feats, y, MODEL_SEED)
        p_held = train_predict(merged, merged_held, feats, y, MODEL_SEED)
        pred_train[f"pred_{name}"] = p_train
        pred_held[f"pred_{name}"] = p_held
        train_metrics.append({"model": name, **metrics(train[TARGET].to_numpy(float), p_train)})
        held_metrics.append({"model": name, **metrics(held[TARGET].to_numpy(float), p_held)})
    save_csv("final_model_feature_sets.csv", pd.DataFrame(model_rows))
    save_csv("model_metrics_training.csv", pd.DataFrame(train_metrics))
    held_metrics_df = pd.DataFrame(held_metrics)
    save_csv("model_metrics_heldout.csv", held_metrics_df)
    save_csv("predictions_training.csv", pred_train)
    save_csv("predictions_heldout.csv", pred_held)

    rgg_ids = pd.Index(sorted(held["rgg"].unique()))
    tad_ids = pd.Index(sorted(held["tad"].unique()))

    def mat(values: np.ndarray) -> np.ndarray:
        tmp = pd.DataFrame({"rgg": held["rgg"], "tad": held["tad"], "v": values})
        return tmp.pivot(index="rgg", columns="tad", values="v").loc[rgg_ids, tad_ids].to_numpy(float)

    ymat = mat(held[TARGET].to_numpy(float))
    pmats = {m: mat(pred_held[f"pred_{m}"].to_numpy(float)) for m in final_models}
    boot_rows = []
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    for rep in range(N_BOOT):
        ri = rng.integers(0, len(rgg_ids), len(rgg_ids))
        ti = rng.integers(0, len(tad_ids), len(tad_ids))
        yy = ymat[np.ix_(ri, ti)].ravel()
        base = metrics(yy, pmats["M0"][np.ix_(ri, ti)].ravel())
        for model, mm in pmats.items():
            m = metrics(yy, mm[np.ix_(ri, ti)].ravel())
            boot_rows.append(
                {
                    "replicate": rep,
                    "model": model,
                    **m,
                    "delta_R2_vs_M0": m["R2"] - base["R2"],
                    "delta_RMSE_vs_M0": m["RMSE"] - base["RMSE"],
                    "delta_MAE_vs_M0": m["MAE"] - base["MAE"],
                }
            )
    boot = pd.DataFrame(boot_rows)
    save_csv("bootstrap_metrics.csv", boot)
    bs = []
    for model, sub in boot.groupby("model"):
        point = held_metrics_df.set_index("model").loc[model]
        base_point = held_metrics_df.set_index("model").loc["M0"]
        for metric in ["R2", "RMSE", "MAE", "Pearson", "Spearman", "delta_R2_vs_M0", "delta_RMSE_vs_M0", "delta_MAE_vs_M0"]:
            vals = sub[metric].to_numpy(float)
            point_value = point[metric] if metric in point else point[metric.split("_vs_")[0].replace("delta_", "")] - base_point[metric.split("_vs_")[0].replace("delta_", "")]
            mean, med, sd, lo, hi = mean_ci(vals)
            bs.append({"model": model, "metric": metric, "point_estimate": point_value, "bootstrap_mean": mean, "bootstrap_median": med, "bootstrap_sd": sd, "ci95_low": lo, "ci95_high": hi})
    save_csv("bootstrap_summary.csv", pd.DataFrame(bs))

    plt.rcParams.update({"pdf.fonttype": 42, "axes.spines.top": False, "axes.spines.right": False})
    fig, ax = plt.subplots(figsize=(7, 4.8))
    plot_models = [m for m in ["M0", "M1", "M2", "M4", "GLOBAL10_REFERENCE"] if m in final_models]
    vals = [held_metrics_df.set_index("model").loc[m, "R2"] for m in plot_models]
    ax.bar(plot_models, vals, color="#4C78A8")
    ax.axhline(0, color="black", lw=0.8)
    ax.set_ylabel("Held-out R2")
    ax.set_title("Figure 5 reviewer final ablation")
    fig.tight_layout()
    fig.savefig(OUT / "final_model_heldout_R2.png", dpi=240)
    fig.savefig(OUT / "final_model_heldout_R2.pdf")
    plt.close(fig)

    fig, axes = plt.subplots(1, min(4, len(plot_models)), figsize=(4 * min(4, len(plot_models)), 4), sharex=True, sharey=True)
    if len(plot_models) == 1:
        axes = [axes]
    for ax, model in zip(axes, plot_models[:4]):
        ax.scatter(held[TARGET], pred_held[f"pred_{model}"], s=7, alpha=0.3)
        lo = min(held[TARGET].min(), pred_held[f"pred_{model}"].min())
        hi = max(held[TARGET].max(), pred_held[f"pred_{model}"].max())
        ax.plot([lo, hi], [lo, hi], color="black", ls="--", lw=1)
        ax.set_title(model)
        ax.set_xlabel("Simulated")
        ax.set_ylabel("Predicted")
    fig.tight_layout()
    fig.savefig(OUT / "final_model_predicted_vs_simulated.png", dpi=240)
    fig.savefig(OUT / "final_model_predicted_vs_simulated.pdf")
    plt.close(fig)

    save_text(
        "input_provenance.txt",
        f"Python: {sys.executable}\nVersion: {sys.version}\nPlatform: {platform.platform()}\n"
        f"numpy={np.__version__}; pandas={pd.__version__}; scipy={scipy.__version__}; sklearn={sklearn.__version__}; matplotlib={matplotlib.__version__}\n"
        f"Inputs:\n" + "\n".join(f"{p}\tsha256={sha256(p)}" for p in input_files if p.exists()) + "\n"
        f"Held-out B22 access: only after selected_patch_features_for_M4.csv lock was written.\n",
    )
    save_text(
        "analysis_summary.txt",
        f"Final Figure 5 reviewer analysis completed on 917 training pairs and 1,794 sequence-exclusive held-out pairs.\n"
        f"Usable patch descriptors screened: {len(included)}.\n"
        f"Frozen final patch panel: {', '.join(selected_combo) if selected_combo else 'none selected'}.\n\n"
        f"Held-out metrics:\n{held_metrics_df.to_string(index=False)}\n\n"
        "Interpretation: OCN-only is the bulk electrostatics baseline. M1 tests non-redundant global electrostatics. "
        "M2 adds non-redundant global sequence descriptors. M4 adds only the patch panel frozen from training-only GLOBAL10 incremental screening. "
        "GLOBAL10_REFERENCE is retained as the current Figure 5 comparator and contains the known OCN/Normalized_OCN redundancy.\n",
    )
    save_text(
        "verification.txt",
        f"PASS: training rows={len(train)}; held-out rows={len(held)}.\n"
        "PASS: patch tables match one-to-one by name.\n"
        "PASS: no pair, TAD, or RGG overlap between training and held-out datasets.\n"
        f"PASS: selected patch panel locked before held-out B22 evaluation; lock sha256={sha256(lock_path)}.\n"
        f"PASS: all outputs written to {OUT}.\n",
    )
    print(f"completed {OUT}")


if __name__ == "__main__":
    main()
