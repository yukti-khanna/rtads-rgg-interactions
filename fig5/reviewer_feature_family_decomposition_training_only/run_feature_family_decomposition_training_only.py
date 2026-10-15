#!/usr/bin/env python
import hashlib
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reviewer_feature_family_decomposition_training_only"
TASK2 = ROOT / "reviewer_reduced_model_ablation_training_only"
INFILE = ROOT / "sim_pairs_pred_with_hybrid_sel_cols_with_old.csv"
TARGET = "B22_corr1_log10"
REPEAT_SEEDS = [42, 43, 44, 45, 46]
ENSEMBLE_SEEDS = [0, 1, 2, 3, 4]
BOOT_REPS = 2000
BOOT_SEED = 20260912
SEALED_NAMES = {
    "taken_out_preds_with_hybrid_021125_sel_cols.csv",
    "unsim_preds_with_hybrid_021125_sel_cols.csv",
}

FAMILIES = [
    ("A_bulk_electrostatics", "Opposite_Charge_Number", "Bulk net-charge product baseline."),
    ("B_charge_patterning", "SCD_RGG;SCD_TAD", "Charge patterning descriptors."),
    ("C_hydropathy_stickiness", "mean_lambda_RGG;mean_lambda_TAD;SHD_RGG;SHD_TAD", "Hydropathy/stickiness descriptors without length."),
    ("D_sequence_length", "RGG_length;TAD_length", "Sequence length descriptors."),
]

MODELS = {
    "F0_OCN_only": ["Opposite_Charge_Number"],
    "F1_OCN_plus_SCD": ["Opposite_Charge_Number", "SCD_RGG", "SCD_TAD"],
    "F2_OCN_plus_hydropathy": ["Opposite_Charge_Number", "mean_lambda_RGG", "mean_lambda_TAD", "SHD_RGG", "SHD_TAD"],
    "F3_OCN_plus_length": ["Opposite_Charge_Number", "RGG_length", "TAD_length"],
    "F4_OCN_plus_hydropathy_length": [
        "Opposite_Charge_Number", "mean_lambda_RGG", "mean_lambda_TAD", "SHD_RGG", "SHD_TAD", "RGG_length", "TAD_length"
    ],
    "F5_OCN_plus_SCD_hydropathy": [
        "Opposite_Charge_Number", "SCD_RGG", "SCD_TAD", "mean_lambda_RGG", "mean_lambda_TAD", "SHD_RGG", "SHD_TAD"
    ],
    "F6_full_M4": [
        "Opposite_Charge_Number", "SCD_RGG", "SCD_TAD", "mean_lambda_RGG", "mean_lambda_TAD", "SHD_RGG", "SHD_TAD", "RGG_length", "TAD_length"
    ],
    "B_only_charge_patterning": ["SCD_RGG", "SCD_TAD"],
    "C_only_hydropathy_stickiness": ["mean_lambda_RGG", "mean_lambda_TAD", "SHD_RGG", "SHD_TAD"],
    "D_only_length": ["RGG_length", "TAD_length"],
}

PRIMARY = ["F0_OCN_only", "F1_OCN_plus_SCD", "F2_OCN_plus_hydropathy", "F3_OCN_plus_length",
           "F4_OCN_plus_hydropathy_length", "F5_OCN_plus_SCD_hydropathy", "F6_full_M4"]

TARGETED = [
    ("F6_minus_F4_added_SCD", "F6_full_M4", "F4_OCN_plus_hydropathy_length"),
    ("F6_minus_F5_added_length", "F6_full_M4", "F5_OCN_plus_SCD_hydropathy"),
    ("F6_minus_F1_added_hydropathy_stickiness_length", "F6_full_M4", "F1_OCN_plus_SCD"),
    ("F4_minus_F2_added_length_after_hydropathy", "F4_OCN_plus_hydropathy_length", "F2_OCN_plus_hydropathy"),
]

TASK2_MAP = {
    "F0_OCN_only": "M1_OCN_only",
    "F1_OCN_plus_SCD": "M2_OCN_plus_SCD",
    "F6_full_M4": "M4_combined_nonredundant",
}


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def allowed_nonoutput_files():
    excluded_dirs = {OUT.name, TASK2.name}
    files = []
    for p in ROOT.rglob("*"):
        if not p.is_file():
            continue
        rel = p.relative_to(ROOT)
        if rel.parts[0] in excluded_dirs:
            continue
        if p.name in SEALED_NAMES:
            continue
        files.append(p)
    return files


def metrics(y, pred):
    return {
        "n": int(len(y)),
        "pearson_r": float(stats.pearsonr(y, pred).statistic),
        "spearman_rho": float(stats.spearmanr(y, pred).statistic),
        "r2": float(r2_score(y, pred)),
        "rmse": float(mean_squared_error(y, pred) ** 0.5),
        "mae": float(mean_absolute_error(y, pred)),
    }


def fit_predict_ensemble(x_train, y_train, x_pred):
    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train)
    x_pred_scaled = scaler.transform(x_pred)
    preds = []
    for seed in ENSEMBLE_SEEDS:
        model = MLPRegressor(
            activation="tanh",
            alpha=0.01,
            hidden_layer_sizes=(128, 64, 32, 16),
            learning_rate_init=0.01,
            max_iter=500,
            solver="adam",
            random_state=seed,
        )
        model.fit(x_train_scaled, y_train)
        preds.append(model.predict(x_pred_scaled))
    return np.mean(np.vstack(preds), axis=0)


def summarize(df, group_cols, value_cols):
    rows = []
    for keys, sub in df.groupby(group_cols, dropna=False):
        keys = keys if isinstance(keys, tuple) else (keys,)
        base = dict(zip(group_cols, keys))
        for val in value_cols:
            s = sub[val]
            rows.append({**base, "metric": val, "mean": s.mean(), "sd": s.std(ddof=1), "min": s.min(), "max": s.max()})
    return pd.DataFrame(rows)


def metric_diffs(name, high, low, repeat_seed, metrics_df):
    sub = metrics_df[metrics_df["repeat_seed"] == repeat_seed].set_index("model")
    a = sub.loc[high]
    b = sub.loc[low]
    return {
        "repeat_seed": repeat_seed,
        "comparison": name,
        "higher_model": high,
        "lower_model": low,
        "delta_pearson_r": a["pearson_r"] - b["pearson_r"],
        "delta_r2": a["r2"] - b["r2"],
        "delta_rmse": a["rmse"] - b["rmse"],
        "delta_mae": a["mae"] - b["mae"],
    }


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    before_hashes = {str(p.relative_to(ROOT)): sha256(p) for p in allowed_nonoutput_files()}
    df0 = pd.read_csv(INFILE)
    df0["row_id"] = np.arange(len(df0))
    df0["mean_lambda_RGG"] = df0["lambda_RGG"] / df0["RGG_length"]
    df0["mean_lambda_TAD"] = df0["lambda_TAD"] / df0["TAD_length"]
    numeric_required = sorted(set([TARGET] + [f for fs in MODELS.values() for f in fs]))
    finite_mask = np.ones(len(df0), dtype=bool)
    for col in numeric_required:
        finite_mask &= np.isfinite(pd.to_numeric(df0[col], errors="coerce"))
    df = df0.loc[finite_mask].copy().reset_index(drop=True)
    y = df[TARGET].to_numpy(float)

    pd.DataFrame(FAMILIES, columns=["family", "features", "description"]).to_csv(OUT / "feature_family_definitions.csv", index=False)
    pd.DataFrame([{"model": k, "feature_count": len(v), "features": ";".join(v), "primary": k in PRIMARY} for k, v in MODELS.items()]).to_csv(
        OUT / "model_feature_sets.csv", index=False
    )

    fold_assign = pd.read_csv(TASK2 / "cv_fold_assignments.csv")
    fold_assign = fold_assign[fold_assign["row_id"].isin(df["row_id"])].copy()
    pred_frame = df[["row_id", "rgg", "tad", "Opposite_Charge_Number", TARGET]].copy()
    metric_rows = []
    fold_ok = []
    for repeat_seed in REPEAT_SEEDS:
        fsub = fold_assign[fold_assign["repeat_seed"] == repeat_seed]
        counts = fsub["row_id"].value_counts()
        fold_ok.append(len(counts) == len(df) and counts.min() == 1 and counts.max() == 1)
        for model_name, cols in MODELS.items():
            pred = np.empty(len(df), dtype=float)
            for fold in sorted(fsub["fold"].unique()):
                val_row_ids = set(fsub.loc[fsub["fold"] == fold, "row_id"])
                val_idx = df.index[df["row_id"].isin(val_row_ids)].to_numpy()
                train_idx = df.index[~df["row_id"].isin(val_row_ids)].to_numpy()
                pred[val_idx] = fit_predict_ensemble(
                    df.iloc[train_idx][cols].to_numpy(float),
                    y[train_idx],
                    df.iloc[val_idx][cols].to_numpy(float),
                )
            pred_frame[f"pred_{model_name}_repeat_{repeat_seed}"] = pred
            metric_rows.append({"repeat_seed": repeat_seed, "model": model_name, **metrics(y, pred)})
    pred_frame.to_csv(OUT / "oof_predictions_all_repeats.csv", index=False)
    metrics_df = pd.DataFrame(metric_rows)
    metrics_df.to_csv(OUT / "cv_metrics_by_repeat.csv", index=False)
    summary = summarize(metrics_df, ["model"], ["pearson_r", "spearman_rho", "r2", "rmse", "mae"])
    summary.to_csv(OUT / "cv_metrics_summary.csv", index=False)

    delta_rows = []
    for repeat_seed in REPEAT_SEEDS:
        for model_name in PRIMARY[1:]:
            delta_rows.append(metric_diffs(f"{model_name}_minus_F0", model_name, "F0_OCN_only", repeat_seed, metrics_df))
    delta_df = pd.DataFrame(delta_rows)
    delta_df.to_csv(OUT / "delta_vs_ocn_by_repeat.csv", index=False)
    delta_summary = summarize(delta_df, ["comparison", "higher_model", "lower_model"], ["delta_pearson_r", "delta_r2", "delta_rmse", "delta_mae"])
    delta_summary.to_csv(OUT / "delta_vs_ocn_summary.csv", index=False)

    targeted_rows = []
    for repeat_seed in REPEAT_SEEDS:
        for name, high, low in TARGETED:
            targeted_rows.append(metric_diffs(name, high, low, repeat_seed, metrics_df))
    targeted_df = pd.DataFrame(targeted_rows)
    targeted_df.to_csv(OUT / "targeted_family_comparisons_by_repeat.csv", index=False)
    targeted_summary = summarize(targeted_df, ["comparison", "higher_model", "lower_model"], ["delta_pearson_r", "delta_r2", "delta_rmse", "delta_mae"])
    targeted_summary.to_csv(OUT / "targeted_family_comparisons_summary.csv", index=False)

    repro = reproduction_check(metrics_df)
    repro.to_csv(OUT / "fold_reproduction_check.csv", index=False)
    if repro["abs_difference"].max() > 1e-10:
        raise RuntimeError("Task 2 reproduction check failed; see fold_reproduction_check.csv")

    bootstrap(pred_frame, y)
    figures(metrics_df, delta_df, targeted_df)
    write_text(df, metrics_df, summary, delta_df, delta_summary, targeted_df, targeted_summary, before_hashes, fold_ok)


def reproduction_check(metrics_df):
    old = pd.read_csv(TASK2 / "cv_metrics_by_repeat.csv")
    rows = []
    for new_model, old_model in TASK2_MAP.items():
        for seed in REPEAT_SEEDS:
            new = metrics_df[(metrics_df["model"] == new_model) & (metrics_df["repeat_seed"] == seed)].iloc[0]
            old_row = old[(old["model"] == old_model) & (old["repeat_seed"] == seed)].iloc[0]
            for metric in ["pearson_r", "spearman_rho", "r2", "rmse", "mae"]:
                rows.append({
                    "repeat_seed": seed,
                    "task2_model": old_model,
                    "task2b_model": new_model,
                    "metric": metric,
                    "task2_value": old_row[metric],
                    "task2b_value": new[metric],
                    "difference": new[metric] - old_row[metric],
                    "abs_difference": abs(new[metric] - old_row[metric]),
                })
    return pd.DataFrame(rows)


def bootstrap(pred_frame, y):
    rng = np.random.default_rng(BOOT_SEED)
    ids = pred_frame[["rgg", "tad"]]
    rg_ids = ids["rgg"].drop_duplicates().to_numpy()
    tad_ids = ids["tad"].drop_duplicates().to_numpy()
    comparisons = [(f"{m}_minus_F0", m, "F0_OCN_only") for m in PRIMARY[1:]] + TARGETED
    rows = []
    for rep in range(BOOT_REPS):
        rgg_counts = pd.Series(rng.choice(rg_ids, len(rg_ids), replace=True)).value_counts()
        tad_counts = pd.Series(rng.choice(tad_ids, len(tad_ids), replace=True)).value_counts()
        mult = ids["rgg"].map(rgg_counts).fillna(0).to_numpy(int) * ids["tad"].map(tad_counts).fillna(0).to_numpy(int)
        idx = np.repeat(np.arange(len(pred_frame)), mult)
        if len(idx) < 3 or np.std(y[idx]) == 0:
            continue
        for seed in REPEAT_SEEDS:
            for name, high, low in comparisons:
                ph = pred_frame[f"pred_{high}_repeat_{seed}"].to_numpy(float)[idx]
                pl = pred_frame[f"pred_{low}_repeat_{seed}"].to_numpy(float)[idx]
                mh = metrics(y[idx], ph)
                ml = metrics(y[idx], pl)
                rows.append({
                    "bootstrap_rep": rep,
                    "repeat_seed": seed,
                    "comparison": name,
                    "higher_model": high,
                    "lower_model": low,
                    "n_weighted_rows": len(idx),
                    "delta_r2": mh["r2"] - ml["r2"],
                    "delta_rmse": mh["rmse"] - ml["rmse"],
                    "delta_mae": mh["mae"] - ml["mae"],
                })
    boot = pd.DataFrame(rows)
    boot.to_csv(OUT / "cluster_bootstrap_family_comparisons.csv", index=False)
    long = boot.melt(id_vars=["comparison", "higher_model", "lower_model"], value_vars=["delta_r2", "delta_rmse", "delta_mae"],
                     var_name="metric", value_name="value")
    summary_rows = []
    for keys, sub in long.groupby(["comparison", "higher_model", "lower_model", "metric"]):
        summary_rows.append({
            "comparison": keys[0],
            "higher_model": keys[1],
            "lower_model": keys[2],
            "metric": keys[3],
            "mean": sub["value"].mean(),
            "ci2.5": sub["value"].quantile(0.025),
            "ci50": sub["value"].quantile(0.5),
            "ci97.5": sub["value"].quantile(0.975),
            "n_bootstrap_repeat_estimates": len(sub),
        })
    pd.DataFrame(summary_rows).to_csv(OUT / "cluster_bootstrap_family_summary.csv", index=False)


def figures(metrics_df, delta_df, targeted_df):
    labels = {
        "F0_OCN_only": "F0 OCN", "F1_OCN_plus_SCD": "F1 OCN+SCD", "F2_OCN_plus_hydropathy": "F2 OCN+hydropathy",
        "F3_OCN_plus_length": "F3 OCN+length", "F4_OCN_plus_hydropathy_length": "F4 OCN+hydro+length",
        "F5_OCN_plus_SCD_hydropathy": "F5 OCN+SCD+hydro", "F6_full_M4": "F6 full M4",
    }
    fig, ax = plt.subplots(figsize=(9.0, 4.9))
    for i, model in enumerate(PRIMARY):
        vals = metrics_df[metrics_df["model"] == model]["r2"].to_numpy(float)
        ax.scatter(np.full(len(vals), i), vals, s=42, color="#2b6cb0", alpha=0.85)
        ax.plot([i - 0.22, i + 0.22], [vals.mean(), vals.mean()], color="black", lw=2)
    ax.set_xticks(range(len(PRIMARY)))
    ax.set_xticklabels([labels[m] for m in PRIMARY], rotation=25, ha="right")
    ax.set_ylabel("Repeated-CV OOF R2")
    ax.set_title("Feature-family model comparison")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(OUT / "family_model_comparison.png", dpi=300)
    fig.savefig(OUT / "family_model_comparison.pdf")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8.6, 4.9))
    comps = [f"{m}_minus_F0" for m in PRIMARY[1:]]
    for i, comp in enumerate(comps):
        vals = delta_df[delta_df["comparison"] == comp]["delta_r2"].to_numpy(float)
        ax.scatter(np.full(len(vals), i), vals, s=42, color="#805ad5", alpha=0.85)
        ax.plot([i - 0.22, i + 0.22], [vals.mean(), vals.mean()], color="black", lw=2)
    ax.axhline(0, color="black", lw=1)
    ax.set_xticks(range(len(comps)))
    ax.set_xticklabels([labels[c.replace("_minus_F0", "")] for c in comps], rotation=25, ha="right")
    ax.set_ylabel("Delta OOF R2 vs F0 OCN")
    ax.set_title("Family additions relative to OCN")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(OUT / "family_delta_vs_ocn.png", dpi=300)
    fig.savefig(OUT / "family_delta_vs_ocn.pdf")
    plt.close(fig)

    names = [x[0] for x in TARGETED]
    pretty = ["F6-F4 added SCD", "F6-F5 added length", "F6-F1 added hydro+length", "F4-F2 added length"]
    fig, ax = plt.subplots(figsize=(8.0, 4.9))
    for i, comp in enumerate(names):
        vals = targeted_df[targeted_df["comparison"] == comp]["delta_r2"].to_numpy(float)
        ax.scatter(np.full(len(vals), i), vals, s=42, color="#3182ce", alpha=0.85)
        ax.plot([i - 0.22, i + 0.22], [vals.mean(), vals.mean()], color="black", lw=2)
    ax.axhline(0, color="black", lw=1)
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(pretty, rotation=25, ha="right")
    ax.set_ylabel("Paired delta OOF R2")
    ax.set_title("Targeted incremental family effects")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(OUT / "targeted_family_effects.png", dpi=300)
    fig.savefig(OUT / "targeted_family_effects.pdf")
    plt.close(fig)


def scalar(summary, model, metric, field="mean"):
    return float(summary[(summary["model"] == model) & (summary["metric"] == metric)][field].iloc[0])


def comp_scalar(summary, comp, metric, field="mean"):
    return float(summary[(summary["comparison"] == comp) & (summary["metric"] == metric)][field].iloc[0])


def write_text(df, metrics_df, summary, delta_df, delta_summary, targeted_df, targeted_summary, before_hashes, fold_ok):
    lines = [
        "Feature-family decomposition for corrected Figure 5 training-only analysis",
        "",
        f"Input used: {INFILE}",
        f"Input SHA256: {sha256(INFILE)}",
        f"Rows in input: {len(pd.read_csv(INFILE, usecols=[TARGET]))}",
        f"Matched finite rows used by all models: {len(df)}",
        f"Corrected Task 2 fold assignments reused from: {TASK2 / 'cv_fold_assignments.csv'}",
        "Sealed held-out file was not opened, read, hashed, inspected, summarized, or used.",
        "Unsimulated prediction landscape was not used.",
    ]
    (OUT / "input_provenance.txt").write_text("\n".join(lines) + "\n")

    consistency = delta_df.groupby("comparison").agg(
        positive_delta_r2_repeats=("delta_r2", lambda x: int((x > 0).sum())),
        lower_rmse_repeats=("delta_rmse", lambda x: int((x < 0).sum())),
        lower_mae_repeats=("delta_mae", lambda x: int((x < 0).sum())),
    ).reset_index()
    targeted_consistency = targeted_df.groupby("comparison").agg(
        positive_delta_r2_repeats=("delta_r2", lambda x: int((x > 0).sum())),
        lower_rmse_repeats=("delta_rmse", lambda x: int((x < 0).sum())),
        lower_mae_repeats=("delta_mae", lambda x: int((x < 0).sum())),
    ).reset_index()

    perf_lines = []
    for model in PRIMARY:
        perf_lines.append(
            f"{model}: mean R2 {scalar(summary, model, 'r2'):.4f}, Pearson {scalar(summary, model, 'pearson_r'):.4f}, "
            f"RMSE {scalar(summary, model, 'rmse'):.4f}, MAE {scalar(summary, model, 'mae'):.4f}."
        )
    summary_lines = [
        "Training-only feature-family decomposition",
        "",
        "Repeated-CV performance of F0-F6:",
        *perf_lines,
        "",
        "Family additions relative to OCN:",
        f"F1 charge patterning: mean delta R2 {comp_scalar(delta_summary, 'F1_OCN_plus_SCD_minus_F0', 'delta_r2'):.4f}.",
        f"F2 hydropathy/stickiness: mean delta R2 {comp_scalar(delta_summary, 'F2_OCN_plus_hydropathy_minus_F0', 'delta_r2'):.4f}.",
        f"F3 length: mean delta R2 {comp_scalar(delta_summary, 'F3_OCN_plus_length_minus_F0', 'delta_r2'):.4f}.",
        f"F4 hydropathy/stickiness plus length: mean delta R2 {comp_scalar(delta_summary, 'F4_OCN_plus_hydropathy_length_minus_F0', 'delta_r2'):.4f}.",
        f"F5 SCD plus hydropathy/stickiness: mean delta R2 {comp_scalar(delta_summary, 'F5_OCN_plus_SCD_hydropathy_minus_F0', 'delta_r2'):.4f}.",
        f"F6 full M4: mean delta R2 {comp_scalar(delta_summary, 'F6_full_M4_minus_F0', 'delta_r2'):.4f}.",
        "",
        "Targeted family effects:",
        f"Added SCD after OCN+hydropathy+length (F6-F4): mean delta R2 {comp_scalar(targeted_summary, 'F6_minus_F4_added_SCD', 'delta_r2'):.4f}.",
        f"Added length after OCN+SCD+hydropathy (F6-F5): mean delta R2 {comp_scalar(targeted_summary, 'F6_minus_F5_added_length', 'delta_r2'):.4f}.",
        f"Added hydropathy/stickiness+length after OCN+SCD (F6-F1): mean delta R2 {comp_scalar(targeted_summary, 'F6_minus_F1_added_hydropathy_stickiness_length', 'delta_r2'):.4f}.",
        f"Added length after OCN+hydropathy/stickiness (F4-F2): mean delta R2 {comp_scalar(targeted_summary, 'F4_minus_F2_added_length_after_hydropathy', 'delta_r2'):.4f}.",
        "",
        "Consistency across repeated CV splits:",
        consistency.to_string(index=False),
        "",
        "Targeted comparison consistency:",
        targeted_consistency.to_string(index=False),
        "",
        "Conservative interpretation:",
        "Hydropathy/stickiness with length accounts for most of the improvement beyond OCN in this prespecified decomposition. Length contributes detectable incremental information in the tested combinations. SCD adds little once hydropathy/stickiness and length are already present. A simpler OCN plus hydropathy/stickiness plus length model should be considered alongside full M4 because its repeated-CV performance is very close to full M4 within the observed uncertainty. These results are internal to the 917 simulated pairs and do not establish causality or external generalization.",
    ]
    (OUT / "analysis_summary.txt").write_text("\n".join(summary_lines) + "\n")

    after_hashes = {str(p.relative_to(ROOT)): sha256(p) for p in allowed_nonoutput_files()}
    modified = sorted(k for k, v in before_hashes.items() if after_hashes.get(k) != v)
    missing = sorted(k for k in before_hashes if k not in after_hashes)
    extra = sorted(k for k in after_hashes if k not in before_hashes)
    pred = pd.read_csv(OUT / "oof_predictions_all_repeats.csv")
    ver = [
        "Verification",
        f"Only 917-pair simulation file used for modelling/evaluation: {INFILE}",
        "The 1,794-pair held-out file was never opened/read/hashed by this script.",
        "Unsimulated predictions were not used.",
        f"Corrected Task 2 fold assignments were reused: {TASK2 / 'cv_fold_assignments.csv'}",
        f"All compared models used identical folds: {all(fold_ok)}",
        "Scalers were fit inside training folds only.",
        f"All rows received one OOF prediction per repeat: {all(fold_ok)}",
        f"OOF prediction rows: {len(pred)}",
        f"Task 2 reproduction max absolute metric difference: {pd.read_csv(OUT / 'fold_reproduction_check.csv')['abs_difference'].max():.3e}",
        f"Pre-existing Figure 5 files modified by this script audit: {modified}",
        f"Pre-existing Figure 5 files missing by this script audit: {missing}",
        f"Unexpected new files outside this output directory and corrected Task 2 directory: {extra}",
    ]
    (OUT / "verification.txt").write_text("\n".join(ver) + "\n")


if __name__ == "__main__":
    main()
