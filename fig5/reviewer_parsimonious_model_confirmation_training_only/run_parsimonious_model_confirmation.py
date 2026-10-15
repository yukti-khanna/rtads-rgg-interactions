#!/usr/bin/env python
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


ROOT = Path(__file__).resolve().parents[1]
TASK2B = ROOT / "reviewer_feature_family_decomposition_training_only"
OUT = ROOT / "reviewer_parsimonious_model_confirmation_training_only"
PRED_FILE = TASK2B / "oof_predictions_all_repeats.csv"
METRIC_FILE = TASK2B / "cv_metrics_by_repeat.csv"
TARGET = "B22_corr1_log10"
REPEAT_SEEDS = [42, 43, 44, 45, 46]
BOOT_REPS = 2000
BOOT_SEED = 20260912

MODELS = {
    "F2": "F2_OCN_plus_hydropathy",
    "F5": "F5_OCN_plus_SCD_hydropathy",
    "F6": "F6_full_M4",
}

COMPARISONS = [
    ("F5_minus_F2", "F5", "F2"),
    ("F6_minus_F2", "F6", "F2"),
]


def metrics(y, pred):
    return {
        "pearson_r": float(stats.pearsonr(y, pred).statistic),
        "r2": float(r2_score(y, pred)),
        "rmse": float(mean_squared_error(y, pred) ** 0.5),
        "mae": float(mean_absolute_error(y, pred)),
    }


def summarize(long_df):
    rows = []
    for (comparison, metric), sub in long_df.groupby(["comparison", "metric"]):
        rows.append({
            "comparison": comparison,
            "metric": metric,
            "mean": sub["value"].mean(),
            "sd": sub["value"].std(ddof=1),
            "min": sub["value"].min(),
            "max": sub["value"].max(),
        })
    return pd.DataFrame(rows)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    pred = pd.read_csv(PRED_FILE)
    task2b_metrics = pd.read_csv(METRIC_FILE)
    y = pred[TARGET].to_numpy(float)

    paired_rows = []
    for seed in REPEAT_SEEDS:
        repeat_metrics = {}
        for short, full in MODELS.items():
            p = pred[f"pred_{full}_repeat_{seed}"].to_numpy(float)
            repeat_metrics[short] = metrics(y, p)
            paired_rows.append({
                "repeat_seed": seed,
                "comparison": f"{short}_model_metric",
                "model": short,
                **repeat_metrics[short],
            })
        for comp, high, low in COMPARISONS:
            mh = repeat_metrics[high]
            ml = repeat_metrics[low]
            paired_rows.append({
                "repeat_seed": seed,
                "comparison": comp,
                "model": f"{high}_minus_{low}",
                "delta_pearson_r": mh["pearson_r"] - ml["pearson_r"],
                "delta_r2": mh["r2"] - ml["r2"],
                "delta_rmse": mh["rmse"] - ml["rmse"],
                "delta_mae": mh["mae"] - ml["mae"],
            })
    paired = pd.DataFrame(paired_rows)
    paired.to_csv(OUT / "paired_metrics_by_repeat.csv", index=False)

    model_long = paired[paired["comparison"].str.endswith("_model_metric")].melt(
        id_vars=["repeat_seed", "comparison", "model"],
        value_vars=["pearson_r", "r2", "rmse", "mae"],
        var_name="metric",
        value_name="value",
    )
    diff_long = paired[paired["comparison"].isin([c[0] for c in COMPARISONS])].melt(
        id_vars=["repeat_seed", "comparison", "model"],
        value_vars=["delta_pearson_r", "delta_r2", "delta_rmse", "delta_mae"],
        var_name="metric",
        value_name="value",
    )
    model_summary = model_long.groupby(["model", "metric"], as_index=False).agg(
        mean=("value", "mean"),
        sd=("value", lambda x: x.std(ddof=1)),
        min=("value", "min"),
        max=("value", "max"),
    )
    diff_summary = summarize(diff_long)
    model_summary.to_csv(OUT / "model_metric_summary.csv", index=False)
    pd.concat([
        model_summary.rename(columns={"model": "comparison"}),
        diff_summary,
    ], ignore_index=True, sort=False).to_csv(OUT / "paired_metrics_summary.csv", index=False)

    boot = cluster_bootstrap(pred, y)
    boot[boot["comparison"] == "F5_minus_F2"].to_csv(OUT / "cluster_bootstrap_F5_vs_F2.csv", index=False)
    boot[boot["comparison"] == "F6_minus_F2"].to_csv(OUT / "cluster_bootstrap_F6_vs_F2.csv", index=False)
    boot_summary_rows = []
    for (comparison, metric), sub in boot.melt(
        id_vars=["comparison"], value_vars=["delta_r2", "delta_rmse", "delta_mae"],
        var_name="metric", value_name="value"
    ).groupby(["comparison", "metric"]):
        boot_summary_rows.append({
            "comparison": comparison,
            "metric": metric,
            "mean": sub["value"].mean(),
            "ci2.5": sub["value"].quantile(0.025),
            "ci50": sub["value"].quantile(0.5),
            "ci97.5": sub["value"].quantile(0.975),
            "n_bootstrap_repeat_estimates": len(sub),
        })
    boot_summary = pd.DataFrame(boot_summary_rows)
    boot_summary.to_csv(OUT / "cluster_bootstrap_summary.csv", index=False)

    make_figure(model_long, diff_long)
    write_text_outputs(pred, task2b_metrics, model_summary, diff_summary, boot_summary)


def cluster_bootstrap(pred, y):
    rng = np.random.default_rng(BOOT_SEED)
    rg_ids = pred["rgg"].drop_duplicates().to_numpy()
    tad_ids = pred["tad"].drop_duplicates().to_numpy()
    rows = []
    for rep in range(BOOT_REPS):
        rgg_counts = pd.Series(rng.choice(rg_ids, len(rg_ids), replace=True)).value_counts()
        tad_counts = pd.Series(rng.choice(tad_ids, len(tad_ids), replace=True)).value_counts()
        mult = pred["rgg"].map(rgg_counts).fillna(0).to_numpy(int) * pred["tad"].map(tad_counts).fillna(0).to_numpy(int)
        idx = np.repeat(np.arange(len(pred)), mult)
        if len(idx) < 3 or np.std(y[idx]) == 0:
            continue
        for seed in REPEAT_SEEDS:
            for comparison, high, low in COMPARISONS:
                ph = pred[f"pred_{MODELS[high]}_repeat_{seed}"].to_numpy(float)[idx]
                pl = pred[f"pred_{MODELS[low]}_repeat_{seed}"].to_numpy(float)[idx]
                mh = metrics(y[idx], ph)
                ml = metrics(y[idx], pl)
                rows.append({
                    "bootstrap_rep": rep,
                    "repeat_seed": seed,
                    "comparison": comparison,
                    "n_weighted_rows": len(idx),
                    "delta_r2": mh["r2"] - ml["r2"],
                    "delta_rmse": mh["rmse"] - ml["rmse"],
                    "delta_mae": mh["mae"] - ml["mae"],
                })
    return pd.DataFrame(rows)


def make_figure(model_long, diff_long):
    fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.9), gridspec_kw={"width_ratios": [1.35, 1.0]})
    ax = axes[0]
    model_order = ["F2", "F5", "F6"]
    for i, model in enumerate(model_order):
        vals = model_long[(model_long["model"] == model) & (model_long["metric"] == "r2")]["value"].to_numpy(float)
        ax.scatter(np.full(len(vals), i), vals, s=44, color="#2b6cb0", alpha=0.85)
        ax.plot([i - 0.22, i + 0.22], [vals.mean(), vals.mean()], color="black", lw=2)
    ax.set_xticks(range(len(model_order)))
    ax.set_xticklabels(model_order)
    ax.set_ylabel("Repeated-CV OOF R2")
    ax.set_title("Model performance")
    ax.grid(axis="y", alpha=0.25)

    ax = axes[1]
    comp_order = ["F5_minus_F2", "F6_minus_F2"]
    comp_labels = ["F5-F2", "F6-F2"]
    for i, comp in enumerate(comp_order):
        vals = diff_long[(diff_long["comparison"] == comp) & (diff_long["metric"] == "delta_r2")]["value"].to_numpy(float)
        ax.scatter(np.full(len(vals), i), vals, s=44, color="#805ad5", alpha=0.85)
        ax.plot([i - 0.22, i + 0.22], [vals.mean(), vals.mean()], color="black", lw=2)
    ax.axhline(0, color="black", lw=1)
    ax.set_xticks(range(len(comp_order)))
    ax.set_xticklabels(comp_labels)
    ax.set_ylabel("Paired delta OOF R2")
    ax.set_title("Added families")
    ax.grid(axis="y", alpha=0.25)
    fig.suptitle("Parsimonious model confirmation", y=1.02)
    fig.tight_layout()
    fig.savefig(OUT / "model_comparison.png", dpi=300, bbox_inches="tight")
    fig.savefig(OUT / "model_comparison.pdf", bbox_inches="tight")
    plt.close(fig)


def val(df, key_col, key, metric, field="mean"):
    row = df[(df[key_col] == key) & (df["metric"] == metric)].iloc[0]
    return float(row[field])


def write_text_outputs(pred, task2b_metrics, model_summary, diff_summary, boot_summary):
    task2b_match = []
    for seed in REPEAT_SEEDS:
        for short, full in MODELS.items():
            old = task2b_metrics[(task2b_metrics["repeat_seed"] == seed) & (task2b_metrics["model"] == full)].iloc[0]
            p = pred[f"pred_{full}_repeat_{seed}"].to_numpy(float)
            m = metrics(pred[TARGET].to_numpy(float), p)
            for metric in ["pearson_r", "r2", "rmse", "mae"]:
                task2b_match.append(abs(m[metric] - old[metric]))
    provenance = [
        "Parsimonious model confirmation for corrected Figure 5 training-only analysis.",
        f"OOF predictions used: {PRED_FILE}",
        f"CV metrics used for cross-check: {METRIC_FILE}",
        f"Rows in OOF prediction table: {len(pred)}",
        "No model fitting, fold generation, or fold assignment regeneration was performed.",
        "The 1,794-pair held-out matrix was not opened, read, hashed, inspected, or used.",
        "Unsimulated predictions were not used.",
    ]
    (OUT / "input_provenance.txt").write_text("\n".join(provenance) + "\n")

    consistency = pd.read_csv(OUT / "paired_metrics_by_repeat.csv")
    consistency = consistency[consistency["comparison"].isin(["F5_minus_F2", "F6_minus_F2"])].groupby("comparison").agg(
        positive_delta_r2_repeats=("delta_r2", lambda x: int((x > 0).sum())),
        lower_rmse_repeats=("delta_rmse", lambda x: int((x < 0).sum())),
        lower_mae_repeats=("delta_mae", lambda x: int((x < 0).sum())),
    ).reset_index()

    lines = [
        "Parsimonious model confirmation",
        "",
        f"Mean repeated-CV R2: F2 = {val(model_summary, 'model', 'F2', 'r2'):.4f}; F5 = {val(model_summary, 'model', 'F5', 'r2'):.4f}; F6 = {val(model_summary, 'model', 'F6', 'r2'):.4f}.",
        f"F5-F2 mean delta R2 = {val(diff_summary, 'comparison', 'F5_minus_F2', 'delta_r2'):.4f}; delta Pearson = {val(diff_summary, 'comparison', 'F5_minus_F2', 'delta_pearson_r'):.4f}; delta RMSE = {val(diff_summary, 'comparison', 'F5_minus_F2', 'delta_rmse'):.4f}; delta MAE = {val(diff_summary, 'comparison', 'F5_minus_F2', 'delta_mae'):.4f}.",
        f"F6-F2 mean delta R2 = {val(diff_summary, 'comparison', 'F6_minus_F2', 'delta_r2'):.4f}; delta Pearson = {val(diff_summary, 'comparison', 'F6_minus_F2', 'delta_pearson_r'):.4f}; delta RMSE = {val(diff_summary, 'comparison', 'F6_minus_F2', 'delta_rmse'):.4f}; delta MAE = {val(diff_summary, 'comparison', 'F6_minus_F2', 'delta_mae'):.4f}.",
        "",
        "Repeat consistency:",
        consistency.to_string(index=False),
        "",
        "Cluster-bootstrap 95% intervals:",
        boot_summary.to_string(index=False),
        "",
        "Interpretation:",
        "Adding SCD to F2 does not provide clearly resolved additional predictive value if the paired repeated-CV gain is small and the cluster-bootstrap interval includes no gain. Adding SCD plus explicit length to F2 should be judged the same way from the reported interval and repeat consistency. The parsimonious choice should not be based on the highest mean R2 alone. If the added-feature intervals include zero or the accuracy changes are small, F2 is a defensible parsimonious final predictor for revised Figure 5. These conclusions are internal to the 917-pair simulated panel and do not claim causality or external generalization.",
    ]
    (OUT / "analysis_summary.txt").write_text("\n".join(lines) + "\n")

    verification = [
        "Verification",
        "No model was retrained. This script imports no MLPRegressor and contains no model fitting code.",
        f"Only Task 2b OOF predictions were used for primary calculations: {PRED_FILE}",
        f"Task 2b CV metrics were used only to verify consistency: {METRIC_FILE}",
        f"Maximum absolute difference between recomputed metrics from OOF predictions and Task 2b metrics: {max(task2b_match):.3e}",
        "The 1,794-pair held-out file was never opened/read/hashed by this script.",
        "Unsimulated predictions were not used.",
        "No existing Figure 5 files were modified by this script except files written inside this new output directory.",
    ]
    (OUT / "verification.txt").write_text("\n".join(verification) + "\n")


if __name__ == "__main__":
    main()
