#!/usr/bin/env python3
"""Build corrected standalone Supplementary Figure S5 panels A-D.

This script creates the final publication-facing S5 only. It presents one hybrid
modelling framework with a base component and a full hybrid model. It does not
compare old vs new models, does not show development-diagnostic model labels in
visible figure text, and does not access held-out B22 targets.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import math

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUTDIR = ROOT / "outputs" / "revised_fig5_supplement"
OUTDIR.mkdir(parents=True, exist_ok=True)

HIST_FIG5 = ROOT / "fig5.py"
HIST_FIGS5 = ROOT / "make_FigS5.py"

LINEAGE_DIR = ROOT / "fig5_residual_tuning_local" / "outputs"
OOF_PATH = LINEAGE_DIR / "outer_oof_predictions.csv"
BOOTSTRAP_PATH = LINEAGE_DIR / "tail_bootstrap_differences.csv"
OUTER_FOLDS_PATH = LINEAGE_DIR / "outer_fold_results.csv"
TAIL_METRICS_PATH = LINEAGE_DIR / "tail_metrics.csv"
IMPROVEMENT_PATH = LINEAGE_DIR / "high_OCN_pair_improvement_summary.csv"
INNER_TUNING_PATH = LINEAGE_DIR / "inner_tuning_results.csv"

FINAL_LABEL = "historical_fixed_nn"
HIGH_THRESHOLD = 40.0
EXPECTED_N_PAIRS = 917
EXPECTED_N_HIGH = 93
EXPECTED_N_FOLDS = 25
EXPECTED_DELTA_RMSE = -0.00133
EXPECTED_CI_LOW = -0.00261
EXPECTED_CI_HIGH = -0.00004
EXPECTED_DELTA_R2 = 0.00408
EXPECTED_DELTA_BIAS = -0.00159

FIGSIZE = (2.4, 2.3)
DPI_PNG = 600
COLOR_BLUE = "#0072B2"
COLOR_GREY = "#656565"
COLOR_DARK = "black"
TREND_COLOR = "#D55E00"

mpl.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "DejaVu Sans"],
    "font.size": 6.5,
    "axes.labelsize": 7,
    "axes.titlesize": 7,
    "xtick.labelsize": 6,
    "ytick.labelsize": 6,
    "legend.fontsize": 6,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "svg.fonttype": "none",
    "axes.linewidth": 0.6,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "xtick.major.size": 2.5,
    "ytick.major.size": 2.5,
    "xtick.direction": "out",
    "ytick.direction": "out",
})


@dataclass
class Sources:
    oof: Path
    bootstrap: Path
    outer_folds: Path
    tail_metrics: Path
    improvement: Path | None
    inner_tuning: Path | None
    selection_reason: str
    delta_rmse: float
    delta_rmse_ci_low: float
    delta_rmse_ci_high: float
    delta_r2: float
    delta_bias: float


def fail(message: str) -> None:
    raise RuntimeError(message)


def read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        fail(f"Missing required source file: {path}")
    return pd.read_csv(path)


def clean_figure_axes(ax) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(length=2.5, width=0.6, pad=1.5)


def save_fig(fig, stem: str) -> list[str]:
    paths = []
    for ext in ("pdf", "svg", "png"):
        path = OUTDIR / f"{stem}.{ext}"
        kwargs = {"bbox_inches": "tight"}
        if ext == "png":
            kwargs["dpi"] = DPI_PNG
        fig.savefig(path, **kwargs)
        paths.append(str(path))
    plt.close(fig)
    return paths


def close_to(value: float, expected: float, tol: float) -> bool:
    return math.isfinite(value) and abs(value - expected) <= tol


def metric_row(df: pd.DataFrame, metric: str) -> pd.Series:
    rows = df[(df["residual_model_label"].eq(FINAL_LABEL)) & (df["metric"].eq(metric))]
    if len(rows) != 1:
        fail(f"Expected one {FINAL_LABEL}/{metric} row in bootstrap table, found {len(rows)}")
    return rows.iloc[0]


def candidate_csvs() -> list[Path]:
    keywords = ("residual", "hybrid", "tail", "tuning", "lambda", "calibration")
    files = []
    for base in [ROOT / "outputs", ROOT / "fig5_residual_tuning_local", ROOT / "reviewer_hybrid_residual_ablation_F2_training_only"]:
        if base.exists():
            files.extend(p for p in base.rglob("*.csv") if any(k in str(p).lower() for k in keywords))
    return sorted(set(files))


def summarize_candidates() -> pd.DataFrame:
    rows = []
    for path in candidate_csvs():
        rec: dict[str, object] = {"path": str(path)}
        try:
            df = pd.read_csv(path)
            rec["n_rows"] = len(df)
            rec["columns"] = ";".join(map(str, df.columns[:25]))
            if "residual_model_label" in df.columns:
                rec["residual_model_labels"] = ";".join(sorted(map(str, pd.unique(df["residual_model_label"].dropna()))))
            if "model" in df.columns:
                rec["legacy_model_labels"] = ";".join(sorted(map(str, pd.unique(df["model"].dropna()))))
            if {"residual_model_label", "metric", "observed_difference"}.issubset(df.columns):
                for metric in ["delta_RMSE", "delta_R2", "delta_bias"]:
                    hit = df[(df["residual_model_label"].eq(FINAL_LABEL)) & (df["metric"].eq(metric))]
                    if len(hit) == 1:
                        rec[f"{FINAL_LABEL}_{metric}"] = float(hit["observed_difference"].iloc[0])
            if {"residual_model_label", "outer_seed", "outer_fold"}.issubset(df.columns):
                counts = df.groupby("residual_model_label").size().to_dict()
                rec["rows_by_residual_label"] = ";".join(f"{k}:{v}" for k, v in sorted(counts.items()))
        except Exception as exc:
            rec["error"] = str(exc)
        rows.append(rec)
    out = pd.DataFrame(rows)
    out.to_csv(OUTDIR / "S5_residual_analysis_candidates.csv", index=False)
    return out


def identify_sources() -> Sources:
    summarize_candidates()
    bootstrap = read_csv(BOOTSTRAP_PATH)
    folds = read_csv(OUTER_FOLDS_PATH)
    oof = read_csv(OOF_PATH)
    tail = read_csv(TAIL_METRICS_PATH)

    required_labels = {FINAL_LABEL}
    if not required_labels.issubset(set(bootstrap.get("residual_model_label", []))):
        fail("Authoritative bootstrap file lacks final hybrid residual lineage")
    if FINAL_LABEL not in set(folds.get("residual_model_label", [])):
        fail("Authoritative outer-fold file lacks final hybrid residual lineage")
    if FINAL_LABEL not in set(oof.get("residual_model_label", [])):
        fail("Authoritative OOF file lacks final hybrid residual lineage")
    if FINAL_LABEL not in set(tail.get("residual_model_label", [])):
        fail("Authoritative tail-metrics file lacks final hybrid residual lineage")

    rmse = metric_row(bootstrap, "delta_RMSE")
    r2 = metric_row(bootstrap, "delta_R2")
    bias = metric_row(bootstrap, "delta_bias")
    delta_rmse = float(rmse["observed_difference"])
    ci_low = float(rmse["ci_2_5"])
    ci_high = float(rmse["ci_97_5"])
    delta_r2 = float(r2["observed_difference"])
    delta_bias = float(bias["observed_difference"])

    checks = [
        close_to(delta_rmse, EXPECTED_DELTA_RMSE, 5e-4),
        close_to(ci_low, EXPECTED_CI_LOW, 5e-4),
        close_to(ci_high, EXPECTED_CI_HIGH, 5e-4),
        close_to(delta_r2, EXPECTED_DELTA_R2, 1e-3),
        close_to(delta_bias, EXPECTED_DELTA_BIAS, 7e-4),
    ]
    if not all(checks):
        fail(
            "Could not verify final hybrid residual lineage. "
            f"Observed ΔRMSE={delta_rmse:.6f}, CI=[{ci_low:.6f}, {ci_high:.6f}], "
            f"ΔR²={delta_r2:.6f}, Δbias={delta_bias:.6f}."
        )

    final_folds = folds[folds["residual_model_label"].eq(FINAL_LABEL)]
    if len(final_folds) != EXPECTED_N_FOLDS:
        fail(f"Expected {EXPECTED_N_FOLDS} final hybrid outer folds, found {len(final_folds)}")

    reason = (
        f"Selected {LINEAGE_DIR} because {BOOTSTRAP_PATH.name} reproduces the expected final "
        f"high−NCP result: ΔRMSE={delta_rmse:.6f}, 95% CI=[{ci_low:.6f}, {ci_high:.6f}], "
        f"ΔR²={delta_r2:.6f}, Δbias={delta_bias:.6f}; and {OUTER_FOLDS_PATH.name} contains "
        f"{EXPECTED_N_FOLDS} matching final hybrid outer folds."
    )
    return Sources(OOF_PATH, BOOTSTRAP_PATH, OUTER_FOLDS_PATH, TAIL_METRICS_PATH, IMPROVEMENT_PATH if IMPROVEMENT_PATH.exists() else None, INNER_TUNING_PATH if INNER_TUNING_PATH.exists() else None, reason, delta_rmse, ci_low, ci_high, delta_r2, delta_bias)


def final_oof_unique(sources: Sources) -> pd.DataFrame:
    df = read_csv(sources.oof)
    df = df[df["residual_model_label"].eq(FINAL_LABEL)].copy()
    required = {"name", "true_target", "OCN", "base_prediction", "hybrid_prediction", "applied_correction"}
    missing = required - set(df.columns)
    if missing:
        fail(f"OOF table is missing required columns: {sorted(missing)}")
    df["computed_correction"] = df["hybrid_prediction"] - df["base_prediction"]
    max_identity_error = float(np.max(np.abs(df["computed_correction"] - df["applied_correction"])))
    if max_identity_error > 1e-10:
        fail(f"hybrid prediction = base prediction + applied correction identity failed; max error {max_identity_error}")
    df["base_residual"] = df["true_target"] - df["base_prediction"]
    df["hybrid_residual"] = df["true_target"] - df["hybrid_prediction"]
    grouped = df.groupby("name", as_index=False).agg(
        simulated_target=("true_target", "first"),
        minus_NCP=("OCN", "first"),
        base_prediction=("base_prediction", "mean"),
        full_hybrid_prediction=("hybrid_prediction", "mean"),
        base_residual=("base_residual", "mean"),
        full_hybrid_residual=("hybrid_residual", "mean"),
        applied_residual_correction=("applied_correction", "mean"),
        n_oof_predictions=("name", "size"),
    )
    grouped["high_minus_NCP"] = grouped["minus_NCP"] >= HIGH_THRESHOLD
    if len(grouped) != EXPECTED_N_PAIRS:
        fail(f"Expected {EXPECTED_N_PAIRS} unique development pairs, found {len(grouped)}")
    n_high = int(grouped["high_minus_NCP"].sum())
    if n_high != EXPECTED_N_HIGH:
        fail(f"Expected {EXPECTED_N_HIGH} high−NCP pairs, found {n_high}")
    grouped["correction_identity_max_abs_error_fold_level"] = max_identity_error
    return grouped


def make_bins(df: pd.DataFrame) -> tuple[pd.Series, pd.DataFrame]:
    low = df[df["minus_NCP"] < HIGH_THRESHOLD].copy()
    high = df[df["minus_NCP"] >= HIGH_THRESHOLD].copy()
    low_edges = sorted(set(np.quantile(low["minus_NCP"], [0.0, 0.2, 0.4, 0.6, 0.8, 1.0])))
    if len(low_edges) < 2:
        fail("Unable to create low−NCP quantile bins")
    high_edges = [HIGH_THRESHOLD, 60.0, 90.0, float(high["minus_NCP"].max()) + 1e-9]
    edges = sorted(set([float(x) for x in low_edges if x < HIGH_THRESHOLD] + high_edges))
    if edges[0] > float(df["minus_NCP"].min()):
        edges.insert(0, float(df["minus_NCP"].min()))
    bins = pd.cut(df["minus_NCP"], bins=edges, include_lowest=True, duplicates="drop")
    if bins.isna().any():
        fail("Some pairs were not assigned to −NCP bins")
    bin_table = pd.DataFrame({"bin_assignment": bins.astype(str), "minus_NCP": df["minus_NCP"]})
    return bins.astype(str), bin_table


def bootstrap_ci(values: np.ndarray, stat_func=np.mean, n_resamples: int = 5000, seed: int = 2026) -> tuple[float, float, float]:
    values = np.asarray(values, dtype=float)
    if len(values) == 0:
        return np.nan, np.nan, np.nan
    obs = float(stat_func(values))
    if len(values) == 1:
        return obs, obs, obs
    rng = np.random.default_rng(seed)
    stats = np.empty(n_resamples, dtype=float)
    n = len(values)
    for i in range(n_resamples):
        stats[i] = stat_func(values[rng.integers(0, n, n)])
    return obs, float(np.percentile(stats, 2.5)), float(np.percentile(stats, 97.5))


def binned_summary(df: pd.DataFrame, value_col: str, bin_col: str) -> pd.DataFrame:
    rows = []
    for b, part in df.groupby(bin_col, observed=False):
        mean, lo, hi = bootstrap_ci(part[value_col].to_numpy(), np.mean)
        rows.append({
            "bin_assignment": str(b),
            "bin_center_minus_NCP": float(part["minus_NCP"].median()),
            "bin_min_minus_NCP": float(part["minus_NCP"].min()),
            "bin_max_minus_NCP": float(part["minus_NCP"].max()),
            "bin_n": int(len(part)),
            f"{value_col}_bin_mean": mean,
            f"{value_col}_bin_ci_2_5": lo,
            f"{value_col}_bin_ci_97_5": hi,
        })
    return pd.DataFrame(rows).sort_values("bin_center_minus_NCP")


def attach_bin_summary(df: pd.DataFrame, summary: pd.DataFrame, value_col: str) -> pd.DataFrame:
    cols = ["bin_assignment", "bin_center_minus_NCP", "bin_min_minus_NCP", "bin_max_minus_NCP", "bin_n", f"{value_col}_bin_mean", f"{value_col}_bin_ci_2_5", f"{value_col}_bin_ci_97_5"]
    return df.merge(summary[cols], on="bin_assignment", how="left", validate="many_to_one")


def plot_residual_panel(df: pd.DataFrame, summary: pd.DataFrame, value_col: str, ylabel: str, stem: str, xlim: tuple[float, float], ylim: tuple[float, float]) -> list[str]:
    fig, ax = plt.subplots(figsize=FIGSIZE)
    low = df[~df["high_minus_NCP"]]
    high = df[df["high_minus_NCP"]]
    ax.scatter(low["minus_NCP"], low[value_col], s=7, alpha=0.50, color=COLOR_GREY, edgecolors="none", zorder=1, label="−NCP < 40")
    ax.scatter(high["minus_NCP"], high[value_col], s=7, alpha=0.80, color=COLOR_BLUE, edgecolors="none", zorder=2, label="−NCP ≥ 40")
    ax.plot(
        summary["bin_center_minus_NCP"],
        summary[f"{value_col}_bin_mean"],
        "o-",
        markersize=2.5,
        lw=0.8,
        color=TREND_COLOR,
        zorder=4
    )    
    ax.axhline(0, lw=0.8, color=COLOR_DARK, zorder=3)
    ax.axvline(HIGH_THRESHOLD, lw=0.8, color=COLOR_DARK, ls="--", zorder=3)
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_xlabel("−NCP")
    ax.set_ylabel(ylabel)
    ax.legend(loc="upper left", frameon=False, handletextpad=0.2, borderpad=0.1)
    clean_figure_axes(ax)
    return save_fig(fig, stem)


def residual_plot_limits(df: pd.DataFrame) -> tuple[tuple[float, float], tuple[float, float]]:
    xmin, xmax = float(df["minus_NCP"].min()), float(df["minus_NCP"].max())
    xpad = 0.04 * (xmax - xmin)
    vals = pd.concat([df["base_residual"], df["full_hybrid_residual"]]).to_numpy(dtype=float)
    ymin, ymax = float(np.nanmin(vals)), float(np.nanmax(vals))
    ypad = 0.08 * (ymax - ymin)
    return (xmin - xpad, xmax + xpad), (ymin - ypad, ymax + ypad)


def make_s5a_b(sources: Sources, panel_info: dict[str, object]) -> pd.DataFrame:
    df = final_oof_unique(sources)
    df["bin_assignment"], _ = make_bins(df)
    base_summary = binned_summary(df, "base_residual", "bin_assignment")
    hybrid_summary = binned_summary(df, "full_hybrid_residual", "bin_assignment")
    xlim, ylim = residual_plot_limits(df)

    s5a = df[["name", "minus_NCP", "base_prediction", "simulated_target", "base_residual", "high_minus_NCP", "bin_assignment", "n_oof_predictions"]].rename(columns={"name": "pair_id"})
    s5a = attach_bin_summary(s5a, base_summary, "base_residual")
    s5a.to_csv(OUTDIR / "FigS5A_source.csv", index=False)
    paths_a = plot_residual_panel(df, base_summary, "base_residual", "Base-component OOF residual", "FigS5A_base_residual_vs_minusNCP", xlim, ylim)

    s5b = df[["name", "minus_NCP", "full_hybrid_prediction", "simulated_target", "full_hybrid_residual", "high_minus_NCP", "bin_assignment", "n_oof_predictions"]].rename(columns={"name": "pair_id"})
    s5b = attach_bin_summary(s5b, hybrid_summary, "full_hybrid_residual")
    s5b.to_csv(OUTDIR / "FigS5B_source.csv", index=False)
    paths_b = plot_residual_panel(df, hybrid_summary, "full_hybrid_residual", "Full-hybrid OOF residual", "FigS5B_hybrid_residual_vs_minusNCP", xlim, ylim)

    panel_info["S5A"] = {"paths": paths_a, "source_csv": str(OUTDIR / "FigS5A_source.csv"), "unique_pairs": len(df), "high_minus_NCP_pairs": int(df["high_minus_NCP"].sum()), "residual_definition": "simulated target − base-component prediction", "x_limits": xlim, "y_limits": ylim}
    panel_info["S5B"] = {"paths": paths_b, "source_csv": str(OUTDIR / "FigS5B_source.csv"), "unique_pairs": len(df), "high_minus_NCP_pairs": int(df["high_minus_NCP"].sum()), "residual_definition": "simulated target − full-hybrid prediction", "x_limits": xlim, "y_limits": ylim}
    return df


def make_s5c(sources: Sources, panel_info: dict[str, object]) -> None:
    folds = read_csv(sources.outer_folds)
    final = folds[folds["residual_model_label"].eq(FINAL_LABEL)].copy()
    if len(final) != EXPECTED_N_FOLDS:
        fail(f"S5C expected {EXPECTED_N_FOLDS} final hybrid folds, found {len(final)}")
    final = final.sort_values(["outer_seed", "outer_fold"]).reset_index(drop=True)
    final["outer_fold_index"] = np.arange(1, len(final) + 1)
    final["base_high_minus_NCP_RMSE"] = final["base_tail_RMSE"].astype(float)
    final["full_hybrid_high_minus_NCP_RMSE"] = final["hybrid_tail_RMSE"].astype(float)
    final["paired_delta_RMSE"] = final["full_hybrid_high_minus_NCP_RMSE"] - final["base_high_minus_NCP_RMSE"]
    observed = sources.delta_rmse
    if not close_to(observed, EXPECTED_DELTA_RMSE, 5e-4):
        fail(f"S5C expected ΔRMSE near {EXPECTED_DELTA_RMSE}, found {observed}")
    out = final[["outer_seed", "outer_fold", "outer_fold_index", "base_high_minus_NCP_RMSE", "full_hybrid_high_minus_NCP_RMSE", "paired_delta_RMSE", "n_high_validation", "lambda_hat", "cap"]].copy()
    out["publication_component_1"] = "Base component"
    out["publication_component_2"] = "Full hybrid"
    out["threshold"] = "−NCP >= 40"
    out.to_csv(OUTDIR / "FigS5C_source.csv", index=False)

    fig, ax = plt.subplots(figsize=FIGSIZE)
    x_base, x_hybrid = 1.0, 2.0
    for _, row in out.iterrows():
        ax.plot([x_base, x_hybrid], [row["base_high_minus_NCP_RMSE"], row["full_hybrid_high_minus_NCP_RMSE"]], lw=0.45, color=COLOR_GREY, alpha=0.45, zorder=1)
    ax.scatter(np.full(len(out), x_base), out["base_high_minus_NCP_RMSE"], s=10, color=COLOR_GREY, alpha=0.65, edgecolors="none", zorder=2)
    ax.scatter(np.full(len(out), x_hybrid), out["full_hybrid_high_minus_NCP_RMSE"], s=10, color=COLOR_BLUE, alpha=0.85, edgecolors="none", zorder=3)
    for x, col, color in [(x_base, "base_high_minus_NCP_RMSE", COLOR_DARK), (x_hybrid, "full_hybrid_high_minus_NCP_RMSE", COLOR_DARK)]:
        mean, lo, hi = bootstrap_ci(out[col].to_numpy(), np.mean)
        ax.errorbar([x], [mean], yerr=[[mean - lo], [hi - mean]], fmt="_", markersize=12, lw=0.8, capsize=2, color=color, zorder=4)
    ax.text(0.98, 0.05, f"ΔRMSE = {observed:.4f}\n95% CI [{sources.delta_rmse_ci_low:.4f}, {sources.delta_rmse_ci_high:.4f}]", transform=ax.transAxes, ha="right", va="bottom", fontsize=6)
    ax.set_xlim(0.6, 2.4)
    ax.set_xticks([x_base, x_hybrid], ["Base\ncomponent", "Full\nhybrid"])
    ax.set_ylabel("High−NCP RMSE")
    clean_figure_axes(ax)
    paths = save_fig(fig, "FigS5C_high_minusNCP_RMSE_ablation")
    panel_info["S5C"] = {"paths": paths, "source_csv": str(OUTDIR / "FigS5C_source.csv"), "fold_count": len(out), "observed_delta_RMSE": observed, "ci_2_5": sources.delta_rmse_ci_low, "ci_97_5": sources.delta_rmse_ci_high, "visible_labels": "Base component; Full hybrid"}


def make_s5d(df: pd.DataFrame, panel_info: dict[str, object]) -> None:
    identity_error = float(np.max(np.abs((df["full_hybrid_prediction"] - df["base_prediction"]) - df["applied_residual_correction"])))
    if identity_error > 1e-10:
        fail(f"S5D correction identity failed, max error {identity_error}")
    corr_summary = binned_summary(df, "applied_residual_correction", "bin_assignment")
    src = df[["name", "minus_NCP", "base_prediction", "full_hybrid_prediction", "applied_residual_correction", "high_minus_NCP", "bin_assignment", "correction_identity_max_abs_error_fold_level"]].rename(columns={"name": "pair_id"})
    src = attach_bin_summary(src, corr_summary, "applied_residual_correction")
    src.to_csv(OUTDIR / "FigS5D_source.csv", index=False)

    fig, ax = plt.subplots(figsize=FIGSIZE)
    low = df[~df["high_minus_NCP"]]
    high = df[df["high_minus_NCP"]]
    ax.scatter(low["minus_NCP"], low["applied_residual_correction"], s=7, alpha=0.50, color=COLOR_GREY, edgecolors="none", zorder=1, label="−NCP < 40")
    ax.scatter(high["minus_NCP"], high["applied_residual_correction"], s=7, alpha=0.80, color=COLOR_BLUE, edgecolors="none", zorder=2, label="−NCP ≥ 40")
    ax.plot(
        corr_summary["bin_center_minus_NCP"],
        corr_summary["applied_residual_correction_bin_mean"],
        "o-",
        markersize=2.5,
        lw=0.8,
        color=TREND_COLOR,
        zorder=4
    )    
    ax.axhline(0, lw=0.8, color=COLOR_DARK, zorder=3)
    ax.axvline(HIGH_THRESHOLD, lw=0.8, color=COLOR_DARK, ls="--", zorder=3)
    ymax = float(np.nanmax(np.abs(df["applied_residual_correction"])))
    ax.set_ylim(-1.15 * ymax, 1.15 * ymax)
    ax.set_xlabel("−NCP")
    ax.set_ylabel("Applied residual correction\nΔ predicted log10(−B22)")
    ax.legend(loc="upper left", frameon=False, handletextpad=0.2, borderpad=0.1)
    clean_figure_axes(ax)
    paths = save_fig(fig, "FigS5D_applied_residual_correction")
    panel_info["S5D"] = {"paths": paths, "source_csv": str(OUTDIR / "FigS5D_source.csv"), "applied_correction_definition": "full-hybrid prediction − base-component prediction", "identity_max_abs_error": identity_error}


def write_legend_notes() -> None:
    text = """Figure S5 legend notes for the final hybrid residual component

High−NCP threshold = 40.
Sigmoid gate centre = 40.
Gate scale = 12.
Calibrated residual weight constrained to 0–0.6.
Correction capped at the 85th percentile of absolute calibration residual magnitude.
Residual model architecture: neural-network residual branch from the completed final nested OOF lineage; see S5 final source audit for the exact authoritative files.
Full-development calibration for the frozen external predictor selected λ = 0.

In the final fit to the complete 917-pair development set, residual calibration selected λ = 0; therefore the frozen 1,794-pair external predictions coincide with the base-component predictions.
"""
    (OUTDIR / "FigS5_legend_notes.txt").write_text(text)


def write_audit(sources: Sources, panel_info: dict[str, object]) -> None:
    lines = [
        "Supplementary Figure S5 final hybrid figure audit",
        "",
        f"Workspace: {ROOT}",
        "Scope: Supplementary Figure S5 only.",
        "Main Figure 5, Supplementary Figure S8, manuscript text, supplementary workbooks, and idr-interaction-predict were not modified by this script.",
        "",
        "Historical plotting aesthetics:",
        f"  {HIST_FIG5}",
        f"  {HIST_FIGS5}",
        "  Standalone 2.4 × 2.3 inch panels, 600 dpi PNG, historical grey/blue, hidden top/right spines, outward ticks.",
        "",
        "Authoritative residual-analysis files:",
        f"  OOF predictions: {sources.oof}",
        f"  Bootstrap high−NCP differences: {sources.bootstrap}",
        f"  Outer-fold results: {sources.outer_folds}",
        f"  Tail metrics: {sources.tail_metrics}",
        f"  Improvement summary: {sources.improvement}",
        f"  Inner tuning diagnostics: {sources.inner_tuning}",
        "",
        "Source-selection reason:",
        f"  {sources.selection_reason}",
        "",
        "Definitions:",
        "  base residual = simulated target − base-component prediction",
        "  hybrid residual = simulated target − full-hybrid prediction",
        "  applied correction = full-hybrid prediction − base-component prediction",
        "",
        "Validation:",
    ]
    for panel, info in panel_info.items():
        lines.append(f"  {panel}")
        for key, value in info.items():
            lines.append(f"    {key}: {value}")
    lines.extend([
        "",
        f"Observed high−NCP ΔRMSE = {sources.delta_rmse:.6f}",
        f"Observed high−NCP ΔRMSE 95% CI = [{sources.delta_rmse_ci_low:.6f}, {sources.delta_rmse_ci_high:.6f}]",
        f"Observed high−NCP ΔR² = {sources.delta_r2:.6f}",
        f"Observed high−NCP bias change = {sources.delta_bias:.6f}",
        "Only one final hybrid model lineage is represented in visible panels.",
        "No old/new model comparison is shown.",
        "Final full-development calibration λ = 0 is documented only in FigS5_legend_notes.txt.",
    ])
    (OUTDIR / "S5_final_hybrid_figure_audit.txt").write_text("\n".join(lines) + "\n")


def main() -> None:
    sources = identify_sources()
    panel_info: dict[str, object] = {}
    df = make_s5a_b(sources, panel_info)
    make_s5c(sources, panel_info)
    make_s5d(df, panel_info)
    write_legend_notes()
    write_audit(sources, panel_info)
    print(f"Wrote corrected standalone S5 panels to {OUTDIR}")


if __name__ == "__main__":
    main()
