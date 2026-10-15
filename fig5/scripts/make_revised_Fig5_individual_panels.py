#!/usr/bin/env python3
"""Generate revised Figure 5 panels B-F as individual manuscript-style files.

Panel A is intentionally not generated. The scatter and landscape plotting code
below mirrors the historical implementation in fig5.py while using revised data.
"""
from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path
from typing import Tuple

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / "outputs" / "mplconfig"))

import joblib
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


mpl.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial"],
    "font.size": 6.5,
    "axes.titlesize": 7,
    "axes.labelsize": 7,
    "xtick.labelsize": 6,
    "ytick.labelsize": 6,
    "legend.fontsize": 6,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "svg.fonttype": "none",
    "axes.linewidth": 0.6,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "xtick.minor.width": 0.6,
    "ytick.minor.width": 0.6,
    "xtick.direction": "out",
    "ytick.direction": "out",
})

OUT = ROOT / "outputs" / "revised_fig5_main"
FIGSIZE = (2.4, 2.3)
DPI_PNG = 600
COLOR_BLUE = "#0072B2"
COLOR_GREY = "#656565"
RNG_SEED = 12345
MAX_BG_POINTS_F = 100_000
RASTERIZE_SCATTER = True
RASTERIZE_DPI = 600
RASTERIZE_ZORDER = 1

HIST_SCRIPT = ROOT / "fig5.py"
HIST_TRAIN = ROOT / "sim_pairs_pred_with_hybrid_sel_cols_with_old.csv"
HIST_HELDOUT = ROOT / "taken_out_preds_with_hybrid_021125_sel_cols.csv"
HIST_UNSIM = ROOT / "unsim_preds_with_hybrid_021125_sel_cols.csv"
RBP_FILE = ROOT / "rg_rbp_unids.txt"

GLOBAL_BY_REPEAT = ROOT / "reviewer_feature_family_decomposition_training_only" / "cv_metrics_by_repeat.csv"
LOCAL_BY_REPEAT = ROOT / "reviewer_f2_patch_context_training_only" / "cv_metrics_by_repeat.csv"
DEV_OOF = ROOT / "outputs" / "fig5_tensorflow_residual_final_base" / "outer_oof_predictions_F2_P8.csv"
HELDOUT = ROOT / "fig5_validation_run_model_local" / "outputs" / "heldout" / "heldout_final_predictions.csv"
CURRENT_F_SOURCE = OUT / "Fig5F_full_landscape_source.csv"
PATCH_UNSIM = Path("/Users/yuktikhanna/calvados/patches/datasets/features_unsim_pairs_for_training.csv")
MODEL_DIR = ROOT / "fig5_validation_run_model_local" / "artifacts" / "final_model"


def ensure_dirs() -> None:
    OUT.mkdir(parents=True, exist_ok=True)


def _read_csv(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {path}")
    return pd.read_csv(path, low_memory=False)


def identity_limits(x: np.ndarray, y: np.ndarray) -> Tuple[float, float]:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    m = np.isfinite(x) & np.isfinite(y)
    if not np.any(m):
        return (0.0, 1.0)
    data = np.concatenate([x[m], y[m]])
    dmin, dmax = float(np.min(data)), float(np.max(data))
    if dmin == dmax:
        dmin -= 0.5
        dmax += 0.5
    pad = 0.05 * (dmax - dmin)
    return dmin - pad, dmax + pad


def clean_figure_axes(fig) -> None:
    for ax in fig.axes:
        try:
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
        except Exception:
            pass


def save_fig(fig: mpl.figure.Figure, name: str) -> None:
    clean_figure_axes(fig)
    fig.savefig(OUT / f"{name}.pdf", bbox_inches="tight", pad_inches=0.01, dpi=RASTERIZE_DPI)
    fig.savefig(OUT / f"{name}.png", dpi=DPI_PNG, bbox_inches="tight", pad_inches=0.01)
    fig.savefig(OUT / f"{name}.svg", bbox_inches="tight", pad_inches=0.01, dpi=RASTERIZE_DPI)
    plt.close(fig)


def load_rbp_unids(path: str | Path) -> set[str]:
    path = Path(path)
    if not path.exists():
        return set()
    return {line.strip() for line in path.read_text().splitlines() if line.strip()}


def add_is_rbp(df: pd.DataFrame, rbp_ids: set[str]) -> pd.DataFrame:
    out = df.copy()
    if "name" not in out.columns:
        raise KeyError("name")
    out["is_rbp"] = out["name"].astype(str).str.extract(r"^([^_]+)")[0].isin(rbp_ids)
    return out


def metrics(y, p) -> dict[str, float]:
    y = pd.to_numeric(pd.Series(y), errors="coerce").to_numpy(float)
    p = pd.to_numeric(pd.Series(p), errors="coerce").to_numpy(float)
    ok = np.isfinite(y) & np.isfinite(p)
    y = y[ok]
    p = p[ok]
    ss_res = np.sum((y - p) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    return {
        "n": int(len(y)),
        "R2": float(1 - ss_res / ss_tot) if ss_tot else np.nan,
        "RMSE": float(np.sqrt(np.mean((y - p) ** 2))),
        "MAE": float(np.mean(np.abs(y - p))),
        "Pearson": float(pd.Series(y).corr(pd.Series(p), method="pearson")),
        "Spearman": float(pd.Series(y).corr(pd.Series(p), method="spearman")),
    }


def rgg_uniprot(pair_name: str) -> str:
    return str(pair_name).split("__", 1)[0].split("_", 1)[0]


def final_model_predict(df: pd.DataFrame) -> np.ndarray:
    feature_order = json.loads((MODEL_DIR / "feature_order.json").read_text())
    cal = json.loads((MODEL_DIR / "residual_calibration.json").read_text())
    if cal.get("lambda_hat") != 0:
        raise SystemExit("Panel F recomputation would require residual model because lambda_hat is non-zero.")
    work = df.copy()
    if "mean_lambda_RGG" not in work.columns:
        work["mean_lambda_RGG"] = work["lambda_RGG"] / work["RGG_length"]
    if "mean_lambda_TAD" not in work.columns:
        work["mean_lambda_TAD"] = work["lambda_TAD"] / work["TAD_length"]
    missing = [c for c in feature_order if c not in work.columns]
    if missing:
        raise SystemExit("Missing final-model features: " + ", ".join(missing))
    scaler = joblib.load(MODEL_DIR / "base_scaler.pkl")
    models = joblib.load(MODEL_DIR / "base_model.pkl")
    X = scaler.transform(work[feature_order].to_numpy(float))
    return np.mean([model.predict(X) for model in models], axis=0)


def audit_cross_checks(rbp_ids: set[str]) -> dict[str, str | float | int]:
    audit = {}
    hist_train = _read_csv(HIST_TRAIN)
    dev = _read_csv(DEV_OOF).groupby("name", as_index=False).agg(
        B22_corr1_log10=("true_target", "mean"),
        prediction_final=("base_prediction", "mean"),
    )
    merged = dev.merge(hist_train[["name", "B22_corr1_log10", "B22_pred_hybrid"]], on="name", suffixes=("_current", "_historical"), validate="one_to_one")
    audit["D_pair_overlap"] = len(merged)
    audit["D_unique_current_pairs"] = dev["name"].nunique()
    audit["D_target_max_abs_difference"] = float((merged["B22_corr1_log10_current"] - merged["B22_corr1_log10_historical"]).abs().max())
    audit["D_prediction_mean_abs_new_vs_historical_difference"] = float((merged["prediction_final"] - merged["B22_pred_hybrid"]).abs().mean())
    audit["D_prediction_new_vs_historical_correlation"] = float(merged["prediction_final"].corr(merged["B22_pred_hybrid"]))
    dev_rbp = add_is_rbp(dev, rbp_ids)
    audit["D_RBP_count"] = int(dev_rbp["is_rbp"].sum())
    audit["D_non_RBP_count"] = int((~dev_rbp["is_rbp"]).sum())

    hist_held = _read_csv(HIST_HELDOUT)
    held = _read_csv(HELDOUT)
    merged_h = held[["name", "B22_corr1_log10", "prediction_final"]].merge(
        hist_held[["name", "B22_corr1_log10", "B22_pred_hybrid"]],
        on="name", suffixes=("_current", "_historical"), validate="one_to_one",
    )
    audit["E_pair_overlap"] = len(merged_h)
    audit["E_unique_current_pairs"] = held["name"].nunique()
    audit["E_target_max_abs_difference"] = float((merged_h["B22_corr1_log10_current"] - merged_h["B22_corr1_log10_historical"]).abs().max())
    audit["E_prediction_mean_abs_new_vs_historical_difference"] = float((merged_h["prediction_final"] - merged_h["B22_pred_hybrid"]).abs().mean())
    audit["E_prediction_new_vs_historical_correlation"] = float(merged_h["prediction_final"].corr(merged_h["B22_pred_hybrid"]))
    held_rbp = add_is_rbp(held, rbp_ids)
    audit["E_RBP_count"] = int(held_rbp["is_rbp"].sum())
    audit["E_non_RBP_count"] = int((~held_rbp["is_rbp"]).sum())
    return audit


def audit_current_f_source() -> dict[str, str | float | int | bool]:
    audit = {}
    if not CURRENT_F_SOURCE.exists():
        audit["F_current_source_exists"] = False
        return audit
    current = _read_csv(CURRENT_F_SOURCE)
    hist = _read_csv(HIST_UNSIM)
    merged = current[["name", "frozen_predicted_log10_minus_B22"]].merge(
        hist[["name", "B22_pred_hybrid"]], on="name", validate="one_to_one"
    )
    max_abs = float((merged["frozen_predicted_log10_minus_B22"] - merged["B22_pred_hybrid"]).abs().max())
    audit["F_current_source_exists"] = True
    audit["F_current_vs_historical_rows"] = len(merged)
    audit["F_current_vs_historical_max_abs_difference"] = max_abs
    audit["F_current_source_rejected"] = bool(max_abs < 1e-12 and len(merged) == len(hist))
    msg = "INVALID_FOR_REVISED_FIGURE - current F source is historical hybrid prediction" if audit["F_current_source_rejected"] else "Current F source not identical to historical hybrid prediction"
    (OUT / "Fig5F_provenance_audit.txt").write_text(msg + "\n")
    return audit


def panel_B() -> None:
    df = _read_csv(GLOBAL_BY_REPEAT)
    order = ["F0_OCN_only", "F1_OCN_plus_SCD", "C_only_hydropathy_stickiness", "F2_OCN_plus_hydropathy", "F6_full_M4"]
    labels = ["-NCP only", "-NCP + charge\npatterning", "non-electrostatic\nonly", "-NCP + hydropathy/\nstickiness", "full non-redundant\nglobal model"]
    source = df[df["model"].isin(order)].copy()
    source["display_model"] = source["model"].map(dict(zip(order, labels)))
    source.to_csv(OUT / "Fig5B_global_ablation_source.csv", index=False)
    summary = source.groupby("model")["r2"].agg(["mean", "std"]).reindex(order)
    fig, ax = plt.subplots(figsize=FIGSIZE, constrained_layout=True)
    fig.set_constrained_layout_pads(w_pad=0.02, h_pad=0.02, wspace=0.0, hspace=0.0)
    x = np.arange(len(summary))
    ax.bar(x, summary["mean"], yerr=summary["std"], color=COLOR_GREY, edgecolor="none", capsize=2)
    ax.set_xticks(x, labels=labels, rotation=45, ha="right")
    ax.set_ylabel("Out-of-fold R²")
    ax.set_ylim(0, 1)
    save_fig(fig, "Fig5B_global_ablation")


def panel_C() -> None:
    df = _read_csv(LOCAL_BY_REPEAT)
    order = ["F2", "F2-P1", "F2-P7", "F2-P8"]
    labels = ["Global model", "+ local charge\nheterogeneity", "+ broader local\nsequence context", "+ AH-energy\nheterogeneity"]
    source = df[df["model"].isin(order)].copy()
    source["display_model"] = source["model"].map(dict(zip(order, labels)))
    source.to_csv(OUT / "Fig5C_local_context_source.csv", index=False)
    summary = source.groupby("model")["R2"].agg(["mean", "std"]).reindex(order)
    fig, ax = plt.subplots(figsize=FIGSIZE, constrained_layout=True)
    fig.set_constrained_layout_pads(w_pad=0.02, h_pad=0.02, wspace=0.0, hspace=0.0)
    x = np.arange(len(summary))
    ax.bar(x, summary["mean"], yerr=summary["std"], color=COLOR_GREY, edgecolor="none", capsize=2)
    ax.set_xticks(x, labels=labels, rotation=45, ha="right")
    ax.set_ylabel("Out-of-fold R²")
    ax.set_ylim(0.90, 0.97)
    save_fig(fig, "Fig5C_local_context")


def panel_D(rbp_ids: set[str]) -> dict[str, float | int]:
    df = _read_csv(DEV_OOF)
    source = df.groupby("name", as_index=False).agg(sim=("true_target", "mean"), pred=("base_prediction", "mean"))
    source = add_is_rbp(source, rbp_ids)
    source.to_csv(OUT / "Fig5D_development_OOF_source.csv", index=False)
    x = source["sim"].to_numpy()
    y = source["pred"].to_numpy()
    is_rbp = source["is_rbp"].to_numpy()
    m = metrics(x, y)
    rbp_r = float(pd.Series(x[is_rbp]).corr(pd.Series(y[is_rbp]), method="pearson")) if is_rbp.sum() > 2 else np.nan
    fig, ax = plt.subplots(figsize=FIGSIZE, constrained_layout=True)
    fig.set_constrained_layout_pads(w_pad=0.02, h_pad=0.02, wspace=0.0, hspace=0.0)
    ax.scatter(x[~is_rbp], y[~is_rbp], s=7, alpha=0.50, c=COLOR_GREY, edgecolors="none", label="Non-RBP", rasterized=RASTERIZE_SCATTER, zorder=RASTERIZE_ZORDER)
    ax.scatter(x[is_rbp], y[is_rbp], s=7, alpha=0.80, c=COLOR_BLUE, edgecolors="none", label="RBP", rasterized=RASTERIZE_SCATTER, zorder=RASTERIZE_ZORDER)
    dmin, dmax = identity_limits(x, y)
    ax.plot([dmin, dmax], [dmin, dmax], lw=0.8, color="black", zorder=3)
    ax.set_xlim(dmin, dmax)
    ax.set_ylim(dmin, dmax)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel(r"Simulated $y=\log_{10}(-B_{22})$")
    ax.set_ylabel(r"Predicted $y=\log_{10}(-B_{22})$")
    ax.text(0.97, 0.03, f"All pairs: R²={m['R2']:.3f}, r={m['Pearson']:.3f}\nn={m['n']}", ha="right", va="bottom", transform=ax.transAxes)
    ax.legend(frameon=False, loc="upper left", handletextpad=0.6, borderpad=0.3)
    save_fig(fig, "Fig5D_development_OOF")
    return {**m, "RBP_Pearson": rbp_r, "RBP_n": int(is_rbp.sum())}


def panel_E(rbp_ids: set[str]) -> dict[str, float | int]:
    source = _read_csv(HELDOUT).copy()
    source = add_is_rbp(source, rbp_ids)
    source[["name", "B22_corr1_log10", "prediction_final", "is_rbp"]].to_csv(OUT / "Fig5E_heldout_1794_source.csv", index=False)
    x = source["B22_corr1_log10"].to_numpy()
    y = source["prediction_final"].to_numpy()
    is_rbp = source["is_rbp"].to_numpy()
    m = metrics(x, y)
    rbp_r = float(pd.Series(x[is_rbp]).corr(pd.Series(y[is_rbp]), method="pearson")) if is_rbp.sum() > 2 else np.nan
    fig, ax = plt.subplots(figsize=FIGSIZE, constrained_layout=True)
    fig.set_constrained_layout_pads(w_pad=0.02, h_pad=0.02, wspace=0.0, hspace=0.0)
    ax.scatter(x[~is_rbp], y[~is_rbp], s=7, alpha=0.50, c=COLOR_GREY, edgecolors="none", label="Non-RBP", rasterized=RASTERIZE_SCATTER, zorder=RASTERIZE_ZORDER)
    ax.scatter(x[is_rbp], y[is_rbp], s=7, alpha=0.80, c=COLOR_BLUE, edgecolors="none", label="RBP", rasterized=RASTERIZE_SCATTER, zorder=RASTERIZE_ZORDER)
    dmin, dmax = identity_limits(x, y)
    ax.plot([dmin, dmax], [dmin, dmax], lw=0.8, color="black", zorder=3)
    ax.set_xlim(dmin, dmax)
    ax.set_ylim(dmin, dmax)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel(r"Simulated $y=\log_{10}(-B_{22})$")
    ax.set_ylabel(r"Predicted $y=\log_{10}(-B_{22})$")
    ax.text(0.97, 0.03, f"All pairs: R²={m['R2']:.3f}, r={m['Pearson']:.3f}\nn={m['n']}", ha="right", va="bottom", transform=ax.transAxes)
    ax.legend(frameon=False, loc="upper left", handletextpad=0.6, borderpad=0.3)
    save_fig(fig, "Fig5E_heldout_1794")
    return {**m, "RBP_Pearson": rbp_r, "RBP_n": int(is_rbp.sum())}


def recompute_panel_F_landscape(rbp_ids: set[str]) -> dict[str, float | int | str]:
    held = _read_csv(HELDOUT)
    held_repred = final_model_predict(held)
    max_diff = float(np.max(np.abs(held_repred - held["prediction_final"].to_numpy(float))))
    if max_diff > 1e-10:
        raise SystemExit(f"Frozen model verification against accepted 1,794 predictions failed: max abs diff={max_diff}")

    patch = _read_csv(PATCH_UNSIM)
    patch["mean_lambda_RGG"] = patch["lambda_RGG"] / patch["RGG_length"]
    patch["mean_lambda_TAD"] = patch["lambda_TAD"] / patch["TAD_length"]
    pred = final_model_predict(patch)
    source = pd.DataFrame({
        "name": patch["name"],
        "-NCP": patch["Opposite_Charge_Number"],
        "frozen_prediction": pred,
    })
    source["is_rbp"] = source["name"].map(rgg_uniprot).isin(rbp_ids)
    source.to_csv(OUT / "full_landscape_F2P8_frozen_predictions.csv", index=False)
    return {"heldout_reprediction_max_abs_diff": max_diff, "landscape_prediction_n": len(source), "model_dir": str(MODEL_DIR)}


def panel_F(rbp_ids: set[str]) -> dict[str, float | int | str]:
    stats = recompute_panel_F_landscape(rbp_ids)
    source = _read_csv(OUT / "full_landscape_F2P8_frozen_predictions.csv")
    rng = np.random.default_rng(RNG_SEED)
    is_rbp = source["is_rbp"].to_numpy()
    idx = np.arange(len(source))
    bg = idx[~is_rbp]
    rbp = idx[is_rbp]
    if len(bg) > MAX_BG_POINTS_F:
        bg = rng.choice(bg, size=MAX_BG_POINTS_F, replace=False)
    keep = np.zeros(len(source), dtype=bool)
    keep[bg] = True
    keep[rbp] = True
    plot = source.loc[keep].copy()
    mask = plot["is_rbp"].to_numpy()
    fig, ax = plt.subplots(figsize=FIGSIZE, constrained_layout=True)
    fig.set_constrained_layout_pads(w_pad=0.02, h_pad=0.02, wspace=0.0, hspace=0.0)
    ax.scatter(plot.loc[~mask, "-NCP"], plot.loc[~mask, "frozen_prediction"], s=7, alpha=0.50, c=COLOR_GREY, edgecolors="none", label="All pairs", rasterized=RASTERIZE_SCATTER, zorder=RASTERIZE_ZORDER)
    ax.scatter(plot.loc[mask, "-NCP"], plot.loc[mask, "frozen_prediction"], s=7, alpha=0.80, c=COLOR_BLUE, edgecolors="none", label="RBP", rasterized=RASTERIZE_SCATTER, zorder=RASTERIZE_ZORDER)
    ax.set_xlabel("-NCP")
    ax.set_ylabel(r"Predicted $y=\log_{10}(-B_{22})$")
    ax.legend(frameon=False, loc="best", handletextpad=0.6, borderpad=0.3)
    save_fig(fig, "Fig5F_full_landscape")
    stats["F_plotted_points"] = int(keep.sum())
    stats["F_RBP_points"] = int(is_rbp.sum())
    return stats


def write_audit(cross: dict, f_source: dict, d_metrics: dict, e_metrics: dict, f_stats: dict) -> None:
    lines = [
        "Historical scatter reproduction audit",
        "",
        "historical script",
        f"path: {HIST_SCRIPT}",
        "functions literally reused/copied: panel_D scatter layering, panel_F landscape downsampling/scatter layering, identity_limits, clean_figure_axes, save_fig, add_is_rbp, load_rbp_unids",
        "",
        "D",
        f"917/917 pair overlap: {cross['D_pair_overlap']}/917",
        f"unique current pairs: {cross['D_unique_current_pairs']}",
        f"target max abs discrepancy: {cross['D_target_max_abs_difference']:.16g}",
        f"new-vs-old prediction mean absolute difference: {cross['D_prediction_mean_abs_new_vs_historical_difference']:.16g}",
        f"new-vs-old prediction correlation: {cross['D_prediction_new_vs_historical_correlation']:.16g}",
        f"RBP/non-RBP counts: {cross['D_RBP_count']}/{cross['D_non_RBP_count']}",
        f"current plotted metrics: n={d_metrics['n']} R2={d_metrics['R2']:.6f} RMSE={d_metrics['RMSE']:.6f} Pearson={d_metrics['Pearson']:.6f} RBP_r={d_metrics['RBP_Pearson']:.6f} RBP_n={d_metrics['RBP_n']}",
        "output resolution: individual 2.4 x 2.3 inch panel, 600-dpi PNG plus PDF/SVG",
        "",
        "E",
        f"1794/1794 pair overlap: {cross['E_pair_overlap']}/1794",
        f"unique current pairs: {cross['E_unique_current_pairs']}",
        f"target max abs discrepancy: {cross['E_target_max_abs_difference']:.16g}",
        f"new-vs-old prediction mean absolute difference: {cross['E_prediction_mean_abs_new_vs_historical_difference']:.16g}",
        f"new-vs-old prediction correlation: {cross['E_prediction_new_vs_historical_correlation']:.16g}",
        f"RBP/non-RBP counts: {cross['E_RBP_count']}/{cross['E_non_RBP_count']}",
        f"current plotted metrics: n={e_metrics['n']} R2={e_metrics['R2']:.6f} RMSE={e_metrics['RMSE']:.6f} Pearson={e_metrics['Pearson']:.6f} RBP_r={e_metrics['RBP_Pearson']:.6f} RBP_n={e_metrics['RBP_n']}",
        "output resolution: individual 2.4 x 2.3 inch panel, 600-dpi PNG plus PDF/SVG",
        "",
        "F",
        f"current F source exists: {f_source.get('F_current_source_exists')}",
        f"current-vs-historical rows: {f_source.get('F_current_vs_historical_rows', 'not available')}",
        f"current-vs-historical max abs difference: {f_source.get('F_current_vs_historical_max_abs_difference', 'not available')}",
        f"current source rejected: {f_source.get('F_current_source_rejected', False)}",
        f"exact model used for recomputation: {f_stats.get('model_dir')}",
        f"verification against accepted 1,794 prediction output max abs difference: {f_stats.get('heldout_reprediction_max_abs_diff'):.16g}",
        f"final number of landscape predictions: {f_stats.get('landscape_prediction_n')}",
        f"plotted points after historical downsampling rule: {f_stats.get('F_plotted_points')}",
        "historical hybrid predictions were not substituted.",
    ]
    (OUT / "historical_scatter_reproduction_audit.txt").write_text("\n".join(lines) + "\n")


def main() -> None:
    ensure_dirs()
    rbp_ids = load_rbp_unids(RBP_FILE)
    cross = audit_cross_checks(rbp_ids)
    f_source = audit_current_f_source()
    panel_B()
    panel_C()
    d_metrics = panel_D(rbp_ids)
    e_metrics = panel_E(rbp_ids)
    f_stats = panel_F(rbp_ids)
    write_audit(cross, f_source, d_metrics, e_metrics, f_stats)
    print(f"Wrote individual revised Figure 5 panels to {OUT}")


if __name__ == "__main__":
    main()
