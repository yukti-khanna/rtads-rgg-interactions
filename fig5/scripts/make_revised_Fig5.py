#!/usr/bin/env python3
"""Build revised main Figure 5 from completed, non-control analyses."""
from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parents[1] / "outputs" / "mplconfig"))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from fig5_plotting_style import plot_manuscript_landscape, plot_manuscript_scatter, write_scatter_style_provenance
from fig5_revision_plot_utils import MAIN_OUT, ROOT, add_panel_label, clean_model_label, ensure_dir, metrics, read_csv, rename_charge_columns, save_all


GLOBAL_BY_REPEAT = ROOT / "reviewer_feature_family_decomposition_training_only" / "cv_metrics_by_repeat.csv"
LOCAL_BY_REPEAT = ROOT / "reviewer_f2_patch_context_training_only" / "cv_metrics_by_repeat.csv"
DEV_OOF = ROOT / "outputs" / "fig5_tensorflow_residual_final_base" / "outer_oof_predictions_F2_P8.csv"
HELDOUT_PRED = ROOT / "fig5_validation_run_model_local" / "outputs" / "heldout" / "heldout_final_predictions.csv"
HELDOUT_METRICS = ROOT / "fig5_validation_run_model_local" / "outputs" / "heldout" / "final_model_metrics.csv"
LANDSCAPE = ROOT / "unsim_preds_with_hybrid_021125_sel_cols.csv"


def plot_framework(ax):
    add_panel_label(ax, "A")
    ax.axis("off")
    boxes = [
        (0.02, 0.62, "917 development\nsimulations"),
        (0.23, 0.62, "Non-redundant\nglobal descriptors"),
        (0.45, 0.62, "Base ensemble\npredictor"),
        (0.67, 0.62, "Calibration"),
        (0.86, 0.62, "Frozen\npredictor"),
        (0.86, 0.18, "Independent\nvalidation"),
    ]
    for x, y, text in boxes:
        ax.add_patch(plt.Rectangle((x, y), 0.14, 0.18, facecolor="#f4f4f4", edgecolor="0.25", lw=0.8))
        ax.text(x + 0.07, y + 0.09, text, ha="center", va="center", fontsize=8)
    for x1, x2 in [(0.16, 0.23), (0.37, 0.45), (0.59, 0.67), (0.81, 0.86)]:
        ax.annotate("", xy=(x2, 0.71), xytext=(x1, 0.71), arrowprops=dict(arrowstyle="->", lw=1))
    ax.annotate("", xy=(0.93, 0.36), xytext=(0.93, 0.62), arrowprops=dict(arrowstyle="->", lw=1))
    ax.add_patch(plt.Rectangle((0.45, 0.30), 0.26, 0.14, facecolor="#fff7e6", edgecolor="0.35", lw=0.8))
    ax.text(0.58, 0.37, "Optional high--NCP\nresidual branch", ha="center", va="center", fontsize=8)
    ax.annotate("", xy=(0.58, 0.44), xytext=(0.52, 0.62), arrowprops=dict(arrowstyle="->", lw=0.8))
    ax.text(0.23, 0.26, "Bulk electrostatics: -NCP\nHydropathy/stickiness: mean lambda and SHD\nLocal sequence context: charge heterogeneity, charge patterning,\nRG/aromatic context, SHD heterogeneity, AH-energy heterogeneity", fontsize=8, va="top")
    ax.text(0.45, 0.08, "Final full-development residual weight = 0\nFrozen external predictor = final base predictor", fontsize=8, va="bottom", fontweight="bold")


def plot_global_ablation(ax, outdir):
    df = read_csv(GLOBAL_BY_REPEAT).copy()
    order = ["F0_OCN_only", "F1_OCN_plus_SCD", "C_only_hydropathy_stickiness", "F2_OCN_plus_hydropathy", "F6_full_M4"]
    df = df[df["model"].isin(order)].copy()
    df["display_model"] = df["model"].map(clean_model_label)
    df.to_csv(outdir / "Fig5B_global_ablation_source.csv", index=False)
    summary = df.groupby(["model", "display_model"], sort=False)["r2"].agg(["mean", "std"]).reindex(order, level=0).reset_index()
    x = np.arange(len(summary))
    ax.bar(x, summary["mean"], yerr=summary["std"], color="#5875a4", edgecolor="black", linewidth=0.4, capsize=3)
    ax.set_xticks(x)
    ax.set_xticklabels(summary["display_model"], rotation=35, ha="right", fontsize=7)
    ax.set_ylabel("Out-of-fold R²")
    ax.set_ylim(0, 1.02)
    add_panel_label(ax, "B")


def plot_local_ablation(ax, outdir):
    df = read_csv(LOCAL_BY_REPEAT).copy()
    order = ["F2", "F2-P1", "F2-P7", "F2-P8"]
    df = df[df["model"].isin(order)].copy()
    df["display_model"] = df["model"].map(clean_model_label)
    df.to_csv(outdir / "Fig5C_local_context_source.csv", index=False)
    summary = df.groupby(["model", "display_model"], sort=False)["R2"].agg(["mean", "std"]).reindex(order, level=0).reset_index()
    x = np.arange(len(summary))
    ax.bar(x, summary["mean"], yerr=summary["std"], color="#68a357", edgecolor="black", linewidth=0.4, capsize=3)
    ax.set_xticks(x)
    ax.set_xticklabels(summary["display_model"], rotation=35, ha="right", fontsize=7)
    ax.set_ylabel("Out-of-fold R²")
    ax.set_ylim(0.90, 0.97)
    add_panel_label(ax, "C")


def plot_dev_oof(ax, outdir):
    df = read_csv(DEV_OOF)
    source = df.groupby("name", as_index=False).agg(true=("true_target", "mean"), pred=("base_prediction", "mean"), n_oof=("base_prediction", "size"))
    source.to_csv(outdir / "Fig5D_development_OOF_source.csv", index=False)
    m = metrics(source["true"], source["pred"])
    plot_manuscript_scatter(
        ax, source["true"], source["pred"],
        xlabel=r"Simulated $y=\log_{10}(-B_{22})$",
        ylabel=r"Predicted $y=\log_{10}(-B_{22})$",
        annotation=f"n={m['n']}\nR²={m['R2']:.3f}\nRMSE={m['RMSE']:.3f}\nr={m['Pearson']:.3f}",
    )
    add_panel_label(ax, "D")


def plot_heldout(ax, outdir):
    df = read_csv(HELDOUT_PRED)
    df = rename_charge_columns(df)
    source = df[["name", "B22_corr1_log10", "prediction_final", "-NCP"]].copy()
    source.to_csv(outdir / "Fig5E_heldout_1794_source.csv", index=False)
    m = metrics(source["B22_corr1_log10"], source["prediction_final"])
    metrics_df = read_csv(HELDOUT_METRICS)
    metrics_df.to_csv(outdir / "Fig5E_heldout_1794_metrics_source.csv", index=False)
    plot_manuscript_scatter(
        ax, source["B22_corr1_log10"], source["prediction_final"],
        xlabel=r"Simulated $y=\log_{10}(-B_{22})$",
        ylabel=r"Predicted $y=\log_{10}(-B_{22})$",
        annotation=f"n={m['n']}\nR²={m['R2']:.3f}\nRMSE={m['RMSE']:.3f}\nr={m['Pearson']:.3f}",
    )
    add_panel_label(ax, "E")


def plot_landscape(ax, outdir):
    df = read_csv(LANDSCAPE)
    df = rename_charge_columns(df)
    pred_col = "B22_pred_hybrid" if "B22_pred_hybrid" in df.columns else "B22_pred_base"
    source = df[["name", "-NCP", pred_col]].rename(columns={pred_col: "frozen_predicted_log10_minus_B22"})
    source.to_csv(outdir / "Fig5F_full_landscape_source.csv", index=False)
    plot_manuscript_landscape(
        ax, source,
        x_col="-NCP",
        y_col="frozen_predicted_log10_minus_B22",
        xlabel="-NCP",
        ylabel=r"Predicted $y=\log_{10}(-B_{22})$",
        highlight_col=None,
    )
    add_panel_label(ax, "F")


def main():
    outdir = ensure_dir(MAIN_OUT)
    fig, axes = plt.subplots(3, 2, figsize=(8.2, 10.2), constrained_layout=True)
    plot_framework(axes[0, 0])
    plot_global_ablation(axes[0, 1], outdir)
    plot_local_ablation(axes[1, 0], outdir)
    plot_dev_oof(axes[1, 1], outdir)
    plot_heldout(axes[2, 0], outdir)
    plot_landscape(axes[2, 1], outdir)
    save_all(fig, outdir, "revised_Fig5")
    plt.close(fig)
    write_scatter_style_provenance(outdir / "scatter_style_provenance.txt")
    audit = [
        "Revised Figure 5 audit",
        "A PASS: schematic drawn from final model documentation.",
        f"B PASS: {GLOBAL_BY_REPEAT}",
        f"C PASS: {LOCAL_BY_REPEAT}",
        f"D PASS: {DEV_OOF}",
        f"E PASS: {HELDOUT_PRED}; metrics source {HELDOUT_METRICS}",
        f"F PASS: {LANDSCAPE}",
        "Terminology: labels use negative net-charge product (-NCP). Historical source columns are used only internally.",
        "",
        "Scatter-panel data/style separation:",
        "Fig5D | data source: revised OOF data | plot-style source: /Users/yuktikhanna/rtads-rgg-interactions/fig5/fig5.py old manuscript scatter function",
        "Fig5E | data source: revised frozen 1794 data | plot-style source: /Users/yuktikhanna/rtads-rgg-interactions/fig5/fig5.py old manuscript scatter function",
        "Fig5F | data source: revised full landscape | plot-style source: /Users/yuktikhanna/rtads-rgg-interactions/fig5/fig5.py old manuscript landscape/scatter function",
    ]
    (outdir / "revised_Fig5_audit.txt").write_text("\n".join(audit) + "\n")


if __name__ == "__main__":
    main()
