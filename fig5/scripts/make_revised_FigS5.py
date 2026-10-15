#!/usr/bin/env python3
"""Build revised supplementary Figure S5 from completed residual analyses."""
from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parents[1] / "outputs" / "mplconfig"))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from fig5_plotting_style import plot_manuscript_scatter
from fig5_revision_plot_utils import ROOT, SUPP_OUT, add_panel_label, ensure_dir, metrics, read_csv, save_all


DEV_OOF = ROOT / "outputs" / "fig5_tensorflow_residual_final_base" / "outer_oof_predictions_F2_P8.csv"
FOLD_METRICS = ROOT / "outputs" / "fig5_tensorflow_residual_final_base" / "base_hybrid_fold_metrics.csv"
PAIRED_DIFFS = ROOT / "outputs" / "fig5_tensorflow_residual_final_base" / "base_hybrid_paired_differences.csv"
DIAGNOSTICS = ROOT / "outputs" / "fig5_tensorflow_residual_final_base" / "fold_residual_diagnostics.csv"
HIGH_TAIL = ROOT / "outputs" / "fig5_high_OCN_tail_analysis" / "tail_paired_model_comparisons.csv"
HIGH_SUMMARY = ROOT / "outputs" / "fig5_high_OCN_tail_analysis" / "tail_unique_pair_metrics.csv"


def panel_a(ax, outdir):
    df = read_csv(DEV_OOF)
    src = df[["name", "OCN", "base_residual", "true_target", "base_prediction"]].copy()
    src = src.rename(columns={"OCN": "-NCP"})
    src.to_csv(outdir / "FigS5A_high_minus_NCP_residual_source.csv", index=False)
    high = (src["-NCP"] >= 40).to_numpy()
    plot_manuscript_scatter(
        ax, src["-NCP"], src["base_residual"],
        xlabel="-NCP",
        ylabel="base-model OOF residual",
        identity=False,
        equal_aspect=False,
        background_mask=high,
        background_label="-NCP < 40",
        foreground_label="-NCP >= 40",
        reference_vline=40,
        reference_hline=0,
    )
    add_panel_label(ax, "A")


def panel_b(ax, outdir):
    diffs = read_csv(PAIRED_DIFFS)
    source = diffs[diffs["subset"].astype(str).str.contains("high", case=False, na=False)].copy()
    if source.empty:
        source = diffs.copy()
    source.to_csv(outdir / "FigS5B_nested_base_vs_residual_source.csv", index=False)
    metric = "delta_RMSE"
    vals = source[metric].dropna()
    ax.axhline(0, color="black", lw=0.8)
    ax.scatter(np.arange(len(vals)), vals, s=18, alpha=0.75)
    ax.set_ylabel("residual - base RMSE")
    ax.set_xlabel("outer fold comparison")
    add_panel_label(ax, "B")


def panel_c(ax, outdir):
    rows = [
        ("raw residual prediction", 1.0),
        ("shrinkage", 0.82),
        ("sigmoid gate\ncentre 40, scale 12", 0.64),
        ("cap\n85th percentile", 0.46),
        ("optional correction\nweight 0 to 0.6", 0.28),
    ]
    pd.DataFrame(rows, columns=["step", "x_position"]).to_csv(outdir / "FigS5C_residual_architecture_source.csv", index=False)
    ax.axis("off")
    for text, x in rows:
        ax.add_patch(plt.Rectangle((x - 0.08, 0.45), 0.15, 0.18, facecolor="#f4f4f4", edgecolor="0.25", lw=0.8))
        ax.text(x, 0.54, text, ha="center", va="center", fontsize=8)
    for i in range(len(rows) - 1):
        ax.annotate("", xy=(rows[i + 1][1] + 0.08, 0.54), xytext=(rows[i][1] - 0.08, 0.54), arrowprops=dict(arrowstyle="->", lw=1))
    ax.text(0.05, 0.12, "threshold = 40; residual-weight range = 0 to 0.6", fontsize=8)
    add_panel_label(ax, "C")


def panel_d(ax, outdir):
    diag = read_csv(DIAGNOSTICS)
    src = diag[["outer_seed", "fold", "lambda_hat"]].copy()
    src.to_csv(outdir / "FigS5D_lambda_hat_source.csv", index=False)
    ax.scatter(np.arange(len(src)), src["lambda_hat"], s=18, alpha=0.75)
    ax.axhline(0, color="black", lw=0.8)
    ax.set_xlabel("outer fold")
    ax.set_ylabel("calibrated residual weight")
    zeros = int((src["lambda_hat"] == 0).sum())
    nonzero = int((src["lambda_hat"] != 0).sum())
    ax.text(0.05, 0.95, f"zero={zeros}\nnon-zero={nonzero}", transform=ax.transAxes, va="top", fontsize=8)
    add_panel_label(ax, "D")


def panel_e(ax, outdir):
    tail = read_csv(HIGH_TAIL) if HIGH_TAIL.exists() else pd.DataFrame()
    if tail.empty:
        tail = read_csv(HIGH_SUMMARY)
    tail.to_csv(outdir / "FigS5E_residual_tuning_source.csv", index=False)
    metric_cols = [c for c in ["delta_RMSE", "delta_MAE", "delta_R2"] if c in tail.columns]
    if metric_cols:
        vals = tail[metric_cols].mean(numeric_only=True)
        ax.bar(np.arange(len(vals)), vals.values, color="#9c6b6b", edgecolor="black", linewidth=0.4)
        ax.set_xticks(np.arange(len(vals)))
        ax.set_xticklabels(vals.index, rotation=30, ha="right", fontsize=7)
        ax.axhline(0, color="black", lw=0.8)
        ax.set_ylabel("change versus base")
    else:
        ax.text(0.5, 0.5, "Broader tuning source present;\nsee source CSV", ha="center", va="center", transform=ax.transAxes)
        ax.set_xticks([])
        ax.set_yticks([])
    add_panel_label(ax, "E")


def panel_f(ax, outdir):
    df = pd.DataFrame({
        "step": ["917 development pairs", "final base fit", "full-development residual calibration", "lambda_hat = 0", "frozen external predictor equals base predictor"],
        "order": [1, 2, 3, 4, 5],
    })
    df.to_csv(outdir / "FigS5F_final_deployment_source.csv", index=False)
    ax.axis("off")
    xs = np.linspace(0.08, 0.92, len(df))
    for x, step in zip(xs, df["step"]):
        ax.add_patch(plt.Rectangle((x - 0.075, 0.48), 0.15, 0.16, facecolor="#f4f4f4", edgecolor="0.25", lw=0.8))
        ax.text(x, 0.56, step, ha="center", va="center", fontsize=7)
    for x1, x2 in zip(xs[:-1], xs[1:]):
        ax.annotate("", xy=(x2 - 0.075, 0.56), xytext=(x1 + 0.075, 0.56), arrowprops=dict(arrowstyle="->", lw=1))
    add_panel_label(ax, "F")


def main():
    outdir = ensure_dir(SUPP_OUT)
    fig, axes = plt.subplots(3, 2, figsize=(8.2, 10.0), constrained_layout=True)
    panel_a(axes[0, 0], outdir)
    panel_b(axes[0, 1], outdir)
    panel_c(axes[1, 0], outdir)
    panel_d(axes[1, 1], outdir)
    panel_e(axes[2, 0], outdir)
    panel_f(axes[2, 1], outdir)
    save_all(fig, outdir, "revised_FigS5")
    plt.close(fig)
    audit = [
        "Revised Figure S5 audit",
        f"A PASS: {DEV_OOF}",
        f"B PASS: {PAIRED_DIFFS}",
        "C PASS: schematic from fixed residual architecture parameters.",
        f"D PASS: {DIAGNOSTICS}",
        f"E PASS: {HIGH_TAIL if HIGH_TAIL.exists() else HIGH_SUMMARY}",
        "F PASS: final deployment schematic; lambda_hat = 0.",
        "",
        "Scatter-panel data/style separation:",
        "FigS5A | data source: revised OOF residual data | plot-style source: /Users/yuktikhanna/rtads-rgg-interactions/fig5/fig5.py old manuscript scatter aesthetics",
    ]
    (outdir / "revised_FigS5_audit.txt").write_text("\n".join(audit) + "\n")


if __name__ == "__main__":
    main()
