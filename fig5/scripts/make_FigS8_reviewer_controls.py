#!/usr/bin/env python3
"""Build draft reviewer-control Figure S8 with placeholders for running controls."""
from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parents[1] / "outputs" / "mplconfig"))

import matplotlib.pyplot as plt
import pandas as pd

from fig5_plotting_style import plot_manuscript_scatter
from fig5_revision_plot_utils import ROOT, SUPP_OUT, add_panel_label, ensure_dir, metrics, placeholder, read_csv, save_all


HISTIDINE = ROOT / "outputs" / "fig5_histidine_pH_audit" / "histidine_matched_b22.csv"
HISTIDINE_PRED_METRICS = ROOT / "outputs" / "fig5_histidine_pH_audit" / "histidine_prediction_metrics_pH6p5.csv"


def write_status(outdir, panel, text):
    (outdir / f"FigS8{panel}_status.txt").write_text(text + "\n")


def plot_panel_A_nondataset(ax, outdir):
    placeholder(ax, "A", "Pending completion of non-dataset WT simulations")
    write_status(outdir, "A", "PENDING - non-dataset WT control simulations still running")


def plot_panel_B_shuffle_WT(ax, outdir):
    placeholder(ax, "B", "Pending completion of matched shuffle simulations")
    write_status(outdir, "B", "PENDING - matched shuffle simulations still running")


def plot_panel_C_shuffle_delta(ax, outdir):
    placeholder(ax, "C", "Pending completion of matched shuffle analysis")
    write_status(outdir, "C", "PENDING - matched shuffle analysis still running")


def plot_panel_D_histidine_pH(ax, outdir):
    df = read_csv(HISTIDINE).copy()
    source = df[["pair_id", "B22_corr1_log10_pH7", "B22_corr1_log10_pH6p5", "delta_logB22", "total_H"]].copy()
    source.to_csv(outdir / "FigS8D_histidine_pH_source.csv", index=False)
    m = metrics(source["B22_corr1_log10_pH7"], source["B22_corr1_log10_pH6p5"])
    plot_manuscript_scatter(
        ax, source["B22_corr1_log10_pH7"], source["B22_corr1_log10_pH6p5"],
        xlabel=r"pH 7 simulated $y=\log_{10}(-B_{22})$",
        ylabel=r"pH 6.5 simulated $y=\log_{10}(-B_{22})$",
        annotation=f"n={m['n']}\nr={m['Pearson']:.3f}\nrho={m['Spearman']:.3f}",
    )
    add_panel_label(ax, "D")


def plot_panel_E_histidine_prediction(ax, outdir):
    df = read_csv(HISTIDINE).copy()
    source = df[["pair_id", "B22_corr1_log10_pH6p5", "prediction_pH6p5", "residual_pH6p5", "abs_error_pH6p5"]].copy()
    source.to_csv(outdir / "FigS8E_histidine_prediction_source.csv", index=False)
    m = metrics(source["B22_corr1_log10_pH6p5"], source["prediction_pH6p5"])
    metric_source = read_csv(HISTIDINE_PRED_METRICS)
    metric_source.to_csv(outdir / "FigS8E_histidine_prediction_metrics_source.csv", index=False)
    plot_manuscript_scatter(
        ax, source["B22_corr1_log10_pH6p5"], source["prediction_pH6p5"],
        xlabel=r"pH 6.5 simulated $y=\log_{10}(-B_{22})$",
        ylabel=r"Predicted $y=\log_{10}(-B_{22})$",
        annotation=f"n={m['n']}\nR²={m['R2']:.3f}\nRMSE={m['RMSE']:.3f}\nr={m['Pearson']:.3f}",
    )
    ax.text(0.03, 1.04, "Predictor does not explicitly encode pH", transform=ax.transAxes, fontsize=6.5, va="bottom")
    add_panel_label(ax, "E")


def plot_panel_F_basic_OOD(ax, outdir):
    placeholder(ax, "F", "Pending completion of basic-region control simulations")
    write_status(outdir, "F", "PENDING - basic-region control simulations still running")


def main():
    outdir = ensure_dir(SUPP_OUT)
    fig, axes = plt.subplots(3, 2, figsize=(8.2, 10.0), constrained_layout=True)
    plot_panel_A_nondataset(axes[0, 0], outdir)
    plot_panel_B_shuffle_WT(axes[0, 1], outdir)
    plot_panel_C_shuffle_delta(axes[1, 0], outdir)
    plot_panel_D_histidine_pH(axes[1, 1], outdir)
    plot_panel_E_histidine_prediction(axes[2, 0], outdir)
    plot_panel_F_basic_OOD(axes[2, 1], outdir)
    save_all(fig, outdir, "FigS8_reviewer_controls_DRAFT")
    plt.close(fig)
    audit = [
        "Reviewer-control Figure S8 DRAFT audit",
        "A PENDING - running simulations",
        "B PENDING - running simulations",
        "C PENDING - running simulations",
        f"D PASS: {HISTIDINE}",
        f"E PASS: {HISTIDINE}; metrics source {HISTIDINE_PRED_METRICS}",
        "F PENDING - running simulations",
        "No preliminary non-dataset WT, shuffle, or basic-region outputs were used.",
        "",
        "Scatter-panel data/style separation:",
        "FigS8D | data source: histidine pH audit | plot-style source: /Users/yuktikhanna/rtads-rgg-interactions/fig5/fig5.py old manuscript scatter function",
        "FigS8E | data source: frozen pH6.5 validation | plot-style source: /Users/yuktikhanna/rtads-rgg-interactions/fig5/fig5.py old manuscript scatter function",
    ]
    (outdir / "revised_FigS8_audit.txt").write_text("\n".join(audit) + "\n")


if __name__ == "__main__":
    main()
