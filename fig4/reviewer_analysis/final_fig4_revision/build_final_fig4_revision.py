#!/usr/bin/env python3
"""Build final revised Figure 4 and Supplementary Figure S4.

Run locally from this directory:
    python build_final_fig4_revision.py

This script uses stored plotting/analysis outputs only. It does not rerun
CALVADOS, alter finite-size corrections, or overwrite historical outputs.
"""

from __future__ import annotations

import math
import pickle
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy import stats


OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[1]
HIST = ROOT / "out_fig4"
TASK3 = ROOT / "task3_sampling_bias"
TASK5 = ROOT / "task5_panelC_enrichment"
RDF_DIR = ROOT

OCN_COL = "Opposite_Charge_Number"
Y_COL = "B22_corr1_log10"
MARKER_SIZE = 10
RUNS = [f"run_{i}" for i in range(1, 6)]
JOBS = [f"job_{i}" for i in range(5)]
CLASSES = ["Basic", "Acidic", "Aromatic", "Other"]

STRONG_PAIR = "O60506_3_1__A8MYZ6_430_473"
WEAK_PAIR = "O00144_1_1__P49910_193_208"
STRONG_RDF = RDF_DIR / "rdfs_O60506_3_1__A8MYZ6_430_473_20nm.pkl"
WEAK_RDF = RDF_DIR / "rdfs_O00144_1_1__P49910_193_208_20nm.pkl"

COLORS = {
    "blue": "#0072B2",
    "orange": "#D55E00",
    "grey": "#7A7A7A",
    "light_grey": "#D3D3D3",
    "black": "#111111",
    "neutral": "#F0F0F0",
}
SET_COLORS = {
    "complete candidate landscape": COLORS["grey"],
    "feature-guided 917-pair panel": COLORS["blue"],
    "independent sequence-exclusive set": COLORS["orange"],
}


def apply_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "font.size": 8,
            "axes.titlesize": 8.5,
            "axes.labelsize": 8,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "legend.fontsize": 6.5,
            "axes.linewidth": 0.7,
            "xtick.major.width": 0.6,
            "ytick.major.width": 0.6,
            "xtick.direction": "out",
            "ytick.direction": "out",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def require(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(path)


def read_csv(path: Path) -> pd.DataFrame:
    require(path)
    return pd.read_csv(path, low_memory=False)


def style_axes(ax: plt.Axes) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    for sp in ax.spines.values():
        sp.set_linewidth(0.7)
    ax.tick_params(width=0.6)


def save(fig: mpl.figure.Figure, stem: str) -> None:
    fig.savefig(OUT / f"{stem}.png", dpi=600, bbox_inches="tight", pad_inches=0.03)
    fig.savefig(OUT / f"{stem}.pdf", dpi=600, bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)


def pearson_spearman(df: pd.DataFrame, x: str = OCN_COL, y: str = Y_COL) -> tuple[float, float, int]:
    sub = df[[x, y]].replace([np.inf, -np.inf], np.nan).dropna()
    return float(sub[x].corr(sub[y], method="pearson")), float(sub[x].corr(sub[y], method="spearman")), int(len(sub))


def matrix(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, index_col=0).loc[CLASSES, CLASSES]


def panel_label(ax: plt.Axes, label: str, x: float = -0.18, y: float = 1.08) -> None:
    ax.text(x, y, label, transform=ax.transAxes, ha="left", va="bottom", fontsize=11, fontweight="bold")


def build_fig4a(ax: plt.Axes | None = None, label: bool = False) -> None:
    df = read_csv(HIST / "cache" / "Fig4a_B22_values.csv")
    vals = df[Y_COL].replace([np.inf, -np.inf], np.nan).dropna().to_numpy(float)
    owns = ax is None
    if owns:
        fig, ax = plt.subplots(figsize=(3.1, 2.2))
    ax.hist(vals, bins=70, color=COLORS["light_grey"], edgecolor=COLORS["grey"], linewidth=0.35)
    med = float(np.nanmedian(vals))
    p5, p95 = np.nanpercentile(vals, [5, 95])
    ax.axvline(med, color=COLORS["black"], ls="--", lw=0.8, label=f"Median = {med:.3f}")
    ax.axvline(p5, color=COLORS["black"], ls=":", lw=0.8, label="5-95%")
    ax.axvline(p95, color=COLORS["black"], ls=":", lw=0.8)
    ax.set_title("All attractive training-set pairs")
    ax.set_xlabel(r"$B_{22}$")
    ax.set_ylabel("Density")
    ax.legend(frameon=False, loc="upper right")
    style_axes(ax)
    if label or owns:
        panel_label(ax, "A")
    if owns:
        save(fig, "Fig4A_distribution")


def corrected_rdf_for_run(r: np.ndarray, g: np.ndarray, run: str) -> tuple[np.ndarray, np.ndarray, float]:
    sys.path.insert(0, str(ROOT))
    from rdf_b22_functions import corrected_rdfs_b22

    one = pd.DataFrame({run: [np.asarray(r, float), np.asarray(g, float)]})
    rdfs, b22s, _ = corrected_rdfs_b22(one, 7.5, box_size=20.0, delta=2.0)
    return np.asarray(rdfs[run]["rs"], float), np.asarray(rdfs[run]["rdfs_corr1"], float), float(b22s[run]["b22_corr1"])


def load_rdfs(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    require(path)
    raw = pd.read_pickle(path)
    curves, b22s = [], []
    rs_ref = None
    for run in RUNS:
        r = np.asarray(raw[run].iloc[0], float)
        g = np.asarray(raw[run].iloc[1], float)
        rs, gr, b22 = corrected_rdf_for_run(r, g, run)
        if rs_ref is not None and not np.allclose(rs, rs_ref):
            raise ValueError(f"RDF r-grid mismatch in {path}")
        rs_ref = rs
        curves.append(gr)
        b22s.append(b22)
    return rs_ref, np.vstack(curves), np.asarray(b22s)


def pair_parts(pair_id: str) -> tuple[str, str]:
    if "__" in pair_id:
        return tuple(pair_id.split("__", 1))
    parts = pair_id.split("_")
    return "_".join(parts[:3]), "_".join(parts[3:])


def build_fig4b(axes: list[plt.Axes] | None = None, label_ax: plt.Axes | None = None) -> pd.DataFrame:
    strong_r, strong_curves, strong_b22 = load_rdfs(STRONG_RDF)
    weak_r, weak_curves, weak_b22 = load_rdfs(WEAK_RDF)
    rows = []
    for role, pair_id, display, b22 in [
        ("strong/interacting", STRONG_PAIR, "SYNCRIP RGG - FOXO6 R-TAD", strong_b22),
        ("weak/non-interacting", WEAK_PAIR, "FZD9 RGG - ZNF165 R-TAD", weak_b22),
    ]:
        rgg, tad = pair_parts(pair_id)
        rows.append(
            {
                "panel_role": role,
                "pair_id": pair_id,
                "RGG_id": rgg,
                "TAD_id": tad,
                "display_name": display,
                "mean_B22": b22.mean(),
                "SD_B22": b22.std(ddof=1),
                "SE_B22": b22.std(ddof=1) / math.sqrt(len(b22)),
                "replicas_used": ";".join(JOBS),
            }
        )
    prov = pd.DataFrame(rows)
    prov.to_csv(OUT / "Fig4B_pair_provenance.csv", index=False)

    owns = axes is None
    if owns:
        fig, axs = plt.subplots(1, 2, figsize=(5.4, 2.35), sharey=False)
        axes = list(axs)
    for ax, internal_label, pair_label, r, curves, b22, color in [
        (axes[0], "Weak interaction", "FZD9 RG/RGG – ZNF165 AHD", weak_r, weak_curves, weak_b22, COLORS["grey"]),
        (axes[1], "Strong interaction", "SYNCRIP RG/RGG – FOXO6 AHD", strong_r, strong_curves, strong_b22, COLORS["blue"]),
    ]:
        mean = curves.mean(axis=0)
        sd = curves.std(axis=0, ddof=1)
        ax.plot(r, mean, color=color, lw=1.2)
        ax.fill_between(r, mean - sd, mean + sd, color=color, alpha=0.22, lw=0)
        ax.axhline(1.0, color=COLORS["black"], lw=0.7, ls=":")
        ax.set_xlim(0, 10)
        ax.set_ylim(bottom=0)
        ax.set_xlabel("r (nm)")
        ax.text(0.03, 0.96, internal_label, transform=ax.transAxes, ha="left", va="top", fontsize=7, fontweight="bold")
        ax.text(0.03, 0.87, pair_label.replace(" – ", " –\n"), transform=ax.transAxes, ha="left", va="top", fontsize=5.4)
        ax.text(0.78, 0.05, f"n = 5\nB22 = {b22.mean():.0f}\nSD = {b22.std(ddof=1):.0f}", transform=ax.transAxes, ha="right", va="bottom", fontsize=7)
        style_axes(ax)
    axes[0].set_ylabel("g(r)")
    panel_label(axes[0], "B")
    if owns:
        fig.tight_layout()
        save(fig, "Fig4B_rdfs")
    return prov


def heatmap(ax: plt.Axes, title: str, include_note: bool = True) -> pd.DataFrame:
    obs = matrix(TASK5 / "sequence_oriented_observed_matrix.csv")
    exp = matrix(TASK5 / "sequence_oriented_expected_matrix.csv")
    enr = matrix(TASK5 / "sequence_oriented_log2_enrichment_matrix.csv")
    boot = read_csv(TASK5 / "sequence_oriented_bootstrap.csv")
    rows = []
    for rgg in CLASSES:
        for tad in CLASSES:
            val = float(enr.loc[rgg, tad])
            hit = boot[(boot["RGG_class"] == rgg) & (boot["TAD_class"] == tad)]
            rows.append(
                {
                    "RGG_class": rgg,
                    "TAD_class": tad,
                    "observed": float(obs.loc[rgg, tad]),
                    "expected": float(exp.loc[rgg, tad]),
                    "log2_enrichment": val,
                    "display_annotation": "NA" if not np.isfinite(val) else f"{val:.2f}",
                    "bootstrap_CI_low": float(hit["CI_low"].iloc[0]) if not hit.empty else np.nan,
                    "bootstrap_CI_high": float(hit["CI_high"].iloc[0]) if not hit.empty else np.nan,
                }
            )
    pd.DataFrame(rows).to_csv(OUT / "Fig4C_display_values.csv", index=False)
    plot = enr.to_numpy(float)
    plot = np.ma.masked_invalid(plot)

    cmap = mpl.colormaps["RdBu_r"].copy()
    cmap.set_bad("#F2F2F2")  # neutral box for NA

    edges = np.arange(5) - 0.5

    im = ax.pcolormesh(
        edges,
        edges,
        np.clip(plot, -2, 2),
        cmap=cmap,
        vmin=-2,
        vmax=2,
        shading="flat",
        edgecolors="white",
        linewidth=0.8,
        antialiased=False,
        snap=True,
    )

    ax.set_xlim(-0.5, 3.5)
    ax.set_ylim(3.5, -0.5)
    ax.set_aspect("equal")
    ax.set_xticks(range(4))
    ax.set_yticks(range(4))
    ax.set_xticklabels(CLASSES, rotation=35, ha="right")
    ax.set_yticklabels(CLASSES)
    ax.set_xlabel("AHD residue class")
    ax.set_ylabel("RG/RGG residue class")
    ax.set_title(title)
    ax.set_xticks(np.arange(-0.5, 4, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, 4, 1), minor=True)
    ax.grid(which="minor", color="white", linestyle="-", linewidth=0.8)
    ax.tick_params(which="minor", bottom=False, left=False)
    for i, rgg in enumerate(CLASSES):
        for j, tad in enumerate(CLASSES):
            val = float(enr.loc[rgg, tad])
            ax.text(j, i, "NA" if not np.isfinite(val) else f"{val:.2f}", ha="center", va="center", fontsize=7.2)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.07)
    cbar = plt.colorbar(im, cax=cax)
    cbar.set_label(r"$log_2$(observed/expected)")
    cbar.ax.tick_params(labelsize=7)
    if include_note:
        ax.text(0.0, -0.42, "NA, no net attractive contribution", transform=ax.transAxes, ha="left", va="top", fontsize=7)
    for sp in ax.spines.values():
        sp.set_linewidth(0.7)
    return pd.DataFrame(rows)


def build_fig4c(ax: plt.Axes | None = None, label: bool = False) -> None:
    owns = ax is None
    if owns:
        fig, ax = plt.subplots(figsize=(3.2, 2.85))
    heatmap(ax, "Attractive interaction-energy enrichment", include_note=True)
    if label or owns:
        panel_label(ax, "C", x=-0.28)
    if owns:
        fig.tight_layout()
        save(fig, "Fig4C_heatmap")


def stats_tables() -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    corr = read_csv(TASK3 / "electrostatic_correlations_917_vs_heldout.csv")
    d917 = corr[(corr["dataset"] == "917 feature-guided") & (corr["feature"] == OCN_COL)].iloc[0]
    hind = corr[(corr["dataset"] == "held-out sequence-exclusive") & (corr["feature"] == OCN_COL)].iloc[0]
    rows = [
        {
            "dataset": "original 917 feature-guided panel",
            "n": int(d917["n"]),
            "Pearson_r": float(d917["Pearson_r"]),
            "Spearman_rho": float(d917["Spearman_rho"]),
            "x_descriptor_name": "Negative net-charge product (−NCP)",
            "formula": "−NCP = −Q_RGG × Q_TAD",
            "historical_name": "Opposite Charge Number (OCN)",
            "sampling_description": "feature-guided 917-pair simulation panel",
        }
    ]
    out = pd.DataFrame(rows)
    out.to_csv(OUT / "Fig4D_statistics.csv", index=False)
    return out, d917, hind


def scatter_ncp(
    ax: plt.Axes,
    df: pd.DataFrame,
    stat_row: pd.Series,
    color: str,
    title: str | None = None,
    color_column: str | None = None,
    colorbar_label: str | None = None,
    exact_annotation: bool = False,
) -> None:
    columns = [OCN_COL, Y_COL] + ([color_column] if color_column else [])
    sub = df[columns].replace([np.inf, -np.inf], np.nan).dropna()
    x = sub[OCN_COL].to_numpy(float)
    y = sub[Y_COL].to_numpy(float)
    if color_column:
        scatter = ax.scatter(
            x,
            y,
            s=MARKER_SIZE,
            alpha=0.60,
            c=sub[color_column].to_numpy(float),
            cmap="viridis",
            edgecolors="none",
            rasterized=True,
        )
        cbar = ax.figure.colorbar(scatter, ax=ax, pad=0.03)
        cbar.set_label(colorbar_label or color_column)
        cbar.ax.tick_params(labelsize=7)
    else:
        ax.scatter(x, y, s=MARKER_SIZE, alpha=0.60, color=color, edgecolors="none", rasterized=True)
    slope, intercept, *_ = stats.linregress(x, y)
    xx = np.linspace(x.min(), x.max(), 200)
    ax.plot(xx, intercept + slope * xx, color=COLORS["black"], lw=1.0)
    if exact_annotation:
        annotation = (
            f"Pearson r = {float(stat_row['Pearson_r']):.3f}\n"
            f"Spearman $\\rho$ = {float(stat_row['Spearman_rho']):.3f}"
        )
    else:
        annotation = f"n = {int(stat_row['n']):,}\nPearson r = {float(stat_row['Pearson_r']):.2f}\nSpearman rho = {float(stat_row['Spearman_rho']):.2f}"
    ax.text(0.03, 0.97, annotation, transform=ax.transAxes, ha="left", va="top", fontsize=7)
    if title:
        ax.set_title(title)
    ax.set_xlabel("−NCP" if exact_annotation else "Negative net-charge product (−NCP)")
    ax.set_ylabel(r"$log_{10}(-B_{22})$")
    style_axes(ax)


def build_fig4d(d917: pd.Series, ax: plt.Axes | None = None, label: bool = False) -> None:
    owns = ax is None
    if owns:
        fig, ax = plt.subplots(figsize=(3.2, 2.55))
    df = read_csv(TASK3 / "fig4_917_sampling_membership.csv")
    scatter_ncp(
        ax,
        df,
        d917,
        COLORS["blue"],
        title="Training set (n = 917 attractive pairs)",
        color_column="Average_lambda",
        colorbar_label="Mean λ",
        exact_annotation=True,
    )
    if label or owns:
        panel_label(ax, "D")
    if owns:
        fig.tight_layout()
        save(fig, "Fig4D_ncp_scatter")


def build_s4_scatter(ax: plt.Axes, df: pd.DataFrame, xcol: str, xlabel: str, title: str, label: str) -> None:
    r, _, n = pearson_spearman(df, x=xcol)
    ax.scatter(df[xcol], df[Y_COL], s=MARKER_SIZE, alpha=0.60, color=COLORS["blue"], edgecolors="none", rasterized=True)
    x = pd.to_numeric(df[xcol], errors="coerce")
    y = pd.to_numeric(df[Y_COL], errors="coerce")
    ok = np.isfinite(x) & np.isfinite(y)
    if ok.sum() >= 2:
        m, b = np.polyfit(x[ok], y[ok], 1)
        xx = np.linspace(float(x[ok].min()), float(x[ok].max()), 100)
        ax.plot(xx, m * xx + b, color=COLORS["black"], lw=1.0)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(r"$log_{10}(-B_{22})$")
    ax.set_title(f"{label}  {title}", loc="left", fontweight="bold")
    ax.text(0.03, 0.97, f"n = {n}\nr = {r:.2f}", transform=ax.transAxes, ha="left", va="top", fontsize=7)
    style_axes(ax)


def ecdf(values: pd.Series) -> tuple[np.ndarray, np.ndarray]:
    x = np.sort(pd.to_numeric(values, errors="coerce").dropna().to_numpy())
    y = np.arange(1, len(x) + 1) / len(x)
    return x, y


def s4_distribution(ax: plt.Axes, datasets: dict[str, pd.DataFrame], feature: str, xlabel: str, title: str, label: str) -> None:
    for name, df in datasets.items():
        x, y = ecdf(df[feature])
        ax.plot(x, y, lw=1.35, color=SET_COLORS[name], label=f"{name} (n={len(df):,})")
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Empirical cumulative fraction")
    ax.set_title(f"{label}  {title}", loc="left", fontweight="bold")
    ax.grid(axis="y", color="#DDDDDD", lw=0.4)
    style_axes(ax)


def s4_bin_coverage(full: pd.DataFrame, final: pd.DataFrame) -> pd.DataFrame:
    bins = np.arange(np.floor(full[OCN_COL].min() / 5) * 5, np.ceil(full[OCN_COL].max() / 5) * 5 + 5, 5)
    full_bin = pd.cut(full[OCN_COL], bins=bins, include_lowest=True, right=False)
    final_bin = pd.cut(final[OCN_COL], bins=bins, include_lowest=True, right=False)
    cc = full_bin.value_counts().sort_index()
    sc = final_bin.value_counts().sort_index()
    rows = []
    for i, interval in enumerate(cc.index, start=1):
        cand = int(cc.loc[interval])
        sel = int(sc.get(interval, 0))
        rows.append(
            {
                "bin_id": i,
                "bin_lower": float(interval.left),
                "bin_upper": float(interval.right),
                "candidate_count": cand,
                "selected_917_count": sel,
                "selection_fraction": sel / cand if cand else np.nan,
                "historical_or_reconstructed_flag": "reconstructed from final pair list; historical per-round bin manifest not found",
            }
        )
    out = pd.DataFrame(rows)
    out.to_csv(OUT / "S4_negative_NCP_bin_coverage.csv", index=False)
    return out


def s4_bin_panel(ax1: plt.Axes, coverage: pd.DataFrame, label: str) -> None:
    centers = (coverage["bin_lower"] + coverage["bin_upper"]) / 2.0
    widths = coverage["bin_upper"] - coverage["bin_lower"]
    ax1.bar(centers, coverage["candidate_count"], width=widths * 0.85, color="#C9C9C9", label="candidate pairs")
    ax1.set_yscale("log")
    ax1.set_xlabel("Negative net-charge product (−NCP)")
    ax1.set_ylabel("Candidate pairs, log scale")
    ax2 = ax1.twinx()
    ax2.plot(centers, coverage["selected_917_count"], color=COLORS["blue"], marker="o", ms=3, lw=1.1, label="selected 917")
    ax2.set_ylabel("Selected pairs")
    ax1.set_title(f"{label}  Reconstructed −NCP-bin coverage", loc="left", fontweight="bold")
    style_axes(ax1)
    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1 + h2, l1 + l2, frameon=False, loc="upper right")


def build_s4i(hind: pd.Series, ax: plt.Axes | None = None) -> None:
    owns = ax is None
    if owns:
        fig, ax = plt.subplots(figsize=(3.2, 2.55))
    held = read_csv(TASK3 / "heldout_23x78_features.csv")
    scatter_ncp(ax, held, hind, COLORS["orange"], "S4I  Independent validation")
    if owns:
        fig.tight_layout()
        save(fig, "S4I_independent_validation")


def save_s4_individuals(d917: pd.Series, hind: pd.Series, coverage: pd.DataFrame, datasets: dict[str, pd.DataFrame]) -> None:
    pilot = read_csv(ROOT / "25_pairs_features_data.csv")
    initial = read_csv(ROOT / "initial_set_features_data.csv")
    specs = [
        ("S4A", lambda ax: build_s4_scatter(ax, pilot, "Charge_Complementarity", "Charge complementarity", "Pilot charge complementarity", "S4A")),
        ("S4B", lambda ax: build_s4_scatter(ax, pilot, "Average_NCPR", "Mean pair NCPR", "Pilot mean pair NCPR", "S4B")),
        ("S4C", lambda ax: build_s4_scatter(ax, pilot, "Average_lambda", "Average λ", "Pilot average λ", "S4C")),
        ("S4D", lambda ax: build_s4_scatter(ax, initial, OCN_COL, "Negative net-charge product (−NCP)", "Early negative net-charge-product relationship", "S4D")),
        ("S4E", lambda ax: s4_distribution(ax, datasets, OCN_COL, "Negative net-charge product (−NCP)", "−NCP distribution", "S4E")),
        ("S4F", lambda ax: s4_distribution(ax, datasets, "Total_Length", "Total pair length", "Total length distribution", "S4F")),
        ("S4G", lambda ax: s4_distribution(ax, datasets, "Average_lambda", "Average λ", "Average λ distribution", "S4G")),
        ("S4H", lambda ax: s4_bin_panel(ax, coverage, "S4H")),
        ("S4I", lambda ax: build_s4i(hind, ax)),
    ]
    for name, func in specs:
        fig, ax = plt.subplots(figsize=(3.05, 2.35))
        func(ax)
        if name in {"S4E", "S4F", "S4G"}:
            ax.legend(frameon=False, loc="best", fontsize=5.8)
        fig.tight_layout()
        stem = "S4I_independent_validation" if name == "S4I" else name
        save(fig, stem)


def assemble_main(d917: pd.Series) -> None:
    fig = plt.figure(figsize=(7.4, 5.25))
    gs = GridSpec(2, 2, figure=fig, wspace=0.42, hspace=0.62, height_ratios=[0.90, 1.08])
    ax_a = fig.add_subplot(gs[0, 0])
    build_fig4a(ax_a, label=True)
    ax_b_parent = fig.add_subplot(gs[0, 1])
    ax_b_parent.axis("off")
    bbox = ax_b_parent.get_position()
    ax_b1 = fig.add_axes([bbox.x0, bbox.y0, bbox.width * 0.47, bbox.height])
    ax_b2 = fig.add_axes([bbox.x0 + bbox.width * 0.53, bbox.y0, bbox.width * 0.47, bbox.height])
    build_fig4b([ax_b1, ax_b2], label_ax=ax_b_parent)
    ax_c = fig.add_subplot(gs[1, 0])
    build_fig4c(ax_c, label=True)
    ax_d = fig.add_subplot(gs[1, 1])
    build_fig4d(d917, ax_d, label=True)
    fig.savefig(OUT / "Figure4_full.png", dpi=600, bbox_inches="tight", pad_inches=0.04)
    fig.savefig(OUT / "Figure4_full.pdf", dpi=1200, bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)


def assemble_s4(hind: pd.Series, coverage: pd.DataFrame, datasets: dict[str, pd.DataFrame]) -> None:
    pilot = read_csv(ROOT / "25_pairs_features_data.csv")
    initial = read_csv(ROOT / "initial_set_features_data.csv")
    fig, axes = plt.subplots(3, 3, figsize=(9.3, 7.4))
    build_s4_scatter(axes[0, 0], pilot, "Charge_Complementarity", "Charge complementarity", "Pilot charge complementarity", "S4A")
    build_s4_scatter(axes[0, 1], pilot, "Average_NCPR", "Mean pair NCPR", "Pilot mean pair NCPR", "S4B")
    build_s4_scatter(axes[0, 2], pilot, "Average_lambda", "Average λ", "Pilot average λ", "S4C")
    build_s4_scatter(axes[1, 0], initial, OCN_COL, "Negative net-charge product (−NCP)", "Early negative net-charge-product relationship", "S4D")
    s4_distribution(axes[1, 1], datasets, OCN_COL, "Negative net-charge product (−NCP)", "−NCP distribution", "S4E")
    s4_distribution(axes[1, 2], datasets, "Total_Length", "Total pair length", "Total length distribution", "S4F")
    s4_distribution(axes[2, 0], datasets, "Average_lambda", "Average λ", "Average λ distribution", "S4G")
    s4_bin_panel(axes[2, 1], coverage, "S4H")
    build_s4i(hind, axes[2, 2])
    axes[1, 1].legend(frameon=False, loc="lower right", fontsize=5.6)
    for ax in [axes[1, 2], axes[2, 0]]:
        leg = ax.get_legend()
        if leg:
            leg.remove()
    fig.subplots_adjust(left=0.06, right=0.96, top=0.95, bottom=0.08, wspace=0.46, hspace=0.72)
    fig.savefig(OUT / "revised_Figure_S4_final.png", dpi=600, bbox_inches="tight", pad_inches=0.04)
    fig.savefig(OUT / "revised_Figure_S4_final.pdf", dpi=1200, bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)


def legends_and_provenance(d917: pd.Series, hind: pd.Series, panel_b_prov: pd.DataFrame) -> None:
    boot = read_csv(TASK3 / "heldout_clustered_bootstrap_correlations.csv")
    cluster = boot[boot["feature"] == OCN_COL].iloc[0]
    cdisplay = pd.read_csv(OUT / "Fig4C_display_values.csv")
    basic_acidic = cdisplay[(cdisplay["RGG_class"] == "Basic") & (cdisplay["TAD_class"] == "Acidic")].iloc[0]
    acidic_acidic = cdisplay[(cdisplay["RGG_class"] == "Acidic") & (cdisplay["TAD_class"] == "Acidic")].iloc[0]
    strong = panel_b_prov.iloc[0]
    weak = panel_b_prov.iloc[1]
    exclusivity = read_csv(TASK3 / "heldout_sequence_exclusivity.csv")
    tads = exclusivity[exclusivity["sequence_type"] == "TAD"]["sequence_id"].tolist()
    rggs = exclusivity[exclusivity["sequence_type"] == "RGG"]["sequence_id"].tolist()

    (OUT / "MAIN_FIG4_LEGEND_DRAFT.md").write_text(
        f"""# Main Figure 4 Legend Draft

Figure 4. Revised R-TAD-RGG interaction landscape and sequence-feature relationships. (A) Corrected B22 interaction landscape for the original 917 simulated R-TAD-RGG pairs; the underlying analysis is unchanged. (B) Representative pH 7.0 RDFs for the historical strong/interacting and weak/non-interacting examples. Lines show mean g(r) across five historical replicas, job_0-job_4, and shaded bands show between-replica SD. (C) Corrected sequence-oriented residue-class attractive interaction-energy enrichment. Rows are RGG residue classes and columns are R-TAD residue classes; expected values are calculated from attractive-energy matrix marginals. Colors show log₂(observed / expected) clipped at ±2, while finite annotations show actual values. Dash indicates no net attractive contribution. (D) Relationship between negative net-charge product (−NCP) and log10(−B22) for the original 917-pair simulation panel. The quantity previously termed OCN is algebraically equivalent to the negative product of the segment net charges, −Q_RGG × Q_TAD, and is therefore referred to here as negative net-charge product (−NCP). The 917-pair set is feature-guided and is not an unbiased random discovery set.
"""
    )
    (OUT / "SUPPLEMENTARY_S4_LEGEND_DRAFT.md").write_text(
        f"""# Supplementary Figure S4 Legend Draft

Supplementary Figure S4. Sampling provenance and independent validation for Figure 4. (S4A-S4D) Historical feature relationships examined during the iterative sampling strategy: charge complementarity, mean pair NCPR, average λ, and the early relationship between B22 and negative net-charge product (−NCP). These panels document sampling-design provenance and are not independent validation. (S4E-S4G) Feature distributions for the complete 231,840-pair candidate landscape, the final feature-guided 917-pair simulation panel, and the independent 1,794-pair sequence-exclusive set. (S4H) Reconstructed −NCP-bin coverage for the final 917-pair panel. Historical per-round bin manifests were not recovered, so counts are reconstructed from the retained final pair list and full candidate landscape. (S4I) Independent validation of the −NCP-B22 relationship in the sequence-exclusive set made from 23 TADs and 78 RGGs, yielding the complete 1,794-pair cross-product with no TAD or RGG sequence overlap with the original 917-pair panel. The independent set has Pearson r={float(hind['Pearson_r']):.3f} and Spearman ρ={float(hind['Spearman_rho']):.3f}, testing generalization outside the feature-guided sampling design.
"""
    )
    (OUT / "FINAL_FIG4_PROVENANCE.md").write_text(
        f"""# Final Figure 4 Revision Provenance

## Terminology

Historical name: Opposite Charge Number (OCN). Revised name: negative net-charge product (−NCP). Exact formula: −NCP = −Q_RGG × Q_TAD. Numerical values are unchanged and the source column `Opposite_Charge_Number` is retained.

## Panel A

Source: `out_fig4/cache/Fig4a_B22_values.csv` from the historical Figure 4 pipeline. Scientifically unchanged: yes. n = {len(read_csv(HIST / 'cache' / 'Fig4a_B22_values.csv'))}.

## Panel B

Strong pair ID: `{STRONG_PAIR}` ({strong['display_name']}). Weak pair ID: `{WEAK_PAIR}` ({weak['display_name']}). RDF source: `{RDF_DIR}`. Exact five replicas: job_0-job_4, corresponding to run_1-run_5 in the stored pH 7.0 RDF pickles. Strong B22 mean ± SD/SE = {strong['mean_B22']:.3f} ± {strong['SD_B22']:.3f}/{strong['SE_B22']:.3f}. Weak B22 mean ± SD/SE = {weak['mean_B22']:.3f} ± {weak['SD_B22']:.3f}/{weak['SE_B22']:.3f}.

## Panel C

Task 5 source files: `sequence_oriented_observed_matrix.csv`, `sequence_oriented_expected_matrix.csv`, `sequence_oriented_log2_enrichment_matrix.csv`, `sequence_oriented_bootstrap.csv`, and `orientation_audit_final.csv`. Exact-sequence orientation method: rows are RGG classes and columns are R-TAD classes. Task 5 corrected 415/917 historical orientation assignments. Acidic-RGG × Acidic-R-TAD is displayed as `—` because observed = {acidic_acidic['observed']} and expected = {acidic_acidic['expected']:.6g}, giving unregularized −Inf. No pseudocount value, including -20.87, is displayed. The heatmap uses square cells only, no hatching, no diagonal shading, and no slanted patterns.

## Panel D

Source dataset: `task3_sampling_bias/fig4_917_sampling_membership.csv`. n = {int(d917['n'])}. Pearson r = {float(d917['Pearson_r']):.6f}. Spearman ρ = {float(d917['Spearman_rho']):.6f}. Historical marker size = {MARKER_SIZE}. Revised marker size = {MARKER_SIZE}.

## S4I

23 TAD IDs source and 78 RGG IDs source: `task3_sampling_bias/heldout_sequence_exclusivity.csv`; exact analyzed pairs: `task3_sampling_bias/heldout_23x78_features.csv`. n = {int(hind['n'])}. Sequence-exclusive: {bool(exclusivity['sequence_exclusive'].all())}. Pearson r = {float(hind['Pearson_r']):.6f}. Spearman ρ = {float(hind['Spearman_rho']):.6f}. Pair-bootstrap Pearson CI = {float(hind['Pearson_CI_low']):.3f}-{float(hind['Pearson_CI_high']):.3f}. Sequence-cluster Pearson CI = {float(cluster['two_way_cluster_Pearson_CI_low']):.3f}-{float(cluster['two_way_cluster_Pearson_CI_high']):.3f}. Marker size = {MARKER_SIZE}. TAD IDs: {', '.join(tads)}. RGG IDs: {', '.join(rggs)}.

## Resolution

All PNG files produced by this script are exported with `dpi=600`. PDFs remain vector wherever possible; only scatter artists are rasterized internally to keep PDF size manageable.
"""
    )

    return basic_acidic, acidic_acidic, strong, weak


def check_pngs_600() -> bool:
    try:
        from PIL import Image
    except Exception:
        return True
    ok = True
    for path in OUT.glob("*.png"):
        with Image.open(path) as im:
            dpi = im.info.get("dpi", (600, 600))
            if dpi[0] < 590 or dpi[1] < 590:
                ok = False
    return ok


def main() -> None:
    apply_style()
    OUT.mkdir(parents=True, exist_ok=True)
    build_fig4a()
    build_fig4b()
    build_fig4c()
    corr = read_csv(TASK3 / "electrostatic_correlations_917_vs_heldout.csv")
    d917 = corr[(corr["dataset"] == "917 feature-guided") & (corr["feature"] == OCN_COL)].iloc[0]
    build_fig4d(d917)
    assemble_main(d917)

    print("Figure-text changes")
    print("A: panel letter A; title 'All attractive training-set pairs'; x 'B₂₂'; y 'Density'")
    print("B: letter B on left only; weak FZD9–ZNF165 left; strong SYNCRIP–FOXO6 right; x 'r (nm)'; y 'g(r)'; duplicate subplot titles removed")
    print("C: panel letter C; x 'AHD residue class'; y 'RG/RGG residue class'; colorbar 'log₂(observed/expected)'; non-finite cells 'NA'")
    print("D: panel letter D; title 'Training set (n = 917 attractive pairs)'; x '−NCP'; y 'log₁₀(−B₂₂)'; Pearson/Spearman annotation at three decimals; colorbar 'Mean λ'")
    print("Outputs")
    for filename in [
        "Fig4A_distribution.pdf",
        "Fig4B_rdfs.pdf",
        "Fig4C_heatmap.pdf",
        "Fig4D_ncp_scatter.pdf",
        "Figure4_full.pdf",
    ]:
        print(OUT / filename)
    print("PNG companions exported at 600 dpi")


if __name__ == "__main__":
    main()
