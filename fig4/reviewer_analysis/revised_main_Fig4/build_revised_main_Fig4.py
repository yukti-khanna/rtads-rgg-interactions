#!/usr/bin/env python3
"""Build the revised main Figure 4 package for the Cell Reports revision.

Run locally from this directory with:
    python build_revised_main_Fig4.py

The script does not overwrite historical Figure 4 outputs and does not rerun
CALVADOS simulations. It reads existing Figure 4 cache plus completed reviewer
analysis outputs and writes all products beside this script.
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
TASK2 = ROOT / "task2_b22_convergence"
TASK3 = ROOT / "task3_sampling_bias"
TASK5 = ROOT / "task5_panelC_enrichment"
RDF_DIR = Path("/Users/yuktikhanna/calvados/test_rdfs")

PANEL_A_HIST_PNG = HIST / "Figure4_panel_a.png"
PANEL_A_HIST_PDF = HIST / "Figure4_panel_a.pdf"
STRONG_RDF = RDF_DIR / "rdfs_O60506_3_1__A8MYZ6_430_473_20nm.pkl"
WEAK_RDF = RDF_DIR / "rdfs_O00144_1_1_P49910_193_208_20nm.pkl"
STRONG_PAIR = "O60506_3_1__A8MYZ6_430_473"
WEAK_PAIR = "O00144_1_1_P49910_193_208"
RUNS = [f"run_{i}" for i in range(1, 6)]
JOBS = [f"job_{i}" for i in range(5)]
CLASSES = ["Basic", "Acidic", "Aromatic", "Other"]

COLORS = {
    "blue": "#0072B2",
    "orange": "#D55E00",
    "grey": "#7A7A7A",
    "light_grey": "#D3D3D3",
    "black": "#111111",
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
            "legend.fontsize": 7,
            "axes.linewidth": 0.7,
            "xtick.major.width": 0.6,
            "ytick.major.width": 0.6,
            "xtick.direction": "out",
            "ytick.direction": "out",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def style_axes(ax: plt.Axes) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    for spine in ax.spines.values():
        spine.set_linewidth(0.7)
    ax.tick_params(width=0.6)


def save(fig: mpl.figure.Figure, stem: str, dpi: int = 600) -> None:
    fig.savefig(OUT / f"{stem}.png", dpi=dpi, bbox_inches="tight", pad_inches=0.03)
    fig.savefig(OUT / f"{stem}.pdf", bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)


def require(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(path)


def copy_panel_a() -> None:
    require(PANEL_A_HIST_PNG)
    require(PANEL_A_HIST_PDF)
    shutil.copy2(PANEL_A_HIST_PNG, OUT / "Fig4A_B22_landscape.png")
    shutil.copy2(PANEL_A_HIST_PDF, OUT / "Fig4A_B22_landscape.pdf")


def corrected_rdf_for_run(r: np.ndarray, g: np.ndarray, run: str) -> tuple[np.ndarray, np.ndarray, float]:
    sys.path.insert(0, str(ROOT))
    from rdf_b22_functions import corrected_rdfs_b22

    one = pd.DataFrame({run: [np.asarray(r, float), np.asarray(g, float)]})
    rdfs, b22s, _ = corrected_rdfs_b22(one, 7.5, box_size=20.0, delta=2.0)
    entry = rdfs[run]
    return np.asarray(entry["rs"], float), np.asarray(entry["rdfs_corr1"], float), float(b22s[run]["b22_corr1"])


def load_replicate_rdfs(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    require(path)
    raw = pd.read_pickle(path)
    curves = []
    b22s = []
    rs_ref = None
    for run in RUNS:
        if run not in raw.columns:
            raise ValueError(f"{path.name} is missing historical replicate {run}")
        r = np.asarray(raw[run].iloc[0], float)
        g = np.asarray(raw[run].iloc[1], float)
        rs, gr, b22 = corrected_rdf_for_run(r, g, run)
        if rs_ref is None:
            rs_ref = rs
        elif not np.allclose(rs_ref, rs):
            raise ValueError(f"RDF r-grid mismatch in {path.name}")
        curves.append(gr)
        b22s.append(b22)
    return rs_ref, np.vstack(curves), np.asarray(b22s)


def pair_parts(pair_id: str) -> tuple[str, str]:
    if "__" in pair_id:
        rgg, tad = pair_id.split("__", 1)
    else:
        parts = pair_id.split("_")
        rgg = "_".join(parts[:3])
        tad = "_".join(parts[3:])
    return rgg, tad


def panel_b() -> pd.DataFrame:
    strong_r, strong_curves, strong_b22 = load_replicate_rdfs(STRONG_RDF)
    weak_r, weak_curves, weak_b22 = load_replicate_rdfs(WEAK_RDF)

    rows = []
    for role, pair_id, b22 in [
        ("strong/interacting", STRONG_PAIR, strong_b22),
        ("weak/non-interacting", WEAK_PAIR, weak_b22),
    ]:
        rgg, tad = pair_parts(pair_id)
        rows.append(
            {
                "panel_role": role,
                "pair_id": pair_id,
                "RGG_id": rgg,
                "R-TAD_id": tad,
                "historical_pair_name": (
                    "SYNCRIP RGG - FOXO6 R-TAD"
                    if role.startswith("strong")
                    else "FZD9 RGG - ZNF165 R-TAD"
                ),
                "mean_B22": b22.mean(),
                "SD_B22": b22.std(ddof=1),
                "SE_B22": b22.std(ddof=1) / math.sqrt(len(b22)),
                "replicas_used": ";".join(JOBS),
            }
        )
    provenance = pd.DataFrame(rows)
    provenance.to_csv(OUT / "Fig4B_pair_provenance.csv", index=False)

    fig, axes = plt.subplots(1, 2, figsize=(5.3, 2.25), sharey=False)
    for ax, title, r, curves, b22, color in [
        (axes[0], "Strong/interacting", strong_r, strong_curves, strong_b22, COLORS["blue"]),
        (axes[1], "Weak/non-interacting", weak_r, weak_curves, weak_b22, COLORS["grey"]),
    ]:
        mean = curves.mean(axis=0)
        sd = curves.std(axis=0, ddof=1)
        ax.plot(r, mean, color=color, lw=1.2)
        ax.fill_between(r, mean - sd, mean + sd, color=color, alpha=0.22, lw=0)
        ax.axhline(1.0, color=COLORS["black"], lw=0.7, ls=":")
        ax.set_xlim(0, 10)
        ax.set_ylim(bottom=0)
        ax.set_xlabel("r (nm)")
        ax.set_title(title)
        ax.text(
            0.98,
            0.96,
            f"n=5 replicas\nB22={b22.mean():.0f}\nSD={b22.std(ddof=1):.0f}",
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=7,
        )
        style_axes(ax)
    axes[0].set_ylabel("g(r)")
    fig.tight_layout()
    save(fig, "Fig4B_RDF_replicate_uncertainty")
    return provenance


def read_matrix(path: Path) -> pd.DataFrame:
    require(path)
    df = pd.read_csv(path, index_col=0)
    return df.loc[CLASSES, CLASSES]


def panel_c() -> pd.DataFrame:
    obs = read_matrix(TASK5 / "sequence_oriented_observed_matrix.csv")
    exp = read_matrix(TASK5 / "sequence_oriented_expected_matrix.csv")
    enr = read_matrix(TASK5 / "sequence_oriented_log2_enrichment_matrix.csv")
    boot = pd.read_csv(TASK5 / "sequence_oriented_bootstrap.csv")

    display = []
    for rgg in CLASSES:
        for tad in CLASSES:
            val = float(enr.loc[rgg, tad])
            hit = boot[(boot["RGG_class"] == rgg) & (boot["TAD_class"] == tad)]
            low = float(hit["CI_low"].iloc[0]) if not hit.empty else np.nan
            high = float(hit["CI_high"].iloc[0]) if not hit.empty else np.nan
            display.append(
                {
                    "RGG_class": rgg,
                    "TAD_class": tad,
                    "observed": float(obs.loc[rgg, tad]),
                    "expected": float(exp.loc[rgg, tad]),
                    "log2_enrichment": val,
                    "display_annotation": "-" if not np.isfinite(val) else f"{val:.2f}",
                    "bootstrap_CI_low": low,
                    "bootstrap_CI_high": high,
                }
            )
    display_df = pd.DataFrame(display)
    display_df.to_csv(OUT / "Fig4C_display_values.csv", index=False)

    plot = enr.to_numpy(dtype=float)
    plot = np.where(np.isfinite(plot), plot, np.nan)
    fig, ax = plt.subplots(figsize=(3.0, 2.7))
    im = ax.imshow(np.clip(plot, -2, 2), cmap="RdBu_r", vmin=-2, vmax=2, aspect="equal")
    ax.set_xticks(range(4))
    ax.set_yticks(range(4))
    ax.set_xticklabels(CLASSES, rotation=35, ha="right")
    ax.set_yticklabels(CLASSES)
    ax.set_xlabel("R-TAD residue class")
    ax.set_ylabel("RGG residue class")
    ax.set_title("Residue-class attractive interaction-energy enrichment")
    for i, rgg in enumerate(CLASSES):
        for j, tad in enumerate(CLASSES):
            val = float(enr.loc[rgg, tad])
            ax.text(j, i, "-" if not np.isfinite(val) else f"{val:.2f}", ha="center", va="center", fontsize=7.5)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.08)
    cbar = plt.colorbar(im, cax=cax)
    cbar.set_label("log2(observed / expected)")
    cbar.ax.tick_params(labelsize=7)
    ax.text(0.0, -0.43, "-, no net attractive contribution", transform=ax.transAxes, ha="left", va="top", fontsize=7)
    for spine in ax.spines.values():
        spine.set_linewidth(0.7)
    fig.tight_layout()
    save(fig, "Fig4C_corrected_energy_enrichment")
    return display_df


def corr_rows() -> pd.DataFrame:
    corr = pd.read_csv(TASK3 / "electrostatic_correlations_917_vs_heldout.csv")
    boot = pd.read_csv(TASK3 / "heldout_clustered_bootstrap_correlations.csv")
    rows = []
    descriptions = {
        "917 feature-guided": "feature-guided 917-pair simulation panel",
        "held-out sequence-exclusive": "independent sequence-exclusive 1,794-pair complete cross-product",
    }
    for dataset in ["917 feature-guided", "held-out sequence-exclusive"]:
        hit = corr[(corr["dataset"] == dataset) & (corr["feature"] == "Opposite_Charge_Number")].iloc[0]
        cluster = boot[boot["feature"] == "Opposite_Charge_Number"]
        rows.append(
            {
                "dataset": dataset,
                "n": int(hit["n"]),
                "Pearson_r": float(hit["Pearson_r"]),
                "Pearson_CI_low": float(hit["Pearson_CI_low"]),
                "Pearson_CI_high": float(hit["Pearson_CI_high"]),
                "Spearman_rho": float(hit["Spearman_rho"]),
                "sequence_exclusive": dataset.startswith("held-out"),
                "sampling_description": descriptions[dataset],
                "sequence_cluster_Pearson_CI_low": float(cluster["two_way_cluster_Pearson_CI_low"].iloc[0])
                if dataset.startswith("held-out") and not cluster.empty
                else np.nan,
                "sequence_cluster_Pearson_CI_high": float(cluster["two_way_cluster_Pearson_CI_high"].iloc[0])
                if dataset.startswith("held-out") and not cluster.empty
                else np.nan,
            }
        )
    out = pd.DataFrame(rows)
    out.to_csv(OUT / "Fig4D_statistics.csv", index=False)
    return out


def d_source_table(path: Path, label: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    id_col = "pair_id" if "pair_id" in df.columns else "name"
    table = df[[id_col, "Opposite_Charge_Number", "B22_corr1_log10"]].copy()
    table = table.rename(columns={id_col: "pair_id"})
    table["dataset"] = label
    table = table.replace([np.inf, -np.inf], np.nan).dropna()
    return table


def plot_d_axes(axes: list[plt.Axes], stats_df: pd.DataFrame) -> None:
    data = [
        (
            d_source_table(TASK3 / "fig4_917_sampling_membership.csv", "917 feature-guided"),
            "Feature-guided panel",
            COLORS["blue"],
            axes[0],
        ),
        (
            d_source_table(TASK3 / "heldout_23x78_features.csv", "held-out sequence-exclusive"),
            "Independent sequence-exclusive set",
            COLORS["orange"],
            axes[1],
        ),
    ]
    for table, title, color, ax in data:
        x = table["Opposite_Charge_Number"].to_numpy(float)
        y = table["B22_corr1_log10"].to_numpy(float)
        ax.scatter(x, y, s=8, alpha=0.55, color=color, edgecolors="none", rasterized=True)
        slope, intercept, *_ = stats.linregress(x, y)
        xx = np.linspace(x.min(), x.max(), 200)
        ax.plot(xx, intercept + slope * xx, color=COLORS["black"], lw=1.0)
        stat = stats_df[stats_df["dataset"] == table["dataset"].iloc[0]].iloc[0]
        ax.text(
            0.03,
            0.97,
            f"n={int(stat['n']):,}\nPearson r={stat['Pearson_r']:.2f}\nSpearman rho={stat['Spearman_rho']:.2f}",
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=7,
        )
        ax.set_title(title)
        ax.set_xlabel("Net-charge product")
        ax.set_ylabel("log10(-B22)")
        style_axes(ax)


def panel_d(stats_df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(5.4, 2.45), sharey=True)
    plot_d_axes(list(axes), stats_df)
    axes[1].set_ylabel("")
    fig.supxlabel(r"Net-charge product, $-Q_{RGG}\times Q_{R-TAD}$", y=0.01, fontsize=8)
    fig.tight_layout()
    save(fig, "Fig4D_net_charge_product_validation")


def cropped_panel_image(path: Path) -> np.ndarray:
    import matplotlib.image as mpimg

    img = mpimg.imread(path)
    rgb = img[:, :, :3] if img.ndim == 3 else img
    mask = np.any(rgb < 0.985, axis=2) if rgb.ndim == 3 else rgb < 0.985
    ys, xs = np.where(mask)
    if len(xs) == 0 or len(ys) == 0:
        return img
    pad = 25
    y0 = max(int(ys.min()) - pad, 0)
    y1 = min(int(ys.max()) + pad, img.shape[0])
    x0 = max(int(xs.min()) - pad, 0)
    x1 = min(int(xs.max()) + pad, img.shape[1])
    return img[y0:y1, x0:x1]


def assemble_figure(stats_df: pd.DataFrame) -> str:
    fig = plt.figure(figsize=(8.4, 5.8))
    gs = GridSpec(2, 3, figure=fig, width_ratios=[1.02, 1.0, 1.0], height_ratios=[1.0, 1.08], wspace=0.52, hspace=0.70)

    ax_a = fig.add_subplot(gs[0, 0])
    ax_a.imshow(cropped_panel_image(OUT / "Fig4A_B22_landscape.png"))
    ax_a.axis("off")
    ax_a.text(-0.28, 1.06, "A", transform=ax_a.transAxes, fontweight="bold", fontsize=10, va="bottom")

    ax_b_parent = fig.add_subplot(gs[0, 1:])
    ax_b_parent.axis("off")
    bbox = ax_b_parent.get_position()
    b_axes = [
        fig.add_axes([bbox.x0, bbox.y0, bbox.width * 0.47, bbox.height]),
        fig.add_axes([bbox.x0 + bbox.width * 0.53, bbox.y0, bbox.width * 0.47, bbox.height]),
    ]
    strong_r, strong_curves, strong_b22 = load_replicate_rdfs(STRONG_RDF)
    weak_r, weak_curves, weak_b22 = load_replicate_rdfs(WEAK_RDF)
    for ax, title, r, curves, b22, color in [
        (b_axes[0], "Strong/interacting", strong_r, strong_curves, strong_b22, COLORS["blue"]),
        (b_axes[1], "Weak/non-interacting", weak_r, weak_curves, weak_b22, COLORS["grey"]),
    ]:
        mean = curves.mean(axis=0)
        sd = curves.std(axis=0, ddof=1)
        ax.plot(r, mean, color=color, lw=1.1)
        ax.fill_between(r, mean - sd, mean + sd, color=color, alpha=0.22, lw=0)
        ax.axhline(1.0, color=COLORS["black"], lw=0.6, ls=":")
        ax.set_xlim(0, 10)
        ax.set_ylim(bottom=0)
        ax.set_title(title)
        ax.set_xlabel("r (nm)")
        ax.text(0.98, 0.96, f"n=5\nB22={b22.mean():.0f}\nSD={b22.std(ddof=1):.0f}", transform=ax.transAxes, ha="right", va="top", fontsize=6.5)
        style_axes(ax)
    b_axes[0].set_ylabel("g(r)")
    b_axes[1].set_ylabel("")
    ax_b_parent.text(-0.04, 1.06, "B", transform=ax_b_parent.transAxes, fontweight="bold", fontsize=10, va="bottom")

    ax_c = fig.add_subplot(gs[1, 0])
    plot_c_in_axis(ax_c)
    ax_c.text(-0.34, 1.08, "C", transform=ax_c.transAxes, fontweight="bold", fontsize=10, va="bottom")

    d_axes = [fig.add_subplot(gs[1, 1]), fig.add_subplot(gs[1, 2])]
    plot_d_axes(d_axes, stats_df)
    d_axes[1].set_ylabel("")
    d_axes[0].text(-0.28, 1.08, "D", transform=d_axes[0].transAxes, fontweight="bold", fontsize=10, va="bottom")

    fig.savefig(OUT / "revised_main_Figure4.png", dpi=600, bbox_inches="tight", pad_inches=0.04)
    fig.savefig(OUT / "revised_main_Figure4.pdf", bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)
    return "A | B / C | D-left | D-right"


def plot_c_in_axis(ax: plt.Axes) -> None:
    enr = read_matrix(TASK5 / "sequence_oriented_log2_enrichment_matrix.csv")
    plot = enr.to_numpy(dtype=float)
    plot = np.where(np.isfinite(plot), plot, np.nan)
    im = ax.imshow(np.clip(plot, -2, 2), cmap="RdBu_r", vmin=-2, vmax=2, aspect="equal")
    ax.set_xticks(range(4))
    ax.set_yticks(range(4))
    ax.set_xticklabels(CLASSES, rotation=35, ha="right")
    ax.set_yticklabels(CLASSES)
    ax.set_xlabel("R-TAD residue class")
    ax.set_ylabel("RGG residue class")
    ax.set_title("Attractive energy enrichment")
    for i, rgg in enumerate(CLASSES):
        for j, tad in enumerate(CLASSES):
            val = float(enr.loc[rgg, tad])
            ax.text(j, i, "-" if not np.isfinite(val) else f"{val:.2f}", ha="center", va="center", fontsize=6.7)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    cbar = plt.colorbar(im, cax=cax)
    cbar.set_label("log2(obs/exp)", fontsize=7)
    cbar.ax.tick_params(labelsize=6.5)
    ax.text(0.0, -0.48, "-, no net attractive contribution", transform=ax.transAxes, ha="left", va="top", fontsize=6.5)
    for spine in ax.spines.values():
        spine.set_linewidth(0.7)


def write_provenance(stats_df: pd.DataFrame, panel_b_prov: pd.DataFrame) -> None:
    d917 = stats_df[stats_df["dataset"] == "917 feature-guided"].iloc[0]
    dhold = stats_df[stats_df["dataset"] == "held-out sequence-exclusive"].iloc[0]
    text = f"""# Main Figure 4 Provenance

## A

Source files: `{PANEL_A_HIST_PNG}` and `{PANEL_A_HIST_PDF}`. Historical script/cache: `fig4.py` and `out_fig4/cache/Fig4a_B22_values.csv`. Values changed: no. Status: scientifically unchanged.

## B

Exact pair IDs: `{STRONG_PAIR}` for the strong/interacting example and `{WEAK_PAIR}` for the weak/non-interacting example. RDF directory: `{RDF_DIR}`. Exact historical replicas used: `job_0`-`job_4`, corresponding to `run_1`-`run_5` in the legacy RDF pickles. Additional runs, where present, were excluded. The central RDF curve is the five-replica mean, and the shaded band is between-replica SD. B22 uncertainty was computed from the same five stored pH 7.0 RDF replicates using the existing `rdf_b22_functions.py` correction routine. Pair provenance table: `Fig4B_pair_provenance.csv`.

## C

Task 5 source files: `sequence_oriented_observed_matrix.csv`, `sequence_oriented_expected_matrix.csv`, `sequence_oriented_log2_enrichment_matrix.csv`, `sequence_oriented_bootstrap.csv`, and `orientation_audit_final.csv`. Orientation method: sequence-oriented rows are RGG residue classes and columns are R-TAD residue classes. The Acidic-RGG x Acidic-R-TAD cell is shown as a dash because the observed attractive contribution is zero while the expected value is positive, giving unregularized enrichment of negative infinity. The historical pseudocount-like `-20.87` display value is not used.

## D

Task 3 source files: `fig4_917_sampling_membership.csv`, `heldout_23x78_features.csv`, `electrostatic_correlations_917_vs_heldout.csv`, and `heldout_clustered_bootstrap_correlations.csv`. The original set is the feature-guided 917-pair simulation panel. The independent set is the sequence-exclusive 23 TAD x 78 RGG complete cross-product, 1,794 simulated pairs, with no TAD or RGG sequence overlap with the original 917. Correlations for net-charge product: 917 Pearson r={d917['Pearson_r']:.6f}, Spearman rho={d917['Spearman_rho']:.6f}; independent Pearson r={dhold['Pearson_r']:.6f}, Spearman rho={dhold['Spearman_rho']:.6f}. Independent pair-bootstrap Pearson CI={dhold['Pearson_CI_low']:.3f}-{dhold['Pearson_CI_high']:.3f}; sequence-cluster Pearson CI={dhold['sequence_cluster_Pearson_CI_low']:.3f}-{dhold['sequence_cluster_Pearson_CI_high']:.3f}.
"""
    (OUT / "MAIN_FIG4_PROVENANCE.md").write_text(text)


def write_legend(stats_df: pd.DataFrame) -> None:
    d917 = stats_df[stats_df["dataset"] == "917 feature-guided"].iloc[0]
    dhold = stats_df[stats_df["dataset"] == "held-out sequence-exclusive"].iloc[0]
    text = f"""# Main Figure 4 Legend Draft

Figure 4. Simulated R-TAD-RGG interaction landscape and sequence features. (A) B22 interaction landscape for the original 917 simulated R-TAD-RGG pairs. This panel preserves the unchanged historical analysis and corrected B22 distribution. (B) Representative RDFs for the historical strong/interacting and weak/non-interacting example pairs. Lines show the mean pH 7.0 RDF across five historical replicas (`job_0`-`job_4`), and shaded bands show between-replica SD. (C) Corrected sequence-oriented residue-class attractive interaction-energy enrichment, shown as log2(observed / expected), with expected values from attractive-energy matrix marginals. Rows are RGG residue classes and columns are R-TAD residue classes. The color scale is clipped to +/-2 for display, while finite cell annotations show the actual values. Dash indicates no net attractive contribution. (D) Relationship between net-charge product, the quantity implemented previously as OCN, and log10(-B22) in the feature-guided 917-pair simulation panel and in the independent sequence-exclusive 1,794-pair set. The independent set tests generalization outside the feature-guided sampling design. For net-charge product, the 917 panel has Pearson r={d917['Pearson_r']:.3f} and Spearman rho={d917['Spearman_rho']:.3f}; the independent set has Pearson r={dhold['Pearson_r']:.3f} and Spearman rho={dhold['Spearman_rho']:.3f}.
"""
    (OUT / "MAIN_FIG4_LEGEND_DRAFT.md").write_text(text)


def main() -> None:
    apply_style()
    OUT.mkdir(parents=True, exist_ok=True)

    copy_panel_a()
    panel_b_prov = panel_b()
    panel_c()
    stats_df = corr_rows()
    panel_d(stats_df)
    layout = assemble_figure(stats_df)
    write_provenance(stats_df, panel_b_prov)
    write_legend(stats_df)

    print("Final panel order: top A | B; bottom C | D-left | D-right")
    print(f"Final assembled layout: {layout}")
    print("Panel A values changed: no, historical panel copied unchanged")
    print("Panel B replicas used: job_0-job_4 only")
    print("Panel C source: Task 5 corrected sequence-oriented matrices")
    print("Panel D source: Task 3 917 and independent 1,794 outputs")
    print(f"Output directory: {OUT}")


if __name__ == "__main__":
    main()
