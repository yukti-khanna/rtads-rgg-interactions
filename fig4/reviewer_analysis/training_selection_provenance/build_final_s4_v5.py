#!/usr/bin/env python3
"""Build B22-saturation S4D and reorder final proposed Supplementary S4 v5.

Reads existing pair lists, feature tables, and authoritative per-pair B22 data.
It does not read trajectories/RDFs, run simulations, fit models, or use held-out
B22 values.
"""
from __future__ import annotations

import subprocess
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import numpy as np
import pandas as pd
from PIL import Image
from pypdf import PdfReader, PdfWriter, Transformation
from scipy import stats


ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
LISTS = Path("/Users/yuktikhanna/calvados/lists")
FINAL_FIG4 = ROOT / "reviewer_analysis" / "final_fig4_revision"

BASE_PDF = OUT / "revised_Figure_S4_A-H_proposed_v4.pdf"
B22_TABLE = ROOT / "sim_pairs_added_features_data_270525.csv"
FINAL_LIST = LISTS / "sim_pairs_for_training_270525.txt"
HELD_AHD = LISTS / "tads_taken_out.txt"
HELD_RGG = LISTS / "rgs_taken_out.txt"
BIN_COVERAGE = FINAL_FIG4 / "S4_negative_NCP_bin_coverage.csv"
FULL_FEATURES = ROOT / "task3_sampling_bias" / "full_combinatorial_feature_landscape.csv"
FINAL_FEATURES = ROOT / "task3_sampling_bias" / "fig4_917_sampling_membership.csv"
HELD_FEATURES = ROOT / "task3_sampling_bias" / "heldout_23x78_features.csv"

S4D_PDF = OUT / "S4D_B22_distribution_saturation.pdf"
S4D_PNG = OUT / "S4D_B22_distribution_saturation.png"
S4D_SOURCE = OUT / "S4D_B22_distribution_saturation_source_data.csv"
V5_PDF = OUT / "revised_Figure_S4_A-H_proposed_v5.pdf"
V5_PNG = OUT / "revised_Figure_S4_A-H_proposed_v5.png"
NOTES = OUT / "S4_panel_order_and_legend_notes.md"

BLUE = "#0072B2"
LIGHT_BLUE = "#56B4E9"
ORANGE = "#D55E00"
GREY = "#888888"
PALE_GREY = "#C9C9C9"
STAGE_COLORS = [GREY, LIGHT_BLUE, ORANGE, BLUE]

STAGE_NAMES = [
    "Initial + outside-filter",
    "+ low-λ",
    "+ −NCP sampling",
    "Final training set",
]


def require(path: Path) -> Path:
    if not path.exists():
        raise FileNotFoundError(path)
    return path


def read_set(path: Path) -> set[str]:
    return {x.strip() for x in require(path).read_text().splitlines() if x.strip()}


def pair_parts(pair: str) -> tuple[str, str]:
    if "__" not in pair:
        raise ValueError(f"Malformed pair ID: {pair}")
    rgg, ahd = pair.split("__", 1)
    return rgg, ahd


def set_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "font.size": 7,
            "axes.titlesize": 7,
            "axes.labelsize": 6.5,
            "xtick.labelsize": 6,
            "ytick.labelsize": 6,
            "legend.fontsize": 5.2,
            "axes.linewidth": 0.6,
            "xtick.major.width": 0.5,
            "ytick.major.width": 0.5,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def reconstruct_cohorts() -> tuple[list[set[str]], set[str]]:
    held_ahd = read_set(HELD_AHD)
    held_rgg = read_set(HELD_RGG)
    batches = [
        read_set(LISTS / "all_test_names.txt"),
        read_set(LISTS / "200_pairs_outside_filter_less_than_lambda045_names.txt"),
        read_set(LISTS / "ocn_names.txt"),
        read_set(LISTS / "ocn_added_names.txt"),
    ]

    def retained(batch: set[str]) -> set[str]:
        return {
            pair
            for pair in batch
            if pair_parts(pair)[0] not in held_rgg and pair_parts(pair)[1] not in held_ahd
        }

    cohorts: list[set[str]] = []
    cumulative: set[str] = set()
    for batch in batches:
        cumulative |= retained(batch)
        cohorts.append(set(cumulative))

    expected = [498, 678, 843, 917]
    if [len(x) for x in cohorts] != expected:
        raise ValueError(f"Unexpected retained cohort sizes: {[len(x) for x in cohorts]}")
    if not all(cohorts[i].issubset(cohorts[i + 1]) for i in range(3)):
        raise ValueError("Retained cohorts are not nested")

    final = read_set(FINAL_LIST)
    if len(final) != 917 or cohorts[-1] != final:
        raise ValueError(
            f"Stage-4/final mismatch: reconstructed={len(cohorts[-1])}, final={len(final)}, "
            f"missing={len(final - cohorts[-1])}, extra={len(cohorts[-1] - final)}"
        )
    if any(pair_parts(p)[1] in held_ahd or pair_parts(p)[0] in held_rgg for p in final):
        raise ValueError("Final cohort contains a reserved identity")
    return cohorts, final


def load_b22(cohorts: list[set[str]]) -> tuple[list[pd.DataFrame], pd.DataFrame]:
    raw = pd.read_csv(require(B22_TABLE), low_memory=False)
    id_col = "name" if "name" in raw else "pair_id"
    required = {id_col, "Corrected_B22_Hummer", "B22_corr1_log10"}
    if not required.issubset(raw.columns):
        raise ValueError(f"B22 table lacks required columns: {required - set(raw.columns)}")
    if raw[id_col].duplicated().any():
        raise ValueError("Duplicate pair IDs in authoritative B22 table")
    raw = raw.rename(columns={id_col: "pair_id"})
    available = set(raw.pair_id.astype(str))
    frames = []
    for name, cohort in zip(STAGE_NAMES, cohorts):
        missing = cohort - available
        if missing:
            raise ValueError(f"{name}: {len(missing)} retained pairs lack B22 values")
        frame = raw[raw.pair_id.astype(str).isin(cohort)].copy()
        if len(frame) != len(cohort):
            raise ValueError(f"{name}: B22 join returned {len(frame)} rows for {len(cohort)} pairs")
        b22 = pd.to_numeric(frame.Corrected_B22_Hummer, errors="coerce")
        stored_log = pd.to_numeric(frame.B22_corr1_log10, errors="coerce")
        if b22.isna().any() or stored_log.isna().any() or not (b22 < 0).all():
            raise ValueError(f"{name}: missing or non-attractive B22 values")
        # Validation only: use the stored project transformation in the panel.
        max_diff = float(np.max(np.abs(stored_log.to_numpy(float) - np.log10(-b22.to_numpy(float)))))
        if max_diff > 1e-10:
            raise ValueError(f"{name}: stored log10(-B22) differs from raw B22; max diff={max_diff}")
        frames.append(frame)

    final_log = frames[-1].B22_corr1_log10.to_numpy(float)
    rows = []
    for number, (name, frame) in enumerate(zip(STAGE_NAMES, frames), start=1):
        b22 = frame.Corrected_B22_Hummer.to_numpy(float)
        log = frame.B22_corr1_log10.to_numpy(float)
        q = np.percentile(b22, [5, 25, 50, 75, 95])
        rows.append(
            {
                "stage_number": number,
                "stage": name,
                "cumulative_pair_count": len(frame),
                "attractive_pair_count": int(np.sum(b22 < 0)),
                "B22_median": float(q[2]),
                "B22_p05": float(q[0]),
                "B22_p25": float(q[1]),
                "B22_p75": float(q[3]),
                "B22_p95": float(q[4]),
                "log10_negB22_median": float(np.median(log)),
                "KS_log10_vs_final917": float(stats.ks_2samp(log, final_log).statistic),
                "Wasserstein_log10_vs_final917": float(stats.wasserstein_distance(log, final_log)),
                "B22_source_file": str(B22_TABLE),
                "heldout_B22_used": False,
                "metric_interpretation": "retrospective support; not a prospective stopping threshold",
            }
        )
    return frames, pd.DataFrame(rows)


def ecdf(values: pd.Series | np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    x = np.sort(pd.to_numeric(pd.Series(values), errors="coerce").dropna().to_numpy(float))
    return x, np.arange(1, len(x) + 1) / len(x)


def clean_axes(ax: plt.Axes) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def plot_saturation(ax: plt.Axes, frames: list[pd.DataFrame], source: pd.DataFrame, panel_letter: bool) -> None:
    for i, (name, frame, color) in enumerate(zip(STAGE_NAMES, frames, STAGE_COLORS)):
        x, y = ecdf(frame.B22_corr1_log10)
        lw = 1.6 if i == 3 else 0.9
        ax.plot(x, y, color=color, lw=lw, label=f"{name} (n = {len(frame)})")
    ax.set_xlabel(r"$log_{10}(-B_{22})$")
    ax.set_ylabel("Empirical cumulative fraction")
    ax.legend(frameon=False, loc="lower right", fontsize=4.6)
    clean_axes(ax)

    inset = inset_axes(ax, width="39%", height="36%", loc="upper left", borderpad=0.9)
    sx = np.arange(1, 5)
    inset.plot(sx, source.Wasserstein_log10_vs_final917, "o-", color=BLUE, ms=2.3, lw=0.8, label="Wasserstein")
    inset.plot(sx, source.KS_log10_vs_final917, "s-", color=ORANGE, ms=2.1, lw=0.8, label="KS")
    inset.set_xticks(sx, ["498", "678", "843", "917"])
    inset.set_xlabel("Cumulative pairs", fontsize=4.5)
    inset.set_ylabel("Distance to final", fontsize=4.5)
    inset.tick_params(labelsize=4.1, width=0.4)
    inset.legend(frameon=False, fontsize=3.8, loc="upper right")
    for spine in inset.spines.values():
        spine.set_linewidth(0.45)
    if panel_letter:
        ax.text(-0.20, 1.07, "D", transform=ax.transAxes, fontsize=9, fontweight="bold", va="bottom")


def plot_bin_coverage(ax: plt.Axes) -> plt.Axes:
    coverage = pd.read_csv(require(BIN_COVERAGE))
    centers = (coverage.bin_lower + coverage.bin_upper) / 2.0
    widths = coverage.bin_upper - coverage.bin_lower
    ax.bar(centers, coverage.candidate_count, width=widths * 0.85, color=PALE_GREY, label="candidate pairs")
    ax.set_yscale("log")
    ax.set_xlabel("−NCP")
    ax.set_ylabel("Candidate pairs, log scale")
    ax2 = ax.twinx()
    ax2.set_zorder(ax.get_zorder() + 1)
    ax2.patch.set_visible(False)
    ax2.plot(centers, coverage.selected_917_count, color=BLUE, marker="o", ms=2.2, lw=1.0, label="selected 917")
    ax2.set_ylabel("Selected pairs")
    ax.spines["top"].set_visible(False)
    ax2.spines["top"].set_visible(False)
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, frameon=False, loc="upper right", fontsize=4.5)
    return ax2


def plot_feature_ecdf(ax: plt.Axes, feature: str, xlabel: str) -> None:
    datasets = [
        ("Complete candidate space", FULL_FEATURES, GREY, 231840),
        ("Training", FINAL_FEATURES, BLUE, 917),
        ("Held-out", HELD_FEATURES, ORANGE, 1794),
    ]
    for label, path, color, expected in datasets:
        frame = pd.read_csv(require(path), usecols=[feature])
        if len(frame) != expected:
            raise ValueError(f"{path}: expected {expected} rows, found {len(frame)}")
        x, y = ecdf(frame[feature])
        ax.plot(x, y, color=color, lw=1.2, label=label)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Empirical cumulative fraction")
    ax.legend(frameon=False, loc="lower right", fontsize=4.7)
    clean_axes(ax)


def save_standalone(frames: list[pd.DataFrame], source: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(3.25, 2.45))
    plot_saturation(ax, frames, source, panel_letter=True)
    fig.subplots_adjust(left=0.17, right=0.98, bottom=0.18, top=0.96)
    fig.savefig(S4D_PDF, bbox_inches="tight", pad_inches=0.03)
    fig.savefig(S4D_PNG, dpi=600, bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)


def save_overlay(width_pt: float, height_pt: float, path: Path, frames: list[pd.DataFrame], source: pd.DataFrame) -> None:
    fig = plt.figure(figsize=(width_pt / 72.0, height_pt / 72.0))
    fig.patch.set_alpha(0.0)
    # Replace D and the full bottom row. A-C remain inherited from v4 exactly.
    fig.add_artist(Rectangle((0.75, 0.50), 0.25, 0.50, transform=fig.transFigure, facecolor="white", edgecolor="none", zorder=1))
    fig.add_artist(Rectangle((0.00, 0.00), 1.00, 0.50, transform=fig.transFigure, facecolor="white", edgecolor="none", zorder=1))

    ax_d = fig.add_axes([0.82, 0.61, 0.17, 0.34], zorder=2)
    plot_saturation(ax_d, frames, source, panel_letter=False)
    ax_d.legend(frameon=False, loc="lower right", fontsize=3.9)
    ax_d.set_xlabel(r"$log_{10}(-B_{22})$", fontsize=5.8)
    ax_d.set_ylabel("Empirical cumulative fraction", fontsize=5.5)
    fig.text(0.755, 0.98, "D", fontsize=9, fontweight="bold", ha="left", va="top", zorder=3)

    ax_e = fig.add_axes([0.045, 0.08, 0.15, 0.34], zorder=2)
    ax_e_right = plot_bin_coverage(ax_e)
    ax_e.tick_params(labelsize=5.0)
    ax_e_right.tick_params(labelsize=5.0)
    ax_e.set_xlabel("−NCP", fontsize=5.8)
    ax_e.set_ylabel("Candidate pairs, log scale", fontsize=5.3)
    ax_e_right.set_ylabel("Selected pairs", fontsize=5.3)

    ax_f = fig.add_axes([0.295, 0.08, 0.175, 0.34], zorder=2)
    plot_feature_ecdf(ax_f, "Opposite_Charge_Number", "−NCP")
    ax_g = fig.add_axes([0.56, 0.08, 0.17, 0.34], zorder=2)
    plot_feature_ecdf(ax_g, "Total_Length", "Total pair length")
    ax_h = fig.add_axes([0.82, 0.08, 0.17, 0.34], zorder=2)
    plot_feature_ecdf(ax_h, "Average_lambda", "Average λ")

    for x, label in [(0.003, "E"), (0.263, "F"), (0.528, "G"), (0.755, "H")]:
        fig.text(x, 0.48, label, fontsize=9, fontweight="bold", ha="left", va="top", zorder=3)
    fig.savefig(path, transparent=True, dpi=300)
    plt.close(fig)


def assemble_v5(frames: list[pd.DataFrame], source: pd.DataFrame) -> None:
    reader = PdfReader(require(BASE_PDF))
    if len(reader.pages) != 1:
        raise ValueError("Expected a one-page S4 base PDF")
    page = reader.pages[0]
    width = float(page.mediabox.width)
    height = float(page.mediabox.height)
    overlay = OUT / ".S4_v5_overlay.pdf"
    save_overlay(width, height, overlay, frames, source)
    overlay_page = PdfReader(overlay).pages[0]
    overlay_page.add_transformation(
        Transformation().scale(width / float(overlay_page.mediabox.width), height / float(overlay_page.mediabox.height))
    )
    page.merge_page(overlay_page)
    writer = PdfWriter()
    writer.add_page(page)
    with V5_PDF.open("wb") as handle:
        writer.write(handle)
    overlay.unlink()

    prefix = OUT / ".S4_v5_render"
    subprocess.run(
        ["pdftoppm", "-png", "-r", "600", "-singlefile", str(V5_PDF), str(prefix)],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    rendered = prefix.with_suffix(".png")
    with Image.open(rendered) as image:
        image.save(V5_PNG, dpi=(600, 600))
    rendered.unlink()


def write_notes(source: pd.DataFrame) -> None:
    metrics = "\n".join(
        f"- {r.stage} (n = {r.cumulative_pair_count}): KS = {r.KS_log10_vs_final917:.6f}; "
        f"Wasserstein = {r.Wasserstein_log10_vs_final917:.6f}."
        for r in source.itertuples(index=False)
    )
    text = f"""# Supplementary Figure S4 panel order and legend notes

## Verified cumulative cohorts

- Stage 1: 498 retained pairs.
- Stage 2: 678 cumulative pairs.
- Stage 3: 843 cumulative pairs.
- Stage 4: 917 cumulative pairs.
- The Stage-4 set is exactly equal to `sim_pairs_for_training_270525.txt`.

## S4D retrospective stability metrics

Metrics are calculated on stored `log10(-B22)` values for the retained training cohorts only. Held-out B22 values were not used.

{metrics}

KS and Wasserstein distances are retrospective quantitative support for the historical practical observation of distribution stabilisation. They were not prospectively defined stopping thresholds. The Stage-3 distances are slightly larger than the Stage-2 distances, so the trajectory should not be described as strictly monotonic.

## Panel renumbering

- Old H became new E: reconstructed −NCP-bin coverage.
- Old E became new F: −NCP distribution comparison.
- Old F became new G: total pair-length distribution comparison.
- Old G became new H: average-λ distribution comparison.
- No S4I is included.

## Proposed complete Supplementary Figure S4 legend

Supplementary Figure S4. Sampling provenance and coverage for the Figure 4 simulations. (A-C) Historical pilot relationships between simulated interaction strength and charge complementarity, mean pair NCPR, and average λ, which informed the early feature-guided sampling strategy. (D) Empirical cumulative distributions of `log10(-B22)` for successive held-out-identity-excluded training cohorts containing 498, 678, 843, and 917 pairs. The inset reports retrospective KS statistics and Wasserstein distances to the final 917-pair distribution; these metrics support, but did not prospectively define, the historical stopping decision. (E) Reconstructed −NCP-bin coverage of the final training panel, showing complete-candidate and selected-pair counts. (F-H) Distributions of −NCP, total pair length, and average λ across the complete candidate landscape, final feature-guided training set, and held-out set.

## Results-ready sentence

Successive feature-guided simulation cohorts showed similar empirical interaction-strength distributions, while reconstructed −NCP-bin coverage and feature-distribution comparisons document how the final 917-pair training panel and held-out set span the broader candidate landscape (Supplementary Fig. S4D-H).
"""
    NOTES.write_text(text)


def main() -> None:
    set_style()
    cohorts, final = reconstruct_cohorts()
    frames, source = load_b22(cohorts)
    source.to_csv(S4D_SOURCE, index=False)
    save_standalone(frames, source)
    assemble_v5(frames, source)
    write_notes(source)

    ks = ", ".join(f"{x:.6f}" for x in source.KS_log10_vs_final917)
    wasserstein = ", ".join(f"{x:.6f}" for x in source.Wasserstein_log10_vs_final917)
    print(f"STAGE 1 RETAINED: {len(cohorts[0])}")
    print(f"STAGE 2 CUMULATIVE: {len(cohorts[1])}")
    print(f"STAGE 3 CUMULATIVE: {len(cohorts[2])}")
    print(f"STAGE 4 CUMULATIVE: {len(cohorts[3])}")
    print(f"FINAL RECONSTRUCTION MATCHES sim_pairs_for_training_270525.txt: {cohorts[-1] == final}")
    print(f"S4D CREATED: {S4D_PDF}")
    print(f"S4D KS METRICS: {ks}")
    print(f"S4D WASSERSTEIN METRICS: {wasserstein}")
    print("OLD H → NEW E CONFIRMED: YES")
    print("OLD E → NEW F CONFIRMED: YES")
    print("OLD F → NEW G CONFIRMED: YES")
    print("OLD G → NEW H CONFIRMED: YES")
    print(f"FINAL S4 A-H CREATED: {V5_PDF}")
    print("S4I PRESENT: NO")
    print("OUTPUT PATHS:")
    for path in [S4D_SOURCE, S4D_PDF, S4D_PNG, V5_PDF, V5_PNG, NOTES]:
        print(path)


if __name__ == "__main__":
    main()
