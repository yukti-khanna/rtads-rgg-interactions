#!/usr/bin/env python3
"""Build replacement S4D and update only panel D in the proposed S4 A-H.

This script consumes the completed provenance-audit outputs. It does not run
the audit, read trajectories, or calculate B22 values.
"""
from __future__ import annotations

import subprocess
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
import pandas as pd
from PIL import Image
from pypdf import PdfReader, PdfWriter, Transformation


OUT = Path(__file__).resolve().parent
STAGE_COUNTS = OUT / "training_selection_stage_counts.csv"
PAIR_PROVENANCE = OUT / "training_selection_pair_provenance.csv"
REPORT = OUT / "TRAINING_SET_SELECTION_PROVENANCE.md"
BASE_PDF = OUT / "revised_Figure_S4_A-H_proposed.pdf"

S4D_PDF = OUT / "S4D_sampling_stage_reconstruction.pdf"
S4D_PNG = OUT / "S4D_sampling_stage_reconstruction.png"
V2_PDF = OUT / "revised_Figure_S4_A-H_proposed_v2.pdf"
V2_PNG = OUT / "revised_Figure_S4_A-H_proposed_v2.png"
SOURCE_CSV = OUT / "S4D_sampling_stage_reconstruction_source.csv"

BLUE = "#0072B2"
ORANGE = "#D55E00"
BLACK = "#111111"

STAGES = [
    (
        "Feature-guided cohort",
        "Feature-guided\ncohort",
        "Dated training list 2025-02-14",
        "stage_dated_training_list_2025_02_14",
        746,
    ),
    (
        "Expanded cohort",
        "Expanded\ncohort",
        "Dated training list 2025-04-09",
        "stage_dated_training_list_2025_04_09",
        911,
    ),
    (
        "Pre-exclusion cohort",
        "Pre-exclusion\ncohort",
        "Pre-exclusion training list 2025-04-22",
        "stage_pre_exclusion_training_list_2025_04_22",
        1039,
    ),
    (
        "Final training set",
        "Final training\nset",
        "Final training list 2025-05-27",
        "stage_final_training_list_2025_05_27",
        917,
    ),
]


def require(path: Path) -> Path:
    if not path.exists():
        raise FileNotFoundError(path)
    return path


def set_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "font.size": 7,
            "axes.titlesize": 7,
            "axes.labelsize": 6.5,
            "xtick.labelsize": 5.5,
            "ytick.labelsize": 6,
            "axes.linewidth": 0.6,
            "xtick.major.width": 0.5,
            "ytick.major.width": 0.5,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def load_stage_data() -> pd.DataFrame:
    counts = pd.read_csv(require(STAGE_COUNTS))
    provenance = pd.read_csv(require(PAIR_PROVENANCE))
    require(REPORT)

    count_rows = counts.set_index("stage")
    sets: list[set[str]] = []
    rows = []
    for order, (name, tick, audit_stage, membership_col, expected) in enumerate(STAGES, start=1):
        if audit_stage not in count_rows.index:
            raise ValueError(f"Missing audited stage: {audit_stage}")
        audited_n = int(count_rows.loc[audit_stage, "n_unique"])
        ids = set(provenance.loc[provenance[membership_col].eq(True), "pair_id"].astype(str))
        if audited_n != expected or len(ids) != expected:
            raise ValueError(
                f"Stage-count mismatch for {name}: stage table={audited_n}, "
                f"pair provenance={len(ids)}, expected={expected}"
            )
        if sets and order <= 3 and not sets[-1].issubset(ids):
            raise ValueError(f"Audited dated stages are not nested at {name}")
        if order == 4 and not ids.issubset(sets[-1]):
            raise ValueError("Final 917 is not a subset of the audited pre-exclusion cohort")

        added = len(ids - sets[-1]) if sets else len(ids)
        removed = len(sets[-1] - ids) if sets else 0
        rows.append(
            {
                "stage_order": order,
                "publication_stage": name,
                "tick_label": tick.replace("\n", " "),
                "audit_stage": audit_stage,
                "pair_count": len(ids),
                "new_pairs_vs_previous_supported_stage": added,
                "removed_pairs_vs_previous_supported_stage": removed,
                "ordering_source": str(STAGE_COUNTS),
                "membership_source": str(PAIR_PROVENANCE),
            }
        )
        sets.append(ids)

    stage = pd.DataFrame(rows)
    expected_added = [746, 165, 128, 0]
    expected_removed = [0, 0, 0, 122]
    if stage.new_pairs_vs_previous_supported_stage.tolist() != expected_added:
        raise ValueError("Unexpected additions between audited dated stages")
    if stage.removed_pairs_vs_previous_supported_stage.tolist() != expected_removed:
        raise ValueError("Unexpected removals between audited dated stages")
    stage.to_csv(SOURCE_CSV, index=False)
    return stage


def plot_stage_panel(ax: plt.Axes, stage: pd.DataFrame, panel_letter: bool) -> None:
    x = np.arange(len(stage))
    y = stage.pair_count.to_numpy(int)
    ax.plot(x[:3], y[:3], color=BLUE, lw=1.25, marker="o", ms=3.8, zorder=3)
    ax.plot(x[2:], y[2:], color=ORANGE, lw=1.25, marker="o", ms=3.8, zorder=3)
    ax.fill_between(x[:3], 0, y[:3], color=BLUE, alpha=0.08, zorder=1)
    ax.fill_between(x[2:], 0, y[2:], color=ORANGE, alpha=0.07, zorder=1)

    labels = ["746\nbaseline", "911\n(+165)", "1,039\n(+128)", "917\n(−122)"]
    offsets = [(0, 7), (0, 7), (0, 7), (0, -21)]
    for xi, yi, text, offset in zip(x, y, labels, offsets):
        ax.annotate(
            text,
            (xi, yi),
            xytext=offset,
            textcoords="offset points",
            ha="center",
            va="bottom" if offset[1] >= 0 else "top",
            fontsize=5.3,
            color=BLACK,
        )

    ax.set_xticks(x, [s[1] for s in STAGES])
    ax.set_xlim(-0.25, 3.25)
    ax.set_ylim(0, 1175)
    ax.set_ylabel("Cumulative selected pairs")
    ax.set_title("Staged feature-guided construction", loc="left", fontweight="bold", pad=5)
    ax.grid(axis="y", color="#DDDDDD", lw=0.4, zorder=0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis="x", pad=2)
    if panel_letter:
        ax.text(-0.20, 1.08, "D", transform=ax.transAxes, fontsize=9, fontweight="bold", va="bottom")


def save_standalone(stage: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(3.25, 2.45))
    plot_stage_panel(ax, stage, panel_letter=True)
    fig.subplots_adjust(left=0.17, right=0.98, bottom=0.25, top=0.87)
    fig.savefig(S4D_PDF, bbox_inches="tight", pad_inches=0.03)
    fig.savefig(S4D_PNG, dpi=600, bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)


def save_composite_overlay(stage: pd.DataFrame, width_pt: float, height_pt: float, path: Path) -> None:
    fig = plt.figure(figsize=(width_pt / 72.0, height_pt / 72.0))
    fig.patch.set_alpha(0.0)
    # Erase only the previous top-right D cell. Other seven panels remain the
    # original PDF content and are not redrawn.
    fig.add_artist(
        Rectangle(
            (0.75, 0.50),
            0.25,
            0.50,
            transform=fig.transFigure,
            facecolor="white",
            edgecolor="none",
            zorder=1,
        )
    )
    ax = fig.add_axes([0.82, 0.61, 0.17, 0.34], zorder=2)
    plot_stage_panel(ax, stage, panel_letter=False)
    ax.tick_params(axis="x", labelsize=4.7)
    ax.set_ylabel("Cumulative selected pairs", fontsize=5.8)
    ax.set_title("Staged construction", loc="left", fontsize=6.2, pad=4)
    for item in ax.texts:
        item.set_fontsize(min(item.get_fontsize(), 4.9))
    fig.text(0.755, 0.98, "D", fontsize=9, fontweight="bold", ha="left", va="top", zorder=3)
    fig.savefig(path, transparent=True, dpi=300)
    plt.close(fig)


def assemble_v2(stage: pd.DataFrame) -> None:
    reader = PdfReader(require(BASE_PDF))
    if len(reader.pages) != 1:
        raise ValueError("Expected one-page proposed S4 PDF")
    page = reader.pages[0]
    width_pt = float(page.mediabox.width)
    height_pt = float(page.mediabox.height)
    overlay_path = OUT / ".S4D_v2_overlay.pdf"
    save_composite_overlay(stage, width_pt, height_pt, overlay_path)
    overlay_page = PdfReader(overlay_path).pages[0]
    overlay_width = float(overlay_page.mediabox.width)
    overlay_height = float(overlay_page.mediabox.height)
    overlay_page.add_transformation(
        Transformation().scale(width_pt / overlay_width, height_pt / overlay_height)
    )
    page.merge_page(overlay_page)
    writer = PdfWriter()
    writer.add_page(page)
    with V2_PDF.open("wb") as handle:
        writer.write(handle)
    overlay_path.unlink()

    render_prefix = OUT / ".revised_Figure_S4_A-H_proposed_v2_render"
    subprocess.run(
        ["pdftoppm", "-png", "-r", "600", "-singlefile", str(V2_PDF), str(render_prefix)],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    rendered = render_prefix.with_suffix(".png")
    with Image.open(rendered) as image:
        image.save(V2_PNG, dpi=(600, 600))
    rendered.unlink()


def main() -> None:
    set_style()
    stage = load_stage_data()
    save_standalone(stage)
    assemble_v2(stage)

    print("NEW S4D HISTORICAL STAGES:")
    for row in stage.itertuples(index=False):
        delta = (
            f"+{row.new_pairs_vs_previous_supported_stage}"
            if row.new_pairs_vs_previous_supported_stage
            else f"−{row.removed_pairs_vs_previous_supported_stage}"
        )
        print(f"{row.stage_order}. {row.publication_stage}: {row.pair_count} pairs ({delta})")
    print(f"ORDERING ESTABLISHED BY: {STAGE_COUNTS}")
    print(f"MEMBERSHIP/NESTING VERIFIED FROM: {PAIR_PROVENANCE}")
    print("UNRESOLVED CHRONOLOGY: early pilot/sub-round ordering is not represented as a cumulative stage; only nested dated lists are shown")
    print("S4A-C AND S4E-H: reused from the previous proposed PDF without analytical or rendering changes")
    print(f"S4D PDF: {S4D_PDF}")
    print(f"S4D PNG: {S4D_PNG}")
    print(f"UPDATED FULL S4 PDF: {V2_PDF}")
    print(f"UPDATED FULL S4 PNG: {V2_PNG}")


if __name__ == "__main__":
    main()
