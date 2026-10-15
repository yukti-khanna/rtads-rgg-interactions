#!/usr/bin/env python3
"""Presentation-only revision of the final Figure S3 BLAST panel."""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source",
        type=Path,
        default=Path(
            "outputs/reviewer_family_bias/merged_fragments_vs_9seeds/"
            "FigS3_sequence_similarity_source.csv"
        ),
    )
    parser.add_argument(
        "--outdir",
        type=Path,
        default=Path("outputs/reviewer_family_bias/merged_fragments_vs_9seeds"),
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def verify(data: pd.DataFrame) -> dict[str, float | int]:
    required = {"region_id", "blast_hit", "pident", "shorter_sequence_coverage"}
    missing = required - set(data.columns)
    if missing:
        raise ValueError(f"Source data lacks columns: {sorted(missing)}")
    if data.region_id.duplicated().any():
        raise ValueError("Source table contains duplicate region IDs")

    hit = data[data.blast_hit.astype(bool)].copy()
    no_hit = data[~data.blast_hit.astype(bool)]
    if len(data) != 485 or len(hit) != 174 or len(no_hit) != 311:
        raise ValueError(
            f"Unexpected counts: total={len(data)}, hits={len(hit)}, no_hits={len(no_hit)}"
        )
    if hit[["pident", "shorter_sequence_coverage"]].isna().any().any():
        raise ValueError("At least one BLAST-hit row lacks identity or coverage")
    if no_hit[["pident", "shorter_sequence_coverage"]].notna().any().any():
        raise ValueError("At least one no-hit row contains plotted alignment values")
    if not hit.pident.between(0, 100, inclusive="both").all():
        raise ValueError("Percent identity outside 0-100")
    if not hit.shorter_sequence_coverage.between(0, 1, inclusive="both").all():
        raise ValueError("Shorter-sequence coverage outside 0-1")

    broad = hit[hit.shorter_sequence_coverage.ge(0.8)]
    broad_max = float(broad.pident.max())
    if len(broad) != 35 or not np.isclose(broad_max, 57.143):
        raise ValueError(
            f"Unexpected broad-alignment summary: n={len(broad)}, max identity={broad_max}"
        )
    return {
        "total": len(data),
        "hits": len(hit),
        "no_hits": len(no_hit),
        "hit_percent": 100 * len(hit) / len(data),
        "no_hit_percent": 100 * len(no_hit) / len(data),
        "broad": len(broad),
        "broad_percent": 100 * len(broad) / len(data),
        "broad_max_identity": broad_max,
    }


def configure_style():
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "font.size": 7,
            "axes.labelsize": 7,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "axes.linewidth": 0.6,
            "xtick.major.width": 0.6,
            "ytick.major.width": 0.6,
            "xtick.major.size": 3,
            "ytick.major.size": 3,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
            "savefig.facecolor": "white",
            "figure.facecolor": "white",
            "axes.facecolor": "white",
        }
    )


def make_figure(data: pd.DataFrame, counts: dict[str, float | int]):
    configure_style()
    hit = data[data.blast_hit.astype(bool)]
    aligned_color = "#4F86B5"
    no_alignment_color = "#C8CDD2"

    fig = plt.figure(figsize=(3.75, 4.05))
    grid = fig.add_gridspec(
        nrows=2,
        ncols=1,
        height_ratios=[0.52, 3.0],
        hspace=0.20,
        left=0.18,
        right=0.98,
        top=0.95,
        bottom=0.265,
    )

    bar_ax = fig.add_subplot(grid[0])
    bar_ax.barh(0, counts["hits"], height=0.28, color=aligned_color, edgecolor="none")
    bar_ax.barh(
        0,
        counts["no_hits"],
        left=counts["hits"],
        height=0.28,
        color=no_alignment_color,
        edgecolor="none",
    )
    bar_ax.set_xlim(0, counts["total"])
    bar_ax.set_ylim(-0.42, 0.45)
    bar_ax.axis("off")
    legend = [
        Patch(
            facecolor=aligned_color,
            edgecolor="none",
            label=(
                f"BLASTP alignment reported: {counts['hits']}/{counts['total']} "
                f"({counts['hit_percent']:.1f}%)"
            ),
        ),
        Patch(
            facecolor=no_alignment_color,
            edgecolor="none",
            label=(
                f"No BLASTP alignment reported: {counts['no_hits']}/{counts['total']} "
                f"({counts['no_hit_percent']:.1f}%)"
            ),
        ),
    ]
    bar_ax.legend(
        handles=legend,
        loc="lower left",
        bbox_to_anchor=(0.0, 0.56),
        frameon=False,
        fontsize=7,
        handlelength=1.15,
        handleheight=0.75,
        borderaxespad=0,
        labelspacing=0.28,
        handletextpad=0.45,
        ncol=1,
    )

    ax = fig.add_subplot(grid[1])
    ax.scatter(
        hit.pident,
        100 * hit.shorter_sequence_coverage,
        s=11,
        color=aligned_color,
        alpha=0.58,
        edgecolors="none",
        linewidths=0,
    )
    ax.axhline(80, color="#5A5A5A", linestyle=(0, (4, 3)), linewidth=0.75, zorder=0)
    ax.text(
        98.5,
        81.7,
        "80% coverage",
        ha="right",
        va="bottom",
        fontsize=7,
        color="#4A4A4A",
        bbox={"facecolor": "white", "edgecolor": "none", "pad": 0.5},
    )
    ax.set_xlim(-2, 102)
    ax.set_ylim(-2, 102)
    ax.set_xticks(np.arange(0, 101, 20))
    ax.set_yticks(np.arange(0, 101, 20))
    ax.set_xlabel("Nearest-seed sequence identity (%)")
    ax.set_ylabel("Shorter-sequence coverage (%)")
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(False)

    fig.text(
        0.18,
        0.045,
        (
            f"≥80% coverage: {counts['broad']}/{counts['total']} "
            f"({counts['broad_percent']:.1f}%)\n"
            f"Maximum identity in this subset: {counts['broad_max_identity']:.1f}%"
        ),
        ha="left",
        va="bottom",
        fontsize=7,
        color="#222222",
        linespacing=1.25,
    )
    return fig


def main():
    options = parse_args()
    source = options.source.resolve()
    outdir = options.outdir.resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    data = pd.read_csv(source)
    counts = verify(data)

    png = outdir / "FigS3_sequence_similarity_FINAL.png"
    pdf = outdir / "FigS3_sequence_similarity_FINAL.pdf"
    for path in (png, pdf):
        if path.exists() and not options.overwrite:
            raise FileExistsError(f"Refusing to overwrite existing final figure: {path}")

    fig = make_figure(data, counts)
    fig.savefig(png, dpi=600, facecolor="white")
    fig.savefig(pdf, facecolor="white")
    plt.close(fig)

    print("Figure S3 presentation correction complete")
    for key, value in counts.items():
        print(f"{key}: {value}")
    print(f"scatter points plotted: {int(data.blast_hit.sum())}")
    print(f"PNG: {png}")
    print(f"PDF: {pdf}")


if __name__ == "__main__":
    main()
