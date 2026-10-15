#!/usr/bin/env python3

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize


# ============================================================
# Paths
#
# Run this script from the fig2 root:
#
#   python scripts/make_fig1A_xrgg_csp_matrix.py
# ============================================================

ROOT = Path.cwd()

INPUT_FILE = ROOT / "inputs" / "tables" / "xrgg csps.xlsx"

OUTPUT_DIR = ROOT / "outputs" / "fig1"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OUT_BASENAME = OUTPUT_DIR / "Fig1A_XRGG_CSP_matrix"


# ============================================================
# Figure settings
# ============================================================

CM_TO_INCH = 1 / 2.54

FIG_WIDTH_CM = 9
FIG_HEIGHT_CM = 5

FONT_SIZE = 7
TICK_SIZE = 7
CELL_TEXT_SIZE = 6.2


# ============================================================
# Panel layout
# ============================================================

PROTEINS = [
    "FOXM1",
    "p53",
    "FOXO4",
    "LEF1",
]

PEPTIDES = [
    "RRGG",
    "KRGG",
    "LRGG",
    "CRGG",
    "ERGG",
]


# ============================================================
# Workbook layout
#
# Zero-based pandas column indices.
#
# Checked against the uploaded xrgg CSP workbook.
#
# The residue-ID column is used to select only genuine peak
# rows and exclude any mean / summary rows in the workbook.
# ============================================================

BLOCKS = {

    "FOXM1": {
        "res_col": 1,
        "peptides": {
            "RRGG": 2,
            "LRGG": 3,
            "KRGG": 4,
        },
    },

    "p53": {
        "res_col": 8,
        "peptides": {
            "CRGG": 9,
            "LRGG": 10,
            "KRGG": 11,
            "RRGG": 12,
        },
    },

    "FOXO4": {
        "res_col": 15,
        "peptides": {
            "CRGG": 16,
            "LRGG": 17,
            "KRGG": 18,
            "RRGG": 19,
        },
    },

    "LEF1": {
        "res_col": 24,
        "peptides": {
            "CRGG": 25,
            "KRGG": 26,
            "RRGG": 27,
        },
    },
}


# ============================================================
# FOXM1 ligand-equivalent normalization
#
# FOXM1 RRGG:
#   measured at 0.58 eq -> leave unchanged
#
# FOXM1 KRGG / LRGG:
#   measured at 1.00 eq -> scale to 0.58 eq
#
# Other TFs are not altered.
# ============================================================

TARGET_EQ_FOXM1 = 0.58

LIGAND_EQ = {
    ("FOXM1", "RRGG"): 0.58,
    ("FOXM1", "KRGG"): 1.00,
    ("FOXM1", "LRGG"): 1.00,
}

NORMALIZE_FOXM1 = True


# ============================================================
# Read workbook
# ============================================================

df = pd.read_excel(
    INPUT_FILE,
    sheet_name="foxm1",
    header=None,
)


# ============================================================
# Storage
# ============================================================

shape = (
    len(PROTEINS),
    len(PEPTIDES),
)

mean_csp_measured = np.full(
    shape,
    np.nan,
    dtype=float,
)

sd_csp_measured = np.full(
    shape,
    np.nan,
    dtype=float,
)

median_csp_measured = np.full(
    shape,
    np.nan,
    dtype=float,
)

n_peaks = np.full(
    shape,
    np.nan,
    dtype=float,
)

mean_csp_plot = np.full(
    shape,
    np.nan,
    dtype=float,
)

ligand_eq_used = np.full(
    shape,
    np.nan,
    dtype=float,
)


# ============================================================
# Extract CSP values
#
# Only rows for which the residue ID begins with "@"
# are retained.
# ============================================================

for i, protein in enumerate(PROTEINS):

    block = BLOCKS[protein]

    res_col = block["res_col"]
    peptide_columns = block["peptides"]

    residue_ids = df.iloc[:, res_col]

    valid_residue_rows = (
        residue_ids
        .astype(str)
        .str.startswith("@")
    )

    for j, peptide in enumerate(PEPTIDES):

        # No numerical dataset for this TF/peptide combination
        if peptide not in peptide_columns:
            continue

        csp_col = peptide_columns[peptide]

        values = pd.to_numeric(
            df.loc[
                valid_residue_rows,
                csp_col,
            ],
            errors="coerce",
        ).dropna()

        if len(values) == 0:
            continue

        measured_mean = values.mean()
        measured_sd = values.std(ddof=1)
        measured_median = values.median()
        n = len(values)

        mean_csp_measured[i, j] = measured_mean
        sd_csp_measured[i, j] = measured_sd
        median_csp_measured[i, j] = measured_median
        n_peaks[i, j] = n

        # --------------------------------------------
        # Default plotted CSP
        # --------------------------------------------

        plot_value = measured_mean
        eq_value = np.nan

        # --------------------------------------------
        # FOXM1-specific 0.58-eq normalization
        # --------------------------------------------

        if protein == "FOXM1":

            eq_value = LIGAND_EQ.get(
                (protein, peptide),
                np.nan,
            )

            if (
                NORMALIZE_FOXM1
                and np.isfinite(eq_value)
                and eq_value > 0
            ):

                plot_value = (
                    measured_mean
                    * TARGET_EQ_FOXM1
                    / eq_value
                )

        mean_csp_plot[i, j] = plot_value
        ligand_eq_used[i, j] = eq_value


# ============================================================
# Relative CSP within each TF
#
# Each row is independently normalized to its largest response:
#
# relative CSP = peptide CSP / max peptide CSP for that TF
#
# This controls only CELL COLOUR.
#
# The actual CSP value is still printed inside each cell.
# ============================================================

relative_csp = np.full_like(
    mean_csp_plot,
    np.nan,
)

for i in range(len(PROTEINS)):

    valid = np.isfinite(
        mean_csp_plot[i]
    )

    if not valid.any():
        continue

    row_max = np.nanmax(
        mean_csp_plot[i]
    )

    if row_max > 0:

        relative_csp[i, valid] = (
            mean_csp_plot[i, valid]
            / row_max
        )


# ============================================================
# Print values for checking
# ============================================================

print("\nValues used for Fig. 1A")
print("=" * 90)

for i, protein in enumerate(PROTEINS):

    for j, peptide in enumerate(PEPTIDES):

        if not np.isfinite(
            mean_csp_measured[i, j]
        ):
            continue

        eq_text = (
            f"{ligand_eq_used[i, j]:.2f}"
            if np.isfinite(ligand_eq_used[i, j])
            else "NA"
        )

        print(
            f"{protein:6s}  "
            f"{peptide:4s}  "
            f"n={int(n_peaks[i, j]):3d}  "
            f"measured={mean_csp_measured[i, j]:.6f}  "
            f"eq={eq_text:>4}  "
            f"plot={mean_csp_plot[i, j]:.6f}  "
            f"relative={relative_csp[i, j]:.3f}"
        )


# ============================================================
# Save source data
# ============================================================

source_rows = []

for i, protein in enumerate(PROTEINS):

    for j, peptide in enumerate(PEPTIDES):

        has_data = np.isfinite(
            mean_csp_measured[i, j]
        )

        source_rows.append(
            {
                "protein": protein,
                "peptide": peptide,

                "n_peaks": (
                    int(n_peaks[i, j])
                    if has_data
                    else np.nan
                ),

                "mean_CSP_measured": (
                    mean_csp_measured[i, j]
                    if has_data
                    else np.nan
                ),

                "SD_CSP_measured": (
                    sd_csp_measured[i, j]
                    if has_data
                    else np.nan
                ),

                "median_CSP_measured": (
                    median_csp_measured[i, j]
                    if has_data
                    else np.nan
                ),

                "ligand_equivalents_measured": (
                    ligand_eq_used[i, j]
                    if np.isfinite(
                        ligand_eq_used[i, j]
                    )
                    else np.nan
                ),

                "mean_CSP_used_for_plot": (
                    mean_csp_plot[i, j]
                    if has_data
                    else np.nan
                ),

                "relative_CSP_within_TF": (
                    relative_csp[i, j]
                    if has_data
                    else np.nan
                ),

                "status": (
                    "measured"
                    if has_data
                    else "not available"
                ),
            }
        )


source_df = pd.DataFrame(
    source_rows
)

SOURCE_CSV = (
    OUTPUT_DIR
    / "Fig1A_XRGG_CSP_matrix_source.csv"
)

source_df.to_csv(
    SOURCE_CSV,
    index=False,
)


# ============================================================
# Matplotlib style
# ============================================================

plt.rcParams.update(
    {
        "font.family": "Arial",
        "font.size": FONT_SIZE,

        "axes.linewidth": 0.7,

        "xtick.major.width": 0.7,
        "ytick.major.width": 0.7,

        "pdf.fonttype": 42,
        "ps.fonttype": 42,

        # Keep text editable in Illustrator
        "svg.fonttype": "none",
    }
)


# ============================================================
# Create figure
# ============================================================

fig, ax = plt.subplots(
    figsize=(
        FIG_WIDTH_CM * CM_TO_INCH,
        FIG_HEIGHT_CM * CM_TO_INCH,
    )
)


# ============================================================
# Continuous colour mapping
#
# IMPORTANT:
#
# The COLOR SCALE is continuous.
#
# However, every matrix cell is rendered as ONE SOLID colour.
# There is no within-cell gradient.
# ============================================================

masked = np.ma.masked_invalid(
    relative_csp
)

cmap = plt.get_cmap(
    "Blues"
).copy()

cmap.set_bad(
    "white"
)

norm = Normalize(
    vmin=0,
    vmax=1,
)


im = ax.imshow(
    masked,

    cmap=cmap,
    norm=norm,

    aspect="auto",

    # Prevent interpolation / blending between cells
    interpolation="nearest",
)


# ============================================================
# Axis labels
# ============================================================

ax.set_xticks(
    np.arange(
        len(PEPTIDES)
    )
)

ax.set_xticklabels(
    [
        rf"$[\mathrm{{{p}}}]_7$"
        for p in PEPTIDES
    ],
    fontsize=TICK_SIZE,
)


ax.set_yticks(
    np.arange(
        len(PROTEINS)
    )
)

ax.set_yticklabels(
    PROTEINS,
    fontsize=TICK_SIZE,
)


ax.tick_params(
    axis="both",
    length=0,
    pad=3,
)


# ============================================================
# Numbers inside cells
#
# These are the actual mean CSP values used for plotting:
#
# FOXM1 KRGG / LRGG:
#     0.58-eq normalized values
#
# FOXM1 RRGG:
#     actual measured 0.58-eq value
#
# Other TFs:
#     measured values
# ============================================================

for i in range(len(PROTEINS)):

    for j in range(len(PEPTIDES)):

        if np.isfinite(
            mean_csp_plot[i, j]
        ):

            value = mean_csp_plot[i, j]
            rel = relative_csp[i, j]

            # White text on darker blue cells
            text_color = (
                "white"
                if rel >= 0.60
                else "black"
            )

            ax.text(
                j,
                i,
                f"{value:.4f}",

                ha="center",
                va="center",

                fontsize=CELL_TEXT_SIZE,
                color=text_color,
            )

        else:

            ax.text(
                j,
                i,
                "—",

                ha="center",
                va="center",

                fontsize=CELL_TEXT_SIZE + 0.4,
                color="0.55",
            )


# ============================================================
# White separators
#
# This makes every cell visually distinct and reinforces the
# appearance of individual solid-colour squares.
# ============================================================

ax.set_xticks(
    np.arange(
        -0.5,
        len(PEPTIDES),
        1,
    ),
    minor=True,
)

ax.set_yticks(
    np.arange(
        -0.5,
        len(PROTEINS),
        1,
    ),
    minor=True,
)


ax.grid(
    which="minor",
    color="white",
    linewidth=1.2,
)


ax.tick_params(
    which="minor",
    bottom=False,
    left=False,
)


# ============================================================
# CONTINUOUS GRADIENT COLOUR KEY
#
# This is intentionally a gradient.
#
# Cells = individual solid colours
# Key   = continuous gradient
# ============================================================

cbar = fig.colorbar(
    im,
    ax=ax,

    orientation="vertical",

    fraction=0.045,
    pad=0.040,
)


cbar.set_ticks(
    [
        0,
        0.5,
        1.0,
    ]
)

cbar.set_ticklabels(
    [
        "0",
        "0.5",
        "1.0",
    ]
)


cbar.ax.tick_params(
    labelsize=6,
    width=0.6,
    length=2,
    pad=2,
)


cbar.outline.set_linewidth(
    0.6
)


cbar.set_label(
    "Relative CSP response",
    fontsize=6.3,
    labelpad=3,
)


# ============================================================
# Clean styling
# ============================================================

for spine in ax.spines.values():
    spine.set_visible(False)

ax.set_xlabel("")
ax.set_ylabel("")


# ============================================================
# Layout
#
# Leave room on the right for the gradient colour key.
# ============================================================

fig.subplots_adjust(
    left=0.17,
    right=0.84,
    bottom=0.20,
    top=0.98,
)


# ============================================================
# Export
#
# Do NOT use bbox_inches="tight" because the requested physical
# panel dimensions are exactly 9 × 5 cm.
# ============================================================

fig.savefig(
    f"{OUT_BASENAME}.png",
    dpi=600,
    facecolor="white",
)

fig.savefig(
    f"{OUT_BASENAME}.pdf",
    facecolor="white",
)

fig.savefig(
    f"{OUT_BASENAME}.svg",
    facecolor="white",
)


plt.close(fig)


# ============================================================
# Output paths
# ============================================================

print("\nSaved:")
print(f"  {OUT_BASENAME}.png")
print(f"  {OUT_BASENAME}.pdf")
print(f"  {OUT_BASENAME}.svg")
print(f"  {SOURCE_CSV}")