#!/usr/bin/env python3
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib as mpl

mpl.rcParams["svg.fonttype"] = "none"
import matplotlib.pyplot as plt

def _resolve_project_root() -> Path:
    here = Path(__file__).resolve().parent
    return here.parent if here.name == "scripts" else here

PROJECT_ROOT = _resolve_project_root()
INPUT_TABLES_DIR = PROJECT_ROOT / "inputs" / "tables"
OUTPUT_SUPP_DIR = PROJECT_ROOT / "outputs" / "supplement"
OUTPUT_CACHE_DIR = PROJECT_ROOT / "outputs" / "cache"
OUTPUT_SUPP_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_CACHE_DIR.mkdir(parents=True, exist_ok=True)

IN_XLSX = INPUT_TABLES_DIR / "Table_S4.xlsx"
OUT_PREFIX = OUTPUT_SUPP_DIR / "figS3_rg_compact_tract_properties"

COLOR_BLUE = "#4C92C3"
COLOR_ORANGE = "#F28E2B"

def ecdf(x: np.ndarray):
    x = np.asarray(x)
    x = x[np.isfinite(x)]
    x = np.sort(x)
    if x.size == 0:
        return np.array([]), np.array([])
    y = np.arange(1, x.size + 1) / x.size
    return x, y

def main(in_xlsx: Path = IN_XLSX, out_prefix: Path = OUT_PREFIX):
    # Expected sheets follow your current workbook
    df_tracts = pd.read_excel(in_xlsx, sheet_name="S4A_Selected_RG_Tracts_1008", skiprows=6)
    df_idrs = pd.read_excel(in_xlsx, sheet_name="S4D_Curated_RGG_IDRs_233", skiprows=6)
    link_tract = pd.read_excel(in_xlsx, sheet_name="S4C_RG_Linker_Lengths_1008", skiprows=6)
    link_idrs = pd.read_excel(in_xlsx, sheet_name="S4E_RG_Linker_Lengths_IDRs", skiprows=6)

    # Cache underlying table extracts
    for name, df in {
        "figS3A_selected_tracts": df_tracts,
        "figS3A_curated_idrs": df_idrs,
        "figS3B_selected_linkers": link_tract,
        "figS3B_curated_linkers": link_idrs,
    }.items():
        df.to_csv(OUTPUT_CACHE_DIR / f"{name}.tsv", sep="\t", index=False)

    fig, axes = plt.subplots(2, 2, figsize=(7.0, 4.6))

    # A: RG density
    ax = axes[0, 0]
    max_x = float(np.nanmax([df_tracts["rg_per_100aa"].max(), df_idrs["rg_per_100aa"].max()]))
    bins = np.linspace(0, max_x, 30) if np.isfinite(max_x) and max_x > 0 else 30
    ax.hist(df_idrs["rg_per_100aa"].dropna(), bins=bins, alpha=0.72, label="Curated RG/RGG IDRs (233)", color=COLOR_BLUE)
    ax.hist(df_tracts["rg_per_100aa"].dropna(), bins=bins, alpha=0.72, label="Selected RG tracts (1008)", color=COLOR_ORANGE)
    ax.set_xlabel("RG motifs per 100 aa")
    ax.set_ylabel("Count")
    ax.legend(frameon=True, fontsize=7)
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)

    # B: linker spacing
    ax = axes[0, 1]
    x1, y1 = ecdf(link_idrs["linker_len"])
    x2, y2 = ecdf(link_tract["linker_len"])
    if x1.size:
        ax.plot(x1, y1, label="Curated RG/RGG IDRs", color=COLOR_BLUE, lw=1.0)
    if x2.size:
        ax.plot(x2, y2, label="Selected RG tracts", color=COLOR_ORANGE, lw=1.0)
    ax.set_xlabel("Inter-RG linker length (aa)")
    ax.set_ylabel("Cumulative fraction")
    ax.set_xlim(0, 30)
    ax.legend(frameon=True, fontsize=7)
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)

    # C: charge density
    ax = axes[1, 0]
    ax.hist(df_tracts["charge_per_res"].dropna(), bins=30, color=COLOR_BLUE)
    ax.set_xlabel("Net charge per residue")
    ax.set_ylabel("Count")
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)

    # D: charge vs RG density
    ax = axes[1, 1]
    ax.scatter(df_tracts["charge_per_res"], df_tracts["rg_per_100aa"], s=10, alpha=0.65, color=COLOR_BLUE, edgecolors="none")
    ax.set_xlabel("Net charge per residue")
    ax.set_ylabel("RG motifs per 100 aa")
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)

    fig.tight_layout()
    pdf_path = out_prefix.with_suffix(".pdf")
    png_path = out_prefix.with_suffix(".png")
    fig.savefig(pdf_path, bbox_inches="tight")
    fig.savefig(png_path, dpi=600, bbox_inches="tight")
    fig.savefig(pdf_path.with_suffix(".svg"), bbox_inches="tight")
    plt.close(fig)
    print(pdf_path)
    print(png_path)

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Supplementary Fig. S3 RG compact-tract property panels.")
    ap.add_argument("--table-s4", type=Path, default=IN_XLSX)
    ap.add_argument("--out-prefix", type=Path, default=OUT_PREFIX)
    args = ap.parse_args()
    main(args.table_s4, args.out_prefix)
