#!/usr/bin/env python3
from __future__ import annotations
import argparse
from pathlib import Path
from typing import Iterable, Tuple
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt

# ---------- Project paths ----------
def _resolve_project_root() -> Path:
    here = Path(__file__).resolve().parent
    return here.parent if here.name == "scripts" else here

PROJECT_ROOT = _resolve_project_root()
INPUT_FASTA_DIR = PROJECT_ROOT / "inputs" / "fasta"
INPUT_TABLES_DIR = PROJECT_ROOT / "inputs" / "tables"
INPUT_META_DIR = PROJECT_ROOT / "inputs" / "metadata"
OUTPUT_MAIN_DIR = PROJECT_ROOT / "outputs" / "main"
OUTPUT_CACHE_DIR = PROJECT_ROOT / "outputs" / "cache"
OUTPUT_MAIN_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_CACHE_DIR.mkdir(parents=True, exist_ok=True)

# ---------- Inputs ----------
FILE_IDR_FASTA     = INPUT_FASTA_DIR / "modified_headers_disordered_regions.fasta"
FILE_RGG_FASTA     = INPUT_FASTA_DIR / "RGG_disordered_regions.fasta"
FILE_REGIONS_DATA  = INPUT_TABLES_DIR / "combined_rg_regions_ll_7_data.csv"
FILE_REGIONS_FASTA = INPUT_FASTA_DIR / "combined_rg_regions_ll_7_all.fasta"
FILE_RBP_LIST      = INPUT_META_DIR / "rbp_rgs_list.txt"

# ---------- Outputs ----------
OUT_A = OUTPUT_MAIN_DIR / "fig3A_rg_chance_cumulative"
OUT_C = OUTPUT_MAIN_DIR / "fig3C_hexbin_len_vs_numrg_charge"
OUT_D1 = OUTPUT_MAIN_DIR / "fig3D_left_charge_vs_len_rbp_highlight"
OUT_D2 = OUTPUT_MAIN_DIR / "fig3D_right_numrg_vs_len_rbp_highlight"

# ---------- Style ----------
DPI_SAVE = 600
FIGSIZE_PANEL = (2.4, 2.3)
FIGSIZE_PANEL_A = (3.6, 2.3)

DPI_SAVE = 600
COLOR_GREY = "#8F8F8F"   # darker print-safe grey
COLOR_GREY_LIGHT = "#9A9A9A"
COLOR_BLUE = "#0072B2"
COLOR_BLACK = "#222222"

# ---------- Typography / export settings ----------
FONT_FAMILY = "Arial"
FONT_SIZE_BASE = 7.0
FONT_SIZE_LABEL = 7.0
FONT_SIZE_TICK = 6.0
FONT_SIZE_LEGEND = 6.5
FONT_SIZE_COLORBAR = 6.5
FONT_SIZE_TITLE = 7.0

mpl.rcParams.update({
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "font.family": FONT_FAMILY,
    "font.size": FONT_SIZE_BASE,
    "axes.labelsize": FONT_SIZE_LABEL,
    "axes.titlesize": FONT_SIZE_TITLE,
    "xtick.labelsize": FONT_SIZE_TICK,
    "ytick.labelsize": FONT_SIZE_TICK,
    "legend.fontsize": FONT_SIZE_LEGEND,
    "figure.dpi": DPI_SAVE,
    "savefig.dpi": DPI_SAVE,
})

def _stem(path: str | Path) -> str:
    return Path(path).with_suffix("").name

def style_axes(ax, xlabel: str = "", ylabel: str = "") -> None:
    ax.set_xlabel(xlabel, fontsize=FONT_SIZE_LABEL)
    ax.set_ylabel(ylabel, fontsize=FONT_SIZE_LABEL)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(0.6)
    ax.spines["bottom"].set_linewidth(0.6)
    ax.tick_params(width=0.5, length=3, labelsize=FONT_SIZE_TICK)

def style_colorbar(cbar, label: str = "") -> None:
    cbar.set_label(label, fontsize=FONT_SIZE_LABEL)
    cbar.outline.set_linewidth(0.5)
    cbar.ax.tick_params(width=0.5, length=3, labelsize=FONT_SIZE_COLORBAR)

def style_legend(leg, size: float = FONT_SIZE_LEGEND) -> None:
    if leg is None:
        return
    leg.get_frame().set_linewidth(0.5)
    for t in leg.get_texts():
        t.set_fontsize(size)

def save_scatter_data(prefix: str, df: pd.DataFrame) -> None:
    out = OUTPUT_CACHE_DIR / f"{prefix}_data.tsv"
    df.to_csv(out, sep="\t", index=False)

def read_fasta_records(path: Path) -> Iterable[Tuple[str, str]]:
    header = None
    seq = []
    with path.open() as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if header is not None:
                    yield header, "".join(seq)
                header = line[1:].strip()
                seq = []
            else:
                seq.append(line)
        if header is not None:
            yield header, "".join(seq)

def count_rg_per_record(path: Path) -> pd.DataFrame:
    rows = []
    for header, seq in read_fasta_records(path):
        seq = seq.upper()
        n_rg = seq.count("RG")
        per100 = (n_rg / max(len(seq), 1)) * 100.0
        rows.append({"Header": header, "Length": len(seq), "num_rg": n_rg, "rg_per_100aa": per100})
    return pd.DataFrame(rows)

def read_regions_from_csv_and_fasta(csv_path: Path, fasta_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    required = {"Header", "Length", "Charge"}
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise KeyError(f"CSV missing required columns: {missing}. Found {list(df.columns)}")
    rows = []
    for header, seq in read_fasta_records(fasta_path):
        rows.append({"Header": header, "num_rg": seq.upper().count("RG")})
    df_rg = pd.DataFrame(rows)
    merged = df.merge(df_rg, on="Header", how="inner")
    for col in ["Length", "Charge", "num_rg"]:
        merged[col] = pd.to_numeric(merged[col], errors="coerce")
    merged = merged.dropna(subset=["Length", "Charge", "num_rg"])
    return merged[["Header", "Length", "Charge", "num_rg"]].copy()

def read_rbp_headers(path: Path) -> set[str]:
    if not path.is_file():
        return set()
    return {line.strip() for line in path.read_text().splitlines() if line.strip()}

# ---------- Panel A ----------
def plot_rg_chance_cumulative(idr_fasta: Path, rgg_fasta: Path, outfile: Path) -> Tuple[Path, Path]:
    idr = count_rg_per_record(idr_fasta)
    rgg = count_rg_per_record(rgg_fasta)

    idr_x = np.sort(idr["rg_per_100aa"].to_numpy())
    idr_y = 100.0 * np.arange(1, len(idr_x) + 1) / max(len(idr_x), 1)
    rgg_x = np.sort(rgg["rg_per_100aa"].to_numpy())
    rgg_y = 100.0 * np.arange(1, len(rgg_x) + 1) / max(len(rgg_x), 1)

    mean_rg_len_idr = idr.loc[idr["num_rg"] > 0, "Length"].sum() / max(idr.loc[idr["num_rg"] > 0, "num_rg"].sum(), 1)
    mean_rg_len_rgg = rgg.loc[rgg["num_rg"] > 0, "Length"].sum() / max(rgg.loc[rgg["num_rg"] > 0, "num_rg"].sum(), 1)
    x_idr = 100.0 / mean_rg_len_idr if np.isfinite(mean_rg_len_idr) and mean_rg_len_idr > 0 else np.nan
    x_rgg = 100.0 / mean_rg_len_rgg if np.isfinite(mean_rg_len_rgg) and mean_rg_len_rgg > 0 else np.nan

    fig, ax = plt.subplots(figsize=FIGSIZE_PANEL_A)
    ax.plot(idr_x, idr_y, color=COLOR_GREY, lw=1.0, label="All disordered regions")
    ax.plot(rgg_x, rgg_y, color=COLOR_BLUE, lw=1.0, label="RG/RGG proteins")
    if np.isfinite(x_idr):
        ax.axvline(x_idr, color=COLOR_GREY, ls="--", lw=0.8, label=f"RG chance~1 per {int(round(mean_rg_len_idr))} aa")
    if np.isfinite(x_rgg):
        ax.axvline(x_rgg, color=COLOR_BLUE, ls="--", lw=0.8, label=f"RG chance~1 per {int(round(mean_rg_len_rgg))} aa")
    style_axes(ax, xlabel="RG dipeptides per 100 aa", ylabel=f"Cumulative % of Regions")
    leg = ax.legend(loc="lower right", frameon=True)
    style_legend(leg, size=FONT_SIZE_LEGEND)
    fig.tight_layout()
    pdf_path = outfile.with_suffix(".pdf")
    png_path = outfile.with_suffix(".png")
    fig.savefig(pdf_path, bbox_inches="tight")
    fig.savefig(png_path, dpi=DPI_SAVE, bbox_inches="tight")
    plt.close(fig)

    cache = OUTPUT_CACHE_DIR / "fig3A_rg_chance_cumulative_data.tsv"
    pd.DataFrame({
        "dataset": (["All disordered regions"] * len(idr_x)) + (["RG/RGG proteins"] * len(rgg_x)),
        "x_rg_per_100aa": np.concatenate([idr_x, rgg_x]),
        "y_cdf": np.concatenate([idr_y, rgg_y]),
    }).to_csv(cache, sep="\t", index=False)
    return pdf_path, png_path

# ---------- Panel C ----------
def fig_hexbin_len_vs_numrg_charge(df: pd.DataFrame, outfile: Path):
    save_scatter_data(prefix=_stem(outfile),
                      df=df.rename(columns={"Length":"x_length","num_rg":"y_num_rg","Charge":"val_charge"})[["Header","x_length","y_num_rg","val_charge"]])

    x = df["Length"].to_numpy()
    y = df["num_rg"].to_numpy()
    c = df["Charge"].to_numpy()

    fig, ax = plt.subplots(figsize=FIGSIZE_PANEL)
    hb = ax.hexbin(x, y, C=c, reduce_C_function=np.mean, gridsize=35, cmap="viridis")
    cbar = fig.colorbar(hb, ax=ax, fraction=0.05, pad=0.02)
    style_colorbar(cbar, label="Mean charge per bin")
    style_axes(ax, xlabel="RG region length (aa)", ylabel="RG motifs per region")
    ax.grid(alpha=0.25, linestyle=":", color=COLOR_GREY_LIGHT)
    fig.tight_layout()
    pdf_path = outfile.with_suffix(".pdf")
    png_path = outfile.with_suffix(".png")
    fig.savefig(pdf_path, dpi=DPI_SAVE, bbox_inches="tight")
    fig.savefig(png_path, dpi=DPI_SAVE, bbox_inches="tight")
    plt.close(fig)
    return pdf_path, png_path

# ---------- Panel D ----------
def scatter_len_charge_highlight_rbps(df: pd.DataFrame, rbp_headers: set[str], outfile: Path):
    data = df.copy()
    data["is_rbp"] = data["Header"].isin(rbp_headers)

    save_scatter_data(prefix=_stem(outfile),
                      df=data.rename(columns={"Length":"x_length","Charge":"y_charge"})[["Header","x_length","y_charge","is_rbp"]])

    fig, ax = plt.subplots(figsize=FIGSIZE_PANEL)
    m_other = ~data["is_rbp"]
    ax.scatter(data.loc[m_other, "Length"], data.loc[m_other, "Charge"],
               s=16, alpha=0.78, color=COLOR_GREY, edgecolors="none", label="Other regions")
    m_rbp = data["is_rbp"]
    ax.scatter(data.loc[m_rbp, "Length"], data.loc[m_rbp, "Charge"],
               s=16, alpha=0.92, color=COLOR_BLUE, edgecolors="none", label="RBP regions")
    style_axes(ax, xlabel="RG region length (aa)", ylabel="Charge of RG region")
    leg = ax.legend(loc="best", frameon=True)
    style_legend(leg, size=FONT_SIZE_LEGEND)
    fig.tight_layout()
    pdf_path = outfile.with_suffix(".pdf")
    png_path = outfile.with_suffix(".png")
    fig.savefig(pdf_path, dpi=DPI_SAVE, bbox_inches="tight", pad_inches=0.08)
    fig.savefig(png_path, dpi=DPI_SAVE, bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)
    return pdf_path, png_path

def scatter_numrg_len_highlight_rbps(df: pd.DataFrame, rbp_headers: set[str], outfile: Path):
    data = df.copy()
    data["is_rbp"] = data["Header"].isin(rbp_headers)

    save_scatter_data(prefix=_stem(outfile),
                      df=data.rename(columns={"Length":"x_length","num_rg":"y_num_rg"})[["Header","x_length","y_num_rg","is_rbp"]])

    fig, ax = plt.subplots(figsize=FIGSIZE_PANEL)
    m_other = ~data["is_rbp"]
    ax.scatter(data.loc[m_other, "Length"], data.loc[m_other, "num_rg"],
               s=16, alpha=0.78, color=COLOR_GREY, edgecolors="none", label="Other regions")
    m_rbp = data["is_rbp"]
    ax.scatter(data.loc[m_rbp, "Length"], data.loc[m_rbp, "num_rg"],
               s=16, alpha=0.92, color=COLOR_BLUE, edgecolors="none", label="RBP regions")
    style_axes(ax, xlabel="RG region length (aa)", ylabel="RG motifs per region")
    leg = ax.legend(loc="best", frameon=True)
    style_legend(leg, size=FONT_SIZE_LEGEND)
    fig.tight_layout()
    pdf_path = outfile.with_suffix(".pdf")
    png_path = outfile.with_suffix(".png")
    fig.savefig(pdf_path, dpi=DPI_SAVE, bbox_inches="tight", pad_inches=0.08)
    fig.savefig(png_path, dpi=DPI_SAVE, bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)
    return pdf_path, png_path

def main():
    df = read_regions_from_csv_and_fasta(FILE_REGIONS_DATA, FILE_REGIONS_FASTA)
    rbp_headers = read_rbp_headers(FILE_RBP_LIST)
    generated = []
    generated += list(plot_rg_chance_cumulative(FILE_IDR_FASTA, FILE_RGG_FASTA, OUT_A))
    generated += list(fig_hexbin_len_vs_numrg_charge(df, OUT_C))
    generated += list(scatter_len_charge_highlight_rbps(df, rbp_headers, OUT_D1))
    generated += list(scatter_numrg_len_highlight_rbps(df, rbp_headers, OUT_D2))
    print("Generated:")
    for g in generated:
        print(g)

if __name__ == "__main__":
    main()
