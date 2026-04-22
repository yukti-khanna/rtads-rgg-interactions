#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
make_figure4_fixed_v6.py

Build Figure 4 (2×2) and save plotting tables to out/cache/.

Fixes vs earlier broken versions:
- Panel A: has title + legend; LaTeX labels for B22.
- Panel B: uses the SAME pipeline as plot_rdf_simple.py:
    pickle -> corrected_rdfs_b22 -> plot g(r)
  (no guessing / no bogus r-axis).
- Panel C: actually built from energies_{name}_{box}nm_sliced.pkl; skips missing files.
- Panel D: y-axis label is r"$-\log_{10}[B_{22}]$"; color by lambda_TAD; Pearson box top-left.
- Font embedding: pdf/ps fonttype 42.
"""

from __future__ import annotations

import argparse
import importlib.util
import pickle
from pathlib import Path
from typing import Optional, Dict, List

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy import stats


# ============================ Global Style (EXACT) ============================
FONT_SCALE = 1.0
def _fs(pts: float) -> float: return float(pts) * float(FONT_SCALE)

AXIS_LABEL_FONTSIZE = _fs(6.5)
AXIS_TITLE_FONTSIZE = _fs(7.0)
TICK_LABEL_FONTSIZE = _fs(6.0)
LEGEND_FONTSIZE     = _fs(6.0)
FIGSIZE_PANEL       = (3.6,2.3)

AXES_LINEWIDTH = 0.6
TICK_WIDTH     = 0.5
LINE_WIDTH     = 0.8

DPI_SAVE       = 2400
RASTERIZE_HEAVY_ARTISTS = True
RASTER_DPI = 600

_FALLBACK = {
    "blue": "#0072B2",
    "orange": "#E69F00",
    "green": "#009E73",
    "sky": "#56B4E9",
    "vermillion": "#D55E00",
    "purple": "#CC79A7",
    "yellow": "#F0E442",
    "grey": "#7A7A7A",
    "grey_light": "#D3D3D3",
    "black": "#000000",
}
PALETTE: Dict[str, str] = {}

def C(name: str, fallback_key: str) -> str:
    try:
        if name in PALETTE:
            return PALETTE[name]
    except Exception:
        pass
    return _FALLBACK.get(fallback_key, "#000000")

def hide_top_right(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

def style_axes(ax, xlabel: str | None = None, ylabel: str | None = None, title: str | None = None):
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=AXIS_LABEL_FONTSIZE)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=AXIS_LABEL_FONTSIZE)
    if title:
        ax.set_title(title, fontsize=AXIS_TITLE_FONTSIZE)
    for sp in ax.spines.values():
        sp.set_linewidth(AXES_LINEWIDTH)
    ax.tick_params(axis="both", which="major", labelsize=TICK_LABEL_FONTSIZE, width=TICK_WIDTH)
    hide_top_right(ax)

def style_legend(leg, size: float | None = None):
    if leg is None:
        return
    fs = LEGEND_FONTSIZE if size is None else size
    for txt in leg.get_texts():
        txt.set_fontsize(fs)
    if leg.get_title():
        leg.get_title().set_fontsize(fs)

def style_colorbar(cbar, label: str | None = None, ticksize: float | None = None):
    if label:
        cbar.set_label(label, fontsize=(LEGEND_FONTSIZE if ticksize is None else ticksize))
    cbar.ax.tick_params(labelsize=(TICK_LABEL_FONTSIZE if ticksize is None else ticksize), width=TICK_WIDTH)
    cbar.ax.yaxis.get_offset_text().set_size(TICK_LABEL_FONTSIZE if ticksize is None else ticksize)
    return cbar

def apply_mpl_style():
    mpl.rcParams["font.family"] = "sans-serif"
    mpl.rcParams["font.sans-serif"] = ["Arial", "Helvetica", "DejaVu Sans"]
    mpl.rcParams["font.size"] = TICK_LABEL_FONTSIZE
    mpl.rcParams["axes.linewidth"] = AXES_LINEWIDTH
    mpl.rcParams["xtick.major.width"] = TICK_WIDTH
    mpl.rcParams["ytick.major.width"] = TICK_WIDTH
    mpl.rcParams["lines.linewidth"] = LINE_WIDTH
    mpl.rcParams["savefig.dpi"] = RASTER_DPI
    mpl.rcParams["pdf.fonttype"] = 42
    mpl.rcParams["ps.fonttype"] = 42
    mpl.rcParams["svg.fonttype"] = "none"

def label_panel(ax, letter: str):
    ax.text(
        0.0, 1.02, letter,
        transform=ax.transAxes,
        ha="left", va="bottom",
        fontsize=AXIS_TITLE_FONTSIZE,
        fontweight="bold",
    )

# ================================ IO helpers =================================
def ensure_dirs(outdir: Path) -> tuple[Path, Path]:
    outdir = Path(outdir).expanduser().resolve()
    cache = outdir / "cache"
    outdir.mkdir(parents=True, exist_ok=True)
    cache.mkdir(parents=True, exist_ok=True)
    return outdir, cache

def dynamic_import(py_path: Path, module_name: str):
    py_path = Path(py_path).expanduser().resolve()
    spec = importlib.util.spec_from_file_location(module_name, str(py_path))
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not import module from {py_path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

def load_master_table(path: Path) -> pd.DataFrame:
    df = pd.read_csv(Path(path).expanduser().resolve())
    df.columns = [c.strip() for c in df.columns]

    # normalize common alternates
    if "lambda_TAD" not in df.columns:
        for alt in ["lambda_tad", "Lambda_TAD", "Lambda_tad", "lambda_Tad"]:
            if alt in df.columns:
                df = df.rename(columns={alt: "lambda_TAD"})
                break

    required = {"name", "Opposite_Charge_Number", "B22_corr1_log10", "Corrected_B22_Hummer", "lambda_TAD"}
    miss = [c for c in required if c not in df.columns]
    if miss:
        raise KeyError(f"Master CSV missing columns: {miss}")
    return df

def pair_label_from_rdf_filename(p: Path) -> str:
    s = p.stem.replace("rdfs_", "").replace("rdf_", "")
    return s.replace("__", "–")

# =============================== Panel A =====================================
def panel_a(ax: plt.Axes, df: pd.DataFrame, cache: Path):
    vals = df["B22_corr1_log10"].replace([np.inf, -np.inf], np.nan).dropna().to_numpy()
    df[["name", "B22_corr1_log10"]].to_csv(cache / "Fig4a_B22_values.csv", index=False)

    bins = 70 if vals.size >= 300 else 40
    n, bins_hist, patches = ax.hist(
        vals,
        bins=bins,
        color=C("grey_light", "grey_light"),
        edgecolor=C("grey", "grey"),
        linewidth=0.35,
    )
    if RASTERIZE_HEAVY_ARTISTS:
        for p in patches:
            p.set_rasterized(True)

    if vals.size:
        med = float(np.nanmedian(vals))
        p5, p95 = np.nanpercentile(vals, [5, 95])
        ax.axvline(med, color=C("black", "black"), linestyle="--", linewidth=0.7, label=f"Median = {round(med,3)}")
        ax.axvline(p5,  color=C("black", "black"), linestyle=":",  linewidth=0.7, label="5–95%")
        ax.axvline(p95, color=C("black", "black"), linestyle=":",  linewidth=0.7)

    style_axes(ax,
              xlabel=r"$\log_{10}\!\left(-B_{22}\right)$",
              ylabel="Number of Simulated Pairs")
              #title=r"Distribution of $\log_{10}\!\left(-B_{22}\right)$")
    ax.yaxis.set_major_locator(mpl.ticker.MaxNLocator(4))

    

    leg = ax.legend(frameon=False, loc="upper right")
    style_legend(leg)

# =============================== Panel B =====================================
def load_rdf_curves_via_corrected_rdfs(pkl_path: Path, *, last: float, box_nm: float, delta: float) -> dict:
    """
    Exact pipeline from plot_rdf_simple.py:
      pickle -> corrected_rdfs_b22(df,last,box_size,delta) -> rdfs_dict -> arrays
    """
    try:
        from rdf_b22_functions import corrected_rdfs_b22
    except Exception as e:
        raise ImportError(
            "Panel B requires rdf_b22_functions.py (same dependency as plot_rdf_simple.py). "
            "Make sure it is importable in this environment."
        ) from e

    with open(pkl_path, "rb") as f:
        df = pickle.load(f)

    rdfs_dict, b22s_dict, kds_dict = corrected_rdfs_b22(df, last, box_size=box_nm, delta=delta)
    if not rdfs_dict:
        raise ValueError(f"rdfs_dict is empty for {pkl_path}")

    run_key = sorted(rdfs_dict.keys())[0]
    rdf_entry = rdfs_dict[run_key]

    rs = np.asarray(rdf_entry["rs"], dtype=float)
    gr_orig = np.asarray(rdf_entry["rdfs"], dtype=float)
    gr_corr1 = np.asarray(rdf_entry["rdfs_corr1"], dtype=float)
    gr_corr2 = np.asarray(rdf_entry["rdfs_corr2"], dtype=float)

    return {"rs": rs, "orig": gr_orig, "corr1": gr_corr1, "corr2": gr_corr2, "run_key": run_key}

def panel_b0(ax_parent: plt.Axes, rdf_strong: Path, rdf_weak: Path, cache: Path,
            *, box_nm: float, last: float, delta: float):
    """
    Two stacked RDF plots:
      top = Interacting
      bottom = Non-interacting
    """
    ax_parent.axis("off")
    fig = ax_parent.figure
    bbox = ax_parent.get_position()

    pad_y = 0.02 * bbox.height
    mid = bbox.y0 + bbox.height * 0.52
    top_h = bbox.y1 - mid - pad_y
    bot_h = mid - bbox.y0 - pad_y

    ax_top = fig.add_axes([bbox.x0, mid + pad_y/2, bbox.width, top_h])
    ax_bot = fig.add_axes([bbox.x0, bbox.y0, bbox.width, bot_h])

    # --- shrink each axis height ---
    H_SHRINK = 0.80  # tune 0.75–0.90

    # keep top axis anchored to the top, bottom axis anchored to the bottom
    pos = ax_top.get_position()
    new_h = pos.height * H_SHRINK
    ax_top.set_position([pos.x0, pos.y1 - new_h, pos.width, new_h])

    pos = ax_bot.get_position()
    new_h = pos.height * H_SHRINK
    ax_bot.set_position([pos.x0, pos.y0, pos.width, new_h])

    ax_parent.set_ylim(0.0, 10.0)
    RIGHT_TEXT_W = 0.30  # fraction of the axes width reserved for text (tune 0.25–0.35)
    for a in (ax_top, ax_bot):
        pos = a.get_position()
        a.set_position([pos.x0, pos.y0, pos.width * (1 - RIGHT_TEXT_W), pos.height])

    strong = load_rdf_curves_via_corrected_rdfs(rdf_strong, last=last, box_nm=box_nm, delta=delta)
    weak   = load_rdf_curves_via_corrected_rdfs(rdf_weak,   last=last, box_nm=box_nm, delta=delta)

    pd.DataFrame({"r_nm": strong["rs"], "g_orig": strong["orig"], "g_corr1": strong["corr1"], "g_corr2": strong["corr2"]}).to_csv(
        cache / "Fig4b_rdf_interacting.csv", index=False
    )
    pd.DataFrame({"r_nm": weak["rs"], "g_orig": weak["orig"], "g_corr1": weak["corr1"], "g_corr2": weak["corr2"]}).to_csv(
        cache / "Fig4b_rdf_noninteracting.csv", index=False
    )

    strong_pair = pair_label_from_rdf_filename(rdf_strong)
    weak_pair   = pair_label_from_rdf_filename(rdf_weak)

    import re, textwrap

    def _wrap_by_delims(text: str, *, width_chars: int = 30) -> str:
        # 1) force breaks after ":"  (keep the colon)
        text = re.sub(r":\s*", ":\n", text)

        # 2) break on en-dash or hyphen between partners
        #    (handles "A–B", "A – B", "A-B", "A - B")
        text = re.sub(r"\s*[–]\s*", "\n", text)

        # 3) now wrap each line if still too long
        out_lines = []
        for line in [l.strip() for l in text.split("\n") if l.strip()]:
            out_lines.extend(textwrap.wrap(line, width=width_chars) or [line])
        return "\n".join(out_lines)

    def add_right_label(ax, text, *, width_chars=26):
        wrapped = _wrap_by_delims(text, width_chars=width_chars)
        ax.text(
            1.02, 0.5, wrapped,
            transform=ax.transAxes,
            ha="left", va="center",
            fontsize=AXIS_LABEL_FONTSIZE,
            bbox=dict(boxstyle="round,pad=0.25", fc="white", ec=C("grey","grey"), lw=0.1, alpha=0.95),
            clip_on=False,
        )


    def _plot_one(ax, cur, main_color):
        rs = cur["rs"]
        (ln,) = ax.plot(rs, cur["corr1"], color=main_color, lw=1.1, label="RDF")  # capture handle
        if RASTERIZE_HEAVY_ARTISTS:
            ln.set_rasterized(True)
        ax.axhline(1.0, color=C("grey","grey"), ls=":", lw=0.7)

        style_axes(ax, ylabel=r"$g(r)$")   # no title
        ax.set_xlim(0.0, box_nm / 2.0)
        ax.set_ylim(0.0, 10.0)
        return ln

    ln_top = _plot_one(ax_top, strong, C("blue","blue"))
    add_right_label(ax_top, "Interacting pair: SYNCRIP RGG–FOXO6 R-TAD")

    ln_bot = _plot_one(ax_bot, weak, C("grey","grey"))
    add_right_label(ax_bot, "Non-interacting pair: FZD9 RGG–ZNF165 R-TAD")

    # x label only bottom
    ax_top.tick_params(labelbottom=False)
    style_axes(ax_bot, xlabel=r"$r$ (nm)", ylabel=r"$g(r)$")

    # Explicit legends using the actual line handles (guaranteed to appear)
    leg  = ax_top.legend(handles=[ln_top], labels=["RDF"], frameon=False,
                        loc="upper right", bbox_to_anchor=(0.98, 0.98), borderaxespad=0.0)
    style_legend(leg, size=_fs(5.8))

    leg2 = ax_bot.legend(handles=[ln_bot], labels=["RDF"], frameon=False,
                        loc="upper right", bbox_to_anchor=(0.98, 0.98), borderaxespad=0.0)
    style_legend(leg2, size=_fs(5.8))


def panel_b(ax_parent: plt.Axes, rdf_strong: Path, rdf_weak: Path, cache: Path,
            *, box_nm: float, last: float, delta: float):
    """
    Two side-by-side RDF plots:
      left  = Interacting
      right = Non-interacting
    Titles are the pair labels.
    No hard y-limit on RDF.
    """
    ax_parent.axis("off")
    fig = ax_parent.figure
    bbox = ax_parent.get_position()

    # --- left/right axes inside the parent bbox ---
    pad_x = 0.03 * bbox.width
    mid_x = bbox.x0 + 0.5 * bbox.width

    left_w  = (mid_x - bbox.x0) - pad_x / 2
    right_w = (bbox.x1 - mid_x) - pad_x / 2

    ax_left  = fig.add_axes([bbox.x0,              bbox.y0, left_w,  bbox.height])
    ax_right = fig.add_axes([mid_x + pad_x / 2.0,  bbox.y0, right_w, bbox.height])

    strong = load_rdf_curves_via_corrected_rdfs(rdf_strong, last=last, box_nm=box_nm, delta=delta)
    weak   = load_rdf_curves_via_corrected_rdfs(rdf_weak,   last=last, box_nm=box_nm, delta=delta)

    pd.DataFrame(
        {"r_nm": strong["rs"], "g_orig": strong["orig"], "g_corr1": strong["corr1"], "g_corr2": strong["corr2"]}
    ).to_csv(cache / "Fig4b_rdf_interacting.csv", index=False)

    pd.DataFrame(
        {"r_nm": weak["rs"], "g_orig": weak["orig"], "g_corr1": weak["corr1"], "g_corr2": weak["corr2"]}
    ).to_csv(cache / "Fig4b_rdf_noninteracting.csv", index=False)

    import re, textwrap

    def _wrap_by_delims(text: str, *, width_chars: int = 45) -> str:
        # break after ":" and on dash/en-dash between partners, then wrap
        #text = re.sub(r":\s*", ":\n", text)
        #text = re.sub(r"\s*[–-]\s*", "\n", text)
        out_lines = []
        for line in [l.strip() for l in text.split("\n") if l.strip()]:
            out_lines.extend(textwrap.wrap(line, width=width_chars) or [line])
        return "\n".join(out_lines)

    def _plot_one(ax, cur, main_color, *, xlabel: bool, ylabel: bool):
        rs = cur["rs"]
        (ln,) = ax.plot(rs, cur["corr1"], color=main_color, lw=1.1, label="RDF")
        if RASTERIZE_HEAVY_ARTISTS:
            ln.set_rasterized(True)
        ax.axhline(1.0, color=C("grey", "grey"), ls=":", lw=0.7)

        # keep x physical if you want; remove this too if you truly want fully auto x-range
        ax.set_xlim(0.0, box_nm / 2.0)

        style_axes(
            ax,
            xlabel=(r"$r$ (nm)" if xlabel else None),
            ylabel=(r"$g(r)$" if ylabel else None),
        )

        # IMPORTANT: no hard y-cap (removes the old set_ylim(0, 10))
        ax.set_ylim(bottom=0.0)  # no upper limit; comment out if you want fully auto including bottom

        return ln

    # Left = interacting
    ln_left = _plot_one(ax_left, strong, C("blue", "blue"), xlabel=True, ylabel=True)
    ax_left.set_title(
        _wrap_by_delims("Interacting pair", width_chars=45),
        fontsize=AXIS_LABEL_FONTSIZE,
        pad=2.0,
    )

    # Right = non-interacting
    ln_right = _plot_one(ax_right, weak, C("grey", "grey"), xlabel=True, ylabel=False)
    ax_right.set_title(
        _wrap_by_delims("Non-interacting pair", width_chars=45),
        fontsize=AXIS_LABEL_FONTSIZE,
        pad=2.0,
    )

    # --- make each plot thinner (increase the gap between them) ---
    W_SHRINK = 0.86  # tune 0.75–0.95 (smaller = thinner, more gap)

    # left: keep anchored to left
    pos = ax_left.get_position()
    new_w = pos.width * W_SHRINK
    ax_left.set_position([pos.x0, pos.y0, new_w, pos.height])

    # right: keep anchored to right
    pos = ax_right.get_position()
    new_w = pos.width * W_SHRINK
    ax_right.set_position([pos.x1 - new_w, pos.y0, new_w, pos.height])

    # --- prevent any y tick labels from the right plot appearing in the middle gap ---
    ax_right.tick_params(axis="y", labelleft=True)
    ax_right.set_ylabel("")


    # Legends (explicit handles)
    leg1 = ax_left.legend(handles=[ln_left], labels=["RDF"], frameon=False,
                          loc="upper right", bbox_to_anchor=(0.98, 0.98), borderaxespad=0.0)
    style_legend(leg1, size=_fs(5.8))

    leg2 = ax_right.legend(handles=[ln_right], labels=["RDF"], frameon=False,
                           loc="upper right", bbox_to_anchor=(0.98, 0.98), borderaxespad=0.0)
    style_legend(leg2, size=_fs(5.8))


# =============================== Panel C =====================================
# =============================== Panel C (enrichment: obs vs expected) ==========================
AA20 = list("ACDEFGHIKLMNPQRSTVWY")
AA_CLASS = {}
for aa in "RKH": AA_CLASS[aa] = "Basic"
for aa in "DE":  AA_CLASS[aa] = "Acidic"
for aa in "FYW": AA_CLASS[aa] = "Aromatic"
for aa in AA20:
    AA_CLASS.setdefault(aa, "Other")
CLASS_ORDER = ["Basic", "Acidic", "Aromatic", "Other"]

def collapse_to_classes(M: np.ndarray) -> np.ndarray:
    idx = {aa: i for i, aa in enumerate(AA20)}
    cidx = {c: i for i, c in enumerate(CLASS_ORDER)}
    G = np.zeros((len(CLASS_ORDER), len(CLASS_ORDER)), dtype=float)
    for a in AA20:
        for b in AA20:
            i, j = idx[a], idx[b]
            ci, cj = cidx[AA_CLASS[a]], cidx[AA_CLASS[b]]
            v = float(M[i, j])
            if np.isfinite(v):
                G[ci, cj] += v
    return G

def independence_expected(M: np.ndarray) -> np.ndarray:
    """Expected matrix under independence given row/col marginals."""
    x = np.nan_to_num(M, nan=0.0)
    row = x.sum(axis=1, keepdims=True)
    col = x.sum(axis=0, keepdims=True)
    tot = x.sum()
    if tot <= 0:
        return np.zeros_like(x)
    return (row @ col) / float(tot)

def panel_c(
    ax: plt.Axes,
    *,
    df: pd.DataFrame,
    energy_dir: Path,
    aa_script: Path,
    cache: Path,
    box_nm: int,
    clip: float = 2.0,
    mode: str = "attractive",
):
    """
    Enrichment map (random vs observed): log2( observed / expected ) at residue-class level (4×4).

    - observed: mean AA-pair attractive matrix across selected pairs
    - expected: independence model based on row/col totals (composition-corrected)
    - selection: top `top_n` strongest binders by B22_corr1_log10 (higher = stronger), restricted to pairs with energy PKLs
    - visualization: values clipped to [-clip, +clip] to avoid a single extreme cell dominating the colormap
    """
    energy_dir = Path(energy_dir).expanduser().resolve()
    aa_script  = Path(aa_script).expanduser().resolve()

    mod = dynamic_import(aa_script, "aa_energy_mod_enrich")
    if not hasattr(mod, "per_sim_AA_pair_matrices"):
        raise AttributeError("amino_acids_energy_interactions.py must define per_sim_AA_pair_matrices")

    work = df[["name", "B22_corr1_log10"]].replace([np.inf, -np.inf], np.nan).dropna()
    if work.empty:
        ax.text(0.5, 0.5, "Panel C unavailable\n(no B22 data)",
                ha="center", va="center", fontsize=AXIS_LABEL_FONTSIZE)
        style_axes(ax); label_panel(ax, "c"); return

    # Require energy PKLs
    present, missing = [], []
    for n in work["name"].tolist():
        p = energy_dir / f"energies_{n}_{box_nm}nm_sliced.pkl"
        (present if p.exists() else missing).append(n)
    pd.Series(missing, name="missing_energy_pkls").to_csv(cache / "Fig4c_missing_energy_pkls.csv", index=False)

    work = work[work["name"].isin(present)].copy()
    if work.empty:
        ax.text(0.5, 0.5, "Panel C unavailable\n(no energy PKLs found)",
                ha="center", va="center", fontsize=AXIS_LABEL_FONTSIZE)
        style_axes(ax); label_panel(ax, "c"); return

    
    names = work["name"].tolist()
    work.to_csv(cache / "Fig4c_selected_topN_by_B22corr1log10.csv", index=False)

    # Load per-sim AA matrices and average
    Ms, *_ = mod.per_sim_AA_pair_matrices(
        names, str(energy_dir), box_nm, mode=mode, assume_name_order=False
    )
    Ms = np.asarray(Ms, dtype=float)
    if Ms.ndim != 3 or Ms.shape[1:] != (20, 20):
        raise ValueError(f"Expected per-sim matrices as (n,20,20), got {Ms.shape}")

    M = Ms.mean(axis=0)
    pd.DataFrame(M, index=AA20, columns=AA20).to_csv(cache / "Fig4c_observed_AA20_mean_topN.csv")

    # Class-level observed and expected
    G_obs = collapse_to_classes(M)
    G_exp = independence_expected(G_obs)

    eps = 1e-9
    G_log2 = np.log2((G_obs + eps) / (G_exp + eps))
    pd.DataFrame(G_obs, index=CLASS_ORDER, columns=CLASS_ORDER).to_csv(cache / "Fig4c_observed_classes_topN.csv")
    pd.DataFrame(G_exp, index=CLASS_ORDER, columns=CLASS_ORDER).to_csv(cache / "Fig4c_expected_classes_topN.csv")
    pd.DataFrame(G_log2, index=CLASS_ORDER, columns=CLASS_ORDER).to_csv(cache / "Fig4c_log2_enrichment_classes_topN.csv")

    # Clip for visualization (avoid one extreme cell dominating)
    clip = float(clip)
    if clip <= 0:
        clip = 2.0
    G_plot = np.clip(G_log2, -clip, clip)

    im = ax.imshow(
        G_plot,
        origin="lower",
        cmap="RdBu_r",
        vmin=-clip,
        vmax=clip,
        aspect="equal",
        rasterized=RASTERIZE_HEAVY_ARTISTS,
    )

    ax.set_xticks(range(len(CLASS_ORDER)))
    ax.set_yticks(range(len(CLASS_ORDER)))
    ax.set_xticklabels(CLASS_ORDER, rotation=45, ha="right", fontsize=TICK_LABEL_FONTSIZE)
    ax.set_yticklabels(CLASS_ORDER, fontsize=TICK_LABEL_FONTSIZE)
    ax.tick_params(axis="x", pad=2)   # gives x tick labels breathing room
    ax.tick_params(axis="y", pad=1)


    style_axes(
        ax,
        xlabel="TAD residue class",
        ylabel="RGG residue class",
        #title=rf"Residue-class enrichment (log$_2$ obs/exp)"
    )

    
    # --- Annotate with unclipped values (smaller text) ---
    ann_fs = _fs(5.0)  # was ~5.2; reduce so x-axis labels remain visible
    for i in range(len(CLASS_ORDER)):
        for j in range(len(CLASS_ORDER)):
            v = G_log2[i, j]
            if np.isfinite(v):
                ax.text(
                    j, i, f"{v:.2f}",
                    ha="center", va="center",
                    fontsize=ann_fs,
                    color=C("grey", "grey"),
                )

    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    cbar = plt.colorbar(im, cax=cax)
    style_colorbar(cbar, label=r"$\log_2(\mathrm{obs/exp})$")

# =============================== Panel D =====================================
def panel_d(ax: plt.Axes, df: pd.DataFrame, cache: Path):
    table = df[["name","Opposite_Charge_Number","lambda_TAD","B22_corr1_log10"]].replace([np.inf, -np.inf], np.nan).dropna()
    table.to_csv(cache / "Fig4d_scatter_table.csv", index=False)

    x = table["Opposite_Charge_Number"].to_numpy(dtype=float)
    y = table["B22_corr1_log10"].to_numpy(dtype=float)
    c = table["lambda_TAD"].to_numpy(dtype=float)

    sc = ax.scatter(
        x,
        y,
        c=c,
        s=10,
        alpha=0.6,
        linewidths=0.0,
        cmap="viridis",
        rasterized=RASTERIZE_HEAVY_ARTISTS,
    )
#    style_axes(ax, xlabel="Opposite-charge number (OCN)", ylabel=r"$\log_{10}\!\left(-B_{22}\right)$", title=r"$\log_{10}\!\left(-B_{22}\right)$"+" Vs. OCN: Simulated pairs")
    style_axes(ax, xlabel="Opposite-charge number (OCN)", ylabel=r"$\log_{10}\!\left(-B_{22}\right)$")

    if x.size >= 3:
        slope, intercept, r, p, _ = stats.linregress(x, y)
        xx = np.linspace(np.nanmin(x), np.nanmax(x), 200)
        ax.plot(xx, intercept + slope*xx, color=C("black","black"), lw=0.9, alpha=0.85)
        ax.text(
            0.02, 0.98,
            f"Pearson r = {r:.2f}\n(n = {x.size})",
            transform=ax.transAxes,
            ha="left", va="top",
            fontsize=_fs(5.4),
            bbox=dict(boxstyle="round,pad=0.2", fc="white", ec=C("grey","grey"), lw=0.5, alpha=0.9),
        )

    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    cbar = plt.colorbar(sc, cax=cax)
    style_colorbar(cbar, label=r"$\lambda_{\mathrm{TAD}}$")


def _save_panel(fig, out_base: Path):
    out_base.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out_base) + ".pdf", dpi=RASTER_DPI)
    fig.savefig(str(out_base) + ".png", dpi=DPI_SAVE)
    fig.savefig(str(out_base) + ".svg", bbox_inches="tight", dpi=RASTER_DPI)
    plt.close(fig)

def build_panel_a_figure(df: pd.DataFrame, cache: Path, out_panels: Path):
    fig, ax = plt.subplots(figsize=FIGSIZE_PANEL)
    fig.subplots_adjust(left=0.22, bottom=0.22, right=0.96, top=0.90)
    panel_a(ax, df, cache)
    _save_panel(fig, out_panels / "Figure4_panel_a")


def build_panel_b_figure(
    rdf_strong: Path, rdf_weak: Path, cache: Path, out_panels: Path,
    *, box_nm: int, rdf_last: float, rdf_delta: float
):
    fig = plt.figure(figsize=FIGSIZE_PANEL)
    ax_parent = fig.add_axes([0.3, 0.3, 0.88, 0.88])  # outer margins
    panel_b(ax_parent, rdf_strong, rdf_weak, cache, box_nm=float(box_nm), last=rdf_last, delta=rdf_delta)
    _save_panel(fig, out_panels / "Figure4_panel_b")


def build_panel_c_figure(
    df: pd.DataFrame, energy_dir: Path, aa_script: Path, cache: Path, out_panels: Path,
    *, box_nm: int
):
    fig, ax = plt.subplots(figsize=FIGSIZE_PANEL)
    # leave room for rotated x ticklabels + colorbar
    fig.subplots_adjust(left=0.30, bottom=0.32, right=0.84, top=0.90)
    panel_c(
        ax,
        df=df,
        energy_dir=energy_dir,
        aa_script=aa_script,
        cache=cache,
        box_nm=box_nm,

    )
    _save_panel(fig, out_panels / "Figure4_panel_c")


def build_panel_d_figure(df: pd.DataFrame, cache: Path, out_panels: Path):
    fig, ax = plt.subplots(figsize=FIGSIZE_PANEL)
    fig.subplots_adjust(left=0.05, bottom=0.22, right=0.9, top=0.90)
    panel_d(ax, df, cache)
    _save_panel(fig, out_panels / "Figure4_panel_d")



# =============================== Compose =====================================
def build_figure(
    df: pd.DataFrame,
    *,
    rdf_strong: Path,
    rdf_weak: Path,
    energy_dir: Path,
    aa_script: Path,
    outdir: Path,
    cache: Path,
    box_nm: int,
    rdf_last: float,
    rdf_delta: float,
):
    fig = plt.figure(figsize=(7.2, 4.6))
    gs = GridSpec(2, 2, figure=fig, wspace=0.35, hspace=0.45)

    axA = fig.add_subplot(gs[0, 0])
    axB_parent = fig.add_subplot(gs[0, 1])
    axC = fig.add_subplot(gs[1, 0])
    axD = fig.add_subplot(gs[1, 1])

    panel_a(axA, df, cache)
    panel_b(axB_parent, rdf_strong, rdf_weak, cache, box_nm=float(box_nm), last=rdf_last, delta=rdf_delta)
    panel_c(
        axC,
        df=df,
        energy_dir=energy_dir,
        aa_script=aa_script,
        cache=cache,
        box_nm=box_nm,
        clip=2.0,
    )    
    panel_d(axD, df, cache)
    fig.subplots_adjust(bottom=0.15)  # increase to 0.14 if needed


    fig.savefig(outdir / "Figure4_full.pdf", dpi=RASTER_DPI)
    fig.savefig(outdir / "Figure4_full.png", dpi=DPI_SAVE)
    fig.savefig(outdir / "Figure4_full.svg", bbox_inches="tight", dpi=RASTER_DPI)
    plt.close(fig)

# =============================== CLI =========================================
def main(argv: Optional[List[str]] = None):
    apply_mpl_style()

    p = argparse.ArgumentParser(description="Build Figure 4 (2×2) and save plotting tables to out/cache/")
    p.add_argument("--table", type=Path, default=Path("~/calvados/predicted_datasets_021125/sim_pairs_pred_with_hybrid_sel_cols_with_old.csv"),
                   help="Master CSV with features + B22 columns.")
    p.add_argument("--outdir", type=Path, default=Path("out_fig4"), help="Output directory.")
    p.add_argument("--rdf_strong", type=Path, default=Path("~/calvados/test_rdfs/rdfs_O60506_3_1__A8MYZ6_430_473_20nm.pkl"),
                   help="Pickle file for strong RDF example.")
    p.add_argument("--rdf_weak", type=Path, default=Path("~/calvados/test_rdfs/rdfs_O00144_1_1_P49910_193_208_20nm.pkl"),
                   help="Pickle file for weak RDF example.")
    p.add_argument("--energy_dir", type=Path, default=Path("~/calvados/test_energies"),
                   help="Directory containing energies_{name}_{box}nm_sliced.pkl files (for Panel C enrichment/metrics). Default: ~/calvados/test_energies")
    p.add_argument("--aa_script", type=Path, default=Path("~/calvados/scripts/amino_acids_energy_interactions.py"),
                   help="Path to amino_acids_energy_interactions.py (for Panel C enrichment/metrics). Default: amino_acids_energy_interactions.py")
    p.add_argument("--box_nm", type=int, default=20, help="Simulation box (nm) used in energy pickle filenames.")
    p.add_argument("--rdf_last", type=float, default=7.5, help="Upper integration limit (nm) for vertical lines.")
    p.add_argument("--rdf_delta", type=float, default=2.0, help="Delta shift for second vertical line.")
    p.add_argument("--export_metrics", action="store_true", help="If set, compute and save aa_interaction_metrics_per_sim.csv to out/cache using energy pickles.")
    p.add_argument("--metrics_csv", type=Path, default=None,
                   help="Optional metrics CSV with columns like RGG_basic_share, etc. Used only as Panel C fallback.")
    args = p.parse_args(argv)

    outdir, cache = ensure_dirs(args.outdir)
    out_panels = outdir
    df = load_master_table(args.table)

    rdf_strong = Path(args.rdf_strong).expanduser().resolve()
    rdf_weak   = Path(args.rdf_weak).expanduser().resolve()
    if not rdf_strong.exists():
        raise FileNotFoundError(f"Missing rdf_strong: {rdf_strong}")
    if not rdf_weak.exists():
        raise FileNotFoundError(f"Missing rdf_weak: {rdf_weak}")

    energy_dir = Path(args.energy_dir).expanduser().resolve()
    aa_script  = Path(args.aa_script).expanduser().resolve()
    if not energy_dir.exists():
        raise FileNotFoundError(f"energy_dir not found: {energy_dir}")
    if not aa_script.exists():
        raise FileNotFoundError(f"aa_script not found: {aa_script}")

    # Optional: export per-simulation metrics table derived from energies_* PKLs
    if args.export_metrics:
        modm = dynamic_import(aa_script, "aa_metrics_mod")
        if not hasattr(modm, "compute_metrics_table"):
            raise AttributeError("amino_acids_energy_interactions.py missing compute_metrics_table()")
        mdf = modm.compute_metrics_table(
            names=df["name"].tolist(),
            out_folder=str(energy_dir),
            box=args.box_nm,
            mode="attractive",
            assume_name_order=False,
        )
        outm = cache / "aa_interaction_metrics_per_sim.csv"
        # Your example file is tab-separated
        mdf.to_csv(outm, sep="\t")
        print(f"Wrote metrics CSV: {outm}")

    build_figure(
        df,
        rdf_strong=rdf_strong,
        rdf_weak=rdf_weak,
        energy_dir=energy_dir,
        aa_script=aa_script,
        outdir=outdir,
        cache=cache,
        box_nm=args.box_nm,
        rdf_last=args.rdf_last,
        rdf_delta=args.rdf_delta,
    )
    #build_panel_a_figure(df, cache, out_panels)
    build_panel_b_figure(rdf_strong, rdf_weak, cache, out_panels,box_nm=args.box_nm, rdf_last=args.rdf_last, rdf_delta=args.rdf_delta)
    #build_panel_c_figure(df, energy_dir, aa_script, cache, out_panels,box_nm=args.box_nm)
    #build_panel_d_figure(df, cache, out_panels)

    print(f"Done. Wrote: {outdir/'Figure4_full.pdf'}, {outdir/'Figure4_full.png'}, and {outdir/'Figure4_full.svg'}")

if __name__ == "__main__":
    main()
