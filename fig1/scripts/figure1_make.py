#!/usr/bin/env python3
# file: scripts/figure1_make.py
"""
Figure 1 composite (2×2, no histogram):
Row 1: a = interactor classes (boxplot) | c = pie (RBP fraction bins)
Row 2: d = top TF hubs (degree/betweenness) | e = top RBP hubs (degree/betweenness)

Inputs (default: input/):
  TF_list.txt, RBP_list.txt, UNIPROTIDS_GENENAME.txt, ppis.txt, PTM_list.txt,
  all_unids_human.txt, random_list.txt (auto-created on first run)

Outputs (default: output/):
  Figure1_full.pdf, cache_figure1/*.tsv, random_list_audit.tsv
"""

from __future__ import annotations

import argparse
import os
import sys
import textwrap
from pathlib import Path
from typing import Dict, Set, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt

# -----------------------------------------------------------------------------
# Import local modules robustly
# -----------------------------------------------------------------------------
_THIS = Path(__file__).resolve()
# This script lives in: <repo>/fig1/scripts/figure1_make.py
_FIG_DIR = _THIS.parents[1]          # <repo>/fig1
_REPO_ROOT = _FIG_DIR.parent         # <repo>

# Ensure local fig1 modules are importable (they live in <repo>/fig1)
sys.path.insert(0, str(_FIG_DIR))

# Default IO locations (relative to fig1/), independent of where you run the script from
DEFAULT_INPUT_DIR = _FIG_DIR / "input"
DEFAULT_OUTPUT_DIR = _FIG_DIR / "output"
DEFAULT_CACHE_DIR = DEFAULT_OUTPUT_DIR / "cache_figure1"

import tf_rbp_full_pipeline_with_tables as pipeline  # noqa: E402
import tf_interactor_classes as ic                   # noqa: E402


# -----------------------------------------------------------------------------
# Nature-style rcParams + unified palette (per style guide)
# -----------------------------------------------------------------------------
mpl.rcParams.update({
    # --- Fonts ---
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 6.5,
    "axes.labelsize": 6.5,
    "axes.titlesize": 7.0,
    "xtick.labelsize": 6.0,
    "ytick.labelsize": 6.0,
    "legend.fontsize": 6.0,

    # --- Lines & ticks ---
    "axes.linewidth": 0.6,
    "xtick.major.width": 0.5,
    "ytick.major.width": 0.5,

    # --- Figure defaults ---
    "figure.dpi": 600,
    "savefig.dpi": 600,
    "savefig.bbox": "tight",

    # --- PDF font embedding (TrueType 42) ---
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

COLOR_TF_NODE       = "#E69F00"   # TFs = orange
COLOR_RBP_NODE      = "#0072B2"   # RBPs = blue
COLOR_PME           = "#009E73"   # PMEs = bluish green
COLOR_RANDOM        = "#B0B0B0"   # Random / background
COLOR_EDGE          = "#D3D3D3"   # Light grey edges

COLOR_TF_BETWEENNESS    = COLOR_TF_NODE
COLOR_TF_INTERACTORS    = "#F6C86F"   # lighter orange
COLOR_RBP_BETWEENNESS   = COLOR_RBP_NODE
COLOR_RBP_INTERACTORS   = "#56B4E9"   # sky blue

PIE_COLORS = [COLOR_RBP_NODE, "#56B4E9", "#D3D3D3"]
HIST_BLUE  = COLOR_RBP_NODE  # not used here
COLOR_EDGE_INTRA = "#E0E0E0"


# -----------------------------------------------------------------------------
# Cache helpers
# -----------------------------------------------------------------------------
def save_plot_matrix(df: pd.DataFrame, cache_dir: str, name: str) -> str:
    os.makedirs(cache_dir, exist_ok=True)
    path = os.path.join(cache_dir, f"{name}.tsv")
    df.to_csv(path, sep="\t", index=False)
    return path

def load_plot_matrix(cache_dir: str, name: str) -> pd.DataFrame:
    path = os.path.join(cache_dir, f"{name}.tsv")
    return pd.read_csv(path, sep="\t")


# -----------------------------------------------------------------------------
# IO helpers
# -----------------------------------------------------------------------------
def _safe_load_uniprot_list(p: Path) -> List[str]:
    if p.is_file():
        try:
            return ic.load_list_uniprot(p)
        except Exception:
            return pd.read_csv(p, sep="\t", header=None, dtype=str).iloc[:, 0].dropna().astype(str).str.strip().tolist()
    return []

def _map_uniprot_to_gene(ids: List[str], mapping: Dict[str, str]) -> Set[str]:
    return {mapping[x] for x in ids if x in mapping}

def _ensure_dirs(input_dir: Path, output_dir: Path, cache_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)
    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")


# -----------------------------------------------------------------------------
# Random list management (create once in input/, then reuse) + audit
# -----------------------------------------------------------------------------
def _build_universe(
    mode: str,
    winners: Dict[str, str],
    all_unip_path: Path,
    graph_nodes: Optional[Set[str]],
) -> Set[str]:
    if mode == "graph":
        if not graph_nodes:
            raise FileNotFoundError("Graph nodes unavailable for --random-universe graph.")
        return set(graph_nodes)
    if all_unip_path.is_file():
        all_unip = _safe_load_uniprot_list(all_unip_path)
        return _map_uniprot_to_gene(all_unip, winners)
    if graph_nodes:
        return set(graph_nodes)
    raise FileNotFoundError(f"Universe file not found: {all_unip_path}")

def _write_random_audit(
    output_dir: Path,
    mode: str,
    seed: int,
    universe: Set[str],
    tf_genes: Set[str],
    rbp_genes: Set[str],
    pme_genes: Set[str],
    pool: Set[str],
    sample: Set[str],
    reused_existing: bool,
    random_path: Path,
) -> Path:
    row = {
        "mode": mode,
        "seed": seed,
        "universe_size": len(universe),
        "excluded_tf": len(set(universe) & set(tf_genes)),
        "excluded_rbp": len(set(universe) & set(rbp_genes)),
        "excluded_pme": len(set(universe) & set(pme_genes)),
        "pool_size": len(pool),
        "sample_size": len(sample),
        "reused_existing": reused_existing,
        "random_list_path": str(random_path),
    }
    out = output_dir / "random_list_audit.tsv"
    pd.DataFrame([row]).to_csv(out, sep="\t", index=False)
    return out

def _load_or_create_random_list(
    input_dir: Path,
    output_dir: Path,
    winners: Dict[str, str],
    tf_genes: Set[str],
    rbp_genes: Set[str],
    pme_genes: Set[str],
    all_unip_path: Path,
    graph_nodes: Optional[Set[str]],
    seed: int,
    desired_size: Optional[int],
    universe_mode: str,
) -> Set[str]:
    rand_path = input_dir / "random_list.txt"
    universe_genes = _build_universe(universe_mode, winners, all_unip_path, graph_nodes)
    exclude = set(tf_genes) | set(rbp_genes) | set(pme_genes)
    pool = sorted(g for g in universe_genes if g not in exclude)

    if rand_path.is_file():
        genes = pd.read_csv(rand_path, sep="\t", header=None, dtype=str).iloc[:, 0].dropna().str.strip().tolist()
        sample_set = set(genes)
        _write_random_audit(
            output_dir=output_dir,
            mode=universe_mode,
            seed=seed,
            universe=universe_genes,
            tf_genes=tf_genes,
            rbp_genes=rbp_genes,
            pme_genes=pme_genes,
            pool=set(pool),
            sample=sample_set,
            reused_existing=True,
            random_path=rand_path,
        )
        return sample_set

    if not pool:
        raise RuntimeError("Random sampling pool is empty after excluding TF/RBP/PME.")

    n_default = len(rbp_genes) if len(rbp_genes) > 0 else min(1000, len(pool))
    n = desired_size if (desired_size and desired_size > 0) else n_default
    n = min(n, len(pool))

    rng = np.random.default_rng(seed)
    sample = sorted(rng.choice(pool, size=n, replace=False).tolist())

    pd.Series(sample).to_csv(rand_path, sep="\t", index=False, header=False)
    print(f"[created] {rand_path} (n={n}, seed={seed}, mode={universe_mode})")

    _write_random_audit(
        output_dir=output_dir,
        mode=universe_mode,
        seed=seed,
        universe=universe_genes,
        tf_genes=tf_genes,
        rbp_genes=rbp_genes,
        pme_genes=pme_genes,
        pool=set(pool),
        sample=set(sample),
        reused_existing=False,
        random_path=rand_path,
    )
    return set(sample)


# -----------------------------------------------------------------------------
# Cached getters (wrapping compute logic from the two modules)
# -----------------------------------------------------------------------------
def get_tf_rbp_fraction_per_tf(cache_dir: str,
                               G,
                               TF_genes: Set[str],
                               RBP_genes: Set[str],
                               force_recompute: bool=False) -> pd.DataFrame:
    name = "tf_rbp_fraction_per_tf"
    path = os.path.join(cache_dir, f"{name}.tsv")
    if (not force_recompute) and os.path.isfile(path):
        return load_plot_matrix(cache_dir, name)
    df, _, _ = pipeline.compute_tf_rbp_fractions(G, TF_genes, RBP_genes)
    save_plot_matrix(df, cache_dir, name)
    return df

def get_hubs_table(cache_dir: str,
                   G,
                   TF_genes: Set[str],
                   RBP_genes: Set[str],
                   min_degree: int = 3,
                   force_recompute: bool=False) -> pd.DataFrame:
    name = "tf_rbp_subgraph_stats"
    path = os.path.join(cache_dir, f"{name}.tsv")
    if (not force_recompute) and os.path.isfile(path):
        return load_plot_matrix(cache_dir, name)
    _, hubs_df = pipeline.build_tf_rbp_subgraph(G, TF_genes, RBP_genes, min_degree=min_degree)
    save_plot_matrix(hubs_df, cache_dir, name)
    return hubs_df

def get_tf_top_hubs(cache_dir: str,
                    G,
                    TF_genes: Set[str],
                    RBP_genes: Set[str],
                    min_degree: int = 3,
                    force_recompute: bool=False) -> pd.DataFrame:
    name = "tf_top_hubs"
    path = os.path.join(cache_dir, f"{name}.tsv")
    if (not force_recompute) and os.path.isfile(path):
        return load_plot_matrix(cache_dir, name)
    hubs_df = get_hubs_table(cache_dir, G, TF_genes, RBP_genes, min_degree=min_degree, force_recompute=force_recompute)
    tf_plot = hubs_df[hubs_df["type"] == "TF"].sort_values("degree_TF_RBP", ascending=False).head(10)
    save_plot_matrix(tf_plot, cache_dir, name)
    return tf_plot

def get_rbp_top_hubs(cache_dir: str,
                     G,
                     TF_genes: Set[str],
                     RBP_genes: Set[str],
                     min_degree: int = 3,
                     force_recompute: bool=False) -> pd.DataFrame:
    name = "rbp_top_hubs"
    path = os.path.join(cache_dir, f"{name}.tsv")
    if (not force_recompute) and os.path.isfile(path):
        return load_plot_matrix(cache_dir, name)
    hubs_df = get_hubs_table(cache_dir, G, TF_genes, RBP_genes, min_degree=min_degree, force_recompute=force_recompute)
    rbp_plot = hubs_df[hubs_df["type"] == "RBP"].sort_values("degree_TF_RBP", ascending=False).head(10)
    save_plot_matrix(rbp_plot, cache_dir, name)
    return rbp_plot

def get_tf_interactor_class_matrix(cache_dir: str,
                                   tf_labels: List[str],
                                   adjacency: Dict[str, Set[str]],
                                   TF_set: Set[str],
                                   RBP_set: Set[str],
                                   PME_set: Optional[Set[str]] = None,
                                   RAND_set: Optional[Set[str]] = None,
                                   force_recompute: bool = False,
                                   signif_out_prefix: Optional[Path] = None) -> pd.DataFrame:
    name = "tf_interactor_classes"
    path = os.path.join(cache_dir, f"{name}.tsv")
    if (not force_recompute) and os.path.isfile(path):
        return load_plot_matrix(cache_dir, name)
    df = ic.compute_per_tf_lists_and_metrics(
        tf_labels=tf_labels,
        adjacency=adjacency,
        TF_set=TF_set,
        RBP_set=RBP_set,
        PME_set=PME_set or set(),
        RAND_set=RAND_set or set(),
    )
    save_plot_matrix(df, cache_dir, name)
    # Save significance tables next to other outputs (default: <fig1>/output/)
    if signif_out_prefix is None:
        signif_out_prefix = DEFAULT_OUTPUT_DIR / "tf_interactor_classes"
    save_vs_rbp_significance_tables(df, out_prefix=signif_out_prefix, seed=42, n_perm=5000)
    return df


# -----------------------------------------------------------------------------
# Aesthetics helpers
# -----------------------------------------------------------------------------
def _clean_axes_min(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

def _panel_letter(ax, letter: str):
    ax.text(0.02, 0.98, letter, transform=ax.transAxes,
            ha="left", va="top", fontsize=7.8, fontweight="bold")

from matplotlib.ticker import FixedLocator, FixedFormatter

def _yticks_to_100(ax, step: int = 20):
    ticks = np.arange(0, 101, step)  # 0..100 inclusive
    ax.yaxis.set_major_locator(FixedLocator(ticks))
    ax.yaxis.set_major_formatter(FixedFormatter([f"{int(t)}" for t in ticks]))


# -----------------------------------------------------------------------------
# Panel drawers (A, C, D, E)
# -----------------------------------------------------------------------------
# 2) Change the signature of plot_panel_A_interactor_classes to accept middle ---
def plot_panel_A_interactor_classes(ax, df: pd.DataFrame, middle: str = "median", star_gap: float = 0.8):
    import textwrap
    long = pd.melt(
        df[["TF", "pct_rbp", "pct_tf", "pct_pme", "pct_random"]],
        id_vars=["TF"], var_name="class", value_name="percent"
    )
    class_map = {"pct_rbp": "RBP", "pct_tf": "TF", "pct_pme": "PME", "pct_random": "Random"}
    long["class"] = long["class"].map(class_map)

    order = ["RBP", "TF", "PME", "Random"]
    data  = [long.loc[long["class"] == c, "percent"].dropna().values for c in order]

    # median vs mean line
    showmeans   = (middle == "mean")
    meanline    = (middle == "mean")
    medianprops = {"color": "black", "linewidth": 1.0} if middle == "median" else {"linewidth": 0.0}
    meanprops   = {"color": "black", "linewidth": 1.0} if middle == "mean"   else {"linewidth": 0.0}

    bp = ax.boxplot(
        data,
        labels=order,
        vert=True,
        patch_artist=True,
        showfliers=False,
        showmeans=showmeans,
        meanline=meanline,
        meanprops=meanprops,
        whiskerprops={"color": "black", "linewidth": 0.8},
        capprops={"color": "black", "linewidth": 0.8},
        medianprops=medianprops,
    )

    

    colors = {"RBP": COLOR_RBP_NODE, "TF": COLOR_TF_NODE, "PME": COLOR_PME, "Random": COLOR_RANDOM}
    for patch, cls in zip(bp["boxes"], order):
        patch.set_facecolor(colors[cls])
        patch.set_alpha(0.50)
        patch.set_edgecolor("black")
        patch.set_linewidth(0.6)

    # ---- show ALL points, same color as box, faint + BEHIND boxplot ----
    rng = np.random.default_rng(0)
    jitter = 0.09

    # points behind
    for i, cls in enumerate(order, start=1):
        y = long.loc[long["class"] == cls, "percent"].dropna().to_numpy()
        x = rng.normal(loc=i, scale=jitter, size=len(y))

        ax.scatter(
            x, y,
            s=8,
            color=colors[cls],
            alpha=0.18,          # light
            linewidths=0,
            zorder=0.5,          # behind boxes/lines
            rasterized=True
        )

    # make sure the boxplot sits on top
    for patch in bp["boxes"]:
        patch.set_zorder(2)
    for k in ("whiskers", "caps"):
        for line in bp[k]:
            line.set_zorder(2.2)
    for k in ("medians", "means"):
        for line in bp.get(k, []):
            line.set_zorder(2.5)

    # p-values → stars
    pvals_df = ic.compute_vs_rbp_pvalues(long, seed=42, n_perm=5000)
    star_map = {row["class"]: ic.p_to_stars(row["pvalue"]) for _, row in pvals_df.iterrows()}

    ymax_data = max([float(np.nanmax(v)) if len(v) else 0.0 for v in data] + [0.0])
    ymin, ymax = ax.get_ylim()
    yr = ymax - ymin if ymax > ymin else 100.0

    base = max(0.05 * yr + ymax_data, 75.0)
    h    = 0.08 * yr
    pad  = 0.03 * yr
    tick = 0.04 * yr

    idx = {"RBP": 1, "TF": 2, "PME": 3, "Random": 4}
    comps = ["TF", "PME", "Random"]
    left_offsets  = [-0.08,  0.00,  0.08]
    right_offsets = [ 0.08,  0.00, -0.08]

    for i, cls in enumerate(comps):
        s = star_map.get(cls, "")
        if not s:
            continue

        x1 = idx["RBP"] + left_offsets[i]
        x2 = idx[cls]   + right_offsets[i]
        y  = base + i * (h + pad)

        # bracket
        ax.plot([x1, x1], [y - tick, y], color="black", linewidth=0.6)
        ax.plot([x2, x2], [y - tick, y], color="black", linewidth=0.6)
        ax.plot([x1, x2], [y, y],       color="black", linewidth=0.6)

        # star: same offsets; closer to bracket using star_gap
        star_x = 0.5 * (x1 + x2)
        star_y = y + (star_gap * tick)   # <— tighter by default
        ax.text(star_x, star_y, s, ha="center", va="bottom",
                fontsize=7, fontweight="bold", zorder=3)

    ax.set_ylim(0, 100)  # initial

    # ... compute stars, brackets ...

    needed_top = base + 2 * (h + pad) + 0.12 * yr
    ymin, ymax = ax.get_ylim()
    if needed_top > ymax:
        ax.set_ylim(ymin, needed_top)

    # IMPORTANT: force ticks AFTER final ylim
    _yticks_to_100(ax, step=20)


    ax.set_ylabel("% of interactors of TFs")
    ax.set_xlabel("Interactor class")
    raw_labels = ["RBPs", "TFs", "Protein-modifying enzymes", "Random"]
    wrapped_labels = [
        "\n".join(textwrap.wrap(lbl, width=14, break_long_words=False, break_on_hyphens=True))
        for lbl in raw_labels
    ]
    ax.set_xticklabels(wrapped_labels)
    ax.set_title("TF interactor classes")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)



def plot_panel_C_pie_rbp_fraction_bins(ax, tf_frac_df: pd.DataFrame, radius: float = 1.3):
    import numpy as np

    if not tf_frac_df.empty:
        bins = {
            "> 0.5":    int((tf_frac_df["fraction_RBP"] > 0.5).sum()),
            "0.25–0.5": int(((tf_frac_df["fraction_RBP"] >= 0.25) & (tf_frac_df["fraction_RBP"] <= 0.5)).sum()),
            "< 0.25":   int((tf_frac_df["fraction_RBP"] < 0.25).sum()),
        }
        total = sum(bins.values()) or 1
        labels = [
            "\n".join(textwrap.wrap(f"{k} fraction of RBP interactors", width=18, break_long_words=False, break_on_hyphens=True))
            for k, _ in bins.items()
        ]
        sizes  = list(bins.values())
    else:
        labels, sizes = ["No data"], [1]

    # Title padding & label distance unchanged
    title_pad = 6 + int((max(radius, 1.0) - 1.0) * 28)
    labeldistance = 1.05 + (max(radius, 1.0) - 1.0) * 0.20

    # Draw pie (no autopct); we'll place exact percentages at wedge centroids
    wedges, texts = ax.pie(
        sizes,
        labels=labels,
        colors=PIE_COLORS[:len(sizes)],
        startangle=90,
        counterclock=False,
        wedgeprops={"linewidth": 0.6, "edgecolor": "white"},
        textprops={"fontsize": 6},
        radius=radius,
        labeldistance=labeldistance,
    )

    # Inside-wedge text at sector centroid (r = 4R sin(θ/2) / (3θ))
    if not tf_frac_df.empty:
        for w, v in zip(wedges, sizes):
            theta1 = np.deg2rad(w.theta1)
            theta2 = np.deg2rad(w.theta2)
            theta  = max(theta2 - theta1, 1e-6)
            theta_m = 0.5 * (theta1 + theta2)
            r_c = (4.0 * radius * np.sin(theta / 2.0)) / (3.0 * theta)  # area centroid radius
            x = r_c * np.cos(theta_m)
            y = r_c * np.sin(theta_m)
            pct = (v / total) * 100.0
            ax.text(x, y, f"{pct:.1f}% TFs", ha="center", va="center", fontsize=6)

    ax.set_title("RBP interaction bias across TFs", pad=title_pad)
    ax.set_aspect("equal", adjustable="box")

    extra = (max(radius, 1.0) - 1.0)
    ax.margins(x=0.02 + 0.05 * extra, y=0.02 + 0.08 * extra)


def plot_panel_D_top_tf_hubs(ax, tf_plot: pd.DataFrame):
    # Left axis: degree; Right axis: betweenness
    if tf_plot.empty:
        ax.text(0.5, 0.5, "No TF hubs", ha="center", va="center"); ax.set_axis_off(); return
    x = list(range(len(tf_plot))); w = 0.4
    ax2 = ax.twinx()
    ax.bar([i - w/2 for i in x], tf_plot["degree_TF_RBP"], width=w, color=COLOR_TF_INTERACTORS)
    ax2.bar([i + w/2 for i in x], tf_plot["betweenness"],    width=w, color=COLOR_TF_BETWEENNESS)
    ax.set_xticks(x); ax.set_xticklabels(tf_plot["gene"], rotation=45, ha="right")
    ax.set_ylabel("Number of RBP interactors")
    ax2.set_ylabel("Betweenness centrality")
    ax.set_title("Top TF hubs")
    handle_degree = plt.Rectangle((0, 0), 1, 1, color=COLOR_TF_INTERACTORS)
    handle_betw   = plt.Rectangle((0, 0), 1, 1, color=COLOR_TF_BETWEENNESS)
    ax.legend([handle_degree, handle_betw], ["Degree", "Betweenness"], loc="upper right", frameon=False)
    _clean_axes_min(ax)


def plot_panel_E_top_rbp_hubs(ax, rbp_plot: pd.DataFrame):
    # Left axis: degree; Right axis: betweenness
    if rbp_plot.empty:
        ax.text(0.5, 0.5, "No RBP hubs", ha="center", va="center"); ax.set_axis_off(); return
    x = list(range(len(rbp_plot))); w = 0.4
    ax2 = ax.twinx()
    ax.bar([i - w/2 for i in x], rbp_plot["degree_TF_RBP"], width=w, color=COLOR_RBP_INTERACTORS)
    ax2.bar([i + w/2 for i in x], rbp_plot["betweenness"],    width=w, color=COLOR_RBP_BETWEENNESS)
    ax.set_xticks(x); ax.set_xticklabels(rbp_plot["gene"], rotation=45, ha="right")
    ax.set_ylabel("Number of TF interactors")
    ax2.set_ylabel("Betweenness centrality")
    ax.set_title("Top RBP hubs")
    handle_degree = plt.Rectangle((0, 0), 1, 1, color=COLOR_RBP_INTERACTORS)
    handle_betw   = plt.Rectangle((0, 0), 1, 1, color=COLOR_RBP_BETWEENNESS)
    ax.legend([handle_degree, handle_betw], ["Degree", "Betweenness"], loc="upper right", frameon=False)
    _clean_axes_min(ax)


# -----------------------------------------------------------------------------
# Composite figure builder (2×2: a,c / d,e)
# -----------------------------------------------------------------------------
def build_composite_figure(df_inter_classes: pd.DataFrame,
                           df_frac_rbp: pd.DataFrame,
                           df_tf_hubs: pd.DataFrame,
                           df_rbp_hubs: pd.DataFrame,
                           outfile: Path,
                           middle: str = "median",
                           star_gap: float = -0.7) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 4.6))
    fig.subplots_adjust(wspace=0.6, hspace=0.8)

    axA, axC = axes[0, 0], axes[0, 1]
    axD, axE = axes[1, 0], axes[1, 1]

    plot_panel_A_interactor_classes(axA, df_inter_classes, middle=middle, star_gap=star_gap)
    plot_panel_C_pie_rbp_fraction_bins(axC, df_frac_rbp)
    plot_panel_D_top_tf_hubs(axD, df_tf_hubs)
    plot_panel_E_top_rbp_hubs(axE, df_rbp_hubs)

    
    outfile.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(outfile, dpi=2400)
    plt.close(fig)

# -----------------------------------------------------------------------------
# Enrichment tests for Result 1 (TF–RBP enrichment)
# -----------------------------------------------------------------------------
from typing import Iterable, Tuple, Dict

def _tf_partner_set(G, tf_genes: set, exclude_tf: bool = True) -> set:
    P = set()
    for tf in tf_genes:
        if tf in G:
            P.update(G.neighbors(tf))
    if exclude_tf:
        P -= set(tf_genes)
    return P

def _fisher_2x2_partner_enrichment(
    G,
    tf_genes: set,
    rbp_genes: set,
    universe_nodes: set,
    *,
    exclude_tf_partners: bool = True,
    exclude_tf_from_rbp: bool = True,
) -> Dict[str, float]:
    """
    Fisher exact test: are RBPs enriched among TF partners vs background?

    universe_nodes: set of nodes that define "background" U.
      - Use set(G.nodes()) for a clean interactome-background test.
      - Optionally use mapped all_unids_human as proteome-background.
    """
    TF = set(tf_genes)
    RBP = set(rbp_genes) - TF if exclude_tf_from_rbp else set(rbp_genes)

    U = set(universe_nodes)
    # Ensure all TF partners are inside U if you want strict accounting
    P = _tf_partner_set(G, TF, exclude_tf=exclude_tf_partners)
    P = P & U

    a = len(P & RBP)
    b = len(P - RBP)
    c = len((U - P) & RBP)
    d = len((U - P) - RBP)

    # try SciPy fisher_exact; if unavailable, return NaN p but keep counts/OR as NaN too
    odds, pval = float("nan"), float("nan")
    try:
        from scipy.stats import fisher_exact
        odds, pval = fisher_exact([[a, b], [c, d]], alternative="greater")
        odds, pval = float(odds), float(pval)
    except Exception:
        pass

    frac_partners = (a / (a + b)) if (a + b) > 0 else float("nan")
    frac_bg       = ( (a + c) / (a + b + c + d) ) if (a + b + c + d) > 0 else float("nan")

    return {
        "a_partners_rbp": a,
        "b_partners_nonrbp": b,
        "c_nonpartners_rbp": c,
        "d_nonpartners_nonrbp": d,
        "odds_ratio": odds,
        "p_value": pval,
        "frac_rbp_in_partners": frac_partners,
        "frac_rbp_in_background": frac_bg,
        "n_partners": (a + b),
        "n_universe": (a + b + c + d),
    }

def _count_tf_rbp_edges(G, tf_genes: set, rbp_set: set) -> int:
    """Count TF–RBP edges once (from TF side)."""
    cnt = 0
    for tf in tf_genes:
        if tf in G:
            for nb in G.neighbors(tf):
                if nb in rbp_set:
                    cnt += 1
    return cnt

def degree_binned_rbp_label_permutation(
    G,
    tf_genes: set,
    rbp_genes: set,
    *,
    n_perm: int = 5000,
    n_bins: int = 10,
    seed: int = 42,
    exclude_tf_from_rbp: bool = True,
) -> Tuple[int, np.ndarray, float, Dict[str, float]]:
    """
    Degree-controlled permutation: shuffle RBP labels within degree bins
    and test whether TF–RBP edge count is higher than expected.
    """
    rng = np.random.default_rng(seed)
    TF = set(tf_genes)
    RBP_obs = set(rbp_genes) - TF if exclude_tf_from_rbp else set(rbp_genes)

    nodes = np.array(list(G.nodes()))
    deg = np.array([G.degree(n) for n in nodes], dtype=float)

    # Build degree bins by quantiles (robust)
    qs = np.linspace(0, 1, n_bins + 1)
    edges = np.quantile(deg, qs)
    edges[-1] += 1e-9  # ensure max included

    # Assign bin index per node
    bin_idx = np.clip(np.searchsorted(edges, deg, side="right") - 1, 0, n_bins - 1)

    # For each bin: candidate nodes + how many RBPs observed in that bin
    bin2nodes = [nodes[bin_idx == b] for b in range(n_bins)]
    rbp_per_bin = []
    for b in range(n_bins):
        ns = bin2nodes[b]
        rbp_per_bin.append(int(np.sum([n in RBP_obs for n in ns])))

    # observed statistic
    obs = _count_tf_rbp_edges(G, TF, RBP_obs)

    null = np.zeros(n_perm, dtype=int)
    for i in range(n_perm):
        RBP_perm = set()
        for b in range(n_bins):
            ns = bin2nodes[b]
            k = rbp_per_bin[b]
            if k <= 0:
                continue
            # sample k nodes from this bin
            chosen = rng.choice(ns, size=k, replace=False)
            RBP_perm.update(chosen.tolist())
        null[i] = _count_tf_rbp_edges(G, TF, RBP_perm)

    p_emp = (int(np.sum(null >= obs)) + 1) / (n_perm + 1)

    summary = {
        "observed_tf_rbp_edges": int(obs),
        "null_mean": float(null.mean()),
        "null_sd": float(null.std(ddof=1)) if n_perm > 1 else float("nan"),
        "p_empirical": float(p_emp),
        "n_perm": int(n_perm),
        "n_bins": int(n_bins),
    }
    return obs, null, float(p_emp), summary

def plot_null_histogram(null: np.ndarray, observed: int, outpath):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(3.2, 2.2))
    ax.hist(null, bins=40)
    ax.axvline(observed, linewidth=1.6)
    ax.set_xlabel("TF–RBP edge count (null)")
    ax.set_ylabel("Frequency")
    ax.set_title("Degree-binned label permutation")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(outpath)
    plt.close(fig)

from pathlib import Path
import numpy as np
import pandas as pd

def _bh_fdr(pvals: np.ndarray) -> np.ndarray:
    """Benjamini–Hochberg FDR correction; returns q-values."""
    p = np.asarray(pvals, dtype=float)
    q = np.full_like(p, np.nan, dtype=float)
    ok = np.isfinite(p)
    if ok.sum() == 0:
        return q
    p_ok = p[ok]
    m = p_ok.size
    order = np.argsort(p_ok)
    ranked = p_ok[order]
    q_raw = ranked * m / (np.arange(1, m + 1))
    q_mon = np.minimum.accumulate(q_raw[::-1])[::-1]
    q_vals = np.clip(q_mon, 0.0, 1.0)
    tmp = np.empty_like(p_ok)
    tmp[order] = q_vals
    q[ok] = tmp
    return q


def _paired_signflip_pvalue(diffs: np.ndarray, *, stat: str = "median", n_perm: int = 10000, seed: int = 42) -> float:
    """
    Paired sign-flip permutation test on diffs = (class% - RBP%) per TF.
    """
    d = np.asarray(diffs, dtype=float)
    d = d[np.isfinite(d)]
    if d.size == 0:
        return float("nan")

    if stat == "median":
        obs = abs(float(np.median(d)))
        f = np.median
    elif stat == "mean":
        obs = abs(float(np.mean(d)))
        f = np.mean
    else:
        raise ValueError("stat must be 'median' or 'mean'")

    rng = np.random.default_rng(seed)
    ge = 0
    for _ in range(n_perm):
        signs = rng.choice([-1.0, 1.0], size=d.size, replace=True)
        val = abs(float(f(d * signs)))
        if val >= obs:
            ge += 1
    return float((ge + 1) / (n_perm + 1))


def save_vs_rbp_significance_tables(
    df_per_tf: pd.DataFrame,
    *,
    out_prefix: Path,
    seed: int = 42,
    n_perm: int = 10000,
    drop_zero_degree: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Saves:
      1) <prefix>_signif_vs_RBP_mwu.tsv         (unpaired MWU; same schema your star-plot expects)
      2) <prefix>_signif_vs_RBP_multi_tests.tsv (MWU + paired Wilcoxon (if available) + paired sign-flip)

    Returns:
      (mwu_df, multi_df)
    """
    dff = df_per_tf.copy()
    if drop_zero_degree and "n_total" in dff.columns:
        dff = dff.loc[dff["n_total"] > 0].copy()

    # --- Table 1: MWU in the exact format your plot function expects (class, comparator, pvalue, effect_median_diff)
    long = pd.melt(
        dff[["TF", "pct_rbp", "pct_tf", "pct_pme", "pct_random"]],
        id_vars=["TF"], var_name="class", value_name="percent"
    )
    class_map = {"pct_rbp": "RBP", "pct_tf": "TF", "pct_pme": "PME", "pct_random": "Random"}
    long["class"] = long["class"].map(class_map)

    # Reuse your existing function if it exists in the file
    mwu_df = ic.compute_vs_rbp_pvalues(long, seed=seed, n_perm=n_perm)

    # --- Table 2: Multi-test table (paired tests included)
    rbp = dff["pct_rbp"].to_numpy(dtype=float)

    rows = []
    for cls, col in [("TF", "pct_tf"), ("PME", "pct_pme"), ("Random", "pct_random")]:
        arr = dff[col].to_numpy(dtype=float)
        diffs = arr - rbp

        # Effects
        eff_median_unpaired = float(np.nanmedian(arr) - np.nanmedian(rbp))
        eff_mean_unpaired   = float(np.nanmean(arr) - np.nanmean(rbp))
        eff_median_paired   = float(np.nanmedian(diffs))
        eff_mean_paired     = float(np.nanmean(diffs))

        # MWU (unpaired)
        p_mwu = float("nan")
        try:
            from scipy.stats import mannwhitneyu
            p_mwu = float(mannwhitneyu(arr, rbp, alternative="two-sided").pvalue)
        except Exception:
            # If SciPy is missing, keep NaN here (you still have the mwu_df fallback already)
            p_mwu = float("nan")

        # Wilcoxon (paired) if SciPy exists
        p_wilcoxon = float("nan")
        try:
            from scipy.stats import wilcoxon
            p_wilcoxon = float(wilcoxon(diffs, alternative="two-sided", zero_method="wilcox").pvalue)
        except Exception:
            p_wilcoxon = float("nan")

        # Paired sign-flip permutation (always)
        p_signflip_median = _paired_signflip_pvalue(diffs, stat="median", n_perm=n_perm, seed=seed)
        p_signflip_mean   = _paired_signflip_pvalue(diffs, stat="mean",   n_perm=n_perm, seed=seed + 1)

        rows.append({
            "class": cls,
            "comparator": "RBP",
            "n_TFs_used": int(len(dff)),
            "effect_median_unpaired": eff_median_unpaired,
            "effect_mean_unpaired": eff_mean_unpaired,
            "effect_median_paired": eff_median_paired,
            "effect_mean_paired": eff_mean_paired,
            "p_mwu": p_mwu,
            "p_wilcoxon": p_wilcoxon,
            "p_signflip_median": p_signflip_median,
            "p_signflip_mean": p_signflip_mean,
        })

    multi_df = pd.DataFrame(rows)
    multi_df["q_mwu_bh"] = _bh_fdr(multi_df["p_mwu"].to_numpy())
    multi_df["q_wilcoxon_bh"] = _bh_fdr(multi_df["p_wilcoxon"].to_numpy())
    multi_df["q_signflip_median_bh"] = _bh_fdr(multi_df["p_signflip_median"].to_numpy())
    multi_df["q_signflip_mean_bh"] = _bh_fdr(multi_df["p_signflip_mean"].to_numpy())

    outbase = Path(out_prefix)
    mwu_path   = outbase.with_name(f"{outbase.name}_signif_vs_RBP_mwu.tsv")
    multi_path = outbase.with_name(f"{outbase.name}_signif_vs_RBP_multi_tests.tsv")

    mwu_df.to_csv(mwu_path, sep="\t", index=False)
    multi_df.to_csv(multi_path, sep="\t", index=False)

    print(f"[saved] MWU table        → {mwu_path}")
    print(f"[saved] Multi-test table → {multi_path}")

    return mwu_df, multi_df


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="Build Figure 1 (2×2: a,c / d,e) from cached matrices or raw inputs.")
    ap.add_argument("--input-dir",  type=Path, default=DEFAULT_INPUT_DIR)
    ap.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    ap.add_argument("--cache-dir",  type=Path, default=DEFAULT_CACHE_DIR)
    ap.add_argument("--seed", type=int, default=42, help="Seed for random_list sampling")
    ap.add_argument("--random-size", type=int, default=None, help="Size of random_list; default = len(RBP_set)")
    ap.add_argument("--random-universe", choices=["all", "graph"], default="all",
                    help="Sampling universe: 'all' uses input/all_unids_human.txt (fallback graph), 'graph' uses graph nodes only.")
    ap.add_argument("--force-recompute-cache", action="store_true")
    ap.add_argument("--also-png", action="store_true")
    ap.add_argument("--min-degree-h", type=int, default=3)
    ap.add_argument(
        "--box-middle",
        choices=["median", "mean"],
        default="median",
        help="Center line in Panel A boxplot: 'median' (default) or 'mean'."
    )

    args = ap.parse_args()

    _ensure_dirs(args.input_dir, args.output_dir, args.cache_dir)

    # Input files
    tf_path   = args.input_dir / "TF_list.txt"
    rbp_path  = args.input_dir / "RBP_list.txt"
    map_path  = args.input_dir / "UNIPROTIDS_GENENAME.txt"
    ppi_path  = args.input_dir / "ppis.txt"
    pme_path  = args.input_dir / "PTM_list.txt"
    all_path  = args.input_dir / "all_unids_human.txt"

    # Load mapping and lists
    winners: Dict[str, str] = pipeline.load_mapping(map_path)
    tf_genes: Set[str]      = pipeline.load_tf_list(tf_path, winners)
    rbp_genes: Set[str]     = pipeline.load_rbp_list(rbp_path)

    # PPI graph
    ppis_mapped: pd.DataFrame = pipeline.load_and_map_ppis(ppi_path, winners)
    G = pipeline.build_graph(ppis_mapped)

    # PME set (optional)
    pme_unip = _safe_load_uniprot_list(pme_path)
    pme_genes = _map_uniprot_to_gene(pme_unip, winners)

    # Random list: load existing or create fixed sample in input/
    random_genes = _load_or_create_random_list(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        winners=winners,
        tf_genes=tf_genes,
        rbp_genes=rbp_genes,
        pme_genes=pme_genes,
        all_unip_path=all_path,
        graph_nodes=set(G.nodes()),
        seed=args.seed,
        desired_size=args.random_size,
        universe_mode=args.random_universe,
    )

    # Cached matrices
    df_frac = get_tf_rbp_fraction_per_tf(
        cache_dir=str(args.cache_dir), G=G, TF_genes=tf_genes, RBP_genes=rbp_genes, force_recompute=args.force_recompute_cache
    )
    df_tf_hubs  = get_tf_top_hubs(str(args.cache_dir), G, tf_genes, rbp_genes, min_degree=args.min_degree_h, force_recompute=args.force_recompute_cache)
    df_rbp_hubs = get_rbp_top_hubs(str(args.cache_dir), G, tf_genes, rbp_genes, min_degree=args.min_degree_h, force_recompute=args.force_recompute_cache)

    # Panel A matrix
    edges = ppis_mapped.rename(columns={"proteinA": "a", "proteinB": "b"})[["a", "b"]]
    adjacency = ic.build_adjacency_from_edges(edges)
    df_inter = get_tf_interactor_class_matrix(
        cache_dir=str(args.cache_dir),
        tf_labels=sorted(tf_genes),
        adjacency=adjacency,
        TF_set=tf_genes,
        RBP_set=rbp_genes,
        PME_set=pme_genes,
        RAND_set=random_genes,
        force_recompute=args.force_recompute_cache,
        signif_out_prefix=args.output_dir / "tf_interactor_classes",
    )

    # Outputs
    pdf_out = args.output_dir / "Figure1_full.pdf"
    build_composite_figure(
        df_inter_classes=df_inter,
        df_frac_rbp=df_frac,
        df_tf_hubs=df_tf_hubs,
        df_rbp_hubs=df_rbp_hubs,
        outfile=pdf_out,
        middle=args.box_middle,   # <---
    )
    print(f"[saved] {pdf_out}")

    if args.also_png:
        png_out = args.output_dir / "Figure1_full.png"
        build_composite_figure(
            df_inter_classes=df_inter,
            df_frac_rbp=df_frac,
            df_tf_hubs=df_tf_hubs,
            df_rbp_hubs=df_rbp_hubs,
            outfile=png_out,       # <-- use png_out here
            middle=args.box_middle,
        )
        print(f"[saved] {png_out}")
        


if __name__ == "__main__":
    main()
