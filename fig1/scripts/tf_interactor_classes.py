#!/usr/bin/env python3
# file: scripts/tf_interactor_class_percentages_mean_only_vsRBP_stars.py

import argparse
import sys
import random
from pathlib import Path
from collections import Counter, defaultdict
from typing import Dict, Set, List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ============================= IO & Mapping ==================================

def load_mapping(path: Path) -> Dict[str, str]:
    """UniProt -> gene_name; majority vote if ambiguous."""
    df = pd.read_csv(path, sep="\t", header=None, names=["uniprot_id", "gene_name"], dtype=str)
    df = df.dropna().applymap(str.strip)
    winners: Dict[str, str] = {}
    for uid, sub in df.groupby("uniprot_id"):
        g = Counter(sub["gene_name"])
        winners[uid] = g.most_common(1)[0][0]
        if len(g) > 1:
            print(f"[warn] UniProt '{uid}' maps to multiple genes {dict(g)}; using '{winners[uid]}'.", file=sys.stderr)
    return winners


def load_list_uniprot(path: Path) -> List[str]:
    return pd.read_csv(path, sep="\t", header=None, names=["uniprot_id"], dtype=str)["uniprot_id"].dropna().str.strip().tolist()


def load_list_genes(path: Path) -> List[str]:
    return pd.read_csv(path, sep="\t", header=None, names=["gene_name"], dtype=str)["gene_name"].dropna().str.strip().tolist()


def map_or_keep(ids: List[str], mapping: Dict[str, str]) -> List[str]:
    """Map UniProt→gene if present; else keep UniProt (do not drop)."""
    return [mapping.get(x, x) for x in ids]


def load_and_map_ppis_keep_all(path: Path, mapping: Dict[str, str]) -> pd.DataFrame:
    """Map endpoints if possible; keep originals. Drop only self-loops; de-duplicate undirected edges."""
    df = pd.read_csv(path, sep="\t", header=None, names=["A", "B"], dtype=str).dropna()
    df["A"] = df["A"].str.strip(); df["B"] = df["B"].str.strip()
    u = df["A"].map(lambda x: mapping.get(x, x))
    v = df["B"].map(lambda x: mapping.get(x, x))
    dfm = pd.DataFrame({"u": u, "v": v})
    dfm = dfm[dfm["u"] != dfm["v"]].copy()
    dfm["x"] = dfm[["u", "v"]].min(axis=1)
    dfm["y"] = dfm[["u", "v"]].max(axis=1)
    dfm = dfm.drop_duplicates(subset=["x", "y"])[["x", "y"]].rename(columns={"x": "proteinA", "y": "proteinB"})
    return dfm


# ============================ Random selection ===============================

def sample_random_uniprot(universe_uniprot: List[str], k: int, seed: int) -> List[str]:
    """Sample k UniProt IDs from ALL; prefer w/o replacement; fallback to with replacement if k > N."""
    rng = random.Random(seed)
    n = len(universe_uniprot)
    if n == 0:
        return []
    if k <= n:
        return rng.sample(universe_uniprot, k)
    return [rng.choice(universe_uniprot) for _ in range(k)]


# ============================ Core computation ===============================

def build_adjacency_from_edges(ppi_edges: pd.DataFrame) -> Dict[str, Set[str]]:
    """Simple undirected adjacency dict (no NetworkX)."""
    adj: Dict[str, Set[str]] = defaultdict(set)
    for a, b in ppi_edges.itertuples(index=False, name=None):
        adj[a].add(b); adj[b].add(a)
    return adj


def compute_per_tf_lists_and_metrics(
    tf_labels: List[str],
    adjacency: Dict[str, Set[str]],
    TF_set: Set[str],
    RBP_set: Set[str],
    PME_set: Set[str],
    RAND_set: Set[str],
) -> pd.DataFrame:
    """For ALL TFs (zero-degree included), compute sublists, counts, fractions, and percentages."""
    rows: List[Dict[str, object]] = []
    for tf in tf_labels:
        neigh = sorted(adjacency.get(tf, set()))
        n_total = len(neigh)
        ns = set(neigh)

        sub_rbp = sorted(ns & RBP_set)
        sub_tf  = sorted(ns & TF_set)
        sub_pme = sorted(ns & PME_set)
        sub_rnd = sorted(ns & RAND_set)

        n_rbp, n_tf, n_pme, n_rnd = len(sub_rbp), len(sub_tf), len(sub_pme), len(sub_rnd)
        pct = (lambda n: 100.0 * n / n_total if n_total > 0 else 0.0)

        rows.append({
            "TF": tf,
            "neighbors": ";".join(neigh),
            "n_total": n_total,

            "rbp_list": ";".join(sub_rbp),
            "tf_list": ";".join(sub_tf),
            "pme_list": ";".join(sub_pme),
            "random_list": ";".join(sub_rnd),

            "n_rbp": n_rbp, "frac_rbp": (n_rbp / n_total if n_total > 0 else 0.0), "pct_rbp": pct(n_rbp),
            "n_tf":  n_tf,  "frac_tf":  (n_tf  / n_total if n_total > 0 else 0.0), "pct_tf":  pct(n_tf),
            "n_pme": n_pme, "frac_pme": (n_pme / n_total if n_total > 0 else 0.0), "pct_pme": pct(n_pme),
            "n_random": n_rnd, "frac_random": (n_rnd / n_total if n_total > 0 else 0.0), "pct_random": pct(n_rnd),
        })
    return pd.DataFrame(rows)


# ============================= Significance (vs RBP) =========================

def mann_whitney_or_perm(x: np.ndarray, y: np.ndarray, n_perm: int = 5000, seed: int = 42) -> Tuple[float, float]:
    """Return (p_value, effect=median(x)-median(y)). Try SciPy MWU; fallback to permutation on medians."""
    x = np.asarray(x, dtype=float); y = np.asarray(y, dtype=float)
    effect = float(np.nanmedian(x) - np.nanmedian(y))
    try:
        from scipy.stats import mannwhitneyu
        res = mannwhitneyu(x, y, alternative="two-sided")
        return float(res.pvalue), effect
    except Exception:
        rng = np.random.default_rng(seed)
        pooled = np.concatenate([x, y]); n_x = len(x); obs = abs(effect); ge = 0
        for _ in range(n_perm):
            rng.shuffle(pooled)
            x_p = pooled[:n_x]; y_p = pooled[n_x:]
            stat = abs(float(np.nanmedian(x_p) - np.nanmedian(y_p)))
            if stat >= obs:
                ge += 1
        p = (ge + 1) / (n_perm + 1)
        return float(p), effect
def _bh_fdr(pvals: np.ndarray) -> np.ndarray:
    """Benjamini–Hochberg FDR correction; returns q-values (same shape)."""
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


def paired_signflip_pvalue(diffs: np.ndarray, *, stat: str = "median", n_perm: int = 10000, seed: int = 42) -> float:
    """
    Paired randomization test (sign-flip) for whether the paired differences are centered at 0.
    - diffs = (class_percent - rbp_percent) per TF
    - stat: "median" or "mean"
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
    # exact-ish Monte Carlo sign-flips
    for _ in range(n_perm):
        signs = rng.choice([-1.0, 1.0], size=d.size, replace=True)
        val = abs(float(f(d * signs)))
        if val >= obs:
            ge += 1
    return float((ge + 1) / (n_perm + 1))


def compute_vs_rbp_tests_wide(df: pd.DataFrame, *, seed: int = 42, n_perm: int = 10000, drop_zero_degree: bool = True) -> pd.DataFrame:
    """
    Compute multiple tests comparing each class vs RBP.
    Uses:
      - MWU (unpaired) on distributions across TFs (as you currently do)
      - Wilcoxon signed-rank (paired) if SciPy exists
      - Sign-flip permutation test (paired) on diffs (always)

    Returns a table with p-values + effect sizes.
    """
    dff = df.copy()
    if drop_zero_degree and "n_total" in dff.columns:
        dff = dff.loc[dff["n_total"] > 0].copy()

    rbp = dff["pct_rbp"].to_numpy(dtype=float)

    out = []
    for cls, col in [("TF", "pct_tf"), ("PME", "pct_pme"), ("Random", "pct_random")]:
        arr = dff[col].to_numpy(dtype=float)

        # Effects
        eff_median_unpaired = float(np.nanmedian(arr) - np.nanmedian(rbp))
        eff_mean_unpaired   = float(np.nanmean(arr) - np.nanmean(rbp))
        diffs = arr - rbp
        eff_median_paired = float(np.nanmedian(diffs))
        eff_mean_paired   = float(np.nanmean(diffs))

        # 1) MWU (unpaired)
        p_mwu = float("nan")
        try:
            from scipy.stats import mannwhitneyu
            p_mwu = float(mannwhitneyu(arr, rbp, alternative="two-sided").pvalue)
        except Exception:
            # if you want: keep your existing mann_whitney_or_perm fallback instead
            p_mwu = float("nan")

        # 2) Wilcoxon (paired)
        p_wilcoxon = float("nan")
        try:
            from scipy.stats import wilcoxon
            # wilcoxon expects paired samples; it will drop zero diffs automatically depending on zero_method
            p_wilcoxon = float(wilcoxon(diffs, alternative="two-sided", zero_method="wilcox").pvalue)
        except Exception:
            p_wilcoxon = float("nan")

        # 3) Paired sign-flip permutation (always)
        p_signflip_median = paired_signflip_pvalue(diffs, stat="median", n_perm=n_perm, seed=seed)
        p_signflip_mean   = paired_signflip_pvalue(diffs, stat="mean",   n_perm=n_perm, seed=seed + 1)

        out.append({
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

    res = pd.DataFrame(out)

    # Optional: FDR correction per test family
    res["q_mwu_bh"] = _bh_fdr(res["p_mwu"].to_numpy())
    res["q_wilcoxon_bh"] = _bh_fdr(res["p_wilcoxon"].to_numpy())
    res["q_signflip_median_bh"] = _bh_fdr(res["p_signflip_median"].to_numpy())
    res["q_signflip_mean_bh"] = _bh_fdr(res["p_signflip_mean"].to_numpy())

    return res


def compute_vs_rbp_pvalues(df_long: pd.DataFrame, seed: int = 42, n_perm: int = 5000) -> pd.DataFrame:
    """
    df_long: ['class','percent'] with classes in {'RBP','TF','PME','Random'}.
    Compare TF/PME/Random vs RBP only. Returns class, comparator=RBP, pvalue, effect_median_diff.
    """
    out = []
    rbp = df_long.loc[df_long["class"] == "RBP", "percent"].dropna().to_numpy()
    for cls in ["TF", "PME", "Random"]:
        arr = df_long.loc[df_long["class"] == cls, "percent"].dropna().to_numpy()
        if arr.size == 0 or rbp.size == 0:
            p, eff = float("nan"), float("nan")
        else:
            p, eff = mann_whitney_or_perm(arr, rbp, n_perm=n_perm, seed=seed)
        out.append({"class": cls, "comparator": "RBP", "pvalue": p, "effect_median_diff": eff})
    return pd.DataFrame(out)


def p_to_stars(p: float) -> str:
    """Map p-value to stars; return '' for non-significant or NaN."""
    if not np.isfinite(p):
        return ""
    if p < 1e-4: return "****"
    if p < 1e-3: return "***"
    if p < 1e-2: return "**"
    if p < 5e-2: return "*"
    return ""


# ============================== Plotting (vertical) ==========================

# Colors as requested
COLOR_TF_NODE  = "#fdae6b"  # TF = orange
COLOR_RBP_NODE = "#1f78b4"  # RBP = blue
COLOR_PME      = "#ffd54f"  # PME = yellow
COLOR_RANDOM   = "#2ca02c"  # Random = green

def _clean_axes(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", linestyle=":", linewidth=0.6, alpha=0.35)

def plot_vertical_boxplot_mean_only(df: pd.DataFrame, out_png: Path, dpi: int = 500) -> None:
    """
    Vertical boxplot: X=class (RBP, TF, PME, Random), Y=percent (0..100).
    - Outliers hidden (showfliers=False)
    - Mean line in black; median hidden
    - Legend with class colors + mean
    - High DPI and clean aesthetics
    """
    if df.empty:
        fig = plt.figure(figsize=(9, 5), dpi=dpi); plt.text(0.5, 0.5, "No data", ha="center", va="center"); plt.axis("off")
        fig.savefig(out_png, bbox_inches="tight"); plt.close(fig); return

    long = pd.melt(
        df[["TF", "pct_rbp", "pct_tf", "pct_pme", "pct_random"]],
        id_vars=["TF"], var_name="class", value_name="percent"
    )
    class_map = {"pct_rbp": "RBP", "pct_tf": "TF", "pct_pme": "PME", "pct_random": "Random"}
    long["class"] = long["class"].map(class_map)

    order = ["RBP", "TF", "PME", "Random"]
    data  = [long.loc[long["class"] == c, "percent"].values for c in order]

    fig, ax = plt.subplots(figsize=(9, 6), dpi=dpi)
    bp = ax.boxplot(
        data,
        labels=order,
        vert=True,
        patch_artist=True,
        showfliers=False,            # hide outliers
        showmeans=True,
        meanline=True,
        meanprops={"color": "black", "linewidth": 2.0},
        whiskerprops={"color": "black", "linewidth": 1.4},
        capprops={"color": "black", "linewidth": 1.8},
        medianprops={"color": "black", "linewidth": 0.0},  # hide median
    )

    colors = {"RBP": COLOR_RBP_NODE, "TF": COLOR_TF_NODE, "PME": COLOR_PME, "Random": COLOR_RANDOM}
    for patch, cls in zip(bp["boxes"], order):
        patch.set_facecolor(colors[cls]); patch.set_alpha(0.95); patch.set_edgecolor("black"); patch.set_linewidth(1.0)

    ax.set_ylim(0, 100)
    ax.set_ylabel("% of interactors of TFs")
    ax.set_xlabel("Interactor class")
    ax.set_title("Per-TF % of interactor classes")
    _clean_axes(ax)

    # Legend: class colors + mean
    from matplotlib.lines import Line2D
    legend_elems = [
        Line2D([0], [0], marker='s', color='w', label='RBP', markerfacecolor=COLOR_RBP_NODE, markersize=12),
        Line2D([0], [0], marker='s', color='w', label='TF', markerfacecolor=COLOR_TF_NODE, markersize=12),
        Line2D([0], [0], marker='s', color='w', label='PME', markerfacecolor=COLOR_PME, markersize=12),
        Line2D([0], [0], marker='s', color='w', label='Random', markerfacecolor=COLOR_RANDOM, markersize=12),
        Line2D([0], [0], color='black', lw=2.0, label='Mean'),
    ]
    ax.legend(handles=legend_elems, frameon=False, loc="upper right")

    fig.tight_layout()
    fig.savefig(out_png, bbox_inches="tight")
    plt.close(fig)


def plot_vertical_boxplot_with_stars_vs_rbp(df: pd.DataFrame, pvals_df: pd.DataFrame, out_png: Path, dpi: int = 500) -> None:
    """
    Same vertical boxplot (mean only, no fliers), plus **stars only** for RBP vs each of TF/PME/Random.
    No numeric p-values on the figure. Stars placed to avoid overlap; y-limit extended as needed.
    """
    long = pd.melt(
        df[["TF", "pct_rbp", "pct_tf", "pct_pme", "pct_random"]],
        id_vars=["TF"], var_name="class", value_name="percent"
    )
    class_map = {"pct_rbp": "RBP", "pct_tf": "TF", "pct_pme": "PME", "pct_random": "Random"}
    long["class"] = long["class"].map(class_map)

    order = ["RBP", "TF", "PME", "Random"]
    data  = [long.loc[long["class"] == c, "percent"].values for c in order]

    fig, ax = plt.subplots(figsize=(9, 6), dpi=dpi)
    bp = ax.boxplot(
        data,
        labels=order,
        vert=True,
        patch_artist=True,
        showfliers=False,
        showmeans=True,
        meanline=True,
        meanprops={"color": "black", "linewidth": 2.0},
        whiskerprops={"color": "black", "linewidth": 1.4},
        capprops={"color": "black", "linewidth": 1.8},
        medianprops={"color": "black", "linewidth": 0.0},  # hide median
    )
    colors = {"RBP": COLOR_RBP_NODE, "TF": COLOR_TF_NODE, "PME": COLOR_PME, "Random": COLOR_RANDOM}
    for patch, cls in zip(bp["boxes"], order):
        patch.set_facecolor(colors[cls]); patch.set_alpha(0.95); patch.set_edgecolor("black"); patch.set_linewidth(1.0)

    ax.set_ylim(0, 100)
    ax.set_ylabel("% of interactors of TFs")
    ax.set_xlabel("Interactor class")
    ax.set_title("TF interactor classes: significance vs RBP (stars)")
    _clean_axes(ax)

    # Add stars centered between RBP (x=1) and each comparator (2,3,4).
    ymax = max([np.nanmax(v) if len(v) else 0 for v in data] + [0])
    # Base far above boxes but allow headroom; extend ylim if needed.
    base = max(75.0, float(ymax) + 5.0)
    step = 3.5
    idx = {"RBP": 1, "TF": 2, "PME": 3, "Random": 4}

    # Compute star strings
    stars = {}
    for cls in ["TF", "PME", "Random"]:
        row = pvals_df[pvals_df["class"] == cls]
        s = ""
        if not row.empty:
            p = row["pvalue"].iloc[0]
            s = p_to_stars(p)
        stars[cls] = s

    # Calculate required ylim extension
    needed_top = base + step * 2 + 3.0  # room for top-most stars
    if needed_top > 100:
        ax.set_ylim(0, needed_top)

    # Draw stars (skip NS to reduce clutter)
    for i, cls in enumerate(["TF", "PME", "Random"]):
        s = stars.get(cls, "")
        if not s:
            continue
        x_mid = 0.5 * (idx["RBP"] + idx[cls])
        y_pos = base + i * step
        ax.text(x_mid, y_pos, s, ha="center", va="bottom", fontsize=14, fontweight="bold")

    # Legend in lower-right to avoid overlapping top stars
    from matplotlib.lines import Line2D
    legend_elems = [
        Line2D([0], [0], marker='s', color='w', label='RBP', markerfacecolor=COLOR_RBP_NODE, markersize=12),
        Line2D([0], [0], marker='s', color='w', label='TF', markerfacecolor=COLOR_TF_NODE, markersize=12),
        Line2D([0], [0], marker='s', color='w', label='PME', markerfacecolor=COLOR_PME, markersize=12),
        Line2D([0], [0], marker='s', color='w', label='Random', markerfacecolor=COLOR_RANDOM, markersize=12),
        Line2D([0], [0], color='black', lw=2.0, label='Mean'),
    ]
    ax.legend(handles=legend_elems, frameon=False, loc="lower right")

    fig.tight_layout()
    fig.savefig(out_png, bbox_inches="tight")
    plt.close(fig)


# =================================== Main ====================================

def main():
    ap = argparse.ArgumentParser(
        description="Per-TF interactor lists & percentages (RBP/TF/PME/Random), mean-only vertical boxplot + stars vs RBP."
    )
    ap.add_argument("--tf", type=Path, default=Path("TF_list.txt"), help="TF list (UniProt IDs)")
    ap.add_argument("--rbp", type=Path, default=Path("RBP_list.txt"), help="RBP list (gene names)")
    ap.add_argument("--pme", type=Path, default=Path("PTM_list.txt"), help="PME list (UniProt IDs)")
    ap.add_argument("--all", type=Path, default=Path("all_unids_human.txt"), help="ALL list (UniProt IDs)")
    ap.add_argument("--map", type=Path, dest="mapping", default=Path("UNIPROTIDS_GENENAME.txt"), help="UniProt→gene mapping (2 cols)")
    ap.add_argument("--ppi", type=Path, default=Path("ppis.txt"), help="PPIs (two cols, tab-separated)")
    ap.add_argument("--out_prefix", default="tf_interactor_classes", help="Output file prefix")
    ap.add_argument("--seed", type=int, default=42, help="Random seed for Random selection from ALL")
    ap.add_argument("--n-perm", type=int, default=5000, help="Permutations for fallback test when SciPy is unavailable")
    ap.add_argument("--no-plot", action="store_true", help="Skip plotting")
    args = ap.parse_args()

    # Load & map inputs
    mapping = load_mapping(args.mapping)
    tf_uniprot  = load_list_uniprot(args.tf)
    rbp_genes   = set(load_list_genes(args.rbp))         # already gene names
    pme_uniprot = load_list_uniprot(args.pme)
    all_uniprot = load_list_uniprot(args.all)

    tf_labels: List[str] = map_or_keep(tf_uniprot, mapping)
    pme_labels: Set[str] = set(map_or_keep(pme_uniprot, mapping))

    # Random list from ALL (size = len(RBP)), then map-or-keep
    rand_uniprot = sample_random_uniprot(all_uniprot, k=len(rbp_genes), seed=args.seed)
    rand_labels: Set[str] = set(map_or_keep(rand_uniprot, mapping))

    # PPI edges (mapped, no filtering) → adjacency
    ppis_mapped = load_and_map_ppis_keep_all(args.ppi, mapping)
    adjacency = build_adjacency_from_edges(ppis_mapped)

    # Sets in the same label space
    TF_set, RBP_set, PME_set, RAND_set = set(tf_labels), set(rbp_genes), set(pme_labels), set(rand_labels)

    # Per-TF lists + metrics for ALL TFs
    df = compute_per_tf_lists_and_metrics(tf_labels, adjacency, TF_set, RBP_set, PME_set, RAND_set)

    # Save tables
    outbase = Path(args.out_prefix)
    df_out = outbase.with_name(f"{outbase.name}_per_TF_interactor_percentages.tsv")
    df.to_csv(df_out, sep="\t", index=False)

    rand_uniprot_out = outbase.with_name(f"{outbase.name}_random_uniprot.txt")
    rand_labels_out  = outbase.with_name(f"{outbase.name}_random_labels.txt")
    pd.Series(rand_uniprot).to_csv(rand_uniprot_out, index=False, header=False)
    pd.Series(sorted(rand_labels)).to_csv(rand_labels_out, index=False, header=False)

    ppis_out = outbase.with_name(f"{outbase.name}_mapped_ppis.tsv")
    ppis_mapped.to_csv(ppis_out, sep="\t", index=False)

    if not args.no_plot:
        # 1) Vertical mean-only boxplot (no outliers)
        box_png = outbase.with_name(f"{outbase.name}_boxplot_vertical_mean_only.png")
        plot_vertical_boxplot_mean_only(df, box_png, dpi=500)

        # 2) P-values vs RBP (separate chart, stars only)
        long = pd.melt(
            df[["TF", "pct_rbp", "pct_tf", "pct_pme", "pct_random"]],
            id_vars=["TF"], var_name="class", value_name="percent"
        )
        class_map = {"pct_rbp": "RBP", "pct_tf": "TF", "pct_pme": "PME", "pct_random": "Random"}
        long["class"] = long["class"].map(class_map)

        pvals_df = compute_vs_rbp_pvalues(long, seed=args.seed, n_perm=args.n_perm)
        pvals_tsv = outbase.with_name(f"{outbase.name}_signif_vs_RBP.tsv")
        pvals_df.to_csv(pvals_tsv, sep="\t", index=False)

        box_sig_png = outbase.with_name(f"{outbase.name}_boxplot_vertical_mean_only_stars_vsRBP.png")
        plot_vertical_boxplot_with_stars_vs_rbp(df, pvals_df, box_sig_png, dpi=500)

    # Console summary (no p-values printed)
    print(f"[done] TFs in input: {len(tf_labels)}; rows in TSV: {df.shape[0]} (zero-degree TFs included)")
    print(f"[saved] metrics            → {df_out}")
    print(f"[saved] random (uniprot)   → {rand_uniprot_out}")
    print(f"[saved] random (labels)    → {rand_labels_out}")
    print(f"[saved] mapped PPIs        → {ppis_out}")
    if not args.no_plot:
        print(f"[saved] box plot           → {box_png}")
        print(f"[saved] significance table → {pvals_tsv}")
        print(f"[saved] significance plot  → {box_sig_png}")


if __name__ == "__main__":
    main()
