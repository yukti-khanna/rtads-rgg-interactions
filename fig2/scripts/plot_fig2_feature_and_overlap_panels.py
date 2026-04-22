#!/usr/bin/env python3
from __future__ import annotations

import argparse
import itertools
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Set, Tuple

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu

# ----------------------------- Project paths ---------------------------------
def _resolve_project_root() -> Path:
    here = Path(__file__).resolve().parent
    return here.parent if here.name == "scripts" else here

PROJECT_ROOT = _resolve_project_root()
INPUT_FASTA_DIR = PROJECT_ROOT / "inputs" / "fasta"
INPUT_TABLES_DIR = PROJECT_ROOT / "inputs" / "tables"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
OUTPUT_MAIN_DIR = OUTPUT_DIR / "main"
OUTPUT_SUPP_DIR = OUTPUT_DIR / "supplement"
OUTPUT_BLAST_DIR = OUTPUT_DIR / "blast"
OUTPUT_CACHE_DIR = OUTPUT_DIR / "cache"

for _p in [OUTPUT_DIR, OUTPUT_MAIN_DIR, OUTPUT_SUPP_DIR, OUTPUT_BLAST_DIR, OUTPUT_CACHE_DIR]:
    _p.mkdir(parents=True, exist_ok=True)

# ----------------------------- Style -----------------------------------------
COLOR_BLUE = "#0072B2"
COLOR_ORANGE = "#E69F00"
COLOR_GREEN = "#009E73"
COLOR_SKYBLUE = "#56B4E9"
COLOR_VERMILLION = "#D55E00"
COLOR_PURPLE = "#CC79A7"
COLOR_BLACK = "#000000"
COLOR_GREY = "#BFBFBF"

DATASET_NAMES_ORDERED = ["R-TADs", "Soto 2022", "Kotha HC 2023", "Kotha GSL 2023", "Staller 2022"]
DATASET_COLOR_MAP = {
    "R-TADs": COLOR_BLUE,
    "Soto 2022": COLOR_GREEN,
    "Kotha HC 2023": COLOR_PURPLE,
    "Kotha GSL 2023": COLOR_SKYBLUE,
    "Staller 2022": COLOR_VERMILLION,
}
DATASET_ABBR = {
    "R-TADs": "R",
    "Soto 2022": "S",
    "Kotha HC 2023": "HC",
    "Kotha GSL 2023": "GSL",
    "Staller 2022": "St",
}

# ----------------------------- Default features for boxplots ----------------
FEATURE_COLUMNS_DEFAULT = ["aromatic", "WF_complexity", "SCD", "kappa*"]

mpl.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 6.5,
    "axes.labelsize": 6.5,
    "axes.titlesize": 7.0,
    "xtick.labelsize": 6.0,
    "ytick.labelsize": 6.0,
    "legend.fontsize": 6.0,
    "axes.linewidth": 0.6,
    "xtick.major.width": 0.5,
    "ytick.major.width": 0.5,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "svg.fonttype": "none",
})

DEFAULT_RTADS_FASTA = INPUT_FASTA_DIR / "pred_tads_15aa.fasta"
DEFAULT_FILE_PAIRS: List[Tuple[str, Tuple[str, str]]] = [
    ("blast_tads_vs_soto.sameprot_sig_all.tsv", ("R-TADs", "Soto 2022")),
    ("blast_tads_vs_kothaHC.sameprot_sig_all.tsv", ("R-TADs", "Kotha HC 2023")),
    ("blast_tads_vs_gsl.sameprot_sig_all.tsv", ("R-TADs", "Kotha GSL 2023")),
    ("blast_tads_vs_staller2022.sameprot_sig_all.tsv", ("R-TADs", "Staller 2022")),
    ("blast_staller22_vs_gsl.sameprot_all.tsv", ("Staller 2022", "Kotha GSL 2023")),
    ("blast_staller22_vs_soto.sameprot_sig_all.tsv", ("Staller 2022", "Soto 2022")),
    ("blast_staller22_vs_kothaHC.sameprot_sig_all.tsv", ("Staller 2022", "Kotha HC 2023")),
    ("blast_gsl_vs_soto.sameprot_sig_all.tsv", ("Kotha GSL 2023", "Soto 2022")),
    ("blast_gsl_vs_kothaHC.sameprot_sig_all.tsv", ("Kotha GSL 2023", "Kotha HC 2023")),
    ("blast_kothastaller_hc_vs_soto.sameprot_sig_all.tsv", ("Kotha HC 2023", "Soto 2022")),
]

SUPP_BLAST_DATASETS: List[Tuple[str, Tuple[str, ...], str]] = [
    ("ADpred", ("blast_tads_vs_adpred.tsv",), COLOR_GREY),
    ("Soto 2022", ("blast_tads_vs_soto.sameprot_sig_all.tsv",), COLOR_GREEN),
    ("Kotha HC 2023", ("blast_tads_vs_kothaHC.sameprot_sig_all.tsv",), COLOR_PURPLE),
    ("Kotha GSL 2023", ("blast_tads_vs_gsl.sameprot_sig_all.tsv",), COLOR_SKYBLUE),
    ("Staller 2022", ("blast_tads_vs_staller2022.sameprot_sig_all.tsv",), COLOR_VERMILLION),
]
# ----------------------------- Utilities -------------------------------------
def _read_table_auto(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(f"Input not found: {path}")
    return pd.read_csv(path, sep=None, engine="python")

def _clean_numeric(x: Sequence) -> np.ndarray:
    a = np.asarray(pd.to_numeric(pd.Series(x), errors="coerce"), dtype=float)
    return a[np.isfinite(a)]

def mwu(a: Sequence, b: Sequence) -> Tuple[float, float]:
    a = _clean_numeric(a)
    b = _clean_numeric(b)
    if a.size == 0 or b.size == 0:
        return float("nan"), float("nan")
    res = mannwhitneyu(a, b, alternative="two-sided")
    return float(res.statistic), float(res.pvalue)

def stars(p: float) -> str:
    if not isinstance(p, (float, np.floating)) or not np.isfinite(p):
        return ""
    if p <= 1e-4: return "****"
    if p <= 1e-3: return "***"
    if p <= 1e-2: return "**"
    if p <= 0.05: return "*"
    return ""

def _ensure_columns(df: pd.DataFrame, cols: Sequence[str], name: str) -> None:
    miss = [c for c in cols if c not in df.columns]
    if miss:
        raise ValueError(f"Missing column(s) {miss} in {name} table.")

def _parse_rtad_protein(header: str) -> str:
    base = header
    if ":" in base:
        base = base.split(":", 1)[0]
    if "-" in base:
        base = base.split("-", 1)[0]
    return base

def _load_rtad_set(fasta_path: Path) -> Set[str]:
    if not fasta_path.is_file():
        raise FileNotFoundError(f"FASTA not found: {fasta_path}")
    proteins: Set[str] = set()
    with open(fasta_path, "r", encoding="utf-8") as fh:
        for line in fh:
            if line.startswith(">"):
                proteins.add(_parse_rtad_protein(line[1:].strip()))
    if not proteins:
        raise ValueError(f"No R-TAD protein headers found in {fasta_path}")
    return proteins


def _read_qprots_from_tsv(tsv_path: Path) -> Set[str]:
    df = pd.read_csv(tsv_path, sep="\t")
    if "qprot" not in df.columns:
        raise ValueError(f"'qprot' column missing in {tsv_path}")
    return set(df["qprot"].astype(str).dropna())


# ---- Helper: Load membership matrix file if present ----
def _load_membership_matrix_file(path: Path) -> Tuple[Dict[str, Set[str]], pd.DataFrame]:
    if not path.is_file():
        raise FileNotFoundError(f"Membership matrix not found: {path}")
    mem = pd.read_csv(path, sep="\t")
    col_map = {
        "R_TADs": "R-TADs",
        "Soto_2022": "Soto 2022",
        "KothaHC_2023": "Kotha HC 2023",
        "KothaGSL_2023": "Kotha GSL 2023",
        "Staller_2022": "Staller 2022",
    }
    _ensure_columns(mem, ["id"] + list(col_map.keys()), "membership_matrix")
    sets: Dict[str, Set[str]] = {v: set() for v in col_map.values()}
    for _, row in mem.iterrows():
        pid = str(row["id"])
        for c_raw, name in col_map.items():
            if int(row.get(c_raw, 0)) == 1:
                sets[name].add(pid)
    return sets, mem


# Prefer an existing membership matrix from several possible locations
def _default_membership_matrix() -> Path:
    candidates = [
        OUTPUT_BLAST_DIR / "venn_RTADs_ADdatasets_fromBLAST_membership_matrix.tsv",
        OUTPUT_BLAST_DIR / "figS2F_overlap_combinations_membership_matrix.tsv",
        PROJECT_ROOT / "venn_RTADs_ADdatasets_fromBLAST_membership_matrix.tsv",
    ]
    for p in candidates:
        if p.is_file():
            return p
    return candidates[0]


# ----------------------------- Boxplots --------------------------------------
def _default_feature_labels(features: Sequence[str]) -> Dict[str, str]:
    pretty: Dict[str, str] = {}
    for f in features:
        if f.lower().startswith("aromatic"):
            pretty[f] = "Aromatic fraction"
        elif f == "WF_complexity":
            pretty[f] = "WF complexity"
        elif f.upper() == "SCD":
            pretty[f] = "SCD"
        elif f.lower().startswith("kappa*"):
            pretty[f] = "κ"
        else:
            pretty[f] = f
    return pretty

def _parse_feature_map(map_str: Optional[str]) -> Optional[Mapping[str, str]]:
    if not map_str:
        return None
    out: Dict[str, str] = {}
    for kv in map_str.split(","):
        if "=" in kv:
            k, v = kv.split("=", 1)
            out[k.strip()] = v.strip()
    return out

def load_feature_tables(a: Path, b: Path) -> Tuple[pd.DataFrame, pd.DataFrame]:
    return _read_table_auto(a), _read_table_auto(b)

def _make_tidy_plot_df(
    d_a: pd.DataFrame, d_b: pd.DataFrame,
    feature_map: Mapping[str, str], group_a: str, group_b: str
) -> pd.DataFrame:
    recs = []
    for feat_label, col in feature_map.items():
        for v in pd.to_numeric(d_a[col], errors="coerce").dropna():
            recs.append({"feature": feat_label, "value": float(v), "group": group_a})
        for v in pd.to_numeric(d_b[col], errors="coerce").dropna():
            recs.append({"feature": feat_label, "value": float(v), "group": group_b})
    return pd.DataFrame.from_records(recs)

def compute_stats(d_a: pd.DataFrame, d_b: pd.DataFrame, feature_map: Mapping[str, str]) -> pd.DataFrame:
    rows = []
    for feat_label, col in feature_map.items():
        U, p = mwu(d_a[col], d_b[col])
        rows.append({"feature": feat_label, "U": U, "p_value": p})
    return pd.DataFrame(rows)

def load_or_compute_boxplot_matrices(
    group_a_path: Path, group_b_path: Path, features: Sequence[str],
    feature_map_str: Optional[str], feature_labels_str: Optional[str],
    out_prefix_box: str, force_recompute: bool, group_a_name: str, group_b_name: str,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    plot_cache = OUTPUT_CACHE_DIR / (Path(out_prefix_box).name + "_plot_data.tsv")
    stats_cache = OUTPUT_CACHE_DIR / (Path(out_prefix_box).name + "_stats.tsv")

    if not force_recompute and plot_cache.is_file() and stats_cache.is_file():
        return pd.read_csv(plot_cache, sep="\t"), pd.read_csv(stats_cache, sep="\t")

    d_a, d_b = load_feature_tables(group_a_path, group_b_path)
    feat_map = _parse_feature_map(feature_map_str)
    if feat_map:
        feature_map = dict(feat_map)
    else:
        feature_map = {f: f for f in features}
        pretty = _default_feature_labels(list(feature_map.keys()))
        for old in list(feature_map.keys()):
            new = pretty.get(old, old)
            if new != old:
                feature_map[new] = feature_map.pop(old)

    if feature_labels_str:
        for kv in feature_labels_str.split(","):
            if "=" in kv:
                k, v = kv.split("=", 1)
                if k.strip() in feature_map:
                    src = feature_map.pop(k.strip())
                    feature_map[v.strip()] = src

    _ensure_columns(d_a, feature_map.values(), group_a_name)
    _ensure_columns(d_b, feature_map.values(), group_b_name)

    plot_df = _make_tidy_plot_df(d_a, d_b, feature_map, group_a_name, group_b_name)
    stats_df = compute_stats(d_a, d_b, feature_map)
    plot_df.to_csv(plot_cache, sep="\t", index=False)
    stats_df.to_csv(stats_cache, sep="\t", index=False)
    return plot_df, stats_df

def _boxplot_one(ax: plt.Axes, values_a, values_b, ylabel: str, labels: Sequence[str], box_width: float):
    groups = [_clean_numeric(values_a), _clean_numeric(values_b)]
    bp = ax.boxplot(
        groups, labels=labels, patch_artist=True, widths=box_width, whis=1.5, showfliers=False,
        medianprops=dict(linewidth=0.8, color="black"), boxprops=dict(linewidth=0.6),
        whiskerprops=dict(linewidth=0.6), capprops=dict(linewidth=0.6),
    )
    for patch, col in zip(bp["boxes"], [COLOR_BLUE, COLOR_ORANGE]):
        patch.set_facecolor(col)
        patch.set_edgecolor(COLOR_BLACK)

    rng = np.random.default_rng(123)
    for i, y in enumerate(groups, start=1):
        if y.size:
            x = np.clip(rng.normal(loc=i, scale=0.03, size=y.size), i - 0.08, i + 0.08)
            ax.scatter(x, y, s=5.5, alpha=0.30, edgecolors="none")
    vals = [g for g in groups if g.size > 0]
    if vals:
        ymax = max(v.max() for v in vals)
        ymin = min(v.min() for v in vals)
        span = (ymax - ymin) if ymax > ymin else 1.0
        _u, p = mwu(groups[0], groups[1])
        lab = stars(p)
        if lab:
            yb = ymax + 0.08 * span
            ax.plot([1, 1, 2, 2], [yb, yb + 0.04 * span, yb + 0.04 * span, yb], lw=0.8, c="black")
            ax.text(1.5, yb + 0.045 * span, lab, ha="center", va="bottom")
        ax.set_ylim(ymin - 0.05 * span, ymax + 0.30 * span)
    ax.set_ylabel(ylabel)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

def plot_boxpanels_two_group(
    plot_df: pd.DataFrame, out_prefix_box: str, order_features: Optional[Sequence[str]],
    labels: Sequence[str], box_width: float, group_a: str, group_b: str,
) -> Tuple[Path, Path]:
    feats_all = list(pd.unique(plot_df["feature"]))
    feats = [f for f in (order_features or feats_all[:4]) if f in feats_all][:4]
    fig, axes = plt.subplots(2, 2, figsize=(4.8, 2.3))
    axes = axes.flatten()
    for ax, feat in zip(axes, feats):
        sub = plot_df[plot_df["feature"] == feat]
        _boxplot_one(ax, sub.loc[sub["group"] == group_a, "value"].values,
                     sub.loc[sub["group"] == group_b, "value"].values, str(feat), labels, box_width)
    for ax in axes[len(feats):]:
        ax.axis("off")
    pdf_path = Path(f"{out_prefix_box}.pdf")
    png_path = Path(f"{out_prefix_box}.png")
    svg_path = Path(f"{out_prefix_box}.svg")
    pdf_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(pdf_path)
    fig.savefig(png_path, dpi=600)
    fig.savefig(svg_path)
    plt.close(fig)
    return pdf_path, png_path

# ----------------------------- Membership/overlap ----------------------------
def load_or_compute_membership_matrix(
    rtads_fasta: Path, file_pairs: Sequence[Tuple[str, Tuple[str, str]]],
    out_prefix_venn: str, restrict_to_rtads: bool, force_recompute: bool,
) -> Tuple[Dict[str, Set[str]], pd.DataFrame]:
    cache_path = OUTPUT_BLAST_DIR / (Path(out_prefix_venn).name + "_membership_matrix.tsv")
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    if not force_recompute and cache_path.is_file():
        mem = pd.read_csv(cache_path, sep="\t")
        col_map = {
            "R_TADs": "R-TADs", "Soto_2022": "Soto 2022", "KothaHC_2023": "Kotha HC 2023",
            "KothaGSL_2023": "Kotha GSL 2023", "Staller_2022": "Staller 2022",
        }
        sets = {v: set() for v in col_map.values()}
        for _, row in mem.iterrows():
            pid = str(row["id"])
            for c_raw, name in col_map.items():
                if int(row.get(c_raw, 0)) == 1:
                    sets[name].add(pid)
        return sets, mem

    rtad_set = _load_rtad_set(rtads_fasta)
    sets: Dict[str, Set[str]] = {name: set() for name in DATASET_NAMES_ORDERED}
    sets["R-TADs"] = set(rtad_set)
    for fname, (ds_a, ds_b) in file_pairs:
        p = OUTPUT_BLAST_DIR / fname
        qprots = _read_qprots_from_tsv(p)
        sets[ds_a].update(qprots)
        sets[ds_b].update(qprots)

    if restrict_to_rtads:
        for k in ["Soto 2022", "Kotha HC 2023", "Kotha GSL 2023", "Staller 2022"]:
            sets[k] &= rtad_set

    all_ids = sorted(set().union(*sets.values()))
    rows = []
    for pid in all_ids:
        rows.append({
            "id": pid,
            "R_TADs": int(pid in sets["R-TADs"]),
            "Soto_2022": int(pid in sets["Soto 2022"]),
            "KothaHC_2023": int(pid in sets["Kotha HC 2023"]),
            "KothaGSL_2023": int(pid in sets["Kotha GSL 2023"]),
            "Staller_2022": int(pid in sets["Staller 2022"]),
        })
    mem = pd.DataFrame(rows)
    mem.to_csv(cache_path, sep="\t", index=False)
    return sets, mem

def _build_full_intersection_table(mem: pd.DataFrame, ordered_names: Sequence[str], include_empty: bool = False) -> pd.DataFrame:
    col_map = {
        "R-TADs": "R_TADs",
        "Soto 2022": "Soto_2022",
        "Kotha HC 2023": "KothaHC_2023",
        "Kotha GSL 2023": "KothaGSL_2023",
        "Staller 2022": "Staller_2022",
    }
    cols = [col_map[n] for n in ordered_names]
    _ensure_columns(mem, cols, "membership_matrix")
    observed = mem.groupby(cols, dropna=False).size().to_dict()

    rows = []
    for bits in itertools.product([0, 1], repeat=len(ordered_names)):
        if not include_empty and sum(bits) == 0:
            continue
        count = int(observed.get(tuple(bits), 0))
        present = [name for bit, name in zip(bits, ordered_names) if bit == 1]
        label = " + ".join(DATASET_ABBR[n] for n in present) if present else "none"
        rows.append({
            **{col: bit for col, bit in zip(cols, bits)},
            "count": count,
            "combination": label,
            "degree": int(sum(bits)),
        })
    # sort observed first by count desc, then degree desc, then label
    return pd.DataFrame(rows).sort_values(
        ["count", "degree", "combination"], ascending=[False, False, True]
    ).reset_index(drop=True)

def plot_detailed_overlap_upset(
    mem: pd.DataFrame, out_prefix: str, show_all_combinations: bool = True
) -> Tuple[Path, Path]:
    ordered_names = list(DATASET_NAMES_ORDERED)
    col_map = {
        "R-TADs": "R_TADs",
        "Soto 2022": "Soto_2022",
        "Kotha HC 2023": "KothaHC_2023",
        "Kotha GSL 2023": "KothaGSL_2023",
        "Staller 2022": "Staller_2022",
    }
    inter = _build_full_intersection_table(mem, ordered_names, include_empty=False)
    if not show_all_combinations:
        inter = inter[inter["count"] > 0].copy()
    n = len(inter)
    fig_w = max(8.0, 0.26 * n + 1.8)
    fig = plt.figure(figsize=(fig_w, 3.2))
    gs = fig.add_gridspec(nrows=2, ncols=1, height_ratios=[2.0, 1.0], hspace=0.06)
    ax_bar = fig.add_subplot(gs[0, 0])
    ax_mat = fig.add_subplot(gs[1, 0], sharex=ax_bar)

    x = np.arange(n)
    bar_colors = ["#000000" if c > 0 else "#D9D9D9" for c in inter["count"].to_numpy()]
    ax_bar.bar(x, inter["count"].to_numpy(), color=bar_colors, width=0.75)
    ax_bar.set_ylabel("Mapped regions")
    ax_bar.spines["top"].set_visible(False)
    ax_bar.spines["right"].set_visible(False)
    ax_bar.tick_params(axis="x", bottom=False, labelbottom=False)
    ymax = max(float(inter["count"].max()), 1.0)
    for xi, v in zip(x, inter["count"].to_numpy()):
        if v > 0:
            ax_bar.text(xi, v + ymax * 0.02, str(int(v)), ha="center", va="bottom", fontsize=5.5)

    ax_mat.set_ylim(-0.5, len(ordered_names) - 0.5)
    ax_mat.set_yticks(np.arange(len(ordered_names)))
    ax_mat.set_yticklabels(ordered_names)
    ax_mat.invert_yaxis()
    ax_mat.spines["top"].set_visible(False)
    ax_mat.spines["right"].set_visible(False)
    ax_mat.tick_params(axis="y", length=0)
    for xi in x:
        ax_mat.scatter(np.full(len(ordered_names), xi), np.arange(len(ordered_names)), s=12, color="#D0D0D0", zorder=1)
    for xi, (_, row) in enumerate(inter.iterrows()):
        present_idx = [i for i, name in enumerate(ordered_names) if int(row[col_map[name]]) == 1]
        if len(present_idx) > 1:
            ax_mat.plot([xi, xi], [min(present_idx), max(present_idx)], color="#606060", lw=0.8, zorder=2)
        for yi in present_idx:
            dataset_name = ordered_names[yi]
            ax_mat.scatter([xi], [yi], s=22, color=DATASET_COLOR_MAP[dataset_name],
                           edgecolors=COLOR_BLACK, linewidths=0.25, zorder=3)
    ax_mat.set_xticks(x)
    ax_mat.set_xticklabels(inter["combination"].tolist(), rotation=90)
    ax_mat.set_xlabel("Dataset combination")
    pdf_path = Path(f"{out_prefix}.pdf")
    png_path = Path(f"{out_prefix}.png")
    svg_path = Path(f"{out_prefix}.svg")
    pdf_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(pdf_path)
    fig.savefig(png_path, dpi=600)
    fig.savefig(svg_path)
    plt.close(fig)
    return pdf_path, png_path

def plot_main_d_support_summary(mem: pd.DataFrame, out_prefix_main: str) -> Tuple[Path, Path]:
    external_cols = ["Soto_2022", "KothaHC_2023", "KothaGSL_2023", "Staller_2022"]
    _ensure_columns(mem, ["R_TADs"] + external_cols, "membership_matrix")
    rtad_only = mem[mem["R_TADs"] == 1].copy()
    rtad_only["external_support"] = rtad_only[external_cols].sum(axis=1)
    labels = ["Unique\nto this study", "Shared with\n1 dataset", "Shared with\n≥2 datasets"]
    counts = [
        int((rtad_only["external_support"] == 0).sum()),
        int((rtad_only["external_support"] == 1).sum()),
        int((rtad_only["external_support"] >= 2).sum()),
    ]
    fig, ax = plt.subplots(figsize=(4.4, 2.8))
    bars = ax.bar(labels, counts, color=COLOR_BLUE, alpha=0.9)
    ax.set_ylabel("Predicted R-TAD regions")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ymax = max(counts) if counts else 1
    ax.set_ylim(0, ymax * 1.18)
    for bar, value in zip(bars, counts):
        ax.text(bar.get_x() + bar.get_width() / 2.0, value + ymax * 0.03, str(value), ha="center", va="bottom")
    pdf_path = Path(f"{out_prefix_main}.pdf")
    png_path = Path(f"{out_prefix_main}.png")
    svg_path = Path(f"{out_prefix_main}.svg")
    pdf_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(pdf_path)
    fig.savefig(png_path, dpi=600)
    fig.savefig(svg_path)
    plt.close(fig)
    return pdf_path, png_path

# ----------------------------- S2A and S2B -------------------------------------------

def _read_sig_blast_table(tsv_names: Sequence[str]) -> pd.DataFrame:
    tried = []
    for name in tsv_names:
        path = OUTPUT_BLAST_DIR / name
        tried.append(str(path))
        if not path.is_file():
            continue

        # First try as a normal tabular file with headers
        df = pd.read_csv(path, sep="\t")
        known_cols = {"qprot", "qseqid", "length", "align_len", "alignment_length", "alnlen", "alen", "qstart", "qend"}
        if any(col in df.columns for col in known_cols):
            return df

        # Fall back to raw BLAST outfmt 6 with no header
        df = pd.read_csv(path, sep="\t", header=None)
        if df.shape[1] >= 4:
            blast_cols = [
                "qseqid", "sseqid", "pident", "length", "mismatch", "gapopen",
                "qstart", "qend", "sstart", "send", "evalue", "bitscore",
            ]
            rename = {i: blast_cols[i] for i in range(min(len(blast_cols), df.shape[1]))}
            df = df.rename(columns=rename)
            if "qseqid" in df.columns and "qprot" not in df.columns:
                df["qprot"] = df["qseqid"].astype(str)
            return df

    raise FileNotFoundError("BLAST summary not found. Tried: " + ", ".join(tried))

def _get_alignment_length_column(df: pd.DataFrame) -> pd.Series:
    for col in ["length", "align_len", "alignment_length", "alnlen", "alen"]:
        if col in df.columns:
            return pd.to_numeric(df[col], errors="coerce")
    if "qstart" in df.columns and "qend" in df.columns:
        qstart = pd.to_numeric(df["qstart"], errors="coerce")
        qend = pd.to_numeric(df["qend"], errors="coerce")
        return (qend - qstart).abs() + 1
    raise ValueError("Could not identify an alignment-length column in BLAST summary table.")


def plot_s2a_mapping_yield(out_prefix: str) -> Tuple[Path, Path]:
    labels: List[str] = []
    counts: List[int] = []
    colors: List[str] = []
    for label, fnames, color in SUPP_BLAST_DATASETS:
        df = _read_sig_blast_table(fnames)
        labels.append(label)
        counts.append(int(len(df)))
        colors.append(color)

    x = np.arange(len(labels))
    fig, (ax_top, ax_bot) = plt.subplots(
        2, 1, sharex=True, figsize=(3.7, 2.9),
        gridspec_kw={"height_ratios": [1.0, 1.8], "hspace": 0.05}
    )
    ax_top.bar(x, counts, color=colors, width=0.72)
    ax_bot.bar(x, counts, color=colors, width=0.72)

    lower_max = 100
    ymax = max(counts) if counts else 1
    ax_bot.set_ylim(0, lower_max)
    ax_top.set_ylim(max(lower_max, lower_max + 1), ymax * 1.08 if ymax > lower_max else lower_max + 5)

    ax_top.spines["bottom"].set_visible(False)
    ax_bot.spines["top"].set_visible(False)
    ax_top.spines["right"].set_visible(False)
    ax_top.spines["top"].set_visible(False)
    ax_bot.spines["right"].set_visible(False)

    ax_top.tick_params(axis="x", bottom=False, labelbottom=False)
    ax_bot.set_xticks(x)
    ax_bot.set_xticklabels(labels, rotation=28, ha="right")
    ax_bot.set_ylabel("Count")

    d = 0.010
    kwargs = dict(transform=ax_top.transAxes, color="k", clip_on=False, linewidth=0.6)
    ax_top.plot((-d, +d), (-d, +d), **kwargs)
    ax_top.plot((1 - d, 1 + d), (-d, +d), **kwargs)
    kwargs.update(transform=ax_bot.transAxes)
    ax_bot.plot((-d, +d), (1 - d, 1 + d), **kwargs)
    ax_bot.plot((1 - d, 1 + d), (1 - d, 1 + d), **kwargs)

    pdf_path = Path(f"{out_prefix}.pdf")
    png_path = Path(f"{out_prefix}.png")
    svg_path = Path(f"{out_prefix}.svg")
    pdf_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(pdf_path)
    fig.savefig(png_path, dpi=600)
    fig.savefig(svg_path)
    plt.close(fig)
    return pdf_path, png_path


def plot_s2b_alignment_lengths(out_prefix: str) -> Tuple[Path, Path]:
    labels: List[str] = []
    arrays: List[np.ndarray] = []
    colors: List[str] = []
    for label, fnames, color in SUPP_BLAST_DATASETS:
        df = _read_sig_blast_table(fnames)
        vals = _clean_numeric(_get_alignment_length_column(df))
        labels.append(label)
        arrays.append(vals)
        colors.append(color)

    fig, ax = plt.subplots(1, 1, figsize=(3.7, 2.9))
    bp = ax.boxplot(
        arrays, labels=labels, patch_artist=True, widths=0.55, whis=1.5, showfliers=False,
        medianprops=dict(linewidth=0.8, color="black"), boxprops=dict(linewidth=0.6),
        whiskerprops=dict(linewidth=0.6), capprops=dict(linewidth=0.6),
    )
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_edgecolor(COLOR_BLACK)

    rng = np.random.default_rng(123)
    for i, arr in enumerate(arrays, start=1):
        if arr.size:
            x = np.clip(rng.normal(loc=i, scale=0.035, size=arr.size), i - 0.10, i + 0.10)
            ax.scatter(x, arr, s=5.0, alpha=0.22, edgecolors="none", color=COLOR_BLACK)

    ax.set_ylabel("Alignment length")
    ax.set_xticklabels(labels, rotation=28, ha="right")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    pdf_path = Path(f"{out_prefix}.pdf")
    png_path = Path(f"{out_prefix}.png")
    svg_path = Path(f"{out_prefix}.svg")
    pdf_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(pdf_path)
    fig.savefig(png_path, dpi=600)
    fig.savefig(svg_path)
    plt.close(fig)
    return pdf_path, png_path
# ----------------------------- CLI -------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description="Figure 2 plotting script for main and supplementary feature panels plus overlap summaries."
    )
    ap.add_argument("--mode", choices=["seed_boxplots", "pred_boxplots", "s2_blast", "detailed_overlap", "main_d", "all"], default="all")
    ap.add_argument("--force-recompute", action="store_true")

    ap.add_argument("--seed-features", type=Path, default=INPUT_TABLES_DIR / "seed_tads_15aa_zscores.tsv")
    ap.add_argument("--pred-features", type=Path, default=INPUT_TABLES_DIR / "pred_tads_15aa_zscores.tsv")
    ap.add_argument("--random-seed-features", type=Path, default=INPUT_TABLES_DIR / "random.csv")
    ap.add_argument("--random-pred-features", type=Path, default=INPUT_TABLES_DIR / "random_pred_tads_15aa_zscores.tsv")
    
    ap.add_argument("--feature-map")
    ap.add_argument("--feature-labels")
    ap.add_argument("--features", nargs="+", default=FEATURE_COLUMNS_DEFAULT,
                    help="Feature columns to plot in the boxplot panels.")
    ap.add_argument("--labels", nargs=2, default=["Seed R-TADs", "Random IDRs"])
    ap.add_argument("--labels-pred", nargs=2, default=["Predicted R-TADs", "Random IDRs"])
    ap.add_argument("--box-width", type=float, default=0.10)
    ap.add_argument("--rtads-fasta", type=Path, default=DEFAULT_RTADS_FASTA)
    ap.add_argument("--membership-matrix", type=Path, default=_default_membership_matrix())
    ap.add_argument("--restrict-to-rtads", action="store_true")

    ap.add_argument("--out-prefix-box", default=str(OUTPUT_MAIN_DIR / "fig2B_seed_vs_random_features_top4"))
    ap.add_argument("--out-prefix-box-pred", default=str(OUTPUT_SUPP_DIR / "figS2_pred_vs_random_features_top4"))
    ap.add_argument("--out-prefix-s2a", default=str(OUTPUT_SUPP_DIR / "figS2A_blast_mapping_yield"))
    ap.add_argument("--out-prefix-s2b", default=str(OUTPUT_SUPP_DIR / "figS2B_alignment_length_distributions"))
    ap.add_argument("--out-prefix-overlap", default=str(OUTPUT_SUPP_DIR / "figS2F_overlap_combinations"))
    ap.add_argument("--out-prefix-main-d", default=str(OUTPUT_MAIN_DIR / "fig2D_support_summary"))
    return ap

def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    generated = []
    order = ["Aromatic fraction", "WF complexity", "SCD", "κ", "aromatic", "kappa", "kappa*"]

    if args.mode in ("seed_boxplots", "all"):
        plot_df, _stats_df = load_or_compute_boxplot_matrices(
            group_a_path=args.seed_features, group_b_path=args.random_seed_features,
            features=args.features, feature_map_str=args.feature_map, feature_labels_str=args.feature_labels,
            out_prefix_box=args.out_prefix_box, force_recompute=args.force_recompute,
            group_a_name="Seed R-TADs", group_b_name="Random IDRs",
        )
        generated.extend(plot_boxpanels_two_group(plot_df, args.out_prefix_box, order, args.labels, args.box_width, "Seed R-TADs", "Random IDRs"))

    if args.mode in ("pred_boxplots", "all"):
        plot_df, _stats_df = load_or_compute_boxplot_matrices(
            group_a_path=args.pred_features, group_b_path=args.random_pred_features,
            features=args.features, feature_map_str=args.feature_map, feature_labels_str=args.feature_labels,
            out_prefix_box=args.out_prefix_box_pred, force_recompute=args.force_recompute,
            group_a_name="Predicted R-TADs", group_b_name="Random IDRs",
        )
        generated.extend(plot_boxpanels_two_group(plot_df, args.out_prefix_box_pred, order, args.labels_pred, args.box_width, "Predicted R-TADs", "Random IDRs"))

    if args.mode in ("s2_blast", "all"):
        generated.extend(plot_s2a_mapping_yield(args.out_prefix_s2a))
        generated.extend(plot_s2b_alignment_lengths(args.out_prefix_s2b))

    if args.mode in ("detailed_overlap", "main_d", "all"):
        use_precomputed = args.membership_matrix.is_file()
        if use_precomputed:
            _sets, mem = _load_membership_matrix_file(args.membership_matrix)
        else:
            _sets, mem = load_or_compute_membership_matrix(
                rtads_fasta=args.rtads_fasta, file_pairs=DEFAULT_FILE_PAIRS,
                out_prefix_venn=args.out_prefix_overlap, restrict_to_rtads=args.restrict_to_rtads,
                force_recompute=args.force_recompute,
            )
        if args.mode in ("detailed_overlap", "all"):
            generated.extend(plot_detailed_overlap_upset(mem, args.out_prefix_overlap, show_all_combinations=True))
        if args.mode in ("main_d", "all"):
            generated.extend(plot_main_d_support_summary(mem, args.out_prefix_main_d))

    if generated:
        print("Outputs:")
        for p in generated:
            print(f"  - {p}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
