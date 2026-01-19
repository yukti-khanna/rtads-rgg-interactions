#!/usr/bin/env python3
# file: scripts/tf_rbp_full_pipeline_with_tables.py

import argparse
import sys
import re
import math
from pathlib import Path
from collections import Counter
from typing import Dict, Set, Tuple, Optional

import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt

# ----------------------------- Colors / Style --------------------------------
COLOR_TF_NODE = "#fdae6b"     # TF = orange
COLOR_RBP_NODE = "#1f78b4"    # RBP = blue
COLOR_OTHER_NODE = "#c7c7c7"  # other = light grey
COLOR_EDGE = "#bfbfbf"

COLOR_TF_BETWEENNESS = "#e6550d"   # dark orange
COLOR_TF_INTERACTORS = "#fdd0a2"   # light orange
COLOR_RBP_BETWEENNESS = "#1f4e79"  # dark blue
COLOR_RBP_INTERACTORS = "#a6cee3"  # light blue

PIE_COLORS = ["#1f4e79", "#6baed6", "#d9d9d9"]
HIST_BLUE = "#1f78b4"
COLOR_EDGE_INTRA = "gainsboro"

_UNIPROT_RE = re.compile(r"^([OPQ][0-9][A-Z0-9]{3}[0-9]|[A-NR-Z][0-9]{5})(-[0-9]+)?$")

# ----------------------------- IO / Mapping ----------------------------------
def load_mapping(path: Path) -> Dict[str, str]:
    df = pd.read_csv(path, sep="\t", header=None, names=["uniprot_id", "gene_name"], dtype=str)
    df = df.dropna(subset=["uniprot_id", "gene_name"])
    df["uniprot_id"] = df["uniprot_id"].str.strip()
    df["gene_name"] = df["gene_name"].str.strip()
    winners: Dict[str, str] = {}
    for uid, sub in df.groupby("uniprot_id"):
        counts = Counter(sub["gene_name"])
        winner, _ = counts.most_common(1)[0]
        if len(counts) > 1:
            print(f"[warn] UniProt '{uid}' maps to multiple gene names {dict(counts)}. Using '{winner}'.", file=sys.stderr)
        winners[uid] = winner
    return winners

def load_tf_list(path: Path, mapping: Dict[str, str]) -> Set[str]:
    s = pd.read_csv(path, sep="\t", header=None, names=["uniprot_id"], dtype=str)["uniprot_id"].dropna().str.strip()
    mapped = s.map(mapping).dropna()
    missing = set(s) - set(mapping.keys())
    if missing:
        print(f"[info] {len(missing)} TF UniProt IDs not in mapping; skipping.", file=sys.stderr)
    return set(mapped)

def load_rbp_list(path: Path) -> Set[str]:
    s = pd.read_csv(path, sep="\t", header=None, names=["gene_name"], dtype=str)["gene_name"].dropna().str.strip()
    return set(s)

def is_uniprot_accession(s: str) -> bool:
    return isinstance(s, str) and bool(_UNIPROT_RE.match(s))

def map_ppi_endpoints(a: str, b: str, mapping: Dict[str, str]) -> Tuple[str, str, bool, bool]:
    a_mapped = mapping.get(a, a)
    b_mapped = mapping.get(b, b)
    return a_mapped, b_mapped, (a != a_mapped), (b != b_mapped)

def load_and_map_ppis(path: Path, mapping: Dict[str, str]) -> pd.DataFrame:
    df = pd.read_csv(path, sep="\t", header=None, names=["proteinA", "proteinB"], dtype=str)
    df = df.dropna(subset=["proteinA", "proteinB"])
    df["proteinA"] = df["proteinA"].str.strip()
    df["proteinB"] = df["proteinB"].str.strip()

    mapped = df.apply(lambda r: map_ppi_endpoints(r["proteinA"], r["proteinB"], mapping), axis=1, result_type="expand")
    mapped.columns = ["a", "b", "a_was_mapped", "b_was_mapped"]
    dfm = pd.concat([df, mapped], axis=1)

    mask_unmapped_both = (
        (~dfm["a_was_mapped"]) & (~dfm["b_was_mapped"])
        & dfm["a"].map(is_uniprot_accession) & dfm["b"].map(is_uniprot_accession)
    )
    dropped = int(mask_unmapped_both.sum())
    if dropped:
        print(f"[info] Dropping {dropped} edges where neither endpoint mapped and both look like UniProt IDs.", file=sys.stderr)
    dfm = dfm.loc[~mask_unmapped_both, ["a", "b"]]

    dfm = dfm[dfm["a"] != dfm["b"]].copy()
    dfm["u"] = dfm[["a", "b"]].min(axis=1)
    dfm["v"] = dfm[["a", "b"]].max(axis=1)
    dfm = dfm.drop_duplicates(subset=["u", "v"])[["u", "v"]].rename(columns={"u": "proteinA", "v": "proteinB"})
    return dfm

# ------------------------------- Graph & Stats -------------------------------
def build_graph(ppi_df: pd.DataFrame) -> nx.Graph:
    g = nx.Graph()
    g.add_edges_from(ppi_df[["proteinA", "proteinB"]].itertuples(index=False, name=None))
    return g

def compute_tf_rbp_fractions(G: nx.Graph, TF_genes: Set[str], RBP_genes: Set[str]):
    records = []
    for tf in [t for t in TF_genes if t in G]:
        partners = list(G.neighbors(tf))
        n_total = len(partners)
        if n_total == 0:
            continue
        n_rbp = sum(1 for p in partners if p in RBP_genes)
        records.append({"TF": tf, "fraction_RBP": n_rbp / n_total, "n_total": n_total, "n_RBP": n_rbp})
    df = pd.DataFrame(records).sort_values("fraction_RBP", ascending=False, ignore_index=True)
    nodes = set(G.nodes())
    background = len(nodes & RBP_genes) / len(nodes) if nodes else float("nan")
    pct_over_0_5 = 100.0 * (df["fraction_RBP"] > 0.5).mean() if not df.empty else 0.0
    return df, background, pct_over_0_5

def build_tf_rbp_subgraph(G: nx.Graph, TF_genes: Set[str], RBP_genes: Set[str], min_degree: int = 3):
    edges = [(u, v) for u, v in G.edges()
             if (u in TF_genes and v in RBP_genes) or (u in RBP_genes and v in TF_genes)]
    H = nx.Graph(); H.add_edges_from(edges)
    if min_degree > 0:
        H.remove_nodes_from([n for n, d in H.degree() if d < min_degree])
    if H.number_of_nodes() == 0:
        return H, pd.DataFrame(columns=["gene", "type", "betweenness", "degree_TF_RBP"])
    bet = nx.betweenness_centrality(H, normalized=True)
    rows = []
    for n in H.nodes():
        t = "TF" if n in TF_genes else ("RBP" if n in RBP_genes else "other")
        if t == "other":
            continue
        rows.append({"gene": n, "type": t, "betweenness": bet.get(n, 0.0), "degree_TF_RBP": H.degree(n)})
    return H, pd.DataFrame(rows)

def full_network_stats(G: nx.Graph, TF_genes: Set[str], RBP_genes: Set[str]) -> pd.DataFrame:
    return pd.DataFrame([{"gene": n,
                          "type": ("TF" if n in TF_genes else ("RBP" if n in RBP_genes else "other")),
                          "degree_full": d}
                         for n, d in G.degree()])

# ----------------------- WHOLE network plot (betweenness) --------------------
def plot_full_network_betweenness_filtered(
    G: nx.Graph,
    TF_genes: Set[str],
    RBP_genes: Set[str],
    out_path: str,
    *,
    min_degree: int = 10,        # degree-based filtering kept
    top_label_count: int = 5,    # label top-K TFs & RBPs by betweenness
    layout_seed: int = 42,
    layout_k_mult: float = 12.0,
    layout_iterations: int = 3000,
    layout_scale: float = 8.0,
    spread: float = 2.0,
    betw_sample_k: Optional[int] = None,  # None => auto sample
    save_nodes_path: Optional[Path] = None,
    save_edges_path: Optional[Path] = None,
) -> None:
    if G.number_of_nodes() == 0:
        fig = plt.figure(figsize=(12, 10))
        plt.text(0.5, 0.5, "Empty graph", ha="center", va="center"); plt.axis("off")
        fig.savefig(out_path, dpi=300, bbox_inches="tight"); plt.close(fig); return

    # Degree filter (single-pass)
    sub = G.copy()
    to_drop = [n for n, d in sub.degree() if d < min_degree]
    if to_drop:
        sub.remove_nodes_from(to_drop)
    if sub.number_of_nodes() == 0:
        fig = plt.figure(figsize=(12, 10))
        plt.text(0.5, 0.5, f"No nodes with degree ≥ {min_degree}", ha="center", va="center"); plt.axis("off")
        fig.savefig(out_path, dpi=300, bbox_inches="tight"); plt.close(fig); return

    # Betweenness for size/labels (sample for speed on big graphs)
    n_sub = sub.number_of_nodes()
    if betw_sample_k is None:
        betw_sample_k = min(50, max(1, n_sub // 10))
    betw = nx.betweenness_centrality(sub, k=betw_sample_k, endpoints=True, normalized=True)

    # Layout
    k_eff = layout_k_mult * math.sqrt(1.0 / max(1, n_sub))
    pos = nx.spring_layout(sub, seed=layout_seed, k=k_eff, iterations=layout_iterations, scale=layout_scale)
    if spread and spread != 1.0:
        for n_ in pos:
            x, y = pos[n_]; pos[n_] = (x * spread, y * spread)

    # Visuals
    deg = dict(sub.degree())  # still useful for TSV
    size_fn = lambda b: max(1e-6, b) * 20000.0
    node_colors = [COLOR_TF_NODE if n_ in TF_genes else (COLOR_RBP_NODE if n_ in RBP_genes else COLOR_OTHER_NODE) for n_ in sub.nodes()]
    node_sizes = [size_fn(betw.get(n_, 0.0)) for n_ in sub.nodes()]

    fig = plt.figure(figsize=(20, 18))
    ax = plt.gca()
    nx.draw_networkx_edges(sub, pos, alpha=0.35, width=0.6, edge_color=COLOR_EDGE, ax=ax)
    nx.draw_networkx_nodes(sub, pos, node_color=node_colors, node_size=node_sizes, linewidths=0.0, ax=ax)

    # Labels: top-K TFs & RBPs by betweenness
    tf_nodes = [n_ for n_ in sub.nodes() if n_ in TF_genes]
    rbp_nodes = [n_ for n_ in sub.nodes() if n_ in RBP_genes]
    tf_top = sorted(tf_nodes, key=lambda n_: betw.get(n_, 0.0), reverse=True)[:top_label_count]
    rbp_top = sorted(rbp_nodes, key=lambda n_: betw.get(n_, 0.0), reverse=True)[:top_label_count]
    label_nodes = {n_: n_ for n_ in (tf_top + rbp_top)}
    if label_nodes:
        nx.draw_networkx_labels(sub, pos, labels=label_nodes, font_size=9, ax=ax)

    # Legends
    tf_proxy = plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=COLOR_TF_NODE, markersize=10)
    rbp_proxy = plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=COLOR_RBP_NODE, markersize=10)
    other_proxy = plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=COLOR_OTHER_NODE, markersize=10)
    color_legend = ax.legend([tf_proxy, rbp_proxy, other_proxy], ["TF", "RBP", "other"], loc="upper right", frameon=False)
    ax.add_artist(color_legend)

    # Node-size legend: min/median/max betweenness
    if betw:
        s = pd.Series(betw.values())
        vals = sorted(set([float(s.min()), float(s.median()), float(s.max())]))
        size_handles = [plt.scatter([], [], s=size_fn(v), facecolors='none', edgecolors='black', label=f"{v:.3f}") for v in vals]
        ax.legend(handles=size_handles, title="Node size ∝ betweenness", loc="lower right", frameon=False)

    ax.set_title(f"PPI network (degree ≥ {min_degree}; size ∝ betweenness)")
    ax.set_axis_off()
    plt.margins(0.25)
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    # Save plotted matrices (coords, types, degree, betweenness, labeled)
    if save_nodes_path or save_edges_path:
        df_nodes = pd.DataFrame({
            "gene": list(sub.nodes()),
            "x": [pos[n][0] for n in sub.nodes()],
            "y": [pos[n][1] for n in sub.nodes()],
            "type": ["TF" if n in TF_genes else ("RBP" if n in RBP_genes else "other") for n in sub.nodes()],
            "degree": [deg.get(n, 0) for n in sub.nodes()],
            "betweenness": [betw.get(n, 0.0) for n in sub.nodes()],
            "labeled": [n in label_nodes for n in sub.nodes()],
        })
        if save_nodes_path:
            df_nodes.to_csv(save_nodes_path, sep="\t", index=False)
        df_edges = pd.DataFrame([(u, v) for u, v in sub.edges()], columns=["source", "target"])
        if save_edges_path:
            df_edges.to_csv(save_edges_path, sep="\t", index=False)

# ------------------------------ Summary panels -------------------------------
def plot_summary_panels(tf_frac_df: pd.DataFrame,
                        background_frac: float,
                        pct_tf_gt_0_5: float,
                        hubs_tf: pd.DataFrame,
                        hubs_rbp: pd.DataFrame,
                        out_path: str,
                        pie_bins_out: Optional[Path] = None) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    ax = axes[0, 0]
    vals = tf_frac_df["fraction_RBP"].dropna().values if not tf_frac_df.empty else []
    ax.hist(vals, bins=30, edgecolor="white", color=HIST_BLUE)
    if pd.notna(background_frac):
        ax.axvline(background_frac, linestyle="--")
    ax.set_xlabel("Fraction of interactors that are RBPs")
    ax.set_ylabel("Number of transcription factors")
    ax.text(0.98, 0.95, f">0.5: {pct_tf_gt_0_5:.1f}%", transform=ax.transAxes, ha="right", va="top")

    ax = axes[0, 1]
    if not tf_frac_df.empty:
        bins = {
            "> 0.5": (tf_frac_df["fraction_RBP"] > 0.5).sum(),
            "0.25–0.5": ((tf_frac_df["fraction_RBP"] >= 0.25) & (tf_frac_df["fraction_RBP"] <= 0.5)).sum(),
            "< 0.25": (tf_frac_df["fraction_RBP"] < 0.25).sum(),
        }
        if pie_bins_out is not None:
            pd.DataFrame({"bin": list(bins.keys()), "count": list(bins.values())}).to_csv(pie_bins_out, sep="\t", index=False)
        total = sum(bins.values()) or 1
        labels = [f"{k} ({(v/total)*100:.1f}%)" for k, v in bins.items()]
        sizes = list(bins.values())
    else:
        labels, sizes = ["No data"], [1]
    ax.pie(sizes, labels=labels, colors=PIE_COLORS[:len(sizes)], startangle=90)
    ax.add_artist(plt.Circle((0, 0), 0.60, color="white"))
    ax.set_title("RBP interaction bias across TFs")

    ax = axes[1, 0]
    tf_plot = hubs_tf.sort_values("betweenness", ascending=False).head(8)
    if not tf_plot.empty:
        x = list(range(len(tf_plot))); w = 0.4
        ax2 = ax.twinx()
        ax.bar([i - w/2 for i in x], tf_plot["betweenness"], width=w, color=COLOR_TF_BETWEENNESS)
        ax2.bar([i + w/2 for i in x], tf_plot["degree_TF_RBP"], width=w, color=COLOR_TF_INTERACTORS)
        ax.set_xticks(x); ax.set_xticklabels(tf_plot["gene"], rotation=45, ha="right")
        ax.set_ylabel("Betweenness centrality")
        ax2.set_ylabel("Number of RBP interactors")
        ax.set_title("Top TF hubs (betweenness and RBP degree)")
        h1 = plt.Rectangle((0, 0), 1, 1, color=COLOR_TF_BETWEENNESS)
        h2 = plt.Rectangle((0, 0), 1, 1, color=COLOR_TF_INTERACTORS)
        ax.legend([h1, h2], ["Betweenness centrality", "Number of RBP interactors"], loc="upper right", frameon=False)
    else:
        ax.text(0.5, 0.5, "No TF hubs", ha="center", va="center"); ax.set_axis_off()

    ax = axes[1, 1]
    rbp_plot = hubs_rbp.sort_values("betweenness", ascending=False).head(8)
    if not rbp_plot.empty:
        x = list(range(len(rbp_plot))); w = 0.4
        ax2 = ax.twinx()
        ax.bar([i - w/2 for i in x], rbp_plot["betweenness"], width=w, color=COLOR_RBP_BETWEENNESS)
        ax2.bar([i + w/2 for i in x], rbp_plot["degree_TF_RBP"], width=w, color=COLOR_RBP_INTERACTORS)
        ax.set_xticks(x); ax.set_xticklabels(rbp_plot["gene"], rotation=45, ha="right")
        ax.set_ylabel("Betweenness centrality")
        ax2.set_ylabel("Number of TF interactors")
        ax.set_title("Top RBP hubs (betweenness and TF degree)")
        h1 = plt.Rectangle((0, 0), 1, 1, color=COLOR_RBP_BETWEENNESS)
        h2 = plt.Rectangle((0, 0), 1, 1, color=COLOR_RBP_INTERACTORS)
        ax.legend([h1, h2], ["Betweenness centrality", "Number of TF interactors"], loc="upper right", frameon=False)
    else:
        ax.text(0.5, 0.5, "No RBP hubs", ha="center", va="center"); ax.set_axis_off()

    plt.tight_layout()
    fig = plt.gcf()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

# ------------------------------ Community panel ------------------------------
def plot_community_panel_and_save_tables(
    G: nx.Graph,
    TF_genes: Set[str],
    RBP_genes: Set[str],
    outbase: str,
    min_degree_b: int = 10,
    seed: int = 4572321,
    save_nodes_path: Optional[Path] = None,
    save_edges_path: Optional[Path] = None,
) -> None:
    if G.number_of_nodes() == 0:
        return
    sub = G.copy()
    sub.remove_nodes_from([n for n, d in sub.degree() if d < min_degree_b])
    if sub.number_of_nodes() == 0:
        return
    comps = list(nx.connected_components(sub))
    if not comps:
        return
    sub = sub.subgraph(max(comps, key=len)).copy()

    def is_tf(n: str) -> bool:  return n in TF_genes
    def is_rbp(n: str) -> bool: return n in RBP_genes

    k_sample = min(10, max(1, sub.number_of_nodes() // 10))
    betw = nx.betweenness_centrality(sub, k=k_sample, endpoints=True)

    lpc = list(nx.community.label_propagation_communities(sub))
    comm_id = {n: i for i, com in enumerate(lpc) for n in com}

    pos = nx.spring_layout(sub, k=0.15, seed=seed)

    intra = [(u, v) for u, v in sub.edges() if comm_id.get(u) == comm_id.get(v)]
    inter = [(u, v) for u, v in sub.edges() if comm_id.get(u) != comm_id.get(v)]
    cmap = plt.get_cmap("tab20")
    inter_colors = [cmap(comm_id.get(u, 0) % 20) for u, v in inter]

    fig, ax = plt.subplots(figsize=(20, 15), dpi=300)
    nx.draw_networkx_edges(sub, pos, edgelist=intra, width=0.8, alpha=0.35, edge_color=COLOR_EDGE_INTRA, ax=ax)
    nx.draw_networkx_edges(sub, pos, edgelist=inter, width=1.2, alpha=0.35, edge_color=inter_colors, ax=ax)

    node_colors = [COLOR_TF_NODE if is_tf(n) else (COLOR_RBP_NODE if is_rbp(n) else COLOR_OTHER_NODE) for n in sub.nodes()]
    node_sizes = [max(1e-6, betw.get(n, 0.0)) * 20000.0 for n in sub.nodes()]
    nx.draw_networkx_nodes(sub, pos, node_color=node_colors, node_size=node_sizes,
                           edgecolors="#DDDDDD", linewidths=0.6, alpha=0.9, ax=ax)

    from matplotlib.patches import Patch
    legend_elems = [
        Patch(facecolor=COLOR_TF_NODE, edgecolor="#DDDDDD", label="TF"),
        Patch(facecolor=COLOR_RBP_NODE, edgecolor="#DDDDDD", label="RBP"),
        Patch(facecolor=COLOR_OTHER_NODE, edgecolor="#DDDDDD", label="Other"),
    ]
    ax.legend(handles=legend_elems, loc="upper right", frameon=True, fontsize=20, markerscale=2.5, borderpad=0.8, labelspacing=0.6)

    ax.set_axis_off()
    fig.tight_layout()
    plt.savefig(outbase + ".png", dpi=600); plt.savefig(outbase + ".pdf")
    plt.close(fig)

    if save_nodes_path or save_edges_path:
        df_nodes = pd.DataFrame({
            "gene": list(sub.nodes()),
            "x": [pos[n][0] for n in sub.nodes()],
            "y": [pos[n][1] for n in sub.nodes()],
            "type": ["TF" if is_tf(n) else ("RBP" if is_rbp(n) else "other") for n in sub.nodes()],
            "degree": [sub.degree(n) for n in sub.nodes()],
            "betweenness": [betw.get(n, 0.0) for n in sub.nodes()],
            "community_id": [comm_id.get(n, -1) for n in sub.nodes()],
        })
        if save_nodes_path:
            df_nodes.to_csv(save_nodes_path, sep="\t", index=False)
        df_edges = pd.DataFrame([(u, v) for u, v in sub.edges()], columns=["source", "target"])
        if save_edges_path:
            df_edges["type"] = [("intra" if comm_id.get(u) == comm_id.get(v) else "inter") for u, v in sub.edges()]
            df_edges.to_csv(save_edges_path, sep="\t", index=False)

# ---------------------------------- Main -------------------------------------
def main():
    ap = argparse.ArgumentParser(description="TF/RBP PPI summary + whole-network betweenness plot (+ tables)")
    ap.add_argument("--tf", type=Path, default=Path("TF_list.txt"))
    ap.add_argument("--rbp", type=Path, default=Path("RBP_list.txt"))
    ap.add_argument("--map", type=Path, dest="mapping", default=Path("UNIPROTIDS_GENENAME.txt"))
    ap.add_argument("--ppi", type=Path, default=Path("ppis.txt"))
    ap.add_argument("--out_prefix", default="tf_rbp")

    ap.add_argument("--min_degree_h", type=int, default=3)

    # WHOLE-network (betweenness) plot options
    ap.add_argument("--min-degree", type=int, default=10, help="Min degree for the whole-network plot (filtering only)")
    ap.add_argument("--top_labels", type=int, default=5, help="Top TFs/RBPs to label by betweenness")
    ap.add_argument("--layout", choices=["spring", "kamada"], default="spring")
    ap.add_argument("--layout-k-mult", type=float, default=18.0)
    ap.add_argument("--layout-iterations", type=int, default=3000)
    ap.add_argument("--layout-scale", type=float, default=8.0)
    ap.add_argument("--layout-seed", type=int, default=42)
    ap.add_argument("--spread", type=float, default=2.5)

    ap.add_argument("--community-min-degree", type=int, default=10)
    ap.add_argument("--skip-community", action="store_true")

    ap.add_argument("--write-tables", action="store_true")
    ap.add_argument("--tables-dir", type=Path, default=None)

    args = ap.parse_args()

    mapping = load_mapping(args.mapping)
    TF_genes = load_tf_list(args.tf, mapping)
    RBP_genes = load_rbp_list(args.rbp)
    ppis = load_and_map_ppis(args.ppi, mapping)
    G = build_graph(ppis)

    if not TF_genes:
        print("[error] No TFs after mapping.", file=sys.stderr)
    if G.number_of_nodes() == 0:
        print("[error] Empty PPI graph after mapping.", file=sys.stderr)

    tf_frac_df, bg_frac, pct_over_0_5 = compute_tf_rbp_fractions(G, TF_genes, RBP_genes)
    H, hubs_df = build_tf_rbp_subgraph(G, TF_genes, RBP_genes, min_degree=args.min_degree_h)
    hubs_tf = hubs_df[hubs_df["type"] == "TF"].sort_values("betweenness", ascending=False).head(8)
    hubs_rbp = hubs_df[hubs_df["type"] == "RBP"].sort_values("betweenness", ascending=False).head(8)

    # 2×2 panels
    panels_path = f"{args.out_prefix}_summary_panels.png"
    pie_bins_path = None
    if args.write_tables:
        tables_dir = args.tables_dir or Path(f"{args.out_prefix}_tables")
        tables_dir.mkdir(parents=True, exist_ok=True)
        pie_bins_path = tables_dir / "pie_bins.tsv"
    plot_summary_panels(tf_frac_df, bg_frac, pct_over_0_5, hubs_tf, hubs_rbp, panels_path, pie_bins_out=pie_bins_path)

    # WHOLE network (betweenness-sized) plot
    whole_png = f"{args.out_prefix}_whole_network_betweenness.png"
    nodes_out = edges_out = None
    if args.write_tables:
        tables_dir = args.tables_dir or Path(f"{args.out_prefix}_tables")
        tables_dir.mkdir(parents=True, exist_ok=True)
        nodes_out = tables_dir / "whole_plot_nodes.tsv"
        edges_out = tables_dir / "whole_plot_edges.tsv"
    plot_full_network_betweenness_filtered(
        G=G,
        TF_genes=TF_genes,
        RBP_genes=RBP_genes,
        out_path=whole_png,
        min_degree=args.min_degree,
        top_label_count=args.top_labels,
        layout_seed=args.layout_seed,
        layout_k_mult=args.layout_k_mult,
        layout_iterations=args.layout_iterations,
        layout_scale=args.layout_scale,
        spread=args.spread,
        save_nodes_path=nodes_out,
        save_edges_path=edges_out,
    )

    # Optional community panel
    if not args.skip_community:
        community_base = f"{args.out_prefix}_community_panel"
        comm_nodes_out = comm_edges_out = None
        if args.write_tables:
            tables_dir = args.tables_dir or Path(f"{args.out_prefix}_tables")
            tables_dir.mkdir(parents=True, exist_ok=True)
            comm_nodes_out = tables_dir / "community_nodes.tsv"
            comm_edges_out = tables_dir / "community_edges.tsv"
        plot_community_panel_and_save_tables(
            G=G,
            TF_genes=TF_genes,
            RBP_genes=RBP_genes,
            outbase=community_base,
            min_degree_b=args.community_min_degree,
            save_nodes_path=comm_nodes_out,
            save_edges_path=comm_edges_out,
        )

    # Other matrices
    if args.write_tables:
        tables_dir = args.tables_dir or Path(f"{args.out_prefix}_tables")
        tables_dir.mkdir(parents=True, exist_ok=True)
        ppis.to_csv(tables_dir / "mapped_ppis.tsv", sep="\t", index=False)
        tf_frac_df.to_csv(tables_dir / "tf_fraction_per_tf.tsv", sep="\t", index=False)
        pd.DataFrame([{
            "background_RBP_fraction": bg_frac,
            "pct_TFs_fraction_gt_0_5": pct_over_0_5,
        }]).to_csv(tables_dir / "background_summary.tsv", sep="\t", index=False)
        hubs_df.to_csv(tables_dir / "tf_rbp_subgraph_stats.tsv", sep="\t", index=False)
        hubs_tf.to_csv(tables_dir / "hubs_tf_top8.tsv", sep="\t", index=False)
        hubs_rbp.to_csv(tables_dir / "hubs_rbp_top8.tsv", sep="\t", index=False)
        full_network_stats(G, TF_genes, RBP_genes).to_csv(tables_dir / "full_network_node_stats.tsv", sep="\t", index=False)

    print(f"[saved] {panels_path}")
    print(f"[saved] {whole_png}")
    if not args.skip_community:
        print(f"[saved] {community_base}.png")
        print(f"[saved] {community_base}.pdf")
    if args.write_tables:
        print(f"[saved] tables → {(args.tables_dir or Path(f'{args.out_prefix}_tables')).resolve()}")

if __name__ == "__main__":
    main()
