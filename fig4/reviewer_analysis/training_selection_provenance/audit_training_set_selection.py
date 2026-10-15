#!/usr/bin/env python3
"""Read-only provenance reconstruction for Figure 4 training/held-out selection.

The script reads archived lists, feature tables, sequence catalogues and B22
tables. It never edits historical inputs, launches simulations, or uses ML
predictions. All outputs are written beside this script.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Iterable

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from mpl_toolkits.axes_grid1.inset_locator import inset_axes


ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
CALVADOS = Path("/Users/yuktikhanna/calvados")
LISTS = CALVADOS / "lists"
PLOT_FEATURES = CALVADOS / "scripts" / "plot_features.py"

FINAL_LIST = LISTS / "sim_pairs_for_training_270525.txt"
PRE_EXCLUSION_LIST = LISTS / "sim_pairs_for_training_220425.txt"
HELDOUT_TADS = LISTS / "tads_taken_out.txt"
HELDOUT_RGGS = LISTS / "rgs_taken_out.txt"
FASTA = LISTS / "all_tads_rggs_dis.fasta"

FULL = ROOT / "task3_sampling_bias" / "full_combinatorial_feature_landscape.csv"
FINAL_FEATURES = ROOT / "task3_sampling_bias" / "fig4_917_sampling_membership.csv"
HELD_FEATURES = ROOT / "task3_sampling_bias" / "heldout_23x78_features.csv"
HELD_EXCLUSIVITY = ROOT / "task3_sampling_bias" / "heldout_sequence_exclusivity.csv"
CORRELATIONS = ROOT / "task3_sampling_bias" / "electrostatic_correlations_917_vs_heldout.csv"

HIST_B22 = {
    "Completed cohort (546)": CALVADOS / "simulated_pairs_added_features_data.csv",
    "Expanded cohort (911)": CALVADOS / "sim_pairs_added_features_data_090425.csv",
    "Pre-exclusion cohort (1,039)": ROOT / "sim_pairs_added_features_data_270525.csv",
    "Final training set (917)": FINAL_FEATURES,
}

STAGE_FILES = [
    ("Initial 5 x 5 pilot", ROOT / "25_pairs_features_data.csv", "csv"),
    ("Archived initial set", ROOT / "initial_set_names_sliced.txt", "txt"),
    ("Feature-guided base list", LISTS / "final_test_names.txt", "txt"),
    ("Outside-filter additions", LISTS / "200_pairs_outside_filter_less_than_lambda045_names.txt", "txt"),
    ("Initial NCP-directed additions", LISTS / "ocn_names.txt", "txt"),
    ("NCP-bin deficit additions", LISTS / "ocn_added_names.txt", "txt"),
    ("Dated training list 2025-02-14", LISTS / "simulated_pairs_for_training_140225.txt", "txt"),
    ("Dated training list 2025-04-09", LISTS / "sim_pairs_for_training_090425.txt", "txt"),
    ("Pre-exclusion training list 2025-04-22", PRE_EXCLUSION_LIST, "txt"),
    ("Final training list 2025-05-27", FINAL_LIST, "txt"),
]

BLUE = "#0072B2"
ORANGE = "#D55E00"
GREY = "#777777"
COLORS = ["#999999", "#56B4E9", "#D55E00", "#0072B2"]


def require(path: Path) -> Path:
    if not path.exists():
        raise FileNotFoundError(path)
    return path


def read_names(path: Path) -> list[str]:
    return [x.strip() for x in require(path).read_text().splitlines() if x.strip()]


def pair_parts(pair: str) -> tuple[str, str]:
    if "__" not in pair:
        raise ValueError(f"Pair lacks '__' separator: {pair}")
    rgg, ahd = pair.split("__", 1)
    if not rgg or not ahd:
        raise ValueError(f"Malformed pair: {pair}")
    return rgg, ahd


def csv_pair_ids(path: Path) -> list[str]:
    df = pd.read_csv(require(path), low_memory=False)
    for col in ("pair_id", "name"):
        if col in df:
            return df[col].astype(str).tolist()
    raise ValueError(f"No pair ID column in {path}")


def read_fasta(path: Path) -> dict[str, str]:
    records: dict[str, str] = {}
    current = None
    chunks: list[str] = []
    for line in require(path).read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        if line.startswith(">"):
            if current is not None:
                records[current] = "".join(chunks)
            current = line[1:].split()[0]
            chunks = []
        else:
            if current is None:
                raise ValueError(f"Sequence before FASTA header in {path}")
            chunks.append(line)
    if current is not None:
        records[current] = "".join(chunks)
    return records


def style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "font.size": 7,
            "axes.titlesize": 7,
            "axes.labelsize": 6.5,
            "xtick.labelsize": 6,
            "ytick.labelsize": 6,
            "legend.fontsize": 5.5,
            "axes.linewidth": 0.6,
            "xtick.major.width": 0.5,
            "ytick.major.width": 0.5,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def clean_axes(ax: plt.Axes) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def panel_label(ax: plt.Axes, label: str) -> None:
    ax.text(-0.18, 1.06, label, transform=ax.transAxes, fontweight="bold", fontsize=9, va="bottom")


def ecdf(values: Iterable[float]) -> tuple[np.ndarray, np.ndarray]:
    x = np.sort(pd.to_numeric(pd.Series(values), errors="coerce").dropna().to_numpy(float))
    return x, np.arange(1, len(x) + 1) / len(x)


def save(fig: plt.Figure, stem: str) -> None:
    fig.savefig(OUT / f"{stem}.pdf", bbox_inches="tight", pad_inches=0.03)
    fig.savefig(OUT / f"{stem}.png", dpi=600, bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)


def stage_audit(final_ids: set[str], held_tads: set[str], held_rggs: set[str]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, set[str]]]:
    stage_sets: dict[str, set[str]] = {}
    stage_rows = []
    cumulative: set[str] = set()
    for stage, path, kind in STAGE_FILES:
        ids = csv_pair_ids(path) if kind == "csv" else read_names(path)
        unique = set(ids)
        for pair in unique:
            pair_parts(pair)
        stage_sets[stage] = unique
        new = unique - cumulative
        cumulative |= unique
        stage_rows.append(
            {
                "stage": stage,
                "source_file": str(path),
                "n_rows": len(ids),
                "n_unique": len(unique),
                "duplicates_within_file": len(ids) - len(unique),
                "n_new_unique": len(new),
                "cumulative_unique": len(cumulative),
                "n_in_final917": len(unique & final_ids),
                "n_not_in_final917": len(unique - final_ids),
            }
        )
    counts = pd.DataFrame(stage_rows)

    overlap_rows = []
    names = list(stage_sets)
    for i, left in enumerate(names):
        for right in names[i + 1 :]:
            overlap_rows.append(
                {
                    "stage_a": left,
                    "stage_b": right,
                    "n_overlap": len(stage_sets[left] & stage_sets[right]),
                    "n_union": len(stage_sets[left] | stage_sets[right]),
                }
            )
    overlaps = pd.DataFrame(overlap_rows)

    all_pairs = sorted(set().union(*stage_sets.values()))
    provenance_rows = []
    for pair in all_pairs:
        rgg, ahd = pair_parts(pair)
        row = {
            "pair_id": pair,
            "RG_RGG_id": rgg,
            "AHD_id": ahd,
            "final_917_membership": pair in final_ids,
            "heldout_AHD_identity": ahd in held_tads,
            "heldout_RG_RGG_identity": rgg in held_rggs,
        }
        for stage in names:
            row["stage_" + stage.lower().replace(" ", "_").replace("-", "_").replace("(", "").replace(")", "").replace(",", "")] = pair in stage_sets[stage]
        provenance_rows.append(row)
    provenance = pd.DataFrame(provenance_rows)
    return counts, provenance, overlaps, stage_sets


def heldout_audit(
    final_ids: set[str], held_tads: set[str], held_rggs: set[str], fasta: dict[str, str]
) -> tuple[pd.DataFrame, dict]:
    final_parts = [pair_parts(p) for p in final_ids]
    final_rggs = {r for r, _ in final_parts}
    final_tads = {t for _, t in final_parts}
    held_pairs = {f"{r}__{t}" for r in held_rggs for t in held_tads}
    held_table_pairs = set(pd.read_csv(require(HELD_FEATURES)).pair_id.astype(str))
    if held_table_pairs != held_pairs:
        raise ValueError(
            "Authoritative held-out feature table is not the complete 23 x 78 cross-product: "
            f"missing={len(held_pairs - held_table_pairs)}, extra={len(held_table_pairs - held_pairs)}"
        )
    rows = []
    for seq_type, ids, training_ids in [
        ("AHD", held_tads, final_tads),
        ("RG/RGG", held_rggs, final_rggs),
    ]:
        for identity in sorted(ids):
            exact_sequence_matches = sorted(x for x in training_ids if fasta[x] == fasta[identity])
            rows.append(
                {
                    "sequence_type": seq_type,
                    "sequence_id": identity,
                    "appears_in_final917": identity in training_ids,
                    "sequence_exclusive": identity not in training_ids,
                    "exact_amino_acid_sequence_match_in_final917": bool(exact_sequence_matches),
                    "matching_training_identifiers": ";".join(exact_sequence_matches),
                    "heldout_cross_product_partner_count": len(held_rggs) if seq_type == "AHD" else len(held_tads),
                }
            )
    audit = pd.DataFrame(rows)
    final_tad_sequences = {fasta[x] for x in final_tads}
    final_rgg_sequences = {fasta[x] for x in final_rggs}
    held_tad_sequence_overlap = sum(fasta[x] in final_tad_sequences for x in held_tads)
    held_rgg_sequence_overlap = sum(fasta[x] in final_rgg_sequences for x in held_rggs)
    summary = {
        "heldout_AHD_count": len(held_tads),
        "heldout_RG_RGG_count": len(held_rggs),
        "heldout_cross_product_expected": len(held_tads) * len(held_rggs),
        "heldout_cross_product_unique": len(held_pairs),
        "AHD_identity_overlap_with_final917": len(final_tads & held_tads),
        "RG_RGG_identity_overlap_with_final917": len(final_rggs & held_rggs),
        "exact_pair_overlap_with_final917": len(final_ids & held_pairs),
        "heldout_AHD_identifiers_with_exact_sequence_match_in_final917": held_tad_sequence_overlap,
        "heldout_RG_RGG_identifiers_with_exact_sequence_match_in_final917": held_rgg_sequence_overlap,
        "final917_pairs_with_reserved_AHD": sum(t in held_tads for _, t in final_parts),
        "final917_pairs_with_reserved_RG_RGG": sum(r in held_rggs for r, _ in final_parts),
        "independence_basis": "identifier-exclusive identities and no use of held-out outcomes; chronological timing is not required",
        "sequence_string_caveat": "FASTA contains identical amino-acid strings under distinct identifiers; see heldout_sequence_exclusion_audit.csv",
        "unix_shuf_status": "reported historical context; archived lists exist, but no executable shuf command plus seed was recovered",
    }
    return audit, summary


def exclusion_accounting(pre_ids: set[str], final_ids: set[str], held_tads: set[str], held_rggs: set[str]) -> tuple[pd.DataFrame, dict]:
    if final_ids - pre_ids:
        raise ValueError("Final 917 is not a subset of the archived pre-exclusion list")
    removed = pre_ids - final_ids
    categories = {"reserved_AHD_only": 0, "reserved_RG_RGG_only": 0, "both_reserved_categories": 0, "other_or_unresolved": 0}
    for pair in removed:
        rgg, ahd = pair_parts(pair)
        a, r = ahd in held_tads, rgg in held_rggs
        key = "both_reserved_categories" if a and r else "reserved_AHD_only" if a else "reserved_RG_RGG_only" if r else "other_or_unresolved"
        categories[key] += 1
    rows = [
        {"accounting_item": "pre_exclusion_pair_count", "count": len(pre_ids), "status": "directly documented"},
        {"accounting_item": "final_training_pair_count", "count": len(final_ids), "status": "directly documented"},
        *[{"accounting_item": k, "count": v, "status": "reconstructed by exact identity matching"} for k, v in categories.items()],
        {"accounting_item": "total_removed", "count": len(removed), "status": "exact set difference"},
    ]
    return pd.DataFrame(rows), {**categories, "pre": len(pre_ids), "final": len(final_ids), "removed": len(removed)}


def normalize_b22_table(path: Path) -> pd.DataFrame:
    df = pd.read_csv(require(path), low_memory=False)
    id_col = "pair_id" if "pair_id" in df else "name"
    b22_col = "Corrected_B22_Hummer"
    log_col = "B22_corr1_log10"
    if b22_col not in df:
        if log_col not in df:
            raise ValueError(f"No authoritative B22 columns in {path}")
        df[b22_col] = -np.power(10.0, pd.to_numeric(df[log_col], errors="coerce"))
    if log_col not in df:
        b = pd.to_numeric(df[b22_col], errors="coerce")
        df[log_col] = np.where(b < 0, np.log10(-b), np.nan)
    out = df[[id_col, b22_col, log_col]].rename(columns={id_col: "pair_id"}).copy()
    if out["pair_id"].duplicated().any():
        raise ValueError(f"Duplicate B22 pair IDs in {path}")
    return out


def descriptive(values: pd.Series, prefix: str) -> dict:
    x = pd.to_numeric(values, errors="coerce").dropna().to_numpy(float)
    q = np.percentile(x, [5, 25, 50, 75, 95])
    return {
        f"{prefix}_n": len(x),
        f"{prefix}_mean": float(np.mean(x)),
        f"{prefix}_sd": float(np.std(x, ddof=1)),
        f"{prefix}_p05": float(q[0]),
        f"{prefix}_p25": float(q[1]),
        f"{prefix}_median": float(q[2]),
        f"{prefix}_p75": float(q[3]),
        f"{prefix}_p95": float(q[4]),
    }


def distribution_stability() -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    tables = {name: normalize_b22_table(path) for name, path in HIST_B22.items()}
    names = list(tables)
    for a, b in zip(names[:2], names[1:3]):
        missing = set(tables[a].pair_id) - set(tables[b].pair_id)
        if missing:
            raise ValueError(f"Historical B22 stage is not cumulative: {a} has {len(missing)} pairs absent from {b}")
    expected_911 = set(read_names(LISTS / "sim_pairs_for_training_090425.txt"))
    expected_1039 = set(read_names(PRE_EXCLUSION_LIST))
    if set(tables["Expanded cohort (911)"].pair_id) != expected_911:
        raise ValueError("The 911-pair B22 table does not match its archived pair list")
    if set(tables["Pre-exclusion cohort (1,039)"].pair_id) != expected_1039:
        raise ValueError("The 1,039-pair B22 table does not match the pre-exclusion pair list")
    final = tables[names[-1]]
    final_raw = final["Corrected_B22_Hummer"].dropna().to_numpy(float)
    final_log = final["B22_corr1_log10"].dropna().to_numpy(float)
    rows = []
    previous = None
    for order, name in enumerate(names, start=1):
        df = tables[name]
        raw = df["Corrected_B22_Hummer"].dropna().to_numpy(float)
        log = df["B22_corr1_log10"].dropna().to_numpy(float)
        row = {
            "stage_order": order,
            "stage": name,
            "source_file": str(HIST_B22[name]),
            "n_pairs": int(df.pair_id.nunique()),
            **descriptive(pd.Series(raw), "B22"),
            **descriptive(pd.Series(log), "log10_negB22_attractive"),
            "KS_log10_to_final917": float(stats.ks_2samp(log, final_log).statistic),
            "Wasserstein_log10_to_final917": float(stats.wasserstein_distance(log, final_log)),
            "KS_B22_to_final917": float(stats.ks_2samp(raw, final_raw).statistic),
            "Wasserstein_B22_to_final917": float(stats.wasserstein_distance(raw, final_raw)),
            "stopping_metric_interpretation": "retrospective audit; no prospective threshold recovered",
        }
        if previous is None:
            for metric in ["KS_log10_vs_previous", "Wasserstein_log10_vs_previous", "abs_delta_B22_median", "abs_delta_B22_p05", "abs_delta_B22_p25", "abs_delta_B22_p75", "abs_delta_B22_p95"]:
                row[metric] = np.nan
        else:
            prev_raw = previous["Corrected_B22_Hummer"].dropna().to_numpy(float)
            prev_log = previous["B22_corr1_log10"].dropna().to_numpy(float)
            pq = np.percentile(prev_raw, [5, 25, 50, 75, 95])
            cq = np.percentile(raw, [5, 25, 50, 75, 95])
            row.update(
                {
                    "KS_log10_vs_previous": float(stats.ks_2samp(prev_log, log).statistic),
                    "Wasserstein_log10_vs_previous": float(stats.wasserstein_distance(prev_log, log)),
                    "abs_delta_B22_p05": abs(float(cq[0] - pq[0])),
                    "abs_delta_B22_p25": abs(float(cq[1] - pq[1])),
                    "abs_delta_B22_median": abs(float(cq[2] - pq[2])),
                    "abs_delta_B22_p75": abs(float(cq[3] - pq[3])),
                    "abs_delta_B22_p95": abs(float(cq[4] - pq[4])),
                }
            )
        rows.append(row)
        previous = df
    return pd.DataFrame(rows), tables


def split_large_bins(df: pd.DataFrame, col: str, edges: np.ndarray, max_count: int) -> pd.DataFrame:
    out = df.copy()
    out["bin_id"] = pd.cut(out[col], bins=edges, labels=False, include_lowest=True)
    out["final_bin"] = out["bin_id"].astype("Int64").astype(str)
    while True:
        counts = out["final_bin"].value_counts()
        too_big = counts[counts > max_count].index.tolist()
        if not too_big:
            break
        for label in too_big:
            mask = out["final_bin"] == label
            subset = out.loc[mask, col]
            n = len(subset)
            n_sub = math.ceil(n / max_count)
            ranks = subset.rank(method="first")
            sub_ids = np.floor((ranks - 1) * n_sub / n).astype(int)
            out.loc[mask, "final_bin"] = [f"{label}_{i}" for i in sub_ids]
    return out


def assign_by_historical_bounds(df: pd.DataFrame, bounds: pd.DataFrame, col: str) -> pd.Series:
    assigned = pd.Series(index=df.index, dtype="object")
    for label, row in bounds.iterrows():
        mask = pd.to_numeric(df[col], errors="coerce").between(row.min_val, row.max_val)
        assigned.loc[mask] = label
    return assigned


def ncp_audit(final_ids: set[str], fasta: dict[str, str]) -> tuple[pd.DataFrame, dict]:
    full = pd.read_csv(require(FULL), low_memory=False)
    eligible = pd.read_csv(require(CALVADOS / "all_pairs_for_training_features_data.csv"), low_memory=False)
    existing = pd.read_csv(require(CALVADOS / "sim_pairs_added_features_data_090425.csv"), low_memory=False)
    added = pd.read_csv(require(CALVADOS / "ocn_binned_features_data_230425.csv"), low_memory=False)
    final = pd.read_csv(require(FINAL_FEATURES), low_memory=False)

    charges = {identity: sum(1 if aa in "RK" else -1 if aa in "DE" else 0 for aa in seq) for identity, seq in fasta.items()}
    missing = (set(full.rgg) | set(full.tad)) - set(charges)
    if missing:
        raise ValueError(f"FASTA is missing {len(missing)} candidate identities")
    recalculated = -(full.rgg.map(charges).to_numpy(int) * full.tad.map(charges).to_numpy(int))
    max_diff = float(np.max(np.abs(recalculated - full.Opposite_Charge_Number.to_numpy(float))))
    if max_diff != 0:
        raise ValueError(f"Sequence-recalculated -NCP differs from historical OCN; max difference={max_diff}")

    _, edges = pd.qcut(eligible["Opposite_Charge_Number"], q=50, labels=False, retbins=True, duplicates="drop")
    max_count = math.ceil(len(eligible) / 50)
    split = split_large_bins(eligible, "Opposite_Charge_Number", edges, max_count)
    bounds = split.groupby("final_bin")["Opposite_Charge_Number"].agg(min_val="min", max_val="max").sort_index()
    existing_bins = assign_by_historical_bounds(existing, bounds, "Opposite_Charge_Number")
    final_bins = assign_by_historical_bounds(final, bounds, "Opposite_Charge_Number")
    candidate_counts = split.final_bin.value_counts().reindex(bounds.index, fill_value=0)
    existing_counts = existing_bins.value_counts().reindex(bounds.index, fill_value=0)
    target = (candidate_counts / candidate_counts.sum() * 1000).round().astype(int)
    deficit = (target - existing_counts).clip(lower=0)
    added_counts = added.final_bin.astype(str).value_counts().reindex(bounds.index, fill_value=0)
    final_counts = final_bins.value_counts().reindex(bounds.index, fill_value=0)
    audit = bounds.reset_index().rename(columns={"final_bin": "bin"})
    audit["candidate_count"] = candidate_counts.to_numpy(int)
    audit["historical_existing_count"] = existing_counts.to_numpy(int)
    audit["target_count"] = target.to_numpy(int)
    audit["deficit"] = deficit.to_numpy(int)
    audit["added_count"] = added_counts.to_numpy(int)
    audit["final917_count"] = final_counts.to_numpy(int)
    audit["historical_candidate_universe"] = "192,303 sequence-excluded training-eligible pairs"
    added_names = set(read_names(LISTS / "ocn_added_names.txt"))
    if set(added.name.astype(str)) != added_names:
        raise ValueError("Archived NCP-bin additions CSV does not match ocn_added_names.txt")
    meta = {
        "full_candidate_pairs": len(full),
        "training_eligible_candidate_pairs": len(eligible),
        "fasta_records": len(fasta),
        "sequence_NCP_max_abs_difference": max_diff,
        "initial_qcut_requested_bins": 50,
        "initial_qcut_realized_edges": len(edges) - 1,
        "recursive_final_bins": len(bounds),
        "max_count": max_count,
        "target_total": 1000,
        "archived_added_pairs": len(added),
    }
    return audit, meta


def plot_ncp_coverage(ax: plt.Axes, label: str | None = None) -> None:
    datasets = [
        ("Complete candidate space", pd.read_csv(FULL)["Opposite_Charge_Number"], GREY, 231840),
        ("Feature-guided training set", pd.read_csv(FINAL_FEATURES)["Opposite_Charge_Number"], BLUE, 917),
        ("Sequence-exclusive held-out set", pd.read_csv(HELD_FEATURES)["Opposite_Charge_Number"], ORANGE, 1794),
    ]
    for name, values, color, n in datasets:
        x, y = ecdf(values)
        ax.plot(x, y, color=color, lw=1.2, label=f"{name} (n = {n:,})")
    ax.set_xlabel("−NCP")
    ax.set_ylabel("Empirical cumulative fraction")
    ax.legend(frameon=False, loc="lower right")
    clean_axes(ax)
    if label:
        panel_label(ax, label)


def plot_stability(ax: plt.Axes, stability: pd.DataFrame, tables: dict[str, pd.DataFrame], label: str | None = None) -> None:
    for (name, df), color in zip(tables.items(), COLORS):
        x, y = ecdf(df["B22_corr1_log10"])
        lw = 1.6 if name.startswith("Final") else 0.9
        ax.plot(x, y, color=color, lw=lw, label=f"{name.split(' (')[0]} (n = {len(df):,})")
    ax.set_xlabel(r"$log_{10}(-B_{22})$")
    ax.set_ylabel("Empirical cumulative fraction")
    ax.legend(frameon=False, loc="lower right", fontsize=4.9)
    clean_axes(ax)
    inset = inset_axes(ax, width="37%", height="35%", loc="upper left", borderpad=1.0)
    stage_x = np.arange(len(stability))
    inset.plot(stage_x, stability.Wasserstein_log10_to_final917, "o-", ms=2.4, lw=0.8, color=BLUE, label="Wasserstein")
    inset.plot(stage_x, stability.KS_log10_to_final917, "s-", ms=2.2, lw=0.8, color=ORANGE, label="KS")
    inset.set_xticks(stage_x, ["546", "911", "1,039", "917"], rotation=45, ha="right")
    inset.set_xlabel("Chronological stage (n)", fontsize=4.8)
    inset.set_ylabel("Distance to final", fontsize=4.8)
    inset.tick_params(labelsize=4.4, width=0.4)
    inset.legend(frameon=False, fontsize=4.0, loc="upper right")
    for sp in inset.spines.values():
        sp.set_linewidth(0.45)
    if label:
        panel_label(ax, label)


def plot_feature_scatter(ax: plt.Axes, df: pd.DataFrame, xcol: str, xlabel: str, label: str) -> None:
    x = pd.to_numeric(df[xcol], errors="coerce")
    y = pd.to_numeric(df["B22_corr1_log10"], errors="coerce")
    ok = np.isfinite(x) & np.isfinite(y)
    # Vector points avoid a multi-axes PDF raster-clipping artefact.
    ax.scatter(x[ok], y[ok], s=10, alpha=0.6, color=BLUE, edgecolors="none")
    if ok.sum() >= 2:
        m, b = np.polyfit(x[ok], y[ok], 1)
        xx = np.linspace(x[ok].min(), x[ok].max(), 100)
        ax.plot(xx, m * xx + b, color="#111111", lw=0.8)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(r"$log_{10}(-B_{22})$")
    clean_axes(ax)
    panel_label(ax, label)


def plot_feature_ecdf(ax: plt.Axes, feature: str, xlabel: str, label: str) -> None:
    for name, path, color in [
        ("Complete", FULL, GREY),
        ("Training", FINAL_FEATURES, BLUE),
        ("Held-out", HELD_FEATURES, ORANGE),
    ]:
        df = pd.read_csv(path, usecols=[feature])
        x, y = ecdf(df[feature])
        ax.plot(x, y, lw=1.2, color=color, label=name)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Empirical cumulative fraction")
    ax.legend(frameon=False, loc="lower right")
    clean_axes(ax)
    panel_label(ax, label)


def make_figures(stability: pd.DataFrame, tables: dict[str, pd.DataFrame]) -> None:
    style()
    fig, ax = plt.subplots(figsize=(3.25, 2.45))
    plot_ncp_coverage(ax, "E")
    fig.tight_layout()
    save(fig, "S4E_NCP_sampling_coverage")

    fig, ax = plt.subplots(figsize=(3.25, 2.45))
    plot_stability(ax, stability, tables, "H")
    fig.subplots_adjust(left=0.16, right=0.97, bottom=0.18, top=0.96)
    save(fig, "S4H_B22_distribution_stability")

    pilot = pd.read_csv(ROOT / "25_pairs_features_data.csv")
    early = pd.read_csv(ROOT / "initial_set_features_data.csv")
    fig, axes = plt.subplots(2, 4, figsize=(12.0, 5.4))
    plot_feature_scatter(axes[0, 0], pilot, "Charge_Complementarity", "Charge complementarity", "A")
    plot_feature_scatter(axes[0, 1], pilot, "Average_NCPR", "Mean pair NCPR", "B")
    plot_feature_scatter(axes[0, 2], pilot, "Average_lambda", "Average λ", "C")
    plot_feature_scatter(axes[0, 3], early, "Opposite_Charge_Number", "−NCP", "D")
    plot_ncp_coverage(axes[1, 0], "E")
    plot_feature_ecdf(axes[1, 1], "Total_Length", "Total pair length", "F")
    plot_feature_ecdf(axes[1, 2], "Average_lambda", "Average λ", "G")
    plot_stability(axes[1, 3], stability, tables, "H")
    fig.subplots_adjust(left=0.06, right=0.985, bottom=0.12, top=0.94, wspace=0.48, hspace=0.58)
    save(fig, "revised_Figure_S4_A-H_proposed")


def make_heldout_fig4d() -> dict:
    held = pd.read_csv(HELD_FEATURES)
    corr = pd.read_csv(CORRELATIONS)
    stat = corr[(corr.dataset == "held-out sequence-exclusive") & (corr.feature == "Opposite_Charge_Number")].iloc[0]
    x = held.Opposite_Charge_Number.to_numpy(float)
    y = held.B22_corr1_log10.to_numpy(float)
    fig, ax = plt.subplots(figsize=(3.2, 2.55))
    sc = ax.scatter(x, y, c=held.Average_lambda, cmap="viridis", s=10, alpha=0.6, edgecolors="none", rasterized=True)
    m, b = np.polyfit(x, y, 1)
    xx = np.linspace(x.min(), x.max(), 200)
    ax.plot(xx, m * xx + b, color="#111111", lw=1.0)
    ax.set_xlabel("−NCP")
    ax.set_ylabel(r"$log_{10}(-B_{22})$")
    ax.set_title("Sequence-exclusive held-out set")
    ax.text(
        0.03,
        0.97,
        f"n = {len(held):,}\nPearson r = {float(stat.Pearson_r):.3f}\nSpearman $\\rho$ = {float(stat.Spearman_rho):.3f}",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=7,
    )
    cbar = fig.colorbar(sc, ax=ax, pad=0.03)
    cbar.set_label("Mean λ")
    cbar.ax.tick_params(labelsize=6)
    clean_axes(ax)
    panel_label(ax, "D")
    fig.tight_layout()
    save(fig, "Fig4D_heldout_candidate")
    return {"n": len(held), "Pearson_r": float(stat.Pearson_r), "Spearman_rho": float(stat.Spearman_rho)}


def write_report(counts: pd.DataFrame, held_summary: dict, exclusion: dict, ncp_meta: dict, stability: pd.DataFrame, current_main: str) -> None:
    script_lines = {
        "NCP binning": "plot_features.py lines 491-623",
        "outside-filter sampling": "plot_features.py lines 802-841",
        "length diversity and partner uniqueness": "plot_features.py lines 843-945",
    }
    text = f"""# Training-set selection provenance

## 1. Complete candidate space

**Directly documented and independently reconstructed.** The sequence catalogue contains 230 AHDs and 1,008 RG/RGG regions, giving 231,840 pair combinations. Sequence-derived net charges reproduce the historical `Opposite_Charge_Number` column exactly as −NCP = −Q_RG/RGG × Q_AHD (maximum absolute difference {ncp_meta['sequence_NCP_max_abs_difference']:.0f}).

## 2. Initial 5 × 5 pilot

**Directly documented by `25_pairs_features_data.csv`.** Five AHDs and five RG/RGG regions were simulated in all 25 combinations. This pilot is an early exploratory set and is not simply a subset that sums into the later 917-pair list.

## 3. Early feature-guided expansion

**Documented in archived lists and commented code.** The historical script calculates charge complementarity, mean pair NCPR and average lambda. The archived `final_test_names.txt` contains 500 unique pairs. The precise end-to-end order of every early sub-round is not recoverable from a single executable manifest.

## 4. Sampling outside the initial feature filter

**Directly documented in commented code and a 200-pair list.** `{script_lines['outside-filter sampling']}` contains filters outside the initially favoured charge-complementarity/NCPR/lambda region and `random_state=42` sampling. `200_pairs_outside_filter_less_than_lambda045_names.txt` contains 200 unique pairs.

## 5. Length/diversity considerations

**Directly documented as candidate procedures, but not fully linked to a final round manifest.** `{script_lines['length diversity and partner uniqueness']}` records randomized AHD order, total-length diversity, and unused-RG/RGG constraints. No seed is set for the Python `random.shuffle` calls in those blocks, so those exact orderings cannot be replayed from code alone.

## 6. −NCP-stratified expansion

**Directly documented and retrospectively reproduced.** `{script_lines['NCP binning']}` uses 50 quantile bins, recursively subdivides bins larger than ceil(N/50), calculates proportional targets for 1,000 pairs, computes deficits, and samples with `random_state=42`. The archived operation used the 192,303-pair sequence-excluded training-eligible landscape, not the full 231,840-pair cross-product. The archived selected table contains {ncp_meta['archived_added_pairs']} pairs and exactly matches `ocn_added_names.txt`.

## 7. Empirical B22-distribution stabilisation

**Historical decision described qualitatively; numerical audit is retrospective.** Complete B22 tables establish cumulative cohorts of 546, 911 and 1,039 simulated pairs, followed by the 917-pair post-exclusion training set. `B22_distribution_stability_by_stage.csv` reports KS statistics, Wasserstein distances and quantile changes. No prospective numerical stopping threshold was recovered, so these measurements support but do not define the historical stopping decision.

## 8. Unix `shuf` selection of held-out identities

**Historical context supplied by the project, not reproducibly documented by an archived command plus seed.** The archived lists contain exactly 23 reserved AHD identities and 78 reserved RG/RGG identities. The audit does not rerun `shuf` and does not infer a replacement draw.

## 9. Sequence-level exclusion from training

**Exactly reconstructed.** The archived pre-exclusion list has {exclusion['pre']:,} pairs. Removing every pair containing a reserved AHD and/or reserved RG/RGG removes {exclusion['removed']} pairs: {exclusion['reserved_AHD_only']} AHD-only, {exclusion['reserved_RG_RGG_only']} RG/RGG-only and {exclusion['both_reserved_categories']} in both categories. There are {exclusion['other_or_unresolved']} unresolved removals, leaving {exclusion['final']:,} pairs.

## 10. Final 917-pair training set

**Directly documented.** `sim_pairs_for_training_270525.txt` and `fig4_917_sampling_membership.csv` contain the same 917 unique pairs. The set is staged and feature-guided, not a single random or unbiased sample.

## 11. Complete 1,794-pair held-out cross-product

**Directly documented and verified at the identifier level.** The 23 reserved AHD identifiers and 78 reserved RG/RGG identifiers generate exactly 1,794 cross-product pairs. AHD identifier overlap, RG/RGG identifier overlap and exact pair overlap with the 917 training set are all zero. However, literal FASTA comparison finds {held_summary['heldout_AHD_identifiers_with_exact_sequence_match_in_final917']} held-out AHD identifier and {held_summary['heldout_RG_RGG_identifiers_with_exact_sequence_match_in_final917']} held-out RG/RGG identifiers whose amino-acid strings are also present under different training identifiers. Independence is outcome-independent and identifier-exclusive; it does not depend on chronological timing.

## 12. Provenance gaps and limitations

- No single executable manifest reconstructs every early selection sub-round.
- The archived files do not contain a reproducible Unix `shuf` command plus seed for the 23/78 identity draw.
- Some commented `random.shuffle` selection blocks have no documented seed.
- The 50-bin −NCP procedure is reproducible from archived inputs, but its candidate universe is the 192,303-pair sequence-excluded training-eligible landscape.
- Current Supplementary S4 mapping is A-D early feature relationships, E −NCP distribution, F length distribution, G average-lambda distribution, H reconstructed −NCP-bin coverage, and I held-out −NCP/B22 validation. The proposed A-H figure replaces H with distribution stability and omits I.
- Current main Figure 4D is: **{current_main}**.
- The archived FASTA contains identical sequence strings under distinct identifiers. The exact matches are listed in `heldout_sequence_exclusion_audit.csv`; therefore “zero exact amino-acid-sequence overlap” is not supported literally, although identifier and exact-pair overlaps are zero.

## Manuscript-safe description

The complete sequence-derived candidate landscape comprised 230 AHDs and 1,008 RG/RGG regions (231,840 possible pairs). The 917-pair CALVADOS training panel was assembled through staged, feature-guided sampling informed by early relationships with charge complementarity, mean pair NCPR, average lambda and the negative net-charge product (−NCP = −Q_RG/RGG × Q_AHD). Archived code documents later −NCP-stratified deficit filling toward an approximate 1,000-pair target. The archived 1,039-pair pre-exclusion set was reduced to 917 by removing 122 pairs containing one or both of 23 reserved AHD and 78 reserved RG/RGG identifiers. The held-out simulations comprise the complete 23 × 78 cross-product (1,794 pairs), with zero AHD-identifier, RG/RGG-identifier or exact-pair overlap with the training set. The decision to stop adding training simulations was based on empirical stabilisation of the B22 distribution; KS and Wasserstein measurements reported here are retrospective diagnostics rather than prospective stopping thresholds.
"""
    (OUT / "TRAINING_SET_SELECTION_PROVENANCE.md").write_text(text)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    final_ids = set(read_names(FINAL_LIST))
    pre_ids = set(read_names(PRE_EXCLUSION_LIST))
    held_tads = set(read_names(HELDOUT_TADS))
    held_rggs = set(read_names(HELDOUT_RGGS))
    fasta = read_fasta(FASTA)

    if len(final_ids) != 917 or len(held_tads) != 23 or len(held_rggs) != 78:
        raise ValueError("Authoritative final/held-out counts do not match 917/23/78")
    final_table_ids = set(pd.read_csv(FINAL_FEATURES).pair_id.astype(str))
    if final_ids != final_table_ids:
        raise ValueError("Final 917 list and authoritative feature table differ")

    counts, provenance, overlaps, stage_sets = stage_audit(final_ids, held_tads, held_rggs)
    held_audit, held_summary = heldout_audit(final_ids, held_tads, held_rggs, fasta)
    exclusion_df, exclusion = exclusion_accounting(pre_ids, final_ids, held_tads, held_rggs)
    stability, b22_tables = distribution_stability()
    ncp_bins, ncp_meta = ncp_audit(final_ids, fasta)

    counts.to_csv(OUT / "training_selection_stage_counts.csv", index=False)
    provenance.to_csv(OUT / "training_selection_pair_provenance.csv", index=False)
    overlaps.to_csv(OUT / "training_selection_pairwise_overlaps.csv", index=False)
    held_audit.to_csv(OUT / "heldout_sequence_exclusion_audit.csv", index=False)
    (OUT / "heldout_sequence_exclusion_summary.json").write_text(json.dumps(held_summary, indent=2) + "\n")
    exclusion_df.to_csv(OUT / "final917_exclusion_accounting.csv", index=False)
    stability.to_csv(OUT / "B22_distribution_stability_by_stage.csv", index=False)
    ncp_bins.to_csv(OUT / "ncp_sampling_bin_audit.csv", index=False)

    current_script = (ROOT / "reviewer_analysis" / "final_fig4_revision" / "build_final_fig4_revision.py").read_text()
    current_main = "training-set relationship (917 pairs)" if 'fig4_917_sampling_membership.csv' in current_script else "not resolved"
    make_figures(stability, b22_tables)
    held_stats = make_heldout_fig4d()
    write_report(counts, held_summary, exclusion, ncp_meta, stability, current_main)

    unique_training_rgg = {pair_parts(p)[0] for p in final_ids}
    unique_training_tad = {pair_parts(p)[1] for p in final_ids}
    print(f"FINAL TRAINING PAIRS: {len(final_ids)}")
    print(f"UNIQUE TRAINING AHDs: {len(unique_training_tad)}")
    print(f"UNIQUE TRAINING RG/RGGs: {len(unique_training_rgg)}")
    print(f"HELD-OUT AHDs: {len(held_tads)}")
    print(f"HELD-OUT RG/RGGs: {len(held_rggs)}")
    print(f"HELD-OUT PAIRS: {held_summary['heldout_cross_product_unique']}")
    print(f"AHD IDENTITY OVERLAP: {held_summary['AHD_identity_overlap_with_final917']}")
    print(f"RG/RGG IDENTITY OVERLAP: {held_summary['RG_RGG_identity_overlap_with_final917']}")
    print(f"EXACT PAIR OVERLAP: {held_summary['exact_pair_overlap_with_final917']}")
    print(f"HELD-OUT AHD IDs WITH AN EXACT FASTA-SEQUENCE MATCH IN TRAINING: {held_summary['heldout_AHD_identifiers_with_exact_sequence_match_in_final917']}")
    print(f"HELD-OUT RG/RGG IDs WITH AN EXACT FASTA-SEQUENCE MATCH IN TRAINING: {held_summary['heldout_RG_RGG_identifiers_with_exact_sequence_match_in_final917']}")
    print(f"RECONSTRUCTED PRE-EXCLUSION PAIR COUNT: {exclusion['pre']}")
    print(f"PAIRS REMOVED BY HELD-OUT AHD EXCLUSION: {exclusion['reserved_AHD_only'] + exclusion['both_reserved_categories']}")
    print(f"PAIRS REMOVED BY HELD-OUT RG/RGG EXCLUSION: {exclusion['reserved_RG_RGG_only'] + exclusion['both_reserved_categories']}")
    print(f"OVERLAP OF THOSE REMOVALS: {exclusion['both_reserved_categories']}")
    print(f"UNRESOLVED FINAL-917 PROVENANCE COUNT: {exclusion['other_or_unresolved']}")
    print("B22 STABILITY DIRECTLY DOCUMENTED OR RETROSPECTIVE: retrospective quantitative audit of a documented qualitative stopping decision")
    print(f"S4E CREATED: {OUT / 'S4E_NCP_sampling_coverage.pdf'}")
    print(f"S4H CREATED: {OUT / 'S4H_B22_distribution_stability.pdf'}")
    print(f"PROPOSED S4 A-H CREATED: {OUT / 'revised_Figure_S4_A-H_proposed.pdf'}")
    print(f"MAIN FIG4D CURRENTLY TRAINING OR HELD-OUT: {current_main}")
    print(f"HELD-OUT FIG4D CANDIDATE CREATED: {OUT / 'Fig4D_heldout_candidate.pdf'}")
    print(f"HELD-OUT FIG4D STATS: n={held_stats['n']}, Pearson={held_stats['Pearson_r']:.3f}, Spearman={held_stats['Spearman_rho']:.3f}")
    print("OUTPUT PATHS:")
    for path in sorted(OUT.iterdir()):
        if path.is_file() and path.name != Path(__file__).name:
            print(path)


if __name__ == "__main__":
    main()
