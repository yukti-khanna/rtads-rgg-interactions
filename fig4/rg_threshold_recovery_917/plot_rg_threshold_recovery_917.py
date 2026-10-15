#!/usr/bin/env python3
"""Render S4J/S4K from completed RG-threshold recovery tables only."""
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
FAMILIES = [("compact", "Compact linker", "compact_linker_cutoff"),
            ("bridge", "Bridge linker", "bridge_linker_cutoff"),
            ("flank", "Flank extension", "flank_extension")]
COLORS = {"compact": "#1b9e77", "bridge": "#d95f02", "flank": "#7570b3"}

def load():
    frames = []
    for factor, label, pname in FAMILIES:
        d = pd.read_csv(HERE / f"one_factor_{factor}_recovery.csv")
        d["parameter_family"], d["parameter_name"], d["parameter_value"] = label, pname, d["factor_value"]
        d["n_unique_regions_total"] = d["total_downstream_RG_RGG_regions"]
        d["n_any_overlap"] = (d["unique_recovered_percent"] * d["n_unique_regions_total"] / 100).round().astype(int)
        d["fraction_any_overlap"] = d["unique_recovered_percent"] / 100
        d["n_pairs_total"] = d["pair_total_pairs"]
        d["n_pairs_any_overlap"] = (d["pair_recovered_percent"] * d["n_pairs_total"] / 100).round().astype(int)
        d["fraction_pairs_any_overlap"] = d["pair_recovered_percent"] / 100
        d["n_exact_match"] = (d["unique_exact_percent"] * d["n_unique_regions_total"] / 100).round().astype(int)
        d["fraction_exact_match"] = d["unique_exact_percent"] / 100
        d["n_pairs_exact_match"] = (d["pair_exact_percent"] * d["n_pairs_total"] / 100).round().astype(int)
        d["fraction_pairs_exact_match"] = d["pair_exact_percent"] / 100
        d["n_lost"] = (d["unique_lost_percent"] * d["n_unique_regions_total"] / 100).round().astype(int)
        d["n_pairs_lost"] = (d["pair_lost_percent"] * d["n_pairs_total"] / 100).round().astype(int)
        d["is_baseline"] = d["is_baseline"].astype(bool)
        frames.append(d)
    out = pd.concat(frames, ignore_index=True).sort_values(["parameter_family", "parameter_value"])
    if set(out.n_unique_regions_total) != {627} or set(out.n_pairs_total) != {917}:
        raise ValueError("Unexpected denominators in completed analysis")
    base = out[out.is_baseline]
    if not ((base.n_any_overlap == 627).all() and (base.n_pairs_any_overlap == 917).all()):
        raise ValueError("Baseline does not recover all downstream regions/pairs")
    return out

def export_tables(d):
    region_cols = ["parameter_family","parameter_name","parameter_value","compact_linker_cutoff","bridge_linker_cutoff","flank_extension","is_baseline","n_unique_regions_total","n_any_overlap","fraction_any_overlap","n_exact_match","fraction_exact_match","n_lost"]
    pair_cols = ["parameter_family","parameter_name","parameter_value","compact_linker_cutoff","bridge_linker_cutoff","flank_extension","is_baseline","n_pairs_total","n_pairs_any_overlap","fraction_pairs_any_overlap","n_pairs_exact_match","fraction_pairs_exact_match","n_pairs_lost"]
    d[region_cols].to_csv(HERE / "S4J_region_recovery_plot_data.csv", index=False)
    d[pair_cols].to_csv(HERE / "S4K_pair_recovery_plot_data.csv", index=False)

def panel(d, pair=False, ax=None):
    if ax is None: fig, ax = plt.subplots(figsize=(5.2, 3.4))
    for factor, label, _ in FAMILIES:
        q = d[d.parameter_family == label]
        x = list(range(len(q)))
        y = (q.fraction_pairs_any_overlap if pair else q.fraction_any_overlap) * 100
        ax.plot(x, y, "o-", color=COLORS[factor], lw=0.9, ms=4, label=label)
        for xi, yi, baseline in zip(x, y, q.is_baseline):
            if baseline: ax.plot(xi, yi, "o", ms=7, mfc="white", mec="black", mew=0.8, zorder=4)
    ax.set_ylabel("Simulation pairs retained (%)" if pair else "Retained with any overlap (%)", fontsize=8)
    ax.set_xlabel("One-factor perturbation", fontsize=8)
    ax.set_title("Impact of RG-region threshold perturbations on final 917 simulation pairs" if pair else "Recovery of downstream RG/RGG regions across RG-region threshold perturbations", fontsize=9)
    ax.set_ylim(0, 105); ax.tick_params(labelsize=7); ax.grid(axis="y", alpha=.2, lw=.5)
    ax.legend(fontsize=7, frameon=False, loc="lower left")
    ax.set_xticks([])
    return ax

def main():
    d = load(); export_tables(d)
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 3.6), constrained_layout=True)
    panel(d, False, axes[0]); panel(d, True, axes[1])
    axes[0].text(-0.14, 1.04, "S4J", transform=axes[0].transAxes, weight="bold", fontsize=10)
    axes[1].text(-0.14, 1.04, "S4K", transform=axes[1].transAxes, weight="bold", fontsize=10)
    for suffix, dpi in [("pdf", None), ("png", 600)]:
        fig.savefig(HERE / f"S4JK_RG_threshold_robustness.{suffix}", dpi=dpi)
    plt.close(fig)
    for pair, tag in [(False, "S4J_RG_region_threshold_recovery"), (True, "S4K_pair_threshold_recovery")]:
        fig, ax = plt.subplots(figsize=(5.2, 3.4), constrained_layout=True); panel(d, pair, ax); fig.savefig(HERE/f"{tag}.pdf"); fig.savefig(HERE/f"{tag}.png", dpi=600); plt.close(fig)
    print("Source: one_factor_compact_recovery.csv, one_factor_bridge_recovery.csv, one_factor_flank_recovery.csv")
    print("Verified denominators: 627 unique regions; 917 pairs")
    for label, q in d.groupby("parameter_family", sort=False): print(label, list(q.parameter_value.astype(int)), "min region %", q.fraction_any_overlap.min()*100, "min pair %", q.fraction_pairs_any_overlap.min()*100)
    print("PNG dpi: 600")

if __name__ == "__main__": main()
