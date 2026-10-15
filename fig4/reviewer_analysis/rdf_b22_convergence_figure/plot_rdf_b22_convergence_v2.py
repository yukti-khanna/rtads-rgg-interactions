#!/usr/bin/env python3
"""Create the 3-panel B22 reproducibility/convergence supplementary figure.

Inputs are the completed Figure 4 convergence outputs. This script does not run
simulations, recalculate RDFs, recalculate B22 from trajectories, or change
finite-size correction definitions.
"""
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[2]
TASK_PANEL_A = ROOT / "reviewer_analysis" / "fig4_task2_b22_convergence"
TASK2 = ROOT / "task2_b22_convergence"
FINAL = ROOT / "reviewer_analysis" / "final_fig4_revision"
OUT = ROOT / "reviewer_analysis" / "rdf_b22_convergence_figure"
OUT.mkdir(exist_ok=True)

DPI = 600
FIGSIZE = (7.2, 5.2)
WEAK_ID = "O00144_1_1__P49910_193_208"
WEAK_ID_ALT = "O00144_1_1_P49910_193_208"
STRONG_ID = "O60506_3_1__A8MYZ6_430_473"
PANEL_C_PAIR_IDS = [
    "P53814_1_2__P32242_63_78",
    "Q9C0J8_3_6__P49910_193_208",
    "Q9UIQ6_1_1__A6NHT5_163_180",
    "Q96E39_4_2__Q9P016_65_80",
    "Q07092_6_4__Q68CP9_440_457",
    "Q96A19_1_1__Q06945_399_417",
    "A8MXQ7_1_1__Q17RP2_334_357",
    "P49448_1_1__O15391_98_116",
    "O15399_1_1__Q9UIF9_321_335",
    "Q9NQ39_2_1__Q9UJW8_43_61",
    "Q6Y7W6_7_1__Q9H2P0_829_854",
    "P19338_5_1__Q9NRY4_393_408",
    "A6H8Y1_1_2__Q66K89_756_771",
    "Q96JG9_1_2__O14948_55_81",
    "Q9H320_1_1__Q9Y6Q9_796_812",
]
EXPECTED_PANEL_B = pd.DataFrame(
    {
        "k": [1, 2, 3, 4, 5],
        "median_absolute_deviation": [121.073486, 72.685436, 48.456957, 30.268371, 0.0],
        "fraction_within_5pct": [0.244275, 0.368593, 0.514177, 0.710578, 1.0],
        "fraction_within_10pct": [0.441439, 0.634351, 0.794111, 0.928244, 1.0],
        "fraction_within_20pct": [0.710578, 0.883751, 0.963359, 0.994766, 1.0],
    }
)
EXPECTED_PANEL_C_MEDIANS = {"weak": 0.014475, "intermediate": 0.010859, "strong": 0.032567}

PAIR_LABELS = {
    WEAK_ID: "FZD9-ZNF165",
    "O00144_1_1__P35716_353_367": "FZD9-SOX11",
    "Q02388_5_13__P49910_193_208": "COL7A1-ZNF165",
    "P19338_5_1__Q9UJU2_14_65": "NCL-LEF1",
    STRONG_ID: "SYNCRIP-FOXO6",
}

sys.path.insert(0, str(FINAL))
from build_final_fig4_revision import load_rdfs  # noqa: E402


def load_inputs():
    return {
        "rep": pd.read_csv(TASK_PANEL_A / "per_replica_b22.csv"),
        "unc": pd.read_csv(TASK_PANEL_A / "per_pair_b22_uncertainty.csv"),
        "sel": pd.read_csv(TASK_PANEL_A / "convergence_pair_selection.csv"),
        "conv": pd.read_csv(TASK2 / "replicate_count_convergence.csv"),
        "conv_summary": pd.read_csv(TASK2 / "replicate_count_convergence_summary.csv"),
        "time": pd.read_csv(TASK2 / "time_convergence_summary.csv"),
        "time_replicas": pd.read_csv(TASK2 / "time_convergence_per_replica.csv"),
        "panel_c_sel": pd.read_csv(TASK2 / "convergence_pair_selection.csv"),
        "prov": pd.read_csv(FINAL / "Fig4B_pair_provenance.csv"),
    }


def normalize_pair_id(pid):
    if pid == WEAK_ID_ALT:
        return WEAK_ID
    return pid


def provenance_row(prov, pair_id):
    tmp = prov.copy()
    tmp["pair_id_norm"] = tmp["pair_id"].map(normalize_pair_id)
    hit = tmp[tmp["pair_id_norm"] == pair_id]
    if hit.empty:
        return None
    return hit.iloc[0]


def select_panel_a_pairs(rep, unc, sel, prov):
    selected_ids = [
        WEAK_ID,
        "O00144_1_1__P35716_353_367",
        "Q02388_5_13__P49910_193_208",
        "P19338_5_1__Q9UJU2_14_65",
        STRONG_ID,
    ]

    sel_meta = sel.set_index("pair_id")
    unc_meta = unc.set_index("pair_id")
    rows = []
    for pair_id in selected_ids:
        reps = rep[rep["pair_id"] == pair_id].sort_values("replicate_id")
        category = "strong" if pair_id == STRONG_ID else (
            sel_meta.loc[pair_id, "regime"] if pair_id in sel_meta.index else "weak"
        )

        if not reps.empty:
            values = reps["Hummer_B22"].astype(float).to_numpy()
            mean = float(values.mean())
            sd = float(values.std(ddof=1))
            se = float(sd / np.sqrt(len(values)))
            source = "per_replica_b22.csv"
        else:
            prow = provenance_row(prov, pair_id)
            if prow is None:
                raise ValueError(f"No replica rows or provenance summary found for required pair {pair_id}")
            rdf_path = Path("/Users/yuktikhanna/calvados/test_rdfs") / f"rdfs_{pair_id}_20nm.pkl"
            if not rdf_path.exists():
                raise FileNotFoundError(f"Required original Figure 4B RDF file not found: {rdf_path}")
            _, _, values = load_rdfs(rdf_path)
            values = values[:5].astype(float)
            mean = float(values.mean())
            sd = float(values.std(ddof=1))
            se = float(sd / np.sqrt(len(values)))
            prov_mean = float(prow["mean_B22"])
            if not np.isclose(mean, prov_mean, rtol=0, atol=1e-3):
                raise ValueError(
                    f"Recovered replica mean for {pair_id} ({mean}) does not match "
                    f"Fig4B provenance mean ({prov_mean})."
                )
            source = "existing final_fig4_revision load_rdfs on historical RDF pickle"

        if pair_id in PAIR_LABELS:
            label = PAIR_LABELS[pair_id]
        elif pair_id in unc_meta.index:
            tad = pair_id.split("__", 1)[1]
            rgg = pair_id.split("__", 1)[0]
            label = f"{rgg}\n{tad}"
        else:
            label = pair_id.replace("__", "\n")

        row = {
            "pair_label": label.replace("\n", " - "),
            "pair_id": pair_id,
            "category": category,
            "selection_reason": "original Figure 4B pair" if pair_id in {WEAK_ID, STRONG_ID}
            else "fixed user-specified additional Panel A representative pair",
            "mean_B22": mean,
            "SD": sd,
            "SE": se,
            "source": source,
        }
        for i in range(5):
            row[f"replica_{i+1}_B22"] = float(values[i]) if i < len(values) and np.isfinite(values[i]) else np.nan
        rows.append(row)

    out = pd.DataFrame(rows).sort_values("mean_B22", ascending=False).reset_index(drop=True)
    return out


def panel_b_summary(conv_summary):
    cols = [
        "k",
        "n_combinations",
        "median_absolute_deviation",
        "absolute_deviation_p05",
        "absolute_deviation_p95",
        "fraction_within_5pct",
        "fraction_within_10pct",
        "fraction_within_20pct",
    ]
    return conv_summary.loc[conv_summary["k"].isin([1, 2, 3, 4, 5]), cols].sort_values("k")


def panel_c_deviations(time, panel_c_sel):
    overlap = sorted(set(PANEL_C_PAIR_IDS).intersection(set(panel_c_sel["pair_id"])))
    if len(overlap) != 15:
        missing = sorted(set(PANEL_C_PAIR_IDS) - set(panel_c_sel["pair_id"]))
        raise ValueError(f"Panel C fixed pair overlap is {len(overlap)}/15; missing: {missing}")

    time = time[time["pair_id"].isin(PANEL_C_PAIR_IDS)].copy()
    final = (
        time[time["trajectory_fraction"] == 1.0][["pair_id", "mean"]]
        .rename(columns={"mean": "final_mean"})
    )
    dev = time.merge(final, on="pair_id", how="left")
    dev["relative_deviation_from_100pct"] = (
        (dev["mean"] - dev["final_mean"]).abs() / dev["final_mean"].abs()
    )
    dev.loc[dev["trajectory_fraction"] == 1.0, "relative_deviation_from_100pct"] = 0.0
    dev["pair_id"] = pd.Categorical(dev["pair_id"], categories=PANEL_C_PAIR_IDS, ordered=True)
    return dev.sort_values(["pair_id", "trajectory_fraction"]).reset_index(drop=True)


def panel_c_summary(dev):
    at80 = dev[dev["trajectory_fraction"] == 0.8].copy()
    return at80.groupby("regime", as_index=False).agg(
        median_relative_change_80_100=("relative_deviation_from_100pct", "median"),
        n_pairs=("pair_id", "count"),
        n_below_5pct=("relative_deviation_from_100pct", lambda s: int((s < 0.05).sum())),
        n_below_10pct=("relative_deviation_from_100pct", lambda s: int((s < 0.10).sum())),
    )


def plot_panel_a(ax, panel_a):
    x = np.arange(len(panel_a))
    colors = {"weak": "#4477AA", "intermediate": "#CCBB44", "strong": "#AA3377"}
    for i, row in panel_a.iterrows():
        vals = np.array([row[f"replica_{j}_B22"] for j in range(1, 6)], dtype=float)
        finite = vals[np.isfinite(vals)]
        jitter = np.linspace(-0.16, 0.16, max(len(finite), 1))
        color = colors.get(row["category"], "#555555")
        if len(finite):
            ax.scatter(np.full(len(finite), x[i]) + jitter, finite, s=14, color=color,
                       alpha=0.78, linewidths=0, zorder=3)
        ax.errorbar(x[i], row["mean_B22"], yerr=row["SD"], fmt="o", ms=4.0,
                    color="black", ecolor="black", elinewidth=0.8, capsize=3, zorder=4)
    ax.axhline(0, color="0.75", lw=0.6)
    ax.set_xticks(x)
    ax.set_xticklabels(panel_a["pair_label"], rotation=0)
    ax.set_ylabel("B22")
    ax.set_title("A  Replica-level B22 for representative AHD-RG/RGG pairs")


def plot_panel_b(ax, conv, summary):
    positions = summary["k"].to_numpy()
    data = [conv.loc[conv["k"] == k, "absolute_deviation"].dropna().to_numpy() for k in positions]
    ax.boxplot(data, positions=positions, widths=0.55, showfliers=False,
               medianprops={"color": "black", "linewidth": 0.9},
               boxprops={"linewidth": 0.7}, whiskerprops={"linewidth": 0.7},
               capprops={"linewidth": 0.7})
    ax.plot(summary["k"], summary["median_absolute_deviation"], color="#4477AA",
            marker="o", lw=0.9, ms=3.5)
    ax.set_xlabel("Replicas included")
    ax.set_ylabel("|B22(k) - B22(5)|")
    ax.set_title("B  B$_{22}$ stabilises with replica number")


def plot_panel_c(ax, panel_c_dev):
    order = ["weak", "intermediate", "strong"]
    colors = {"weak": "#4477AA", "intermediate": "#CCBB44", "strong": "#AA3377"}
    for pair_id, group in panel_c_dev.groupby("pair_id", sort=False, observed=True):
        regime = group["regime"].iloc[0]
        ax.plot(
            group["time_us"],
            group["relative_deviation_from_100pct"] * 100,
            color=colors[regime],
            lw=0.45,
            marker="o",
            ms=2.0,
            alpha=0.42,
        )
    for regime in order:
        med = panel_c_dev[panel_c_dev["regime"] == regime].groupby("time_us", as_index=False)[
            "relative_deviation_from_100pct"
        ].median()
        ax.plot(
            med["time_us"],
            med["relative_deviation_from_100pct"] * 100,
            color=colors[regime],
            lw=1.35,
            marker="o",
            ms=3.0,
            label=regime,
        )
    ax.axhline(0, color="0.75", lw=0.6)
    ax.set_xticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_xlabel("Trajectory length (us)")
    ax.set_ylabel("Relative deviation from 100% B22 (%)")
    ax.set_title("C  Trajectory-length convergence")
    ax.legend(fontsize=5.8, frameon=False)


def style_axes(axes):
    for ax in axes:
        ax.tick_params(labelsize=6.0, width=0.5, length=2.5)
        for spine in ax.spines.values():
            spine.set_linewidth(0.6)
        ax.title.set_fontsize(7.0)
        ax.xaxis.label.set_fontsize(6.5)
        ax.yaxis.label.set_fontsize(6.5)


def verify_authoritative_values(panel_b, panel_c_dev, panel_c_summary_df, panel_c_sel):
    print("Panel B medians for k = 1-5")
    print(panel_b[["k", "median_absolute_deviation"]].to_string(index=False))
    print("Panel B 5% / 10% / 20% fractions")
    print(panel_b[["k", "fraction_within_5pct", "fraction_within_10pct", "fraction_within_20pct"]].to_string(index=False))
    print("Panel C exact pair IDs")
    for pair_id in PANEL_C_PAIR_IDS:
        print(pair_id)

    overlap = len(set(PANEL_C_PAIR_IDS).intersection(set(panel_c_sel["pair_id"])))
    print(f"Panel C overlap with convergence_pair_selection.csv: {overlap}/15")
    print("Panel C weak/intermediate/strong 80->100% median relative changes")
    print(panel_c_summary_df[["regime", "median_relative_change_80_100"]].to_string(index=False))
    count_below_5 = int((panel_c_dev[panel_c_dev["trajectory_fraction"] == 0.8]["relative_deviation_from_100pct"] < 0.05).sum())
    count_below_10 = int((panel_c_dev[panel_c_dev["trajectory_fraction"] == 0.8]["relative_deviation_from_100pct"] < 0.10).sum())
    print(f"Panel C count below 5%: {count_below_5}/15")
    print(f"Panel C count below 10%: {count_below_10}/15")

    merged_b = panel_b.merge(EXPECTED_PANEL_B, on="k", suffixes=("", "_expected"))
    checks = [
        "median_absolute_deviation",
        "fraction_within_5pct",
        "fraction_within_10pct",
        "fraction_within_20pct",
    ]
    for col in checks:
        if not np.allclose(merged_b[col], merged_b[f"{col}_expected"], rtol=0, atol=5e-7):
            raise ValueError(
                f"Panel B {col} mismatch. Observed {merged_b[['k', col]].to_dict(orient='records')}; "
                f"expected {merged_b[['k', f'{col}_expected']].to_dict(orient='records')}"
            )
    if overlap != 15:
        raise ValueError(f"Panel C overlap mismatch: observed {overlap}/15, expected 15/15")
    observed_medians = dict(zip(panel_c_summary_df["regime"], panel_c_summary_df["median_relative_change_80_100"]))
    for regime, expected in EXPECTED_PANEL_C_MEDIANS.items():
        if not np.isclose(observed_medians[regime], expected, rtol=0, atol=5e-7):
            raise ValueError(
                f"Panel C {regime} median mismatch: observed {observed_medians[regime]}, expected {expected}"
            )
    if count_below_5 != 13 or count_below_10 != 15:
        raise ValueError(
            f"Panel C count mismatch: observed {count_below_5}/15 below 5% and "
            f"{count_below_10}/15 below 10%; expected 13/15 and 15/15"
        )


def write_report(panel_a, panel_b, panel_c, panel_c_dev):
    weak = panel_a[panel_a["pair_id"] == WEAK_ID].iloc[0]
    strong = panel_a[panel_a["pair_id"] == STRONG_ID].iloc[0]
    weak_reps = weak[[f"replica_{i}_B22" for i in range(1, 6)]].dropna().astype(float)
    strong_reps = strong[[f"replica_{i}_B22" for i in range(1, 6)]].dropna().astype(float)
    if len(strong_reps) and len(weak_reps):
        separated = strong_reps.max() < weak_reps.min()
        separation_text = (
            "The strongest and weakest representative pairs remain clearly separated across all individual replicas."
            if separated else
            "The strongest and weakest representative pairs are separated by their five-replica means, but their full replica ranges could not be judged as strictly non-overlapping from the available rows."
        )
    else:
        separation_text = (
            "The strongest and weakest representative pairs are clearly separated by their five-replica means; "
            "strict replica-range separation could not be evaluated because one required representative lacks finite replica rows."
        )

    at80 = panel_c_dev[panel_c_dev["trajectory_fraction"] == 0.8]
    counts_5 = int((at80["relative_deviation_from_100pct"] < 0.05).sum())
    counts_10 = int((at80["relative_deviation_from_100pct"] < 0.10).sum())

    lines = [
        "# B22 reproducibility/convergence report",
        "",
        "Inputs were read from the completed Figure 4 convergence/reproducibility outputs. No simulations were run, no RDFs or B22 values were recalculated from raw trajectories, and no manuscript files were edited.",
        "",
        "## Panel A",
        "",
        "Five fixed representative AHD-RG/RGG pairs were ordered from weakest to strongest five-replica mean B22. The weak FZD9 RG/RGG-ZNF165 AHD and strong SYNCRIP RG/RGG-FOXO6 AHD pairs are retained from original Figure 4B. The three additional fixed representatives are FZD9 RG/RGG-SOX11 AHD, COL7A1 RG/RGG-ZNF165 AHD, and NCL RG/RGG-LEF1 AHD.",
        "",
        panel_a[["pair_label", "pair_id", "category", "mean_B22", "SD", "SE", "selection_reason", "source"]].to_string(index=False),
        "",
        separation_text,
        "",
        "## Panel B",
        "",
        "Exact summary values for absolute deviation from the five-replica estimate:",
        "",
        panel_b.to_string(index=False),
        "",
        "B22 estimates progressively stabilised as additional replicas were included across the training set.",
        "",
        "## Panel C",
        "",
        "Exact 80->100% trajectory-length summary values for the original 15 representative pairs:",
        "",
        panel_c.to_string(index=False),
        "",
        f"Across all 15 pairs, {counts_5}/15 changed by <5% and {counts_10}/15 changed by <10% between 80% and 100% of the trajectory.",
        "",
        "Extending trajectories from 80% to 100% produced only small changes in B22 for representative weak, intermediate and strong pairs.",
        "",
    ]
    (OUT / "B22_REPRODUCIBILITY_CONVERGENCE_REPORT.md").write_text("\n".join(lines))


def write_legend():
    legend = (
        "Supplementary simulation reproducibility and convergence. "
        "(A) Replica-level B22 values for five representative AHD-RG/RGG pairs spanning weak-to-strong mean interaction strengths in the training set. "
        "Dots show individual replicas; black points and error bars show the five-replica mean +/- SD. "
        "(B) Absolute deviation of B22 estimates calculated from all combinations of k=1-5 replicas from the final five-replica mean across the training set. "
        "(C) Relative deviation of cumulative B22 estimates at 20%, 40%, 60%, 80%, and 100% of the trajectory from the final 100% estimate for the original 15 representative pairs (5 weak, 5 intermediate, 5 strong). "
        "All panels use existing completed convergence-analysis outputs."
    )
    (OUT / "B22_REPRODUCIBILITY_CONVERGENCE_LEGEND_DRAFT.md").write_text(legend + "\n")


def save_panel(fig, name):
    fig.savefig(OUT / f"{name}.pdf")
    fig.savefig(OUT / f"{name}.png", dpi=DPI)
    plt.close(fig)


def main():
    tables = load_inputs()
    panel_a = select_panel_a_pairs(tables["rep"], tables["unc"], tables["sel"], tables["prov"])
    panel_b = panel_b_summary(tables["conv_summary"])
    panel_c_dev = panel_c_deviations(tables["time"], tables["panel_c_sel"])
    panel_c = panel_c_summary(panel_c_dev)
    verify_authoritative_values(panel_b, panel_c_dev, panel_c, tables["panel_c_sel"])

    panel_a.to_csv(OUT / "PanelA_selected_pairs_B22.csv", index=False)
    panel_b.to_csv(OUT / "PanelB_replica_number_convergence.csv", index=False)
    panel_c_dev.to_csv(OUT / "PanelC_time_convergence_15pairs.csv", index=False)

    fig = plt.figure(figsize=FIGSIZE, constrained_layout=True)
    gs = fig.add_gridspec(2, 2, height_ratios=[1.25, 1.0])
    ax_a = fig.add_subplot(gs[0, :])
    ax_b = fig.add_subplot(gs[1, 0])
    ax_c = fig.add_subplot(gs[1, 1])
    plot_panel_a(ax_a, panel_a)
    plot_panel_b(ax_b, tables["conv"], panel_b)
    plot_panel_c(ax_c, panel_c_dev)
    style_axes([ax_a, ax_b, ax_c])
    fig.savefig(OUT / "B22_reproducibility_convergence_3panel_corrected.pdf")
    fig.savefig(OUT / "B22_reproducibility_convergence_3panel_corrected.png", dpi=DPI)
    plt.close(fig)

    for name, plotter in [
        ("PanelA_replica_B22_examples", lambda ax: plot_panel_a(ax, panel_a)),
        ("PanelB_replica_number_convergence", lambda ax: plot_panel_b(ax, tables["conv"], panel_b)),
        ("PanelC_trajectory_length_convergence", lambda ax: plot_panel_c(ax, panel_c_dev)),
    ]:
        pfig, pax = plt.subplots(figsize=(4.0, 2.7), constrained_layout=True)
        plotter(pax)
        style_axes([pax])
        save_panel(pfig, name)

    write_report(panel_a, panel_b, panel_c, panel_c_dev)
    write_legend()

    print("Final panel order: A replica-level B22 examples; B replica-number convergence; C trajectory-length convergence.")
    print(f"Panel A example pairs: {', '.join(panel_a['pair_id'])}")
    print(f"Panel B median absolute deviations: {panel_b[['k', 'median_absolute_deviation']].to_dict(orient='records')}")
    print(f"Panel C median 80->100% relative changes: {panel_c[['regime', 'median_relative_change_80_100']].to_dict(orient='records')}")
    print(f"Figure size: {FIGSIZE[0]} x {FIGSIZE[1]} inches")
    print(f"Output directory: {OUT}")
    print(f"PNG dpi: {DPI}")


if __name__ == "__main__":
    main()
