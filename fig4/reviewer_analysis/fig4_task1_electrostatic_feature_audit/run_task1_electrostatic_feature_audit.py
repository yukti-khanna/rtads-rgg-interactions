#!/usr/bin/env python3
"""Audit Figure 4 electrostatic features without importing feature-generation code."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

TOL = 1e-12
EXPECTED_N = 917


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fig4-dir", required=True, type=Path,
                        help="Extracted Figure 4 input directory")
    parser.add_argument("--outdir", required=True, type=Path,
                        help="New audit output directory")
    return parser.parse_args()


def require_columns(frame: pd.DataFrame, columns: list[str], label: str) -> None:
    missing = sorted(set(columns) - set(frame.columns))
    if missing:
        raise ValueError(f"{label} is missing required columns: {missing}")


def correlation(x: pd.Series, y: pd.Series, method: str) -> tuple[float, float]:
    if method == "pearson":
        result = stats.pearsonr(x, y)
    else:
        result = stats.spearmanr(x, y)
    return float(result.statistic), float(result.pvalue)


def savefig_both(fig: plt.Figure, outdir: Path, stem: str) -> None:
    for suffix in ("png", "pdf"):
        fig.savefig(outdir / f"{stem}.{suffix}", dpi=300, bbox_inches="tight")
    plt.close(fig)


def scatter(x, y, xlabel, ylabel, outdir, stem, identity=False, annotation=None):
    fig, ax = plt.subplots(figsize=(5.2, 4.3))
    ax.scatter(x, y, s=13, alpha=0.55, edgecolors="none")
    if identity:
        lo = float(min(np.min(x), np.min(y)))
        hi = float(max(np.max(x), np.max(y)))
        ax.plot([lo, hi], [lo, hi], color="black", lw=1, ls="--", label="identity")
        ax.legend(frameon=False)
    ax.set(xlabel=xlabel, ylabel=ylabel)
    if annotation:
        ax.text(0.03, 0.97, annotation, transform=ax.transAxes, va="top")
    fig.tight_layout()
    savefig_both(fig, outdir, stem)


def write_report(outdir: Path, fig4_dir: Path, values: dict[str, object]) -> None:
    normalized_status = values["normalized_status"]
    report = f"""# Figure 4 electrostatic-feature audit (Reviewer 1, R1-11/R1-12)

## 1. Question

This audit tests whether the Figure 4 implementation's `Opposite_Charge_Number` (OCN) is just the negative product of the two segment net charges, whether normalized OCN is only length-normalized OCN, whether `Charge_Complementarity` is mathematically distinct, and whether the reported OCN–interaction-strength relationship is reproducible in the exact Figure 4D population.

## 2. Inputs

- `{fig4_dir / 'get_features_all.py'}` (definition audit only)
- `{fig4_dir / 'sim_pairs_added_features_data_270525.csv'}` (master pair-level features)
- `{fig4_dir / 'out_fig4/cache/Fig4d_scatter_table.csv'}` (authoritative Figure 4D pair list, OCN, and `B22_corr1_log10`)
- `{fig4_dir / 'residues.csv'}` (inspected as the residue-charge source; independent pair-level checks use stored charges and lengths)

The master table has {values['master_rows']} rows and {values['master_unique']} unique names. The Figure 4D cache has {values['fig_rows']} rows and {values['fig_unique']} unique names. All 917 cached names mapped uniquely and completely: **{values['mapped']}**.

## 3. Implemented definitions

The executable code in `get_features_all.py` defines:

- Total RGG and TAD charges as sums of residue-property `q` values (lines 270–276): \\(Q_{{RGG}}=\\sum_i q_i\\), \\(Q_{{TAD}}=\\sum_j q_j\\). In `residues.csv`, R/K have +1, D/E have -1, and the other listed residues have 0.
- Segment NCPR (lines 283–288): \\(NCPR_{{RGG}}=Q_{{RGG}}/L_{{RGG}}\\), \\(NCPR_{{TAD}}=Q_{{TAD}}/L_{{TAD}}\\), and `Average_NCPR` is their arithmetic mean.
- OCN in `opp_charges` (lines 190–214): \\((p_R n_T+n_R p_T)-(p_R p_T+n_R n_T)\\), where \\(p\\) and \\(n\\) are the magnitudes of positive and negative charge counts.
- Charge complementarity (lines 154–157): \\(C=|Q_R+Q_T|-\\tfrac12(|Q_R|+|Q_T|)\\).
- `Total_Length` (line 412): \\(L_R+L_T\\).
- `Normalized_OCN` (line 417): \\(OCN/(L_R+L_T)\\).

## 4. Algebraic result

With \\(Q_R=p_R-n_R\\) and \\(Q_T=p_T-n_T\\),

\\[
-Q_RQ_T=-(p_R-n_R)(p_T-n_T)
=-p_Rp_T+p_Rn_T+n_Rp_T-n_Rn_T
=(p_Rn_T+n_Rp_T)-(p_Rp_T+n_Rn_T)=OCN.
\\]

Thus the implemented OCN is **exactly algebraically equivalent** to \\(-Q_{{RGG}}Q_{{TAD}}\\); this conclusion does not depend on an observed correlation.

## 5. Numerical verification

For the exact {values['fig_rows']} Figure 4D pairs, the maximum absolute `OCN - (-Q_RGG*Q_TAD)` difference was **{values['ocn_max']:.17g}**, the mean absolute difference was **{values['ocn_mean']:.17g}**, and **{values['ocn_nonzero']}** rows differed at `np.isclose(atol=1e-12, rtol=1e-12)`. The Pearson and Spearman coefficients between the two columns were both **{values['ocn_ncp_pearson']:.17g}** and **{values['ocn_ncp_spearman']:.17g}**, respectively.

Normalized OCN status: **{normalized_status}** Its correlation with OCN was Pearson **{values['ocn_norm_pearson']:.6g}** and Spearman **{values['ocn_norm_spearman']:.6g}**. This is normalization-derived dependence, not algebraic identity unless total length is constant.

## 6. Charge-complementarity comparison

The independent formula reproduced stored `Charge_Complementarity`: maximum absolute error **{values['cc_max']:.17g}**, mean absolute error **{values['cc_mean']:.17g}**, and **{values['cc_nonzero']}** non-equal rows at the stated tolerance.

Charge complementarity is a piecewise-linear function of the two net charges; OCN is their negative product. They are therefore mathematically distinct. In these 917 pairs, OCN versus charge complementarity had Pearson r = **{values['ocn_cc_pearson']:.6g}** (p = {values['ocn_cc_pearson_p']:.6g}) and Spearman rho = **{values['ocn_cc_spearman']:.6g}** (p = {values['ocn_cc_spearman_p']:.6g}). Normalized OCN versus charge complementarity had Pearson r = **{values['norm_cc_pearson']:.6g}** and Spearman rho = **{values['norm_cc_spearman']:.6g}**.

The full Pearson and Spearman matrices distinguish ordinary statistical correlations from the exact OCN/net-charge-product identity. OCN correlations with TAD charge were Pearson **{values['ocn_tad_pearson']:.6g}** and Spearman **{values['ocn_tad_spearman']:.6g}**; with RGG charge they were Pearson **{values['ocn_rgg_pearson']:.6g}** and Spearman **{values['ocn_rgg_spearman']:.6g}**.

## 7. Figure 4D reproduction

Without outlier removal, OCN versus `B22_corr1_log10` for n = **{values['fig_rows']}** gave Pearson r = **{values['b22_pearson']:.9g}** (p = {values['b22_pearson_p']:.9g}), Spearman rho = **{values['b22_spearman']:.9g}** (p = {values['b22_spearman_p']:.9g}), slope = **{values['slope']:.9g}**, intercept = **{values['intercept']:.9g}**, and R² = **{values['r2']:.9g}**. Replacing OCN with \\(-Q_RQ_T\\) produced numerically identical results: **{values['relationship_identical']}**.

## 8. Reviewer-relevant conclusion

The implemented OCN is the negative product of the segment net charges. Normalized OCN is the same value divided by total sequence length. Charge complementarity is a distinct, related net-charge descriptor, not OCN under another name. None of these three definitions uses residue positions or charge ordering; accordingly, none is a true charge-patterning or residue-pairing metric. The observed correlations describe this dataset and should not be mistaken for additional algebraic identities.

## 9. Implications for subsequent analysis

Electrostatic descriptors shown here to be exactly redundant or deterministically derived should not later be interpreted as independent mechanistic discoveries. Model-feature ablation or retraining belongs to the later Figure 5 task and was not performed here.
"""
    (outdir / "TASK1_REPORT.md").write_text(report, encoding="utf-8")


def main() -> None:
    args = parse_args()
    fig4_dir = args.fig4_dir.resolve()
    outdir = args.outdir.resolve()
    master_path = fig4_dir / "sim_pairs_added_features_data_270525.csv"
    cache_path = fig4_dir / "out_fig4/cache/Fig4d_scatter_table.csv"
    source_path = fig4_dir / "get_features_all.py"
    residues_path = fig4_dir / "residues.csv"
    for path in (master_path, cache_path, source_path, residues_path):
        if not path.is_file():
            raise FileNotFoundError(f"Required input not found: {path}")
    if outdir.exists() and any(outdir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty output directory: {outdir}")
    outdir.mkdir(parents=True, exist_ok=True)

    master = pd.read_csv(master_path)
    cache = pd.read_csv(cache_path)
    master_required = ["name", "charge_RGG", "charge_TAD", "RGG_length", "TAD_length",
                       "Opposite_Charge_Number", "Charge_Complementarity", "NCPR_RGG",
                       "NCPR_TAD", "Average_NCPR", "Total_Length", "Average_charge"]
    cache_required = ["name", "Opposite_Charge_Number", "B22_corr1_log10"]
    require_columns(master, master_required, "master feature table")
    require_columns(cache, cache_required, "Figure 4D cache")

    master_dup_names = master.loc[master["name"].duplicated(False), "name"]
    cache_dup_names = cache.loc[cache["name"].duplicated(False), "name"]
    missing_names = sorted(set(cache["name"]) - set(master["name"]))
    fig_master_dups = sorted(set(cache["name"]) & set(master_dup_names))
    if len(cache) != EXPECTED_N:
        raise ValueError(f"Figure 4D cache has {len(cache)} rows; expected {EXPECTED_N}")
    if len(cache_dup_names) or missing_names or fig_master_dups:
        raise ValueError("Figure 4 mapping is not unique and complete: "
                         f"cache duplicates={cache_dup_names.nunique()}, missing={len(missing_names)}, "
                         f"master duplicates for cached names={len(fig_master_dups)}")

    selected = cache.rename(columns={"Opposite_Charge_Number": "OCN_cache"}).merge(
        master[master_required], on="name", how="left", validate="one_to_one")
    selected = selected.rename(columns={"Opposite_Charge_Number": "OCN_stored",
                                        "Charge_Complementarity": "Charge_Complementarity_stored"})
    selected["Net_Charge_Product"] = -(selected["charge_RGG"] * selected["charge_TAD"])
    selected["OCN_minus_NetChargeProduct"] = selected["OCN_stored"] - selected["Net_Charge_Product"]
    selected["Total_Length_recomputed"] = selected["RGG_length"] + selected["TAD_length"]
    selected["Normalized_OCN_recomputed"] = selected["OCN_stored"] / selected["Total_Length_recomputed"]
    selected["Charge_Complementarity_recomputed"] = (
        (selected["charge_RGG"] + selected["charge_TAD"]).abs()
        - 0.5 * (selected["charge_RGG"].abs() + selected["charge_TAD"].abs()))
    selected["CC_difference"] = (selected["Charge_Complementarity_stored"]
                                  - selected["Charge_Complementarity_recomputed"])
    selected["OCN_cache_minus_master"] = selected["OCN_cache"] - selected["OCN_stored"]

    normalized_candidates = []
    normalized_differences = []
    for csv_path in sorted(fig4_dir.rglob("*.csv")):
        try:
            candidate = pd.read_csv(csv_path)
        except Exception:
            continue
        if "Normalized_OCN" in candidate.columns:
            require_columns(candidate, ["Opposite_Charge_Number", "Total_Length"], str(csv_path))
            normalized_candidates.append(str(csv_path))
            normalized_differences.extend(
                (candidate["Normalized_OCN"]
                 - candidate["Opposite_Charge_Number"] / candidate["Total_Length"]).tolist())
    if normalized_candidates:
        normalized_differences = np.asarray(normalized_differences, dtype=float)
        normalized_nonzero = int((~np.isclose(normalized_differences, 0, atol=TOL, rtol=TOL)).sum())
        normalized_status = (
            f"A stored column was found in {len(normalized_candidates)} CSV files ({len(normalized_differences)} "
            f"row occurrences, including duplicated supplementary copies) and compared with OCN/Total_Length; "
            f"max absolute difference={np.max(np.abs(normalized_differences)):.17g}, "
            f"mean absolute difference={np.mean(np.abs(normalized_differences)):.17g}, "
            f"non-equal count={normalized_nonzero} at the stated tolerance. The value was independently "
            "reconstructed for all 917 primary pairs because the 1,039-row master table lacks the column.")
    else:
        normalized_nonzero = 0
        normalized_differences = np.asarray([], dtype=float)
        normalized_status = ("No stored `Normalized_OCN` column was found in supplied CSV data; it was reconstructed "
                             "from the implemented formula (the source code does implement the column).")

    analysis_cols = ["OCN_stored", "Net_Charge_Product", "Normalized_OCN_recomputed",
                     "Charge_Complementarity_stored", "charge_RGG", "charge_TAD", "NCPR_RGG",
                     "NCPR_TAD", "Average_NCPR", "Total_Length_recomputed", "Average_charge"]
    used_cols = analysis_cols + ["B22_corr1_log10", "RGG_length", "TAD_length"]
    missing_counts = selected[used_cols].isna().sum()
    nonfinite_counts = (~np.isfinite(selected[used_cols].astype(float))).sum()
    if int(missing_counts.sum()) or int(nonfinite_counts.sum()):
        raise ValueError(f"Missing/non-finite values in analysis columns: missing={missing_counts.to_dict()}, "
                         f"nonfinite={nonfinite_counts.to_dict()}")

    pretty = {"OCN_stored": "Opposite_Charge_Number",
              "Charge_Complementarity_stored": "Charge_Complementarity",
              "Total_Length_recomputed": "Total_Length"}
    corr_data = selected[analysis_cols].rename(columns=pretty)
    pearson = corr_data.corr(method="pearson")
    spearman = corr_data.corr(method="spearman")
    pearson.to_csv(outdir / "pearson_electrostatic_correlations.csv", float_format="%.17g")
    spearman.to_csv(outdir / "spearman_electrostatic_correlations.csv", float_format="%.17g")

    output_cols = ["name", "charge_RGG", "charge_TAD", "RGG_length", "TAD_length",
                   "Total_Length_recomputed", "OCN_stored", "Net_Charge_Product",
                   "OCN_minus_NetChargeProduct", "Charge_Complementarity_stored",
                   "Charge_Complementarity_recomputed", "CC_difference",
                   "Normalized_OCN_recomputed", "B22_corr1_log10"]
    audit = selected[output_cols].rename(columns={"Total_Length_recomputed": "Total_Length",
                                                  "OCN_stored": "Opposite_Charge_Number_stored"})
    audit.to_csv(outdir / "fig4_917_electrostatic_audit.csv", index=False, float_format="%.17g")

    integrity_rows = [
        ("master_rows", len(master)), ("master_unique_names", master["name"].nunique()),
        ("master_duplicate_rows_by_name", int(master["name"].duplicated(False).sum())),
        ("master_duplicate_name_count", master_dup_names.nunique()),
        ("figure4_cache_rows", len(cache)), ("expected_figure4_rows", EXPECTED_N),
        ("figure4_cache_unique_names", cache["name"].nunique()),
        ("figure4_cache_duplicate_rows_by_name", int(cache["name"].duplicated(False).sum())),
        ("figure4_names_missing_from_master", len(missing_names)),
        ("master_duplicate_names_for_figure4_names", len(fig_master_dups)),
        ("all_917_mapped_uniquely_and_completely", True),
        ("missing_values_in_used_columns", int(missing_counts.sum())),
        ("nonfinite_values_in_used_columns", int(nonfinite_counts.sum())),
        ("cache_vs_master_ocn_max_abs_difference", selected["OCN_cache_minus_master"].abs().max()),
    ]
    pd.DataFrame(integrity_rows, columns=["check", "value"]).to_csv(
        outdir / "dataset_integrity_summary.csv", index=False)

    ocn_equal = np.isclose(selected["OCN_stored"], selected["Net_Charge_Product"], atol=TOL, rtol=TOL)
    cc_equal = np.isclose(selected["Charge_Complementarity_stored"], selected["Charge_Complementarity_recomputed"], atol=TOL, rtol=TOL)
    pr_ocn_cc = correlation(selected["OCN_stored"], selected["Charge_Complementarity_stored"], "pearson")
    sp_ocn_cc = correlation(selected["OCN_stored"], selected["Charge_Complementarity_stored"], "spearman")
    pr_norm_cc = correlation(selected["Normalized_OCN_recomputed"], selected["Charge_Complementarity_stored"], "pearson")
    sp_norm_cc = correlation(selected["Normalized_OCN_recomputed"], selected["Charge_Complementarity_stored"], "spearman")
    pr_b22 = correlation(selected["OCN_stored"], selected["B22_corr1_log10"], "pearson")
    sp_b22 = correlation(selected["OCN_stored"], selected["B22_corr1_log10"], "spearman")
    regression = stats.linregress(selected["OCN_stored"], selected["B22_corr1_log10"])
    regression_ncp = stats.linregress(selected["Net_Charge_Product"], selected["B22_corr1_log10"])
    relationship_identical = bool(np.allclose(
        [regression.slope, regression.intercept, regression.rvalue, regression.pvalue],
        [regression_ncp.slope, regression_ncp.intercept, regression_ncp.rvalue, regression_ncp.pvalue],
        atol=TOL, rtol=TOL))

    def cpair(a, b):
        return correlation(selected[a], selected[b], "pearson")[0], correlation(selected[a], selected[b], "spearman")[0]
    ocn_ncp = cpair("OCN_stored", "Net_Charge_Product")
    ocn_norm = cpair("OCN_stored", "Normalized_OCN_recomputed")
    ocn_tad = cpair("OCN_stored", "charge_TAD")
    ocn_rgg = cpair("OCN_stored", "charge_RGG")
    key_rows = [
        ("primary_n", len(selected)),
        ("ocn_net_charge_product_max_abs_difference", selected["OCN_minus_NetChargeProduct"].abs().max()),
        ("ocn_net_charge_product_mean_abs_difference", selected["OCN_minus_NetChargeProduct"].abs().mean()),
        ("ocn_net_charge_product_non_equal_count", int((~ocn_equal).sum())),
        ("charge_complementarity_max_abs_difference", selected["CC_difference"].abs().max()),
        ("charge_complementarity_mean_abs_difference", selected["CC_difference"].abs().mean()),
        ("charge_complementarity_non_equal_count", int((~cc_equal).sum())),
        ("stored_normalized_ocn_row_occurrences_compared", len(normalized_differences)),
        ("stored_normalized_ocn_max_abs_difference", np.max(np.abs(normalized_differences)) if len(normalized_differences) else np.nan),
        ("stored_normalized_ocn_mean_abs_difference", np.mean(np.abs(normalized_differences)) if len(normalized_differences) else np.nan),
        ("stored_normalized_ocn_non_equal_count", normalized_nonzero),
        ("ocn_vs_net_charge_product_pearson_r", ocn_ncp[0]),
        ("ocn_vs_net_charge_product_spearman_rho", ocn_ncp[1]),
        ("ocn_vs_normalized_ocn_pearson_r", ocn_norm[0]),
        ("ocn_vs_normalized_ocn_spearman_rho", ocn_norm[1]),
        ("ocn_vs_charge_complementarity_pearson_r", pr_ocn_cc[0]),
        ("ocn_vs_charge_complementarity_pearson_p", pr_ocn_cc[1]),
        ("ocn_vs_charge_complementarity_spearman_rho", sp_ocn_cc[0]),
        ("ocn_vs_charge_complementarity_spearman_p", sp_ocn_cc[1]),
        ("normalized_ocn_vs_charge_complementarity_pearson_r", pr_norm_cc[0]),
        ("normalized_ocn_vs_charge_complementarity_spearman_rho", sp_norm_cc[0]),
        ("ocn_vs_tad_charge_pearson_r", ocn_tad[0]),
        ("ocn_vs_tad_charge_spearman_rho", ocn_tad[1]),
        ("ocn_vs_rgg_charge_pearson_r", ocn_rgg[0]),
        ("ocn_vs_rgg_charge_spearman_rho", ocn_rgg[1]),
        ("ocn_vs_b22_pearson_r", pr_b22[0]), ("ocn_vs_b22_pearson_p", pr_b22[1]),
        ("ocn_vs_b22_spearman_rho", sp_b22[0]), ("ocn_vs_b22_spearman_p", sp_b22[1]),
        ("ocn_vs_b22_slope", regression.slope), ("ocn_vs_b22_intercept", regression.intercept),
        ("ocn_vs_b22_r_squared", regression.rvalue ** 2),
        ("net_charge_product_b22_relationship_identical", relationship_identical),
    ]
    pd.DataFrame(key_rows, columns=["statistic", "value"]).to_csv(
        outdir / "key_statistics.csv", index=False, float_format="%.17g")

    scatter(selected["Net_Charge_Product"], selected["OCN_stored"],
            r"$-Q_{RGG}Q_{TAD}$", "Stored OCN", outdir, "ocn_vs_net_charge_product", identity=True)
    scatter(selected["OCN_stored"], selected["Charge_Complementarity_stored"],
            "Opposite Charge Number", "Charge Complementarity", outdir,
            "ocn_vs_charge_complementarity",
            annotation=f"n={len(selected)}\nPearson r={pr_ocn_cc[0]:.3f}\nSpearman ρ={sp_ocn_cc[0]:.3f}")
    scatter(selected["OCN_stored"], selected["B22_corr1_log10"],
            "Opposite Charge Number", "B22_corr1_log10", outdir, "ocn_vs_B22_fig4_917",
            annotation=f"n={len(selected)}\nPearson r={pr_b22[0]:.3f}\nSpearman ρ={sp_b22[0]:.3f}")

    fig, ax = plt.subplots(figsize=(9.2, 7.8))
    im = ax.imshow(pearson.to_numpy(), vmin=-1, vmax=1, cmap="coolwarm")
    ax.set_xticks(range(len(pearson)), pearson.columns, rotation=55, ha="right")
    ax.set_yticks(range(len(pearson)), pearson.index)
    fig.colorbar(im, ax=ax, label="Pearson r", shrink=0.8)
    fig.tight_layout()
    savefig_both(fig, outdir, "electrostatic_descriptor_correlation_heatmap")

    values = dict(master_rows=len(master), master_unique=master["name"].nunique(),
                  fig_rows=len(cache), fig_unique=cache["name"].nunique(), mapped="yes",
                  ocn_max=selected["OCN_minus_NetChargeProduct"].abs().max(),
                  ocn_mean=selected["OCN_minus_NetChargeProduct"].abs().mean(),
                  ocn_nonzero=int((~ocn_equal).sum()), cc_max=selected["CC_difference"].abs().max(),
                  cc_mean=selected["CC_difference"].abs().mean(), cc_nonzero=int((~cc_equal).sum()),
                  normalized_status=normalized_status, ocn_ncp_pearson=ocn_ncp[0],
                  ocn_ncp_spearman=ocn_ncp[1], ocn_norm_pearson=ocn_norm[0],
                  ocn_norm_spearman=ocn_norm[1], ocn_cc_pearson=pr_ocn_cc[0],
                  ocn_cc_pearson_p=pr_ocn_cc[1], ocn_cc_spearman=sp_ocn_cc[0],
                  ocn_cc_spearman_p=sp_ocn_cc[1], norm_cc_pearson=pr_norm_cc[0],
                  norm_cc_spearman=sp_norm_cc[0], ocn_tad_pearson=ocn_tad[0],
                  ocn_tad_spearman=ocn_tad[1], ocn_rgg_pearson=ocn_rgg[0],
                  ocn_rgg_spearman=ocn_rgg[1], b22_pearson=pr_b22[0],
                  b22_pearson_p=pr_b22[1], b22_spearman=sp_b22[0],
                  b22_spearman_p=sp_b22[1], slope=regression.slope,
                  intercept=regression.intercept, r2=regression.rvalue ** 2,
                  relationship_identical="yes" if relationship_identical else "no")
    write_report(outdir, fig4_dir, values)

    print(f"Primary n: {len(selected)}")
    print("All 917 pairs mapped successfully: yes")
    print(f"OCN == -Q_RGG * Q_TAD: {'yes' if ocn_equal.all() else 'no'}")
    print(f"Maximum numerical discrepancy: {selected['OCN_minus_NetChargeProduct'].abs().max():.17g}")
    print(f"Normalized OCN: {normalized_status}")
    print(f"OCN vs Charge_Complementarity: Pearson r={pr_ocn_cc[0]:.9g}; Spearman rho={sp_ocn_cc[0]:.9g}")
    print(f"OCN vs B22: Pearson r={pr_b22[0]:.9g}; Spearman rho={sp_b22[0]:.9g}")
    print(f"Output directory: {outdir}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise
