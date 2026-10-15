#!/usr/bin/env python3
"""Build the single Figure 6 parent master from supplied experimental workbooks.

The two Excel inputs are read-only. RRGG7 rows are intentionally excluded.
Raw experimental wording is preserved alongside a conservative normalized class.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
DEFAULT_PROGRESS = Path("/Users/yuktikhanna/Downloads/CellReports_revision_experimental_pairs_progress.xlsx")
DEFAULT_IRATIOS = Path("/Users/yuktikhanna/Downloads/i ratios.xlsx")

AHD_DISPLAY = {
    "foxo4_cr3": ("FOXO4", "461-505", "WT"),
    "foxo6_cr3": ("FOXO6", "418-472", "WT"),
    "lef1_nter": ("LEF1", "1-101", "WT"),
    "tfe3_cter": ("TFE3", "475-575", "WT"),
    "atf4_tad": ("ATF4", "41-141", "WT"),
    "sox12": ("SOX12", "57-72", "WT"),
    "tp53": ("p53", "1-94", "WT"),
    "foxm1": ("FOXM1", "685-801", "WT"),
}
RGG_DISPLAY = {
    "thoc4_nter": "THOC4 N", "thoc4_cter": "THOC4 C", "g3bp1_rgg": "G3BP1",
    "taf15_rgg": "TAF15", "snrnp70_s": "SNRNP70 375-399", "rbmx": "RBMX",
    "tceal6": "TCEAL6 171-198", "q8n319": "Q8N319", "rab44": "RAB44 308-333",
}

PROGRESS_AHD = {
    "SOX12 57–72": "sox12", "ATF4": "atf4_tad", "FOXO4": "foxo4_cr3",
    "FOXO6": "foxo6_cr3", "LEF1": "lef1_nter", "TFE3": "tfe3_cter",
    "FOXO4 shuffle 461–505": "foxo4_cr3_shuffle_2",
    "FOXO6 shuffle 418–472": "foxo6_cr3_shuffle_1",
    "ATF4 shuffle 41–141": "atf4_tad_shuffle_1",
    "TFE3 shuffle 475–575": "tfe3_cter_shuffle_2",
    "p53 shuffle": "tp53_shuffle_1", "FOXM1 shuffle": "foxm1_shuffle_1",
    "LEF1 shuffle 1–101": "lef1_nter_shuffle_1",
    "p53 WT 1–94": "tp53", "FOXM1 WT 685–801": "foxm1",
}
PROGRESS_RGG = {
    "TAF15": "taf15_rgg", "THOC4 N": "thoc4_nter", "THOC4 C": "thoc4_cter",
    "G3BP1": "g3bp1_rgg", "RAB44 308–333": "rab44", "TCEAL6 171–198": "tceal6",
    "SNRNP70 375–399": "snrnp70_s", "RBMX": "rbmx",
}
IRATIO_PAIR = {
    "TFE3 RAB44 308–333": "rab44__tfe3_cter",
    "LEF1 RAB44 308–333": "rab44__lef1_nter",
    "FOXO6 shuffle 418–472 THOC4 C": "thoc4_cter__foxo6_cr3_shuffle_1",
    "TFE3 shuffle 475–575 TAF15": "taf15_rgg__tfe3_cter_shuffle_2",
    # The source sheet omits the final 5 in 475–575; the construct identity is otherwise exact.
    "TFE3 shuffle 475–57 THOC4 N": "thoc4_nter__tfe3_cter_shuffle_2",
    "ATF4 TCEAL6 171–198": "tceal6__atf4_tad",
    "ATF4 SNRNP70 375–399": "snrnp70_s__atf4_tad",
    "ATF4 RBMX": "rbmx__atf4_tad",
    "FOXO4 WT 461–508 TAF15": "taf15_rgg__foxo4_cr3",
    "FOXO6 WT 418–472 TAF15": "taf15_rgg__foxo6_cr3",
}


def clean(value):
    return "" if pd.isna(value) else str(value).strip()


def normalize_outcome(raw: str, kd: float | None) -> str:
    text = clean(raw)
    low = text.lower()
    if pd.notna(kd) and float(kd) > 0:
        return "Kd_fit"
    if "will follow" in low:
        return "Kd_pending"
    if low == "nb" or low.startswith("nb, minor"):
        return "NB"
    if low == "lb" or low.startswith("lb,") or low.startswith("lb+"):
        return "LB"
    if "weak bind" in low:
        return "Binding_qualitative"
    if any(token in low for token in ("nb?", "nb/", "weak shifts", "minor shifts")):
        return "Ambiguous"
    if low in ("kd_fit", "kd fit"):
        return "Kd_fit"
    return "Unassigned"


def canonical_metadata(pred: pd.DataFrame) -> pd.DataFrame:
    out = pred[["pair_id", "AHD_name", "RGG_name", "final_hybrid_prediction"]].copy()
    out = out.rename(columns={"final_hybrid_prediction": "Pred_log10_negB22"})
    def ahd_fields(name):
        parent = str(name).split("_shuffle_")[0]
        gene, residues, _ = AHD_DISPLAY[parent]
        variant = "WT" if "_shuffle_" not in str(name) else "shuffle_" + str(name).split("_shuffle_")[1]
        return gene, residues, variant
    vals = out.AHD_name.map(ahd_fields)
    out["AHD_gene"] = vals.map(lambda x: x[0])
    out["AHD_residues"] = vals.map(lambda x: x[1])
    out["AHD_variant"] = vals.map(lambda x: x[2])
    out["RGG_partner"] = out.RGG_name.map(RGG_DISPLAY)
    out["pair_display_name"] = out.apply(
        lambda z: f"{z.AHD_gene} {z.AHD_residues} ({z.AHD_variant}) + {z.RGG_partner}", axis=1)
    return out


def read_progress(path: Path) -> pd.DataFrame:
    raw = pd.read_excel(path, sheet_name="Experimental pairs_revision", header=None)
    raw = raw.iloc[19:61, :12].copy()  # rows 20–61; rows 2–19 are the excluded RRGG7 block
    raw.columns = ["experiment_group", "AHD_display_raw", "RGG_display_raw", "Purpose_note",
                   "Outcome_raw", "Notes", "source_predicted_B22", "source_simulated_B22",
                   "Kd_uM", "Kd_uM_err", "koff_fit", "koff_err"]
    raw["AHD_display_raw"] = raw.AHD_display_raw.map(clean)
    raw["RGG_display_raw"] = raw.RGG_display_raw.map(clean)
    unknown_a = sorted(set(raw.AHD_display_raw) - set(PROGRESS_AHD))
    unknown_r = sorted(set(raw.RGG_display_raw) - set(PROGRESS_RGG))
    if unknown_a or unknown_r:
        raise ValueError(f"Unmapped progress names: AHD={unknown_a}, RGG={unknown_r}")
    raw["AHD_name"] = raw.AHD_display_raw.map(PROGRESS_AHD)
    raw["RGG_name"] = raw.RGG_display_raw.map(PROGRESS_RGG)
    raw["pair_id"] = raw.RGG_name + "__" + raw.AHD_name
    if raw.pair_id.duplicated().any():
        raise ValueError("Duplicate pair in experimental progress sheet")
    raw["Outcome"] = [normalize_outcome(o, k) for o, k in zip(raw.Outcome_raw, raw.Kd_uM)]
    raw["experiment_status"] = raw.Outcome.map({
        "Kd_fit": "fitted Kd supplied", "Kd_pending": "titration pending", "NB": "explicit NB",
        "LB": "line broadening without fitted Kd", "Binding_qualitative": "qualitative binding",
        "Ambiguous": "ambiguous qualitative result", "Unassigned": "not yet assigned",
    })
    raw["experimental_source"] = path.name
    raw["experimental_source_location"] = "Experimental pairs_revision rows 20-61"
    return raw


def read_iratio(path: Path) -> pd.DataFrame:
    raw = pd.read_excel(path, sheet_name="new", header=0)
    raw = raw.iloc[:, :5].copy()
    raw.columns = ["intensity_pair_name_raw", "I_ratio_threshold", "I_ratio_mean_below_threshold",
                   "source_predicted_B22_iratio", "source_simulated_B22_iratio"]
    raw["intensity_pair_name_raw"] = raw.intensity_pair_name_raw.map(clean)
    raw = raw[raw.intensity_pair_name_raw.isin(IRATIO_PAIR)].copy()
    raw["pair_id"] = raw.intensity_pair_name_raw.map(IRATIO_PAIR)
    if len(raw) != 10 or raw.pair_id.duplicated().any():
        raise ValueError("Expected exactly 10 unique Figure 6 intensity-ratio rows from sheet 'new'")
    raw["I_ratio_source"] = path.name
    raw["I_ratio_source_location"] = "new"
    return raw


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--progress", type=Path, default=DEFAULT_PROGRESS)
    ap.add_argument("--intensity-ratios", type=Path, default=DEFAULT_IRATIOS)
    ap.add_argument("--output", type=Path, default=ROOT / "fig6_parent_master.csv")
    args = ap.parse_args()
    output = args.output.resolve()
    if not output.is_relative_to(ROOT):
        raise ValueError("Output must stay inside fig6")

    pred = pd.read_csv(ROOT / "preds_val_updated/predictions_minimal.csv")
    master = canonical_metadata(pred)
    sims = pd.read_csv(ROOT / "updated_validation/validation_master.csv")[["pair_id", "Sim_log10_negB22_final"]]
    sims = sims.rename(columns={"Sim_log10_negB22_final": "Sim_log10_negB22"})
    master = master.merge(sims, on="pair_id", how="left", validate="one_to_one")

    old = pd.read_csv(ROOT / "Table_S8.csv")
    old = old[["pair_id", "Outcome", "Notes", "Kd_uM", "Kd_uM_err", "pKd_PPT", "pKd_err_PPT"]].copy()
    old = old.rename(columns={"Outcome": "Outcome_raw", "pKd_PPT": "pKd", "pKd_err_PPT": "pKd_error"})
    old["Outcome"] = [normalize_outcome(o, k) for o, k in zip(old.Outcome_raw, old.Kd_uM)]
    old["experiment_status"] = old.Outcome.map({"Kd_fit": "fitted Kd supplied", "NB": "explicit NB", "LB": "line broadening without fitted Kd"})
    old["experimental_source"] = "Table_S8.csv"
    old["experimental_source_location"] = "legacy Figure 6 experimental source"

    progress = read_progress(args.progress)
    overlap = set(old.pair_id) & set(progress.pair_id)
    if overlap:
        raise ValueError(f"Unexpected duplicate experimental pairs across old/new sources: {sorted(overlap)}")
    exp_cols = ["pair_id", "Outcome", "Outcome_raw", "Notes", "Kd_uM", "Kd_uM_err", "koff_fit", "koff_err",
                "pKd", "pKd_error", "experiment_group", "Purpose_note", "experiment_status",
                "experimental_source", "experimental_source_location", "AHD_display_raw", "RGG_display_raw",
                "source_predicted_B22", "source_simulated_B22"]
    experiments = pd.concat([old.reindex(columns=exp_cols), progress.reindex(columns=exp_cols)], ignore_index=True)
    master = master.merge(experiments, on="pair_id", how="left", validate="one_to_one")
    # The parent master contains every pair with supplied experimental information.
    # Computational-only pairs remain in the 73-pair prediction inventory.
    master = master[master.pair_id.isin(experiments.pair_id)].copy()
    for column in ["Kd_uM", "Kd_uM_err", "pKd", "pKd_error", "koff_fit", "koff_err"]:
        master[column] = pd.to_numeric(master[column], errors="coerce")
    master["pKd"] = master.pKd.where(master.pKd.notna(), 6 - np.log10(master.Kd_uM.astype(float)))
    master["pKd_error"] = master.pKd_error.where(
        master.pKd_error.notna(), master.Kd_uM_err / (master.Kd_uM * np.log(10)))

    iratio = read_iratio(args.intensity_ratios)
    master = master.merge(iratio, on="pair_id", how="left", validate="one_to_one")
    master["I_ratio_threshold_rank_ascending"] = master.I_ratio_threshold.rank(method="min", ascending=True)
    master["I_ratio_mean_below_threshold_rank_ascending"] = master.I_ratio_mean_below_threshold.rank(method="min", ascending=True)
    master["I_ratio_rank_direction"] = np.where(master.I_ratio_threshold.notna(), "1 = lowest intensity ratio", pd.NA)

    # Flag disagreements rather than replacing canonical local prediction/simulation values.
    for column in ["source_predicted_B22", "source_simulated_B22"]:
        master[column] = pd.to_numeric(master[column], errors="coerce")
    master["source_prediction_matches_canonical"] = np.where(
        master.source_predicted_B22.notna(), np.isclose(master.source_predicted_B22, master.Pred_log10_negB22, atol=1e-10), pd.NA)
    master["source_simulation_matches_canonical"] = np.where(
        master.source_simulated_B22.notna(), np.isclose(master.source_simulated_B22, master.Sim_log10_negB22, atol=1e-10), pd.NA)

    order = ["pair_id", "pair_display_name", "AHD_gene", "AHD_residues", "AHD_variant", "AHD_name", "RGG_partner", "RGG_name",
             "Outcome", "Outcome_raw", "Kd_uM", "Kd_uM_err", "pKd", "pKd_error", "koff_fit", "koff_err",
             "I_ratio_threshold", "I_ratio_mean_below_threshold", "I_ratio_threshold_rank_ascending",
             "I_ratio_mean_below_threshold_rank_ascending", "I_ratio_rank_direction",
             "Pred_log10_negB22", "Sim_log10_negB22", "experiment_status", "experiment_group", "Purpose_note", "Notes",
             "experimental_source", "experimental_source_location", "I_ratio_source", "I_ratio_source_location",
             "intensity_pair_name_raw", "AHD_display_raw", "RGG_display_raw", "source_predicted_B22", "source_simulated_B22",
             "source_predicted_B22_iratio", "source_simulated_B22_iratio", "source_prediction_matches_canonical",
             "source_simulation_matches_canonical"]
    master = master.reindex(columns=order)
    if len(master) != 62 or master.pair_id.duplicated().any():
        raise ValueError("Parent master must contain the 62 unique experimentally tracked Figure 6 pairs")
    output.parent.mkdir(parents=True, exist_ok=True)
    master.to_csv(output, index=False, na_rep="")
    print(f"Wrote {output} ({len(master)} rows; {(master.Outcome != 'Unassigned').sum()} annotated; {master.I_ratio_threshold.notna().sum()} intensity ratios)")


if __name__ == "__main__":
    main()
