#!/usr/bin/env python3
"""Prepare new Figure S8 control B22/feature inputs for cluster prediction.

This script is intended to run from the mirrored fig5 directory on the cluster.
It does not load or run the final hybrid model. It computes B22 from existing RDF
pickles using the existing Figure 5/Figure 4 RDF function, generates the final
13 features using the validated historical patch feature implementation, and
writes model-ready input CSVs for prediction.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

FIG5_ROOT = Path(__file__).resolve().parents[2]
WORK = FIG5_ROOT / "s8_extended_controls"
INPUT = WORK / "inputs"
OUTPUT = WORK / "outputs"
TMP = WORK / "_tmp_feature_work"
OUTPUT.mkdir(parents=True, exist_ok=True)
TMP.mkdir(parents=True, exist_ok=True)

FINAL_MODEL_DIR = FIG5_ROOT / "outputs" / "final_hybrid_model" / "frozen_model"
FEATURE_ORDER = json.loads((FINAL_MODEL_DIR / "MODEL_MANIFEST.json").read_text())["feature_order"]

RDF_FUNCTION_DIR = FIG5_ROOT / "fig5_final_model_validation_local" / "scripts"
PATCH_SCRIPT = RDF_FUNCTION_DIR / "historical_patch_get_features.py"
RESIDUE_CSV = FIG5_ROOT / "fig5_final_model_validation_local" / "inputs" / "histidine" / "metadata" / "residues.csv"
DEFAULT_RDF_DIR_CANDIDATES = [
    FIG5_ROOT / "controls" / "rdfs",
    FIG5_ROOT / "fig5_final_model_validation_local" / "inputs" / "controls" / "rdfs",
    FIG5_ROOT.parent / "fig4" / "controls_rdfs",
]

PAIR_FILES = {
    "dataset_new": INPUT / "dataset_new_pairs_to_run_140.txt",
    "nondataset_new": INPUT / "nondataset_new_pairs_to_run_140.txt",
}
FASTA_FILES = [INPUT / "dataset_new_sequences_to_run.fasta", INPUT / "nondataset_new_sequences_to_run.fasta"]


def read_names(path: Path) -> list[str]:
    return [x.strip() for x in path.read_text().splitlines() if x.strip()]


def read_fasta(paths: list[Path]) -> dict[str, str]:
    seqs = {}
    for path in paths:
        key = None
        for line in path.read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                key = line[1:].split()[0]
                seqs[key] = ""
            elif key:
                seqs[key] += line
    return seqs


def write_fasta(seqs: dict[str, str], path: Path) -> None:
    with path.open("w") as f:
        for k, v in seqs.items():
            f.write(f">{k}\n{v}\n")


def parse_rdf_name(path: Path) -> str:
    stem = path.stem
    if stem.startswith("rdfs_"):
        stem = stem[5:]
    if stem.endswith("_20nm"):
        stem = stem[:-5]
    return stem


def find_rdf_dir(user_rdf_dir: str | None) -> Path:
    if user_rdf_dir:
        p = Path(user_rdf_dir)
        if not p.is_absolute():
            p = FIG5_ROOT / p
        return p
    for p in DEFAULT_RDF_DIR_CANDIDATES:
        if p.exists():
            return p
    raise FileNotFoundError("No RDF directory found. Checked: " + "; ".join(str(x) for x in DEFAULT_RDF_DIR_CANDIDATES))


def classify_pair(name: str) -> str:
    left, right = name.split("__", 1)
    if "_SHUF" in left:
        return "RGG_shuffle"
    if "_SHUF" in right:
        return "TAD_shuffle"
    if "basic_AD_control" in right:
        return "basic_WT"
    if "external_AD_core" in right:
        return "external_AD_core_WT"
    return "WT"


def parent_for_pair(name: str) -> str:
    left, right = name.split("__", 1)
    left = re.sub(r"_SHUF\d+", "_WT", left)
    right = re.sub(r"_SHUF\d+", "_WT", right)
    return f"{left}__{right}"


def b22_from_rdfs(names: list[str], rdf_dir: Path) -> pd.DataFrame:
    sys.path.insert(0, str(RDF_FUNCTION_DIR))
    from rdf_b22_functions import corrected_rdfs_b22

    rdf_map = {parse_rdf_name(p): p for p in rdf_dir.glob("*.pkl")}
    rows = []
    for name in names:
        p = rdf_map.get(name)
        if p is None:
            rows.append({"name": name, "rdf_found": False, "rdf_file": "", "Corrected_B22_Hummer": np.nan, "B22_corr1_log10": np.nan, "rdf_error": "missing RDF"})
            continue
        try:
            rdf = pd.read_pickle(p)
            _, b22s, _ = corrected_rdfs_b22(rdf, 7.5, box_size=20, delta=2, rc_nm=4.0)
            vals = []
            for v in b22s.values():
                if "b22_corr1" in v:
                    vals.append(float(v["b22_corr1"]))
                elif "Corrected_B22_Hummer" in v:
                    vals.append(float(v["Corrected_B22_Hummer"]))
            vals = [x for x in vals if np.isfinite(x)]
            mean = float(np.mean(vals)) if vals else np.nan
            rows.append({
                "name": name,
                "rdf_found": True,
                "rdf_file": str(p.relative_to(FIG5_ROOT.parent) if str(p).startswith(str(FIG5_ROOT.parent)) else p),
                "n_replicate_values": len(vals),
                "Corrected_B22_Hummer": mean,
                "B22_corr1_log10": np.log10(-mean) if np.isfinite(mean) and mean < 0 else np.nan,
                "target_evaluable": bool(np.isfinite(mean) and mean < 0),
                "B22_sign": "negative" if np.isfinite(mean) and mean < 0 else "positive_or_missing",
                "rdf_error": "",
            })
        except Exception as e:
            rows.append({"name": name, "rdf_found": True, "rdf_file": str(p), "Corrected_B22_Hummer": np.nan, "B22_corr1_log10": np.nan, "target_evaluable": False, "rdf_error": repr(e)})
    return pd.DataFrame(rows)


def run_feature_generator(names: list[str], seqs: dict[str, str]) -> pd.DataFrame:
    fasta = TMP / "new_s8_control_sequences.fasta"
    names_file = TMP / "new_s8_control_feature_names.txt"
    out_csv = "new_s8_generated_features.csv"
    write_fasta(seqs, fasta)
    names_file.write_text("\n".join(names) + "\n")
    # If the environment lacks numba, historical_patch_get_features only needs njit for speed.
    shim = TMP / "numba.py"
    shim.write_text("def njit(*args, **kwargs):\n    if args and callable(args[0]) and len(args) == 1 and not kwargs:\n        return args[0]\n    def deco(fn):\n        return fn\n    return deco\n")
    env = dict(os.environ)
    env["PYTHONPATH"] = str(TMP) + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    cmd = [sys.executable, str(PATCH_SCRIPT), "--names_file", str(names_file), "--residue_csv", str(RESIDUE_CSV), "--fasta_file", str(fasta), "--out_folder", str(TMP), "--out_csv", out_csv]
    subprocess.run(cmd, check=True, env=env)
    return pd.read_csv(TMP / out_csv)


def normalize_feature_columns(features: pd.DataFrame) -> pd.DataFrame:
    """Create the frozen final-model feature names from validated historical columns.

    The historical generator reports lambda_RGG/lambda_TAD as sequence sums. The
    frozen final model uses mean residue lambda, so these must be divided by the
    corresponding sequence lengths rather than copied directly.
    """
    features = features.copy()
    audit = []
    required_for_mean = ["lambda_RGG", "RGG_length", "lambda_TAD", "TAD_length"]
    missing = [c for c in required_for_mean if c not in features.columns]
    if missing:
        raise RuntimeError(f"Cannot compute mean lambda features; missing historical columns: {missing}")
    for num, den, final_name in [
        ("lambda_RGG", "RGG_length", "mean_lambda_RGG"),
        ("lambda_TAD", "TAD_length", "mean_lambda_TAD"),
    ]:
        denom = pd.to_numeric(features[den], errors="coerce")
        if denom.isna().any() or (denom <= 0).any():
            raise RuntimeError(f"Invalid denominator values while computing {final_name} from {num}/{den}")
        features[final_name] = pd.to_numeric(features[num], errors="coerce") / denom
        audit.append({
            "final_feature": final_name,
            "source_column": f"{num}/{den}",
            "action": "computed_mean_residue_lambda",
            "min": float(features[final_name].min()),
            "max": float(features[final_name].max()),
        })
    pd.DataFrame(audit).to_csv(OUTPUT / "feature_column_alias_audit.csv", index=False)
    return features


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rdf-dir", default=None, help="Optional RDF directory. Defaults to project-relative candidates including ../fig4/controls_rdfs.")
    ap.add_argument("--allow-missing-rdf", action="store_true", help="Write audit and continue feature prep when RDFs are missing. Prediction inputs will include only rows with features; target may be missing.")
    args = ap.parse_args()

    groups = {k: read_names(v) for k, v in PAIR_FILES.items()}
    all_names = groups["dataset_new"] + groups["nondataset_new"]
    if len(all_names) != 280 or len(set(all_names)) != 280:
        raise RuntimeError(f"Expected 280 unique supplied names, found rows={len(all_names)} unique={len(set(all_names))}")

    seqs = read_fasta(FASTA_FILES)
    seq_audit = []
    for name in all_names:
        left, right = name.split("__", 1)
        seq_audit.append({"name": name, "left_sequence_id": left, "right_sequence_id": right, "left_found": left in seqs, "right_found": right in seqs})
    seq_audit_df = pd.DataFrame(seq_audit)
    seq_audit_df.to_csv(OUTPUT / "new_controls_sequence_audit.csv", index=False)
    if not seq_audit_df[["left_found", "right_found"]].all(axis=1).all():
        raise RuntimeError("Missing sequences; see outputs/new_controls_sequence_audit.csv")

    rdf_dir = find_rdf_dir(args.rdf_dir)
    b22 = b22_from_rdfs(all_names, rdf_dir)
    b22.to_csv(OUTPUT / "new_controls_b22.csv", index=False)
    missing_rdf = b22.loc[~b22["rdf_found"].astype(bool), "name"].tolist()
    if missing_rdf and not args.allow_missing_rdf:
        raise RuntimeError(f"Missing {len(missing_rdf)} RDFs. See outputs/new_controls_b22.csv. Missing examples: {missing_rdf[:10]}")

    features = normalize_feature_columns(run_feature_generator(all_names, seqs))
    missing_feature_cols = [c for c in FEATURE_ORDER if c not in features.columns]
    if missing_feature_cols:
        raise RuntimeError(f"Feature generator missing final model columns: {missing_feature_cols}")
    if features[FEATURE_ORDER].isna().any().any():
        raise RuntimeError("Generated features contain missing values")
    features.to_csv(OUTPUT / "new_controls_features.csv", index=False)

    meta_rows = []
    for group, names in groups.items():
        for name in names:
            cls = classify_pair(name)
            meta_rows.append({"name": name, "input_group": group, "control_origin": "new", "control_class": cls, "parent_WT_pair_id": parent_for_pair(name), "is_shuffle": cls in ["TAD_shuffle", "RGG_shuffle"]})
    meta = pd.DataFrame(meta_rows)
    merged = meta.merge(b22, on="name", how="left", validate="one_to_one").merge(features, on="name", how="left", validate="one_to_one")
    merged.to_csv(OUTPUT / "new_controls_model_input.csv", index=False)

    audit = merged[["name", "input_group", "control_class", "parent_WT_pair_id", "rdf_found", "target_evaluable"]].copy()
    audit["all_13_features_present"] = merged[FEATURE_ORDER].notna().all(axis=1)
    audit.to_csv(OUTPUT / "new_controls_input_audit.csv", index=False)
    (OUTPUT / "PREPARE_INPUTS_AUDIT.txt").write_text(
        "New S8 control input preparation audit\n"
        f"fig5_root={FIG5_ROOT}\n"
        f"rdf_dir={rdf_dir}\n"
        f"total_supplied={len(all_names)}\n"
        f"dataset_manifest_n={len(groups['dataset_new'])}\n"
        f"nondataset_manifest_n={len(groups['nondataset_new'])}\n"
        f"rdf_found={int(b22['rdf_found'].sum())}\n"
        f"rdf_missing={len(missing_rdf)}\n"
        f"missing_rdf_ids={json.dumps(missing_rdf)}\n"
        f"all_sequences_found={bool(seq_audit_df[['left_found','right_found']].all(axis=1).all())}\n"
        f"classification_counts={json.dumps(meta['control_class'].value_counts().to_dict(), sort_keys=True)}\n"
        f"all_13_features_complete={bool(merged[FEATURE_ORDER].notna().all(axis=1).all())}\n"
    )
    print(f"Prepared new S8 control inputs in {OUTPUT}")


if __name__ == "__main__":
    main()
