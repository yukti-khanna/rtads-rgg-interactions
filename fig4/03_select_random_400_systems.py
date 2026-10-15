#!/usr/bin/env python3
"""Select the prospective 20-TAD by 5-common-RGG factorial control panel."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import re
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

GLOBAL_SEED = 20260906


def fail(message: str) -> None:
    raise SystemExit(f"ERROR: {message}")


def seed_for(namespace: str, identifier: str, global_seed: int) -> int:
    payload = f"{global_seed}|{namespace}|{identifier}".encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:4], "big")


def sha256_file(path: Path) -> str:
    if not path.is_file():
        fail(f"required file not found: {path}")
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def require_columns(df: pd.DataFrame, columns: list[str], source: Path) -> None:
    missing = [c for c in columns if c not in df.columns]
    if missing:
        fail(f"{source} is missing required columns: {missing}; found: {list(df.columns)}")


def exact_one(df: pd.DataFrame, context: str) -> pd.Series:
    if len(df) != 1:
        fail(f"expected exactly one universe row for {context}, found {len(df)}")
    return df.iloc[0]


def as_optional_int(value: object) -> int | None:
    return None if pd.isna(value) else int(value)


def safe_token(value: object) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value))


VALID_AA = set("ACDEFGHIKLMNPQRSTVWY")


def simulation_variant_id(molecule: str, parent_id: object, variant_type: object,
                          shuffle_number: object) -> str:
    """Return the short, filesystem-safe ID consumed by CALVADOS."""
    prefix = f"{molecule}_{safe_token(parent_id)}"
    if str(variant_type).upper() == "WT":
        suffix = "WT"
    else:
        if pd.isna(shuffle_number):
            fail(f"missing shuffle number for {molecule} parent {parent_id}")
        suffix = f"SHUF{int(shuffle_number):02d}"
    identifier = f"{prefix}_{suffix}"
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", identifier):
        fail(f"generated unsafe simulation variant identifier: {identifier}")
    return identifier


def validate_amino_acid_sequence(sequence: object, identifier: str) -> str:
    seq = str(sequence).strip().upper()
    invalid = sorted(set(seq) - VALID_AA)
    if not seq or invalid:
        fail(f"invalid amino-acid sequence for {identifier}: invalid characters={invalid}")
    return seq


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--universe", type=Path,
                   default=Path("reviewer_analysis/control_simulation_manifest/full_candidate_universe.parquet"))
    p.add_argument("--input-dir", type=Path, default=Path("reviewer_analysis/control_simulation_manifest"))
    p.add_argument("--output-dir", type=Path, default=Path("reviewer_analysis/control_simulation_manifest"))
    p.add_argument("--seed", type=int, default=GLOBAL_SEED)
    p.add_argument("--all-fasta", type=Path, default=Path("all_tads_rggs_dis.fasta"))
    p.add_argument("--rgg-fasta", type=Path, default=Path("combined_rg_regions_ll_7_all.fasta"))
    p.add_argument("--conventional-rbps", type=Path, default=Path("conventional_rbp_uniprot_ids.txt"))
    p.add_argument("--tad-selection", type=Path, default=Path("tad_region_selection.xlsx"))
    p.add_argument("--external-controls", type=Path,
                   default=Path("ETV4_SPI1_CREB3_core_and_longest_sequences.xlsx"))
    p.add_argument("--basic-controls", type=Path, default=Path("basic_AD_IDR_controls.xlsx"))
    return p.parse_args()


def main() -> None:
    fail("This legacy campaign is superseded. Its pair list is now a protected exclusion input. Run 02_build_control_pair_universe.py, 03_select_new_shuffle_400.py, and 04_select_new_basic_controls.py instead.")
    args = parse_args()
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    parent_path = args.input_dir / "tad_parent_sequences.csv"
    tad_shuffle_path = args.input_dir / "tad_shuffles.csv"
    rgg_parent_path = args.input_dir / "eligible_conventional_RGGs.csv"
    rgg_shuffle_path = args.input_dir / "rgg_shuffles.csv"
    script1_paths = [parent_path, tad_shuffle_path, rgg_parent_path, rgg_shuffle_path]
    for path in [args.universe, *script1_paths]:
        if not path.is_file():
            fail(f"required input not found: {path}")
    try:
        universe = pd.read_parquet(args.universe, engine="pyarrow")
    except ImportError as exc:
        fail(f"pyarrow is required to read {args.universe}: {exc}")
    tp, ts, rp, rs = (pd.read_csv(p) for p in script1_paths)
    require_columns(tp, ["parent_id", "gene", "class", "sequence"], parent_path)
    require_columns(ts, ["parent_id", "variant_id", "shuffle_number", "seed", "sequence"], tad_shuffle_path)
    require_columns(rp, ["RGG_ID", "UniProt_ID", "sequence"], rgg_parent_path)
    require_columns(rs, ["RGG_parent_id", "variant_id", "shuffle_number", "seed", "sequence"], rgg_shuffle_path)
    required_universe = ["candidate_system_id", "TAD_parent_id", "TAD_variant_type", "RGG_parent_id",
        "RGG_UniProt_ID", "RGG_variant_type", "TAD_sequence", "RGG_sequence",
        "TAD_shuffle_number", "RGG_shuffle_number", "TAD_seed", "RGG_seed", "TAD_length",
        "RGG_length", "charge_TAD", "charge_RGG", "NCPR_TAD", "NCPR_RGG",
        "Net_Charge_Product", "TAD_aromatic_fraction"]
    require_columns(universe, required_universe, args.universe)

    expected_tad_classes = {"native_RTAD": 13, "external_AD_not_RTAD": 3, "basic_AD_control": 4}
    if len(tp) != 20 or tp["parent_id"].nunique() != 20 or tp["class"].value_counts().to_dict() != expected_tad_classes:
        fail(f"expected 20 TAD parents with class counts {expected_tad_classes}; found {tp['class'].value_counts().to_dict()}")
    if ts.groupby("parent_id").size().to_dict() != {x: 20 for x in tp["parent_id"]}:
        fail("each TAD parent must have exactly 20 shuffles")
    eligible_proteins = sorted(rp["UniProt_ID"].astype(str).unique())
    if len(eligible_proteins) < 5:
        fail(f"fewer than five conventional RBP proteins are eligible: {len(eligible_proteins)}")

    protein_seed = seed_for("common_RGG_protein_selection", "five_proteins", args.seed)
    protein_rng = np.random.default_rng(protein_seed)
    chosen_proteins = np.array(eligible_proteins, dtype=object)[
        protein_rng.choice(len(eligible_proteins), 5, replace=False)
    ].tolist()
    common_rows = []
    for order, uid in enumerate(chosen_proteins, 1):
        candidates = sorted(rp.loc[rp["UniProt_ID"].astype(str) == uid, "RGG_ID"].tolist())
        region_seed = seed_for("common_RGG_region_selection", uid, args.seed)
        region = candidates[int(np.random.default_rng(region_seed).integers(len(candidates)))]
        meta = exact_one(rp[rp["RGG_ID"] == region], f"RGG metadata {region}")
        common_rows.append({"RGG_parent_id": region, "RGG_UniProt_ID": uid,
                            "WT_sequence": meta["sequence"], "random_selection_order": order,
                            "protein_selection_seed": protein_seed, "region_selection_seed": region_seed})
    common = pd.DataFrame(common_rows)
    if common["RGG_parent_id"].nunique() != 5 or common["RGG_UniProt_ID"].nunique() != 5:
        fail("common RGG selection did not yield five regions from five distinct proteins")
    common.to_csv(out / "selected_common_RGGs.csv", index=False)

    tad_choices = {}
    tad_choice_rows = []
    for parent in sorted(tp["parent_id"]):
        selection_seed = seed_for("select_two_TAD_shuffles", parent, args.seed)
        numbers = np.random.default_rng(selection_seed).choice(np.arange(1, 21), 2, replace=False).astype(int).tolist()
        chosen = []
        for order, number in enumerate(numbers, 1):
            meta = exact_one(ts[(ts["parent_id"] == parent) & (ts["shuffle_number"] == number)],
                             f"TAD shuffle {parent}/{number}")
            chosen.append(number)
            tad_choice_rows.append({"TAD_parent_id": parent, "selection_order": order,
                                    "selected_shuffle_number": number,
                                    "selected_variant_id": meta["variant_id"],
                                    "shuffle_seed": int(meta["seed"]), "selection_seed": selection_seed})
        tad_choices[parent] = chosen
    tad_long = pd.DataFrame(tad_choice_rows)
    tad_wide_rows = []
    for parent in sorted(tad_choices):
        rows = tad_long[tad_long["TAD_parent_id"] == parent].sort_values("selection_order")
        tad_wide_rows.append({"TAD_parent_id": parent,
            "selected_shuffle_1": int(rows.iloc[0]["selected_shuffle_number"]),
            "selected_shuffle_2": int(rows.iloc[1]["selected_shuffle_number"]),
            "selected_variant_1": rows.iloc[0]["selected_variant_id"],
            "selected_variant_2": rows.iloc[1]["selected_variant_id"],
            "shuffle_seed_1": int(rows.iloc[0]["shuffle_seed"]),
            "shuffle_seed_2": int(rows.iloc[1]["shuffle_seed"]),
            "selection_seed": int(rows.iloc[0]["selection_seed"]), "selection_order": "A,B"})
    pd.DataFrame(tad_wide_rows).to_csv(out / "selected_TAD_shuffles.csv", index=False)

    rgg_choices = {}
    rgg_choice_rows = []
    for row in common.itertuples(index=False):
        selection_seed = seed_for("select_one_RGG_shuffle", row.RGG_parent_id, args.seed)
        number = int(np.random.default_rng(selection_seed).choice(np.arange(1, 21)))
        meta = exact_one(rs[(rs["RGG_parent_id"] == row.RGG_parent_id) & (rs["shuffle_number"] == number)],
                         f"RGG shuffle {row.RGG_parent_id}/{number}")
        rgg_choices[row.RGG_parent_id] = number
        rgg_choice_rows.append({"RGG_parent_id": row.RGG_parent_id, "RGG_UniProt_ID": row.RGG_UniProt_ID,
            "selected_shuffle_number": number, "selected_variant_id": meta["variant_id"],
            "shuffle_seed": int(meta["seed"]), "selection_seed": selection_seed})
    pd.DataFrame(rgg_choice_rows).to_csv(out / "selected_RGG_shuffles.csv", index=False)

    parent_meta = tp.set_index("parent_id")
    selected_rows = []
    for tad_parent in sorted(tp["parent_id"]):
        tad_numbers = tad_choices[tad_parent]
        for rgg_parent in common.sort_values("random_selection_order")["RGG_parent_id"]:
            base = universe[(universe["TAD_parent_id"] == tad_parent) & (universe["RGG_parent_id"] == rgg_parent)]
            selectors = [
                ("WT_WT", "WT", None, "WT", None),
                ("TAD_SHUFFLE_A", "SHUFFLED", tad_numbers[0], "WT", None),
                ("TAD_SHUFFLE_B", "SHUFFLED", tad_numbers[1], "WT", None),
                ("RGG_SHUFFLE", "WT", None, "SHUFFLED", rgg_choices[rgg_parent]),
            ]
            for condition, tad_type, tad_number, rgg_type, rgg_number in selectors:
                match = base[(base["TAD_variant_type"] == tad_type) & (base["RGG_variant_type"] == rgg_type)]
                if tad_number is not None:
                    match = match[match["TAD_shuffle_number"] == tad_number]
                if rgg_number is not None:
                    match = match[match["RGG_shuffle_number"] == rgg_number]
                selected = exact_one(match, f"{condition} {tad_parent}/{rgg_parent}")
                selection_seed = (protein_seed if condition == "WT_WT" else
                                  seed_for("select_two_TAD_shuffles", tad_parent, args.seed)
                                  if condition.startswith("TAD_") else
                                  seed_for("select_one_RGG_shuffle", rgg_parent, args.seed))
                selected_rows.append({**selected.to_dict(), "condition": condition,
                                      "TAD_gene": parent_meta.loc[tad_parent, "gene"],
                                      "selection_seed": selection_seed})
    manifest = pd.DataFrame(selected_rows)
    expected_conditions = {"WT_WT": 100, "TAD_SHUFFLE_A": 100,
                           "TAD_SHUFFLE_B": 100, "RGG_SHUFFLE": 100}
    if len(manifest) != 400 or manifest["condition"].value_counts().to_dict() != expected_conditions:
        fail(f"invalid final condition counts: {manifest['condition'].value_counts().to_dict()}")
    if ((manifest["TAD_variant_type"] == "SHUFFLED") & (manifest["RGG_variant_type"] == "SHUFFLED")).any():
        fail("double-shuffled system entered the final panel")

    validation_rows = []
    for (tad_parent, rgg_parent), group in manifest.groupby(["TAD_parent_id", "RGG_parent_id"], sort=False):
        reference = exact_one(group[group["condition"] == "WT_WT"], f"WT reference {tad_parent}/{rgg_parent}")
        for _, row in group.iterrows():
            tad_shuffle = row["condition"] in {"TAD_SHUFFLE_A", "TAD_SHUFFLE_B"}
            rgg_shuffle = row["condition"] == "RGG_SHUFFLE"
            checks = {
                "TAD_composition_preserved": Counter(row["TAD_sequence"]) == Counter(reference["TAD_sequence"]) if tad_shuffle else True,
                "TAD_length_preserved": row["TAD_length"] == reference["TAD_length"] if tad_shuffle else True,
                "TAD_charge_preserved": row["charge_TAD"] == reference["charge_TAD"] if tad_shuffle else True,
                "TAD_NCPR_preserved": np.isclose(row["NCPR_TAD"], reference["NCPR_TAD"], atol=1e-15, rtol=0) if tad_shuffle else True,
                "TAD_aromatic_preserved": sum(row["TAD_sequence"].count(x) for x in "FWY") == sum(reference["TAD_sequence"].count(x) for x in "FWY") if tad_shuffle else True,
                "same_WT_RGG": row["RGG_sequence"] == reference["RGG_sequence"] if tad_shuffle else True,
                "RGG_composition_preserved": Counter(row["RGG_sequence"]) == Counter(reference["RGG_sequence"]) if rgg_shuffle else True,
                "RGG_length_preserved": row["RGG_length"] == reference["RGG_length"] if rgg_shuffle else True,
                "RGG_charge_preserved": row["charge_RGG"] == reference["charge_RGG"] if rgg_shuffle else True,
                "RGG_NCPR_preserved": np.isclose(row["NCPR_RGG"], reference["NCPR_RGG"], atol=1e-15, rtol=0) if rgg_shuffle else True,
                "same_WT_TAD": row["TAD_sequence"] == reference["TAD_sequence"] if rgg_shuffle else True,
                "Net_Charge_Product_preserved": row["Net_Charge_Product"] == reference["Net_Charge_Product"],
            }
            validation_rows.append({"candidate_system_id": row["candidate_system_id"], "condition": row["condition"],
                                    **checks, "all_required_checks_passed": all(checks.values())})
    validation = pd.DataFrame(validation_rows)
    if len(validation) != 400 or not validation["all_required_checks_passed"].all():
        fail("matched-control validation failed")

    manifest["TAD_variant_id"] = [simulation_variant_id("TAD", p, t, n) for p, t, n in
        zip(manifest["TAD_parent_id"], manifest["TAD_variant_type"], manifest["TAD_shuffle_number"])]
    manifest["RGG_variant_id"] = [simulation_variant_id("RGG", p, t, n) for p, t, n in
        zip(manifest["RGG_parent_id"], manifest["RGG_variant_type"], manifest["RGG_shuffle_number"])]
    manifest.insert(0, "name", manifest["RGG_variant_id"] + "__" + manifest["TAD_variant_id"])
    manifest = manifest.rename(columns={"TAD_seed": "TAD_shuffle_seed", "RGG_seed": "RGG_shuffle_seed"})
    output_columns = ["name", "TAD_parent_id", "TAD_gene", "TAD_class", "TAD_variant_id",
        "TAD_variant_type", "TAD_shuffle_number", "RGG_parent_id", "RGG_UniProt_ID",
        "RGG_variant_id", "RGG_variant_type", "RGG_shuffle_number", "condition", "TAD_sequence",
        "RGG_sequence", "TAD_length", "RGG_length", "charge_TAD", "charge_RGG", "NCPR_TAD",
        "NCPR_RGG", "Net_Charge_Product", "TAD_shuffle_seed", "RGG_shuffle_seed", "selection_seed"]
    manifest = manifest[output_columns]
    manifest_path = out / "simulation_manifest_400.csv"
    pair_path = out / "simulation_pair_names_400.txt"
    fasta_path = out / "simulation_sequences_400.fasta"
    lookup_path = out / "simulation_sequence_lookup_400.csv"
    manifest.to_csv(manifest_path, index=False)
    pair_path.write_text("\n".join(manifest["name"]) + "\n")
    validation.to_csv(out / "manifest_validation_400.csv", index=False)

    sequence_rows = []
    for molecule in ("TAD", "RGG"):
        for variant_id, group in manifest.groupby(f"{molecule}_variant_id", sort=True):
            sequences = group[f"{molecule}_sequence"].drop_duplicates().tolist()
            if len(sequences) != 1:
                fail(f"{variant_id} maps to multiple sequences")
            first = group.iloc[0]
            is_wt = first[f"{molecule}_variant_type"] == "WT"
            sequence = validate_amino_acid_sequence(sequences[0], variant_id)
            sequence_rows.append({"variant_id": variant_id, "molecule_type": molecule,
                "parent_id": first[f"{molecule}_parent_id"],
                "gene_or_uniprot": first["TAD_gene" if molecule == "TAD" else "RGG_UniProt_ID"],
                "biological_class": first["TAD_class"] if molecule == "TAD" else "conventional_RGG",
                "variant_type": "WT" if is_wt else "shuffle",
                "shuffle_number": pd.NA if is_wt else int(first[f"{molecule}_shuffle_number"]),
                "sequence": sequence, "length": len(sequence),
                "shuffle_seed": pd.NA if is_wt else as_optional_int(first[f"{molecule}_shuffle_seed"])})
    lookup = pd.DataFrame(sequence_rows)
    lookup.to_csv(lookup_path, index=False)
    with fasta_path.open("w") as handle:
        for row in lookup.itertuples(index=False):
            handle.write(f">{row.variant_id}\n{row.sequence}\n")

    pair_lines = pair_path.read_text().splitlines()
    written = pd.read_csv(manifest_path)
    headers = [x[1:] for x in fasta_path.read_text().splitlines() if x.startswith(">")]
    components = set()
    for name in pair_lines:
        if name.count("__") != 1:
            fail(f"pair name does not split exactly once: {name}")
        rgg_id, tad_id = name.split("__")
        if not rgg_id.startswith("RGG_") or not tad_id.startswith("TAD_"):
            fail(f"pair order/prefix invalid: {name}")
        components.update((rgg_id, tad_id))
    if len(pair_lines) != len(set(pair_lines)) or len(pair_lines) != 400:
        fail("pair list does not contain exactly 400 unique names")
    if set(pair_lines) != set(written["name"]) or len(written) != written["name"].nunique() or len(written) != 400:
        fail("manifest and pair-list names differ or are not unique")
    if components != set(headers) or len(headers) != len(set(headers)) or set(headers) != set(lookup["variant_id"]):
        fail("FASTA is not the exact unique set of used pair components")
    if len(lookup) != 70 or (lookup["molecule_type"].value_counts().to_dict() != {"TAD": 60, "RGG": 10}):
        fail(f"expected exactly 70 FASTA identifiers (60 TAD, 10 RGG), found {lookup['molecule_type'].value_counts().to_dict()}")
    wt = manifest[manifest["condition"] == "WT_WT"]
    if wt.groupby("TAD_parent_id")["RGG_parent_id"].nunique().to_dict() != {x: 5 for x in tp["parent_id"]}:
        fail("not every WT TAD is paired with all five common WT RGGs")
    for condition in ("TAD_SHUFFLE_A", "TAD_SHUFFLE_B"):
        subset = manifest[manifest["condition"] == condition]
        if subset.groupby("TAD_variant_id")["RGG_parent_id"].nunique().ne(5).any() or subset["TAD_variant_id"].nunique() != 20:
            fail(f"selected TAD shuffles are not reused across all five RGGs for {condition}")
    rsub = manifest[manifest["condition"] == "RGG_SHUFFLE"]
    if rsub.groupby("RGG_variant_id")["TAD_parent_id"].nunique().ne(20).any() or rsub["RGG_variant_id"].nunique() != 5:
        fail("selected RGG shuffles are not reused across all 20 WT TADs")

    native_pool_path = args.input_dir / "native_RTAD_eligible_pool.csv"
    native_pool = pd.read_csv(native_pool_path)
    unique_tads, unique_rggs, total_sequences = 60, 10, 70

    original_inputs = [
        args.all_fasta, args.rgg_fasta, args.conventional_rbps, args.tad_selection,
        args.external_controls, args.basic_controls,
    ]
    audit = {
        "global_random_seed": args.seed,
        "seed_derivation": "first 32 bits of SHA256(global_seed|namespace|identifier)",
        "common_RGG_protein_selection_seed": protein_seed,
        "original_input_sha256": {str(p): sha256_file(p) for p in original_inputs},
        "script1_output_sha256": {str(p): sha256_file(p) for p in script1_paths},
        "full_candidate_universe_sha256": sha256_file(args.universe),
        "selected_common_RGGs": common.to_dict(orient="records"),
        "selected_TAD_shuffles": tad_choice_rows,
        "selected_RGG_shuffles": rgg_choice_rows,
        "python_version": platform.python_version(),
        "numpy_version": np.__version__,
        "pandas_version": pd.__version__,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "selection_exclusions": [
            "B22", "predicted B22", "ML predictions", "OCN", "Net_Charge_Product",
            "charge", "NCPR", "lambda", "length", "RG density",
            "previously observed interaction properties",
        ],
        "sampling_method": "uniform identifier-only sampling without replacement; no feature weighting",
    }
    (out / "selection_audit_400.json").write_text(json.dumps(audit, indent=2) + "\n")

    counts = manifest["condition"].value_counts()
    report = [
        "# Script 3 final-selection report", "",
        f"- Final systems: {len(manifest)}",
        "- Current/final design: revised 400-system common-RGG factorial panel",
        "- TAD-side parents: 20",
        "- Randomly selected native predicted R-TADs: 13",
        "- Fixed external acidic AD/non-R-TAD controls: 3",
        "- Fixed basic AD controls: 4",
        "- Common conventional RGG regions: 5 from 5 distinct conventional RBP proteins",
        "- All 20 TAD parents were tested against the same five RGGs.",
        "- Distinct WT biological TAD-RGG contexts: 100",
        "- Two independently selected exact-composition TAD shuffles per TAD parent were reused across all five RGGs.",
        "- One exact-composition RGG shuffle per RGG parent was reused across all 20 WT TADs.",
        f"- WT_WT: {int(counts.get('WT_WT', 0))}",
        f"- TAD_SHUFFLE_A: {int(counts.get('TAD_SHUFFLE_A', 0))}",
        f"- TAD_SHUFFLE_B: {int(counts.get('TAD_SHUFFLE_B', 0))}",
        f"- RGG_SHUFFLE: {int(counts.get('RGG_SHUFFLE', 0))}",
        "- Double shuffle: 0",
        "- All composition-preservation and matched-control checks passed.",
        "> Neither TAD/RGG parent selection nor shuffle selection used previous CALVADOS B22 values, machine-learning predictions, OCN, charge complementarity, lambda, NCPR, length, RG density, or any other interaction-strength or physicochemical ranking variable.",
        "",
        "Sequence descriptors were used only after selection for documentation and QC.",
        f"- Global seed: {args.seed}",
        f"- Common-RGG protein selection seed: {protein_seed}", "",
        f"- Unique TAD variants required: {unique_tads}",
        f"- Unique RGG variants required: {unique_rggs}",
        f"- Total unique FASTA sequences: {total_sequences}",
        "- All pair components found exactly once in the final FASTA: yes", "",
        "No CALVADOS simulations, RDFs, B22 calculations, ML predictions, or comparisons to prior interaction strengths were run.",
    ]
    (out / "SCRIPT3_FINAL_SELECTION_REPORT_400.md").write_text("\n".join(report) + "\n")
    print(f"native R-TAD eligible pool: {len(native_pool)}")
    print("native R-TADs selected: 13")
    print("external acidic controls: 3")
    print("basic controls: 4")
    print("total TAD parents: 20")
    print(f"eligible conventional RGGs: {len(rp)}")
    print(f"conventional RBP proteins represented in eligible pool: {len(eligible_proteins)}")
    print("common RGG parents selected: 5")
    print(f"selected RGG parent UniProt proteins: {','.join(chosen_proteins)}")
    print("selected TAD shuffle variants: 40")
    print("selected RGG shuffle variants: 5")
    print("WT pair contexts: 100")
    print("WT_WT systems: 100")
    print("TAD_SHUFFLE_A systems: 100")
    print("TAD_SHUFFLE_B systems: 100")
    print("RGG_SHUFFLE systems: 100")
    print("total systems: 400")
    print(f"unique FASTA identifiers: {total_sequences}")
    print(f"pair-list lines: {len(pair_lines)}")
    print(f"duplicate pair names: {len(pair_lines) - len(set(pair_lines))}")
    print("all pair components found in FASTA: yes")
    print(f"global seed: {args.seed}")
    print(f"FASTA output path: {fasta_path}")
    print(f"pair-list output path: {pair_path}")
    print(f"manifest output path: {manifest_path}")


if __name__ == "__main__":
    main()
