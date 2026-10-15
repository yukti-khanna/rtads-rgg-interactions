#!/usr/bin/env python3
"""Validate inputs, define control parents, and create composition shuffles."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from control_campaign_common import configure_naming, variant, sha

GLOBAL_SEED = 20260906
AA = set("ACDEFGHIKLMNPQRSTVWY")
EXCLUDED_GENES = {"ETV4", "SPI1", "CREB3", "TCF7L2", "TET3", "EOMES", "LEF1"}
EXTERNAL_GENES = ("ETV4", "SPI1", "CREB3")
BASIC_GENES = ("TCF7L2", "TET3", "EOMES", "LEF1")


def fail(message: str) -> None:
    raise SystemExit(f"ERROR: {message}")


def seed_for(namespace: str, identifier: str, global_seed: int) -> int:
    payload = f"{global_seed}|{namespace}|{identifier}".encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:4], "big")


def clean_sequence(value: object, context: str) -> str:
    if pd.isna(value):
        fail(f"missing sequence for {context}")
    seq = re.sub(r"\s+", "", str(value)).upper()
    bad = sorted(set(seq) - AA)
    if not seq or bad:
        fail(f"invalid amino-acid sequence for {context}; unexpected characters: {bad}")
    return seq


def read_fasta(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        fail(f"required FASTA not found: {path}")
    records: list[dict[str, str]] = []
    header = None
    parts: list[str] = []
    for line_number, raw in enumerate(path.read_text().splitlines(), 1):
        line = raw.strip()
        if not line:
            continue
        if line.startswith(">"):
            if header is not None:
                records.append({"sequence_id": header, "sequence": clean_sequence("".join(parts), header)})
            header = line[1:].strip().split()[0]
            if not header:
                fail(f"empty FASTA identifier in {path} at line {line_number}")
            parts = []
        else:
            if header is None:
                fail(f"sequence before first FASTA header in {path} at line {line_number}")
            parts.append(line)
    if header is not None:
        records.append({"sequence_id": header, "sequence": clean_sequence("".join(parts), header)})
    if not records:
        fail(f"no FASTA records found in {path}")
    ids = [r["sequence_id"] for r in records]
    if len(ids) != len(set(ids)):
        dup = pd.Series(ids)[pd.Series(ids).duplicated()].unique().tolist()
        fail(f"duplicate FASTA identifiers in {path}: {dup[:10]}")
    return records


def parent_accession(region_id: str) -> str:
    if "_" not in region_id:
        fail(f"RG region identifier lacks an accession separator: {region_id}")
    return region_id.split("_", 1)[0]


def charge(seq: str) -> int:
    c = Counter(seq)
    return c["K"] + c["R"] - c["D"] - c["E"]


def descriptors(seq: str) -> dict[str, float | int]:
    c = Counter(seq)
    q = charge(seq)
    aromatic = c["F"] + c["W"] + c["Y"]
    return {
        "length": len(seq),
        "net_charge": q,
        "NCPR": q / len(seq),
        "acidic_count": c["D"] + c["E"],
        "basic_count": c["K"] + c["R"],
        "aromatic_count": aromatic,
        "aromatic_fraction": aromatic / len(seq),
    }


def require_columns(df: pd.DataFrame, columns: list[str], source: Path) -> None:
    missing = [c for c in columns if c not in df.columns]
    if missing:
        fail(f"{source} is missing required columns: {missing}; found: {list(df.columns)}")


def unique_shuffles(seq: str, count: int, seed: int, context: str) -> list[str]:
    counts = Counter(seq)
    # A cheap impossibility check that avoids factorial overflow.
    if len(counts) == 1:
        fail(f"cannot generate any non-WT shuffle for {context}: sequence is monomeric")
    rng = np.random.default_rng(seed)
    chars = np.array(list(seq), dtype="U1")
    seen: set[str] = set()
    max_attempts = max(10000, count * 1000)
    for _ in range(max_attempts):
        shuffled = "".join(rng.permutation(chars).tolist())
        if shuffled != seq:
            seen.add(shuffled)
            if len(seen) == count:
                return sorted(seen)
    fail(f"could not generate {count} unique non-WT shuffles for {context} after {max_attempts} attempts")
    return []


def write_fasta(rows: pd.DataFrame, id_col: str, seq_col: str, path: Path) -> None:
    with path.open("w") as handle:
        for row in rows.itertuples(index=False):
            handle.write(f">{getattr(row, id_col)}\n{getattr(row, seq_col)}\n")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--all-fasta", type=Path, default=Path("all_tads_rggs_dis.fasta"))
    p.add_argument("--rgg-fasta", type=Path, default=Path("combined_rg_regions_ll_7_all.fasta"))
    p.add_argument("--conventional-rbps", type=Path, default=Path("conventional_rbp_uniprot_ids.txt"))
    p.add_argument("--tad-selection", type=Path, default=Path("tad_region_selection.xlsx"))
    external_default = Path("ETV4_SPI1_CREB3_core_longest_with_B22.xlsx")
    if not external_default.is_file():
        external_default = Path("ETV4_SPI1_CREB3_core_and_longest_sequences.xlsx")
    basic_default = Path("basic_AD_IDR_controls (1).xlsx")
    if not basic_default.is_file():
        basic_default = Path("basic_AD_IDR_controls.xlsx")
    p.add_argument("--external-controls", type=Path, default=external_default)
    p.add_argument("--basic-controls", type=Path, default=basic_default)
    p.add_argument("--output-dir", type=Path, default=Path("reviewer_analysis/control_simulation_manifest"))
    p.add_argument("--seed", type=int, default=GLOBAL_SEED)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    inputs = [args.all_fasta, args.rgg_fasta, args.conventional_rbps, args.tad_selection,
              args.external_controls, args.basic_controls]
    missing = [str(p) for p in inputs if not p.is_file()]
    if missing:
        fail(f"required input files missing: {missing}")
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)

    all_records = read_fasta(args.all_fasta)
    rgg_records = read_fasta(args.rgg_fasta)
    if len(all_records) != 1238:
        fail(f"expected 1,238 records in {args.all_fasta}, found {len(all_records)}")
    if len(rgg_records) != 1008:
        fail(f"expected 1,008 records in {args.rgg_fasta}, found {len(rgg_records)}")
    all_by_id = {r["sequence_id"]: r["sequence"] for r in all_records}
    rgg_by_id = {r["sequence_id"]: r["sequence"] for r in rgg_records}
    absent = sorted(set(rgg_by_id) - set(all_by_id))
    mismatched = sorted(i for i, s in rgg_by_id.items() if all_by_id.get(i) != s)
    if absent or mismatched:
        fail(f"RG cross-check failed; absent IDs={absent[:10]}, sequence mismatches={mismatched[:10]}")
    rtad_ids = sorted(set(all_by_id) - set(rgg_by_id))
    if len(rtad_ids) != 230:
        fail(f"expected 230 predicted R-TADs after ID/sequence cross-check, found {len(rtad_ids)}")

    inventory_rows = []
    for r in all_records:
        sid = r["sequence_id"]
        inventory_rows.append({
            "sequence_id": sid,
            "UniProt_ID": parent_accession(sid),
            "sequence": r["sequence"],
            "length": len(r["sequence"]),
            "sequence_class": "compact_RG_region" if sid in rgg_by_id else "predicted_RTAD",
            "source_file": str(args.all_fasta),
        })
    inventory = pd.DataFrame(inventory_rows)
    inventory.to_csv(out / "full_sequence_inventory.csv", index=False)

    conventional = {
        line.strip().split()[0] for line in args.conventional_rbps.read_text().splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    }
    if not conventional:
        fail(f"no accessions found in {args.conventional_rbps}")
    eligible_rgg_rows = []
    for rid, seq in sorted(rgg_by_id.items()):
        uid = parent_accession(rid)
        if uid in conventional:
            c = Counter(seq)
            d = descriptors(seq)
            eligible_rgg_rows.append({
                "RGG_ID": rid, "UniProt_ID": uid, "sequence": seq, "length": d["length"],
                "RG_motif_count": seq.count("RG"), "R_count": c["R"], "G_count": c["G"],
                "K_count": c["K"], "D_count": c["D"], "E_count": c["E"],
                "net_charge": d["net_charge"], "NCPR": d["NCPR"],
            })
    eligible_rgg = pd.DataFrame(eligible_rgg_rows)
    if eligible_rgg.empty:
        fail("no compact RG regions have parents in the conventional-RBP list")
    eligible_rgg.to_csv(out / "eligible_conventional_RGGs.csv", index=False)

    tad = pd.read_excel(args.tad_selection, sheet_name="Selected region clusters",
        usecols=lambda c: c in {"Gene name", "UniProt ID", "Category", "External dataset count",
                               "Representative sequence", "Overlapping R-TAD region ID(s)", "Representative source ID"})
    tad_cols = ["Gene name", "UniProt ID", "Category", "External dataset count",
                "Representative sequence", "Overlapping R-TAD region ID(s)", "Representative source ID"]
    require_columns(tad, tad_cols, args.tad_selection)
    candidate_native: list[dict[str, object]] = []
    for _, row in tad.iterrows():
        gene = str(row["Gene name"]).strip()
        uid = str(row["UniProt ID"]).strip()
        if gene in EXCLUDED_GENES or pd.isna(row["Overlapping R-TAD region ID(s)"]):
            continue
        if pd.isna(row["External dataset count"]) or int(row["External dataset count"]) < 2:
            continue
        overlap_ids = sorted(x.strip() for x in str(row["Overlapping R-TAD region ID(s)"]).split(";") if x.strip())
        mapped = [x for x in overlap_ids if x in rtad_ids]
        missing_overlap = [x for x in overlap_ids if x not in rtad_ids]
        if missing_overlap:
            fail(f"documented R-TAD IDs for {gene} are absent from the predicted catalogue: {missing_overlap}")
        if not mapped:
            continue
        # Non-feature tie-break: one catalogue region per protein, lexicographically first ID.
        rid = mapped[0]
        if parent_accession(rid) != uid:
            fail(f"accession mismatch for {gene}: workbook {uid}, R-TAD ID {rid}")
        identical_catalogue_ids = sorted(x for x in rtad_ids if all_by_id[x] == all_by_id[rid])
        if identical_catalogue_ids != [rid]:
            fail(
                f"sequence for documented R-TAD {rid} does not map unambiguously in the "
                f"230-sequence catalogue; identical IDs={identical_catalogue_ids}"
            )
        candidate_native.append({
            "parent_id": rid, "gene": gene, "UniProt_ID": uid, "sequence": all_by_id[rid],
            "external_dataset_count": int(row["External dataset count"]),
            "RTAD_overlap_status": f"documented overlap: {rid}",
            "source_region_id": rid,
        })
    native_pool = pd.DataFrame(candidate_native)
    if native_pool.empty:
        fail("no eligible native R-TAD candidates were constructed")
    # Collapse duplicate workbook rows and enforce at most one candidate per protein.
    native_pool = (native_pool.sort_values(["UniProt_ID", "parent_id"])
                   .drop_duplicates("UniProt_ID", keep="first")
                   .drop_duplicates("gene", keep="first").reset_index(drop=True))
    if len(native_pool) < 13:
        fail(f"fewer than 13 eligible native R-TAD proteins: {len(native_pool)}")
    native_pool["class"] = "native_RTAD"
    native_pool["region_start"] = native_pool["parent_id"].str.extract(r"^[^_]+_(\d+)_\d+$", expand=False)
    native_pool["region_end"] = native_pool["parent_id"].str.extract(r"^[^_]+_\d+_(\d+)$", expand=False)
    native_pool.to_csv(out / "native_RTAD_eligible_pool.csv", index=False)

    parents: list[dict[str, object]] = []
    for _, row in native_pool.sort_values("parent_id").iterrows():
        d = descriptors(row["sequence"])
        parents.append({
            "parent_id": row["parent_id"], "gene": row["gene"], "UniProt_ID": row["UniProt_ID"],
            "class": "native_RTAD", "source": str(args.tad_selection),
            "source_region_id": row["source_region_id"], "sequence": row["sequence"], **d,
            "external_dataset_count": row["external_dataset_count"],
            "RTAD_overlap_status": row["RTAD_overlap_status"],
            "selection_method": "eligible_candidate_not_yet_selected",
            "selection_seed": np.nan,
        })

    ext = pd.read_excel(args.external_controls, sheet_name=0, header=2,
                        usecols=lambda c: c in {"Protein", "UniProt ID", "Core representative sequence"})
    ext_cols = ["Protein", "UniProt ID", "Core representative sequence"]
    require_columns(ext, ext_cols, args.external_controls)
    for gene in EXTERNAL_GENES:
        rows = ext[ext["Protein"].astype(str).str.strip() == gene]
        ann = tad[tad["Gene name"].astype(str).str.strip() == gene]
        if len(rows) != 1:
            fail(f"expected exactly one core representative for {gene} in {args.external_controls}, found {len(rows)}")
        if len(ann) != 1:
            fail(f"expected exactly one annotation row for {gene} in {args.tad_selection}, found {len(ann)}")
        erow, arow = rows.iloc[0], ann.iloc[0]
        seq = clean_sequence(erow["Core representative sequence"], gene)
        if clean_sequence(arow["Representative sequence"], f"{gene} annotation") != seq:
            fail(f"{gene} core representative does not match the supplied annotation sequence")
        if pd.notna(arow["Overlapping R-TAD region ID(s)"]) or "+ R-TAD" in str(arow["Category"]):
            fail(f"{gene} cannot be verified as non-R-TAD from supplied annotations")
        if int(arow["External dataset count"]) < 2:
            fail(f"{gene} lacks external AD support count >=2")
        uid = str(erow["UniProt ID"]).strip()
        d = descriptors(seq)
        parents.append({
            "parent_id": f"{uid}_external_AD_core", "gene": gene, "UniProt_ID": uid,
            "class": "external_AD_not_RTAD", "source": str(args.external_controls),
            "source_region_id": str(arow["Representative source ID"]), "sequence": seq, **d,
            "external_dataset_count": int(arow["External dataset count"]),
            "RTAD_overlap_status": "explicitly no overlap in supplied annotation",
            "selection_method": "prespecified_exact_core_representative", "selection_seed": np.nan,
        })

    basic = pd.read_excel(args.basic_controls, sheet_name="Basic controls", header=2,
        usecols=lambda c: c in {"Gene", "UniProt ID", "Start", "End", "Length", "Sequence"})
    basic_cols = ["Gene", "UniProt ID", "Start", "End", "Length", "Sequence"]
    require_columns(basic, basic_cols, args.basic_controls)
    for gene in BASIC_GENES:
        rows = basic[basic["Gene"].astype(str).str.strip() == gene]
        if len(rows) != 1:
            fail(f"expected exactly one populated basic control for {gene}, found {len(rows)}")
        row = rows.iloc[0]
        seq = clean_sequence(row["Sequence"], gene)
        if int(row["Length"]) != len(seq):
            fail(f"length mismatch for basic control {gene}: sheet={row['Length']}, sequence={len(seq)}")
        uid = str(row["UniProt ID"]).strip()
        d = descriptors(seq)
        parents.append({
            "parent_id": f"{uid}_{int(row['Start'])}_{int(row['End'])}_basic_AD_control",
            "gene": gene, "UniProt_ID": uid, "class": "basic_AD_control",
            "source": str(args.basic_controls),
            "source_region_id": f"{uid}_{int(row['Start'])}_{int(row['End'])}",
            "sequence": seq, **d, "external_dataset_count": np.nan,
            "RTAD_overlap_status": "not applicable; supplied basic AD-derived control",
            "selection_method": "prespecified_exact_supplied_sequence", "selection_seed": np.nan,
        })

    parent_cols = ["parent_id", "gene", "UniProt_ID", "class", "source", "source_region_id",
                   "sequence", "length", "net_charge", "NCPR", "acidic_count", "basic_count",
                   "aromatic_count", "aromatic_fraction", "external_dataset_count",
                   "RTAD_overlap_status", "selection_method", "selection_seed"]
    parent_df = pd.DataFrame(parents)[parent_cols].sort_values(["class", "parent_id"]).reset_index(drop=True)
    expected_classes = {"native_RTAD": len(native_pool), "external_AD_not_RTAD": 3, "basic_AD_control": 4}
    if parent_df["class"].value_counts().to_dict() != expected_classes:
        fail(f"TAD candidate class counts invalid: {parent_df['class'].value_counts().to_dict()}")
    if parent_df["parent_id"].duplicated().any():
        fail("duplicate TAD parent IDs")
    basic_df = parent_df[parent_df["class"] == "basic_AD_control"].copy()
    basic_df.to_csv(out / "basic_control_TAD_parents.csv", index=False)
    parent_df[parent_df["class"] == "external_AD_not_RTAD"].to_csv(out / "external_acidic_AD_parents.csv", index=False)
    parent_df = parent_df[parent_df["class"] != "basic_AD_control"].copy()
    parent_df.to_csv(out / "shuffle_TAD_candidate_parents.csv", index=False)
    configure_naming(out)

    tad_shuffle_rows = []
    tad_seeds: dict[str, int] = {}
    for row in parent_df.itertuples(index=False):
        s = seed_for("TAD_shuffle", row.parent_id, args.seed)
        tad_seeds[row.parent_id] = s
        for number, shuffled in enumerate(unique_shuffles(row.sequence, 20, s, row.parent_id), 1):
            d = descriptors(shuffled)
            comp_ok = Counter(shuffled) == Counter(row.sequence)
            wt_d = descriptors(row.sequence)
            if not comp_ok or any(d[k] != wt_d[k] for k in ["length", "net_charge", "NCPR", "aromatic_count"]):
                fail(f"TAD shuffle invariants failed for {row.parent_id}/{number}")
            tad_shuffle_rows.append({
                "parent_id": row.parent_id, "gene": row.gene,
                "variant_id": variant("TAD", row.parent_id, number),
                "shuffle_number": number, "seed": s, "shuffle_seed": s, "sequence": shuffled, "length": d["length"],
                "net_charge": d["net_charge"], "NCPR": d["NCPR"],
                "aromatic_count": d["aromatic_count"], "composition_valid": comp_ok,
                "identical_to_WT": shuffled == row.sequence, "duplicate_within_parent": False,
            })
    tad_shuffles = pd.DataFrame(tad_shuffle_rows)
    tad_shuffles["class"] = tad_shuffles["parent_id"].map(parent_df.set_index("parent_id")["class"])
    if len(tad_shuffles) != len(parent_df)*20 or not tad_shuffles["composition_valid"].all() or tad_shuffles["identical_to_WT"].any():
        fail("TAD shuffle validation failed")
    if tad_shuffles.duplicated(["parent_id", "sequence"]).any():
        fail("duplicate TAD shuffle sequence within a parent")
    tad_shuffles.to_csv(out / "shuffle_TAD_variants.csv", index=False)
    write_fasta(tad_shuffles, "variant_id", "sequence", out / "shuffle_TAD_variants.fasta")

    rgg_shuffle_rows = []
    rgg_seeds: dict[str, int] = {}
    for row in eligible_rgg.itertuples(index=False):
        s = seed_for("RGG_shuffle", row.RGG_ID, args.seed)
        rgg_seeds[row.RGG_ID] = s
        for number, shuffled in enumerate(unique_shuffles(row.sequence, 20, s, row.RGG_ID), 1):
            d = descriptors(shuffled)
            wt_d = descriptors(row.sequence)
            if Counter(shuffled) != Counter(row.sequence) or any(d[k] != wt_d[k] for k in ["length", "net_charge", "NCPR", "aromatic_count"]):
                fail(f"RGG shuffle invariants failed for {row.RGG_ID}/{number}")
            current_motifs = shuffled.count("RG")
            rgg_shuffle_rows.append({
                "RGG_parent_id": row.RGG_ID,
                "variant_id": variant("RGG", row.RGG_ID, number),
                "shuffle_number": number, "seed": s, "shuffle_seed": s, "sequence": shuffled, "length": d["length"],
                "net_charge": d["net_charge"], "NCPR": d["NCPR"],
                "WT_RG_motif_count": row.RG_motif_count,
                "shuffled_RG_motif_count": current_motifs,
                "delta_RG_motif_count": current_motifs - row.RG_motif_count,
                "composition_valid": Counter(shuffled) == Counter(row.sequence),
                "identical_to_WT": shuffled == row.sequence,
                "duplicate_within_parent": False,
            })
    rgg_shuffles = pd.DataFrame(rgg_shuffle_rows)
    expected_rgg_shuffles = len(eligible_rgg) * 20
    if len(rgg_shuffles) != expected_rgg_shuffles or not rgg_shuffles["composition_valid"].all() or rgg_shuffles["identical_to_WT"].any():
        fail("RGG shuffle validation failed")
    if rgg_shuffles.duplicated(["RGG_parent_id", "sequence"]).any():
        fail("duplicate RGG shuffle sequence within a parent")
    rgg_shuffles.to_csv(out / "shuffle_RGG_variants.csv", index=False)
    write_fasta(rgg_shuffles, "variant_id", "sequence", out / "shuffle_RGG_variants.fasta")

    selected_native = native_pool[["parent_id", "gene", "UniProt_ID"]]
    report = [
        "# Script 1 shuffle report", "",
        f"- Global seed: {args.seed}",
        f"- Seed derivation: first 32 bits of SHA256(global_seed|namespace|identifier)",
        f"- Input total sequences: {len(all_records)}",
        f"- Predicted R-TADs validated: {len(rtad_ids)}",
        f"- Compact RG regions validated: {len(rgg_records)}",
        f"- Eligible conventional RGG regions: {len(eligible_rgg)}",
        f"- Conventional-RBP proteins represented: {eligible_rgg['UniProt_ID'].nunique()}",
        f"- Native R-TAD eligible pool size (one per protein): {len(native_pool)}",
        "- No final native parents were selected in Script 1; selection occurs after exclusions in Script 3.", "",
        "## Complete eligible native R-TAD pool", "",
        selected_native.to_markdown(index=False), "",
        "## All shuffle candidates; four WT basic controls validated separately", "",
        parent_df[["parent_id", "gene", "UniProt_ID", "class"]].to_markdown(index=False), "",
        f"- TAD shuffles generated: {len(tad_shuffles)}",
        f"- RGG shuffles generated: {len(rgg_shuffles)}",
        "- All length, composition, charge, NCPR, and aromatic-count invariants passed.",
        "- WT sequences and duplicate shuffles within a parent are absent.", "",
        "- ETV4/SPI1/CREB3 core sequences and non-R-TAD annotations validated.",
        "- TCF7L2/TET3/EOMES/LEF1 supplied WT controls validated; no basic-control shuffles generated.", "",
        "## TAD parent-specific shuffle seeds", "",
        pd.DataFrame(tad_seeds.items(), columns=["parent_id", "seed"]).to_markdown(index=False), "",
        "## RGG parent-specific shuffle seeds", "",
        pd.DataFrame(rgg_seeds.items(), columns=["RGG_parent_id", "seed"]).to_markdown(index=False), "",
        "No pair combinations were created by this script.",
    ]
    (out / "SCRIPT1_SHUFFLE_GENERATION_REPORT.md").write_text("\n".join(report) + "\n")
    (out / "generation_input_audit.json").write_text(json.dumps({"global_seed": args.seed,
        "original_input_sha256": {str(p.resolve()): sha(p) for p in inputs}}, indent=2)+"\n")
    print(f"Script 1 complete: {len(parent_df)} TAD parents, {len(eligible_rgg)} RGG parents, "
          f"{len(tad_shuffles)} TAD shuffles, {len(rgg_shuffles)} RGG shuffles -> {out}")


if __name__ == "__main__":
    main()
