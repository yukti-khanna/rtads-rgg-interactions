#!/usr/bin/env python3
"""Generate the historical get_features.py schema for Figure 5 held-out pairs."""
from __future__ import annotations
import argparse, hashlib, importlib.util, os, subprocess, sys
from pathlib import Path
import pandas as pd

def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()

def fasta_ids(path: Path):
    ids, seqs, key = set(), {}, None
    for line in path.read_text().splitlines():
        if line.startswith(">"):
            key = line[1:].split()[0]; ids.add(key); seqs[key] = ""
        elif key:
            seqs[key] += line.strip()
    return ids, seqs

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fig5-root", type=Path, required=True)
    ap.add_argument("--patch-root", type=Path, required=True)
    ap.add_argument("--heldout-csv", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--validation-report", type=Path, required=True)
    args = ap.parse_args()
    patch = args.patch_root.resolve(); fig = args.fig5_root.resolve()
    held = pd.read_csv(args.heldout_csv, usecols=["name"])
    names = held["name"].astype(str)
    if len(names) != 1794 or names.nunique() != 1794:
        raise SystemExit(f"held-out IDs failed expected 1794 unique rows: {len(names)}, {names.nunique()}")
    fasta = patch / "features/all_tads_rggs_dis.fasta"
    ids, seqs = fasta_ids(fasta)
    missing = []
    for name in names:
        parts = name.split("__", 1)
        if len(parts) != 2 or parts[0] not in ids or parts[1] not in ids:
            missing.append(name)
    if missing:
        raise SystemExit(f"missing held-out sequence IDs ({len(missing)}): {missing[:10]}")
    work = args.output.parent / "_heldout_generation_work"
    work.mkdir(parents=True, exist_ok=True)
    names_file = work / "heldout_names.txt"; names_file.write_text("\n".join(names) + "\n")
    source = patch / "scripts/get_features.py"
    cmd = [sys.executable, str(source), "--shard", "0", "--nshards", "1", "--patch_size", "5", "--names_file", str(names_file), "--residue_csv", str(patch / "features/residues.csv"), "--fasta_file", str(fasta), "--out_folder", str(work)]
    subprocess.run(cmd, check=True)
    generated = work / "features_shard0_of_1.csv"
    df = pd.read_csv(generated)
    if len(df) != 1794 or df["name"].nunique() != 1794:
        raise SystemExit(f"generated output failed expected coverage: {df.shape}, {df['name'].nunique()}")
    df.to_csv(args.output, index=False)
    report = [
        "Held-out patch feature generation provenance",
        f"generator: {source}", f"generator_sha256: {sha256(source)}",
        f"residue_table: {patch / 'features/residues.csv'}", f"residue_sha256: {sha256(patch / 'features/residues.csv')}",
        f"fasta: {fasta}", f"fasta_sha256: {sha256(fasta)}",
        f"heldout_source: {args.heldout_csv}", f"heldout_source_sha256: {sha256(args.heldout_csv)}",
        "patch_size: 5", "schema_source: get_features.py historical 917-pair-compatible schema",
        f"rows: {len(df)}", f"unique_pairs: {df['name'].nunique()}",
        f"output: {args.output}", f"output_sha256: {sha256(args.output)}",
        f"python: {sys.version}", f"pandas: {pd.__version__}",
    ]
    args.validation_report.write_text("\n".join(report) + "\n")
    print(f"generated {len(df)} held-out rows at {args.output}")

if __name__ == "__main__": main()
