#!/usr/bin/env python3
import subprocess
from pathlib import Path
import pandas as pd

def _resolve_project_root() -> Path:
    here = Path(__file__).resolve().parent
    return here.parent if here.name == "scripts" else here

PROJECT_ROOT = _resolve_project_root()
INPUT_FASTA_DIR = PROJECT_ROOT / "inputs" / "fasta"
OUTPUT_BLAST_DIR = PROJECT_ROOT / "outputs" / "blast"
OUTPUT_BLAST_DIR.mkdir(parents=True, exist_ok=True)

TAD_FASTA = INPUT_FASTA_DIR / "gsl_kotha_staller.fasta"
DATASETS = [
    {"name": "soto", "fasta": INPUT_FASTA_DIR / "soto_ADs_filtered.fasta"},
    {"name": "kothaHC", "fasta": INPUT_FASTA_DIR / "kotha_staller_high_conf.fasta"},
]

BLAST_COLS = [
    "qseqid", "sseqid", "pident", "length", "mismatch", "gapopen",
    "qstart", "qend", "sstart", "send", "evalue", "bitscore", "qlen", "slen",
]

def make_db_for_dataset(dataset):
    db_prefix = OUTPUT_BLAST_DIR / f"db_{dataset['name']}"
    if not (Path(str(db_prefix) + ".phr").exists() or Path(str(db_prefix) + ".pin").exists() or Path(str(db_prefix) + ".psq").exists()):
        cmd = ["makeblastdb", "-in", str(dataset["fasta"]), "-dbtype", "prot", "-out", str(db_prefix)]
        subprocess.run(cmd, check=True)
    return db_prefix

def run_blast(dataset_name, db_prefix):
    out_tsv = OUTPUT_BLAST_DIR / f"blast_gsl_vs_{dataset_name}.tsv"
    cmd = [
        "blastp", "-query", str(TAD_FASTA), "-db", str(db_prefix), "-evalue", "1e-3",
        "-outfmt", "6 " + " ".join(BLAST_COLS), "-out", str(out_tsv)
    ]
    subprocess.run(cmd, check=True)
    return out_tsv

def add_shorter_len_and_filter(df):
    df["shorter_len"] = df[["qlen", "slen"]].min(axis=1)
    df["qcov_short"] = (df["length"] / df["shorter_len"]) * 100.0
    keep = (df["pident"] >= 60) & (df["qcov_short"] >= 80) & (df["evalue"] <= 1e-3)
    return df.loc[keep].copy()

def same_protein(row):
    qid = str(row["qseqid"]).split(":")[0].split("-")[0]
    sid = str(row["sseqid"]).split(":")[0].split("-")[0]
    return qid == sid

def postprocess(tsv_path: Path):
    if not tsv_path.exists() or tsv_path.stat().st_size == 0:
        return
    df = pd.read_csv(tsv_path, sep="\t", names=BLAST_COLS)
    if df.empty:
        return
    df = add_shorter_len_and_filter(df)
    if df.empty:
        return
    df["same_protein"] = df.apply(same_protein, axis=1)
    df["qprot"] = df["qseqid"].astype(str).str.split(":").str[0].str.split("-").str[0]
    same_all = df[df["same_protein"]].copy()
    sig_all = same_all.copy()
    sig_best = sig_all.sort_values(["qprot", "evalue", "bitscore"], ascending=[True, True, False]).drop_duplicates("qprot")
    same_all.to_csv(tsv_path.with_name(tsv_path.stem + ".sameprot_all.tsv"), sep="\t", index=False)
    sig_all.to_csv(tsv_path.with_name(tsv_path.stem + ".sameprot_sig_all.tsv"), sep="\t", index=False)
    sig_best.to_csv(tsv_path.with_name(tsv_path.stem + ".sameprot_sig_best.tsv"), sep="\t", index=False)

def main():
    for ds in DATASETS:
        db = make_db_for_dataset(ds)
        out = run_blast(ds["name"], db)
        postprocess(out)

if __name__ == "__main__":
    main()
