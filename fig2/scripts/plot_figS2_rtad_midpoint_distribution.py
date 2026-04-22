#!/usr/bin/env python3
import argparse
import re
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd

mpl.rcParams["svg.fonttype"] = "none"

def _resolve_project_root() -> Path:
    here = Path(__file__).resolve().parent
    return here.parent if here.name == "scripts" else here

PROJECT_ROOT = _resolve_project_root()
INPUT_FASTA_DIR = PROJECT_ROOT / "inputs" / "fasta"
OUTPUT_SUPP_DIR = PROJECT_ROOT / "outputs" / "supplement"
OUTPUT_CACHE_DIR = PROJECT_ROOT / "outputs" / "cache"

OUTPUT_SUPP_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_CACHE_DIR.mkdir(parents=True, exist_ok=True)

def _default_tf_fasta() -> Path:
    candidates = [
        INPUT_FASTA_DIR / "100_tfs_at2.fasta",
        INPUT_FASTA_DIR / "100_tfs.fasta",
        INPUT_FASTA_DIR / "tf_modified_headers.fasta",
        INPUT_FASTA_DIR / "all_tfs.fasta",
        INPUT_FASTA_DIR / "seed_tads.fasta",
    ]
    for p in candidates:
        if p.is_file():
            return p
    return candidates[-1]

UNIPROT_RE = re.compile(r"(A0A[0-9A-Z]{7}|[OPQ][0-9][0-9A-Z]{3}[0-9]|[A-NR-Z][0-9][0-9A-Z]{3}[0-9][0-9A-Z]{2})")

def read_fasta(path: Path):
    records = []
    header = None
    seq = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if header is not None:
                    records.append((header, "".join(seq)))
                header = line[1:]
                seq = []
            else:
                seq.append(line)
        if header is not None:
            records.append((header, "".join(seq)))
    return records

def parse_pred_header(h: str):
    parts = h.split(":")
    pid = parts[0]
    m = re.search(r"(\d+)[_-](\d+)", h)
    if not m:
        raise ValueError(f"Could not parse coordinates from header: {h}")
    return pid, int(m.group(1)), int(m.group(2))

def get_uid(header: str):
    m = UNIPROT_RE.search(header)
    return m.group(1) if m else header.split()[0]

def main(pred_fasta: Path, tf_fasta: Path, out_prefix_length: Path, out_prefix_midpoint: Path):
    pred = read_fasta(pred_fasta)
    tfs = {get_uid(h): len(seq) for h, seq in read_fasta(tf_fasta)}

    rows = []
    for h, seq in pred:
        pid, start, end = parse_pred_header(h)
        uid = get_uid(pid)
        if uid not in tfs:
            continue
        tf_len = tfs[uid]
        mid = (start + end) / 2.0
        rows.append({
            "uid": uid,
            "start": start,
            "end": end,
            "tf_len": tf_len,
            "mid_frac": mid / tf_len,
            "rtad_len": len(seq),
        })

    df = pd.DataFrame(rows)
    if df.empty:
        raise ValueError(
            "No predicted R-TAD entries could be mapped onto TF lengths. "
            "Check that --tf-fasta points to the full-length TF FASTA for the prediction set."
        )
    out_prefix_length.parent.mkdir(parents=True, exist_ok=True)
    out_prefix_midpoint.parent.mkdir(parents=True, exist_ok=True)

    cache_tsv = OUTPUT_CACHE_DIR / "figS2D_E_rtad_positions_lengths.tsv"
    df.to_csv(cache_tsv, sep="\t", index=False)

    # Panel D: R-TAD length distribution
    fig, ax = plt.subplots(1, 1, figsize=(3.2, 2.6))
    ax.hist(df["rtad_len"], bins=20)
    ax.set_xlabel("Predicted R-TAD length (aa)")
    ax.set_ylabel("Count")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(out_prefix_length.with_suffix(".pdf"))
    fig.savefig(out_prefix_length.with_suffix(".png"), dpi=600)
    fig.savefig(out_prefix_length.with_suffix(".svg"), bbox_inches="tight")
    plt.close(fig)

    # Panel E: midpoint distribution
    fig, ax = plt.subplots(1, 1, figsize=(3.2, 2.6))
    ax.hist(df["mid_frac"], bins=20)
    ax.set_xlim(0, 1)
    ax.set_xlabel("Normalized R-TAD midpoint\n(position / TF length)")
    ax.set_ylabel("Count")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(out_prefix_midpoint.with_suffix(".pdf"))
    fig.savefig(out_prefix_midpoint.with_suffix(".png"), dpi=600)
    fig.savefig(out_prefix_midpoint.with_suffix(".svg"), bbox_inches="tight")
    plt.close(fig)

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Supplementary R-TAD length and midpoint-distribution panels.")
    ap.add_argument("--pred-fasta", type=Path, default=INPUT_FASTA_DIR / "pred_tads_frags.fasta")
    ap.add_argument("--tf-fasta", type=Path, default=_default_tf_fasta())
    ap.add_argument("--out-prefix-length", type=Path, default=OUTPUT_SUPP_DIR / "figS2D_rtad_length_distribution")
    ap.add_argument("--out-prefix-midpoint", type=Path, default=OUTPUT_SUPP_DIR / "figS2E_rtad_midpoint_distribution")
    args = ap.parse_args()
    main(args.pred_fasta, args.tf_fasta, args.out_prefix_length, args.out_prefix_midpoint)
