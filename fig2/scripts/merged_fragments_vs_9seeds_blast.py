#!/usr/bin/env python3
"""BLAST merged predicted fragments against nine full starting AHD regions.

Without --run-blast, this script performs only input QC and prints the query
length summary and nine recovered starting sequences. BLAST is opt-in.
"""
from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


EXPECTED_SEEDS = {
    "Q9NQB0": "TCF7L2", "Q12778": "FOXO1", "P04637": "TP53",
    "O43524": "FOXO3", "A8MYZ6": "FOXO6", "P98177": "FOXO4",
    "Q9UJU2": "LEF1", "O15409": "FOXP2", "Q08050": "FOXM1",
}
AXIN1_ACCESSION = "O15169"
BLAST_COLUMNS = [
    "qseqid", "sseqid", "pident", "length", "mismatch", "gapopen",
    "qstart", "qend", "sstart", "send", "evalue", "bitscore", "qlen", "slen",
]
INTEGER_COLUMNS = [
    "length", "mismatch", "gapopen", "qstart", "qend", "sstart", "send", "qlen", "slen",
]
FLOAT_COLUMNS = ["pident", "evalue", "bitscore"]
METRICS = [
    "pident", "alignment_length", "query_coverage",
    "shorter_sequence_coverage", "bitscore", "evalue",
]


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--query-fasta", type=Path, required=True,
                        help="100_all_at2_frags_dis.fasta")
    parser.add_argument("--starting-fasta", type=Path, default=Path(
        "/Users/yuktikhanna/idr-sequence-predict/inputs/known_tads.fasta"))
    parser.add_argument("--gene-map", type=Path, default=Path("inputs/metadata/UNIPROTIDS_GENENAME.txt"))
    parser.add_argument("--outdir", type=Path, default=Path(
        "outputs/reviewer_family_bias/merged_fragments_vs_9seeds"))
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--blastp", default="blastp")
    parser.add_argument("--makeblastdb", default="makeblastdb")
    parser.add_argument("--run-blast", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def fasta_records(path: Path):
    identifier, sequence = None, []
    with path.open() as handle:
        for raw in handle:
            line = raw.strip()
            if not line:
                continue
            if line.startswith(">"):
                if identifier is not None:
                    yield identifier, "".join(sequence)
                identifier, sequence = line[1:].split()[0], []
            else:
                if identifier is None:
                    raise ValueError(f"Sequence before FASTA header in {path}")
                sequence.append(line.upper())
    if identifier is not None:
        yield identifier, "".join(sequence)


def load_fasta(path: Path) -> list[tuple[str, str]]:
    records = list(fasta_records(path))
    if not records:
        raise ValueError(f"No FASTA records found: {path}")
    ids = [identifier for identifier, _ in records]
    if len(ids) != len(set(ids)):
        duplicates = pd.Series(ids)[pd.Series(ids).duplicated()].unique().tolist()
        raise ValueError(f"Duplicate FASTA identifiers: {duplicates[:20]}")
    empty = [identifier for identifier, sequence in records if not sequence]
    if empty:
        raise ValueError(f"Empty FASTA sequences: {empty[:20]}")
    invalid = [(identifier, sorted(set(sequence) - set("ABCDEFGHIKLMNPQRSTVWXYZUO*")))
               for identifier, sequence in records
               if set(sequence) - set("ABCDEFGHIKLMNPQRSTVWXYZUO*")]
    if invalid:
        raise ValueError(f"Invalid amino-acid symbols: {invalid[:10]}")
    return records


def gene_map(path: Path) -> dict[str, str]:
    result = {}
    with path.open() as handle:
        for line in handle:
            fields = line.rstrip("\n").split("\t")
            if len(fields) >= 2:
                result[fields[0]] = fields[1]
    return result


def seed_parts(identifier: str) -> tuple[str, int, int]:
    match = re.fullmatch(r"(.+?)[_-](\d+)[_-](\d+)", identifier)
    if not match:
        raise ValueError(f"Starting-region identifier lacks coordinates: {identifier}")
    return match.group(1), int(match.group(2)), int(match.group(3))


def nine_seeds(path: Path, genes: dict[str, str]) -> tuple[pd.DataFrame, list[tuple[str, str]]]:
    rows, records = [], []
    for identifier, sequence in load_fasta(path):
        accession, start, end = seed_parts(identifier)
        if accession == AXIN1_ACCESSION:
            continue
        rows.append({
            "seed_id": identifier,
            "gene": genes.get(accession, EXPECTED_SEEDS.get(accession, "")),
            "protein_accession": accession,
            "coordinates": f"{start}-{end}",
            "sequence_length": len(sequence),
            "sequence": sequence,
        })
        records.append((identifier, sequence))
    table = pd.DataFrame(rows)
    if len(table) != 9:
        raise ValueError(f"Expected 9 non-AXIN1 starting regions, found {len(table)}")
    if set(table.protein_accession) != set(EXPECTED_SEEDS):
        raise ValueError("Recovered starting accessions do not match the expected nine-region set")
    for row in table.itertuples(index=False):
        if row.gene.upper() != EXPECTED_SEEDS[row.protein_accession]:
            raise ValueError(f"Unexpected gene for {row.protein_accession}: {row.gene}")
        start, end = map(int, row.coordinates.split("-"))
        if end - start + 1 != row.sequence_length:
            raise ValueError(f"Coordinate/length mismatch for {row.seed_id}")
    return table, records


def query_qc(records: list[tuple[str, str]]) -> dict[str, float | int]:
    lengths = pd.Series([len(sequence) for _, sequence in records], dtype=float)
    return {
        "n": len(records), "minimum_length": int(lengths.min()),
        "maximum_length": int(lengths.max()), "median_length": float(lengths.median()),
        "IQR_length": float(lengths.quantile(.75) - lengths.quantile(.25)),
        "Q1_length": float(lengths.quantile(.25)), "Q3_length": float(lengths.quantile(.75)),
    }


def print_preflight(query_path: Path, start_path: Path, qc: dict, seeds: pd.DataFrame) -> None:
    print("=" * 78)
    print("INPUT AUDIT — BLAST HAS NOT RUN")
    print("=" * 78)
    print(f"Merged-fragment query FASTA: {query_path.resolve()}")
    print(f"Merged fragments: {qc['n']:,}")
    print(f"Length range: {qc['minimum_length']}-{qc['maximum_length']} aa")
    print(f"Median length: {qc['median_length']:.3g} aa")
    print(f"IQR: {qc['IQR_length']:.3g} aa (Q1={qc['Q1_length']:.3g}, Q3={qc['Q3_length']:.3g})")
    print(f"\nOriginal starting-region source: {start_path.resolve()}")
    print("AXIN1 excluded: yes")
    print(seeds[["seed_id", "gene", "coordinates", "sequence_length", "sequence"]].to_string(index=False))
    print(f"Verified seed count: {len(seeds)}")
    print("\nMain BLAST settings: blastp; BLOSUM62; E-value reporting ceiling 1000; SEG off")
    print("No post-BLAST identity, coverage, or E-value filter will be applied.")
    print("Best-hit order: bitscore descending, e-value ascending, pident descending, aligned span descending")
    print("=" * 78)


def refuse(path: Path, overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(f"Refusing to overwrite {path}; add --overwrite intentionally")


def write_fasta(path: Path, records: list[tuple[str, str]], overwrite: bool) -> None:
    refuse(path, overwrite)
    with path.open("w") as handle:
        for identifier, sequence in records:
            handle.write(f">{identifier}\n{sequence}\n")


def command(parts: list[str]) -> None:
    print("RUN:", " ".join(parts), flush=True)
    subprocess.run(parts, check=True)


def run_blast(query: Path, db: Path, output: Path, seg: str, args: argparse.Namespace) -> None:
    refuse(output, args.overwrite)
    command([
        args.blastp, "-query", str(query), "-db", str(db),
        "-evalue", "1000", "-seg", seg, "-matrix", "BLOSUM62",
        "-num_threads", str(args.threads),
        "-outfmt", "6 " + " ".join(BLAST_COLUMNS), "-out", str(output),
    ])


def read_hits(path: Path) -> pd.DataFrame:
    if path.stat().st_size == 0:
        return pd.DataFrame(columns=BLAST_COLUMNS)
    hits = pd.read_csv(path, sep="\t", names=BLAST_COLUMNS)
    for column in INTEGER_COLUMNS:
        hits[column] = pd.to_numeric(hits[column], errors="raise").astype(int)
    for column in FLOAT_COLUMNS:
        hits[column] = pd.to_numeric(hits[column], errors="raise").astype(float)
    hits["query_aligned_span"] = (hits.qend - hits.qstart).abs() + 1
    hits["seed_aligned_span"] = (hits.send - hits.sstart).abs() + 1
    hits["query_coverage"] = hits.query_aligned_span / hits.qlen
    hits["seed_coverage"] = hits.seed_aligned_span / hits.slen
    hits["shorter_sequence_coverage"] = np.where(
        hits.qlen <= hits.slen, hits.query_coverage, hits.seed_coverage)
    cover = hits[["query_coverage", "seed_coverage", "shorter_sequence_coverage"]]
    if (cover.to_numpy(float) > 1 + 1e-12).any() or (cover.to_numpy(float) < 0).any():
        raise ValueError("Calculated coverage outside [0, 1]")
    return hits


def best_hits(hits: pd.DataFrame) -> pd.DataFrame:
    if hits.empty:
        return hits.copy()
    ranked = hits.sort_values(
        ["qseqid", "bitscore", "evalue", "pident", "length", "sseqid"],
        ascending=[True, False, True, False, False, True], kind="stable")
    return ranked.drop_duplicates("qseqid", keep="first").copy()


def build_best_table(query_records, seeds: pd.DataFrame, hits: pd.DataFrame) -> pd.DataFrame:
    query = pd.DataFrame(query_records, columns=["region_id", "region_sequence"])
    query["region_length"] = query.region_sequence.str.len()
    wanted = [
        "qseqid", "sseqid", "pident", "length", "query_aligned_span", "seed_aligned_span",
        "query_coverage", "seed_coverage", "shorter_sequence_coverage", "evalue", "bitscore",
    ]
    selected = best_hits(hits)[wanted] if len(hits) else pd.DataFrame(columns=wanted)
    result = query.merge(selected, left_on="region_id", right_on="qseqid", how="left", validate="one_to_one")
    seed_index = seeds.set_index("seed_id")
    result["nearest_seed_id"] = result.sseqid
    result["nearest_seed_gene"] = result.nearest_seed_id.map(seed_index.gene)
    result["seed_length"] = result.nearest_seed_id.map(seed_index.sequence_length)
    result["blast_hit"] = result.nearest_seed_id.notna()
    result["alignment_length"] = result.length
    seed_sequences = seeds.sequence.tolist()
    result["exact_or_overlapping_original_seed"] = result.region_sequence.map(
        lambda query_seq: any(query_seq == seed or query_seq in seed or seed in query_seq
                              for seed in seed_sequences))
    columns = [
        "region_id", "region_sequence", "region_length", "nearest_seed_id",
        "nearest_seed_gene", "seed_length", "pident", "alignment_length",
        "query_aligned_span", "seed_aligned_span", "query_coverage", "seed_coverage",
        "shorter_sequence_coverage", "evalue", "bitscore", "blast_hit",
        "exact_or_overlapping_original_seed",
    ]
    return result[columns]


def describe(best: pd.DataFrame, method: str) -> pd.DataFrame:
    hit = best[best.blast_hit]
    no_hits = int((~best.blast_hit).sum())
    rows = []
    for metric in METRICS:
        values = hit[metric].dropna()
        rows.append({
            "method": method, "metric": metric, "n_total": len(best),
            "n_with_blast_hit": len(hit), "n_without_blast_hit": no_hits,
            "mean": values.mean(), "SD": values.std(ddof=1), "median": values.median(),
            "IQR": values.quantile(.75)-values.quantile(.25),
            "percentile_5": values.quantile(.05), "percentile_10": values.quantile(.10),
            "percentile_25": values.quantile(.25), "percentile_50": values.quantile(.50),
            "percentile_75": values.quantile(.75), "percentile_90": values.quantile(.90),
            "percentile_95": values.quantile(.95), "maximum": values.max(),
        })
    return pd.DataFrame(rows)


def threshold_counts(best: pd.DataFrame) -> pd.DataFrame:
    hit = best[best.blast_hit]
    rows = []
    for identity in [20, 30, 40, 50, 60, 70, 80]:
        identity_mask = hit.pident.ge(identity)
        rows.append({
            "pident_threshold": identity,
            "n_pident_at_least_threshold": int(identity_mask.sum()),
            "n_with_shorter_coverage_at_least_0_5": int(
                (identity_mask & hit.shorter_sequence_coverage.ge(.5)).sum()),
            "n_with_shorter_coverage_at_least_0_8": int(
                (identity_mask & hit.shorter_sequence_coverage.ge(.8)).sum()),
            "denominator_blast_hit_fragments": len(hit),
        })
    return pd.DataFrame(rows)


def save_table(data: pd.DataFrame, path: Path, overwrite: bool, sep=",") -> None:
    refuse(path, overwrite)
    data.to_csv(path, index=False, sep=sep)


def plot_diagnostics(best: pd.DataFrame, outdir: Path) -> None:
    mpl.rcParams.update({"font.family": "sans-serif", "font.size": 8,
                         "axes.spines.top": False, "axes.spines.right": False})
    hit = best[best.blast_hit].copy()
    special = hit[hit.exact_or_overlapping_original_seed]

    fig, ax = plt.subplots(figsize=(5, 3.2))
    ax.hist(hit.pident, bins=35, color="#0072B2", edgecolor="white")
    ax.set(xlabel="Nearest-seed percent identity", ylabel="Merged fragments")
    save(fig, outdir / "diagnostic_A_pident_histogram.png")

    scatter(hit, special, "pident", "shorter_sequence_coverage", outdir / "diagnostic_B_pident_vs_shorter_coverage.png")
    scatter(hit, special, "pident", "alignment_length", outdir / "diagnostic_C_pident_vs_alignment_length.png")
    scatter(hit, special, "pident", "bitscore", outdir / "diagnostic_D_pident_vs_bitscore.png")

    fig, ax = plt.subplots(figsize=(7, 3.6))
    grouped = list(hit.groupby("nearest_seed_gene", sort=True))
    ax.boxplot([part.pident for _, part in grouped], labels=[name for name, _ in grouped], showfliers=False)
    ax.set(xlabel="Nearest starting region", ylabel="Percent identity")
    ax.tick_params(axis="x", rotation=35)
    save(fig, outdir / "diagnostic_E_pident_by_nearest_seed.png")

    fig, ax = plt.subplots(figsize=(5, 3.6))
    density = ax.hexbin(hit.pident, hit.shorter_sequence_coverage, gridsize=35,
                        mincnt=1, bins="log", cmap="viridis")
    fig.colorbar(density, ax=ax, label="log10 count")
    if len(special):
        ax.scatter(special.pident, special.shorter_sequence_coverage, marker="*", s=55,
                   color="#D55E00", edgecolor="black", linewidth=.4, label="Exact/overlapping seed")
        ax.legend(frameon=False)
    ax.set(xlabel="Percent identity", ylabel="Shorter-sequence coverage")
    save(fig, outdir / "diagnostic_F_pident_coverage_density.png")


def scatter(hit: pd.DataFrame, special: pd.DataFrame, x: str, y: str, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(5, 3.2))
    ax.scatter(hit[x], hit[y], s=9, alpha=.35, color="#0072B2", linewidths=0)
    if len(special):
        ax.scatter(special[x], special[y], marker="*", s=55, color="#D55E00",
                   edgecolor="black", linewidth=.4, label="Exact/overlapping seed")
        ax.legend(frameon=False)
    ax.set(xlabel=x.replace("_", " "), ylabel=y.replace("_", " "))
    save(fig, path)


def save(fig, path: Path) -> None:
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def seg_comparison(main: pd.DataFrame, filtered: pd.DataFrame) -> pd.DataFrame:
    paired = main[["region_id", "blast_hit", "pident", "shorter_sequence_coverage"]].merge(
        filtered[["region_id", "blast_hit", "pident", "shorter_sequence_coverage"]],
        on="region_id", suffixes=("_seg_no", "_seg_yes"), validate="one_to_one")
    both = paired[paired.blast_hit_seg_no & paired.blast_hit_seg_yes]
    return pd.DataFrame([{
        "n_fragments": len(paired), "hits_seg_no": int(paired.blast_hit_seg_no.sum()),
        "hits_seg_yes": int(paired.blast_hit_seg_yes.sum()), "paired_hits": len(both),
        "median_pident_seg_no": both.pident_seg_no.median(),
        "median_pident_seg_yes": both.pident_seg_yes.median(),
        "median_paired_pident_change_seg_yes_minus_no": (
            both.pident_seg_yes - both.pident_seg_no).median(),
        "median_shorter_coverage_seg_no": both.shorter_sequence_coverage_seg_no.median(),
        "median_shorter_coverage_seg_yes": both.shorter_sequence_coverage_seg_yes.median(),
        "median_paired_coverage_change_seg_yes_minus_no": (
            both.shorter_sequence_coverage_seg_yes - both.shorter_sequence_coverage_seg_no).median(),
    }])


def print_extremes(best: pd.DataFrame) -> None:
    columns = ["region_id", "nearest_seed_gene", "pident", "alignment_length",
               "query_coverage", "shorter_sequence_coverage", "evalue", "bitscore"]
    hits = best[best.blast_hit]
    strongest = hits.sort_values(
        ["bitscore", "evalue", "pident", "alignment_length", "region_id"],
        ascending=[False, True, False, False, True]).head(30)
    weakest = hits.sort_values(
        ["bitscore", "shorter_sequence_coverage", "pident", "region_id"],
        ascending=[True, True, True, True]).head(30)
    print("\n30 strongest detectable matches (primarily bitscore):")
    print(strongest[columns].to_string(index=False))
    print("\n30 weak detectable matches (lowest bitscore/coverage):")
    print(weakest[columns].to_string(index=False))


def print_final(qc: dict, best: pd.DataFrame, summary: pd.DataFrame,
                counts: pd.DataFrame, seg: pd.DataFrame) -> None:
    by_metric = summary.set_index("metric")
    identity = by_metric.loc["pident"]
    coverage = by_metric.loc["shorter_sequence_coverage"]
    count = counts.set_index("pident_threshold")
    print("\nFINAL DESCRIPTIVE SUMMARY")
    print(f"Merged fragments: n = {qc['n']:,}")
    print(f"length range = {qc['minimum_length']}-{qc['maximum_length']} aa")
    print(f"median length = {qc['median_length']:.3g} aa")
    print(f"BLAST hits: n = {int(identity.n_with_blast_hit):,}")
    print(f"no hits = {int(identity.n_without_blast_hit):,}")
    print(f"Nearest-seed pident: median = {identity['median']:.3g}; IQR = {identity.IQR:.3g}; "
          f"90th percentile = {identity.percentile_90:.3g}; 95th percentile = {identity.percentile_95:.3g}")
    print(f"Shorter-sequence coverage: median = {coverage['median']:.3g}; IQR = {coverage.IQR:.3g}; "
          f"90th percentile = {coverage.percentile_90:.3g}")
    for threshold in [40, 50, 60, 80]:
        row = count.loc[threshold]
        print(f">={threshold}% identity = {int(row.n_pident_at_least_threshold):,}")
        print(f">={threshold}% identity + >=50% shorter-sequence coverage = "
              f"{int(row.n_with_shorter_coverage_at_least_0_5):,}")
    print("\nSEG sensitivity:")
    print(seg.to_string(index=False))
    print_extremes(best)
    print("\nSTOP: no sequence-similarity cutoff was defined.")


def main() -> None:
    args = arguments()
    query_path = args.query_fasta.resolve()
    start_path = args.starting_fasta.resolve()
    gene_path = args.gene_map.resolve()
    for path in (query_path, start_path, gene_path):
        if not path.is_file():
            raise FileNotFoundError(path)

    queries = load_fasta(query_path)
    seeds, seed_records = nine_seeds(start_path, gene_map(gene_path))
    qc = query_qc(queries)
    print_preflight(query_path, start_path, qc, seeds)
    if not args.run_blast:
        print("AUDIT-ONLY COMPLETE: BLAST was not run. Add --run-blast after reviewing QC.")
        return

    for executable in (args.blastp, args.makeblastdb):
        if shutil.which(executable) is None:
            raise FileNotFoundError(f"Required executable not found: {executable}")
    outdir = args.outdir.resolve()
    outdir.mkdir(parents=True, exist_ok=True)
    seed_fasta = outdir / "original_9_starting_regions_no_AXIN1.fasta"
    write_fasta(seed_fasta, seed_records, args.overwrite)
    save_table(seeds, outdir / "original_9_starting_regions.csv", args.overwrite)

    db = outdir / "original_9_starting_regions_db"
    existing_db = list(outdir.glob("original_9_starting_regions_db.*"))
    if existing_db and not args.overwrite:
        raise FileExistsError(f"BLAST database already exists: {existing_db[0]}")
    command([args.makeblastdb, "-in", str(seed_fasta), "-dbtype", "prot",
             "-parse_seqids", "-out", str(db)])

    raw_main = outdir / ".blast_seg_no_native.tsv"
    raw_seg = outdir / ".blast_seg_yes_native.tsv"
    run_blast(query_path, db, raw_main, "no", args)
    run_blast(query_path, db, raw_seg, "yes", args)
    main_hits, seg_hits = read_hits(raw_main), read_hits(raw_seg)
    main_best = build_best_table(queries, seeds, main_hits)
    seg_best = build_best_table(queries, seeds, seg_hits)

    enriched_columns = BLAST_COLUMNS + ["query_aligned_span", "seed_aligned_span",
                                        "query_coverage", "seed_coverage",
                                        "shorter_sequence_coverage"]
    save_table(main_hits[enriched_columns],
               outdir / "merged_fragments_vs_9seeds_all_blast_hits.tsv", args.overwrite, sep="\t")
    save_table(main_best, outdir / "merged_fragments_vs_9seeds_best_hit.csv", args.overwrite)
    save_table(main_best[~main_best.blast_hit],
               outdir / "merged_fragments_vs_9seeds_no_hits.csv", args.overwrite)
    save_table(seg_hits[enriched_columns],
               outdir / "merged_fragments_vs_9seeds_all_blast_hits_seg_yes.tsv", args.overwrite, sep="\t")
    save_table(seg_best, outdir / "merged_fragments_vs_9seeds_best_hit_seg_yes.csv", args.overwrite)

    summary = pd.concat([describe(main_best, "historical_seg_no"),
                         describe(seg_best, "sensitivity_seg_yes")], ignore_index=True)
    counts = threshold_counts(main_best)
    seg = seg_comparison(main_best, seg_best)
    save_table(summary, outdir / "merged_fragments_vs_9seeds_descriptive_summary.csv", args.overwrite)
    save_table(counts, outdir / "merged_fragments_vs_9seeds_identity_counts.csv", args.overwrite)
    save_table(seg, outdir / "merged_fragments_vs_9seeds_SEG_sensitivity.csv", args.overwrite)
    plot_diagnostics(main_best, outdir)

    metadata = {
        "query_fasta": str(query_path), "merged_fragments": len(queries),
        "starting_fasta": str(start_path), "starting_regions": 9, "AXIN1_excluded": True,
        "main_BLAST": "blastp; BLOSUM62; SEG off; E-value reporting ceiling 1000; no post-filter",
        "best_hit_order": ["bitscore desc", "evalue asc", "pident desc", "alignment length desc"],
        "coverage": "coordinate spans, not BLAST alignment length",
        "SEG_sensitivity": "same settings with SEG on",
        "blastp_version": subprocess.check_output([args.blastp, "-version"], text=True).splitlines()[0],
        "python": sys.version,
    }
    metadata_path = outdir / "merged_fragments_vs_9seeds_metadata.json"
    refuse(metadata_path, args.overwrite)
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")
    print_final(qc, main_best, summary[summary.method.eq("historical_seg_no")], counts, seg)
    print(f"\nOutputs: {outdir}")


if __name__ == "__main__":
    main()
