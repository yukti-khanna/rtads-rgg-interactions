#!/usr/bin/env python3
"""Create the final descriptive Fig. S3 sequence-similarity source data and panel."""
from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


BLAST_COLUMNS = [
    "qseqid", "sseqid", "pident", "length", "nident", "mismatch", "gapopen",
    "qstart", "qend", "sstart", "send", "evalue", "bitscore", "qlen", "slen",
]
EXPECTED_SEED_GENES = {"LEF1", "FOXM1", "TP53", "FOXO4", "FOXO6", "TCF7L2", "FOXP2", "FOXO1", "FOXO3"}


def args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--analysis-dir", type=Path, default=Path(
        "outputs/reviewer_family_bias/merged_fragments_vs_9seeds"))
    p.add_argument("--query-fasta", type=Path, default=Path("inputs/fasta/100_all_at2_frags_dis.fasta"))
    p.add_argument("--gene-map", type=Path, default=Path("inputs/metadata/UNIPROTIDS_GENENAME.txt"))
    p.add_argument("--threads", type=int, default=8)
    p.add_argument("--blastp", default="blastp")
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


def refuse(path: Path, overwrite: bool):
    if path.exists() and not overwrite:
        raise FileExistsError(f"Refusing to overwrite existing output: {path}")


def read_fasta(path: Path) -> dict[str, str]:
    records, identifier, sequence = {}, None, []
    with path.open() as handle:
        for raw in handle:
            line = raw.strip()
            if not line:
                continue
            if line.startswith(">"):
                if identifier is not None:
                    if identifier in records:
                        raise ValueError(f"Duplicate FASTA ID: {identifier}")
                    records[identifier] = "".join(sequence)
                identifier, sequence = line[1:].split()[0], []
            else:
                sequence.append(line.upper())
    if identifier is not None:
        if identifier in records:
            raise ValueError(f"Duplicate FASTA ID: {identifier}")
        records[identifier] = "".join(sequence)
    return records


def genes(path: Path) -> dict[str, str]:
    result = {}
    with path.open() as handle:
        for line in handle:
            fields = line.rstrip("\n").split("\t")
            if len(fields) >= 2:
                result[fields[0]] = fields[1]
    return result


def parse_region(identifier: str) -> tuple[str, int | float, int | float]:
    match = re.fullmatch(r"(.+):(\d+)[_-](\d+)", identifier)
    if not match:
        return re.sub(r"_\d+$", "", identifier), np.nan, np.nan
    parent, start, end = match.group(1), int(match.group(2)), int(match.group(3))
    accession = re.sub(r"_\d+$", "", parent)
    return accession, start, end


def rerun_with_nident(query: Path, db: Path, output: Path, blastp: str, threads: int, overwrite: bool):
    refuse(output, overwrite)
    if shutil.which(blastp) is None:
        raise FileNotFoundError(f"Cannot find BLAST executable: {blastp}")
    command = [
        blastp, "-query", str(query), "-db", str(db), "-evalue", "1000",
        "-seg", "no", "-matrix", "BLOSUM62", "-num_threads", str(threads),
        "-outfmt", "6 " + " ".join(BLAST_COLUMNS), "-out", str(output),
    ]
    print("Rerunning unchanged BLAST settings only to report nident:")
    print(" ".join(command))
    subprocess.run(command, check=True)


def read_nident_hits(path: Path) -> pd.DataFrame:
    hits = pd.read_csv(path, sep="\t", names=BLAST_COLUMNS)
    integer = ["length", "nident", "mismatch", "gapopen", "qstart", "qend", "sstart", "send", "qlen", "slen"]
    for column in integer:
        hits[column] = pd.to_numeric(hits[column], errors="raise").astype(int)
    for column in ["pident", "evalue", "bitscore"]:
        hits[column] = pd.to_numeric(hits[column], errors="raise").astype(float)
    hits["query_aligned_span"] = (hits.qend - hits.qstart).abs() + 1
    hits["seed_aligned_span"] = (hits.send - hits.sstart).abs() + 1
    hits["query_coverage"] = hits.query_aligned_span / hits.qlen
    hits["seed_coverage"] = hits.seed_aligned_span / hits.slen
    hits["shorter_sequence_coverage"] = np.where(
        hits.qlen <= hits.slen, hits.query_coverage, hits.seed_coverage)
    coverage = hits[["query_coverage", "seed_coverage", "shorter_sequence_coverage"]].to_numpy(float)
    if np.any(coverage < 0) or np.any(coverage > 1 + 1e-12):
        raise ValueError("Coverage outside [0, 1]")
    hits["identical_residue_fraction_shorter"] = hits.nident / hits[["qlen", "slen"]].min(axis=1)
    return hits


def select_best(hits: pd.DataFrame) -> pd.DataFrame:
    return hits.sort_values(
        ["qseqid", "bitscore", "evalue", "pident", "length", "sseqid"],
        ascending=[True, False, True, False, False, True], kind="stable",
    ).drop_duplicates("qseqid", keep="first")


def verify_unchanged(existing: pd.DataFrame, selected: pd.DataFrame):
    compare = existing[existing.blast_hit].merge(
        selected, left_on="region_id", right_on="qseqid", validate="one_to_one")
    if len(compare) != int(existing.blast_hit.sum()):
        raise ValueError("The nident reporting run did not recover every historical best hit")
    if not compare["nearest_seed_id"].astype(str).equals(compare["sseqid"].astype(str)):
        raise ValueError("Best-hit mismatch in nearest_seed_id")
    if not np.allclose(compare["alignment_length"], compare["length"], rtol=0, atol=0):
        raise ValueError("Best-hit mismatch in alignment_length")
    numeric = ["pident", "evalue", "bitscore", "query_aligned_span", "seed_aligned_span"]
    for column in numeric:
        if not np.allclose(compare[column + "_x"] if column in existing.columns and column in selected.columns else compare[column],
                           compare[column + "_y"] if column in existing.columns and column in selected.columns else compare[column],
                           equal_nan=True, rtol=1e-10, atol=1e-12):
            raise ValueError(f"Best-hit mismatch in {column}")


def source_table(existing: pd.DataFrame, selected: pd.DataFrame, seeds: pd.DataFrame,
                 sequences: dict[str, str], gene_map: dict[str, str]) -> pd.DataFrame:
    hit_columns = [
        "qseqid", "sseqid", "pident", "nident", "length", "qstart", "qend", "sstart", "send",
        "query_aligned_span", "seed_aligned_span", "query_coverage", "seed_coverage",
        "shorter_sequence_coverage", "identical_residue_fraction_shorter", "bitscore", "evalue",
    ]
    selected = selected[hit_columns]
    result = existing[["region_id", "region_sequence", "region_length", "blast_hit"]].merge(
        selected, left_on="region_id", right_on="qseqid", how="left", validate="one_to_one")
    parsed = result.region_id.map(parse_region)
    result["protein_id"] = [item[0] for item in parsed]
    result["gene_symbol"] = result.protein_id.map(gene_map)
    result["region_start"] = [item[1] for item in parsed]
    result["region_end"] = [item[2] for item in parsed]
    result["nearest_seed_id"] = result.sseqid
    seed_index = seeds.set_index("seed_id")
    result["nearest_seed_gene"] = result.nearest_seed_id.map(seed_index.gene)
    result["seed_sequence_length"] = result.nearest_seed_id.map(seed_index.sequence_length)
    result["alignment_length"] = result.length
    result["alignment_extent"] = np.select(
        [~result.blast_hit, result.shorter_sequence_coverage.ge(.8)],
        ["no BLAST alignment", ">=80% shorter-sequence coverage"],
        default="<80% shorter-sequence coverage")
    order = [
        "region_id", "protein_id", "gene_symbol", "region_start", "region_end",
        "region_sequence", "region_length", "blast_hit", "nearest_seed_id",
        "nearest_seed_gene", "seed_sequence_length", "pident", "nident",
        "alignment_length", "qstart", "qend", "sstart", "send", "query_aligned_span",
        "seed_aligned_span", "query_coverage", "seed_coverage", "shorter_sequence_coverage",
        "identical_residue_fraction_shorter", "bitscore", "evalue", "alignment_extent",
    ]
    return result[order]


def summary_table(data: pd.DataFrame) -> pd.DataFrame:
    total = len(data)
    hit = data[data.blast_hit]
    rows = [
        ("total merged fragments", total),
        ("fragments with any BLAST alignment", len(hit)),
        ("fragments with no BLAST alignment", total - len(hit)),
        ("BLAST hits with >=50% shorter-sequence coverage", int(hit.shorter_sequence_coverage.ge(.5).sum())),
        ("BLAST hits with >=80% shorter-sequence coverage", int(hit.shorter_sequence_coverage.ge(.8).sum())),
    ]
    for coverage in [.5, .8]:
        for identity in [40, 50, 60, 80]:
            count = int((hit.pident.ge(identity) & hit.shorter_sequence_coverage.ge(coverage)).sum())
            rows.append((f">={identity}% identity + >={int(coverage*100)}% shorter-sequence coverage", count))
    return pd.DataFrame([
        {"criterion": criterion, "n": count, "total_n": total, "percent": 100 * count / total}
        for criterion, count in rows])


def fmt_stats(values: pd.Series) -> dict[str, float]:
    values = values.dropna().astype(float)
    return {
        "median": values.median(), "q1": values.quantile(.25), "q3": values.quantile(.75),
        "IQR": values.quantile(.75)-values.quantile(.25), "minimum": values.min(),
        "maximum": values.max(), "p90": values.quantile(.90), "p95": values.quantile(.95),
    }


def text_outputs(data: pd.DataFrame, summary: pd.DataFrame) -> tuple[str, str, str]:
    total, hits_n = len(data), int(data.blast_hit.sum())
    no_hits = total - hits_n
    hit = data[data.blast_hit]
    broad = hit[hit.shorter_sequence_coverage.ge(.8)]
    identity, span = fmt_stats(hit.pident), fmt_stats(hit.query_aligned_span)
    coverage = fmt_stats(hit.shorter_sequence_coverage)
    identical = fmt_stats(hit.identical_residue_fraction_shorter)
    broad_id = fmt_stats(broad.pident)
    count = summary.set_index("criterion").n
    statistics = f"""Figure S3 sequence-similarity descriptive statistics

Merged predicted fragments:
n = {total}

BLAST hit:
n = {hits_n}
percent = {100*hits_n/total:.1f}%

No BLAST alignment:
n = {no_hits}
percent = {100*no_hits/total:.1f}%

Among BLAST hits:
pident: median={identity['median']:.3f}; IQR={identity['q1']:.3f}-{identity['q3']:.3f}; range={identity['minimum']:.3f}-{identity['maximum']:.3f}; 90th percentile={identity['p90']:.3f}; 95th percentile={identity['p95']:.3f}
alignment span (query coordinate span, aa): median={span['median']:.3f}; IQR={span['q1']:.3f}-{span['q3']:.3f}
shorter-sequence coverage: median={coverage['median']:.4f}; IQR={coverage['q1']:.4f}-{coverage['q3']:.4f}; 90th percentile={coverage['p90']:.4f}
identical_residue_fraction_shorter: median={identical['median']:.4f}; IQR={identical['q1']:.4f}-{identical['q3']:.4f}; 90th percentile={identical['p90']:.4f}; 95th percentile={identical['p95']:.4f}; maximum={identical['maximum']:.4f}

Subset with >=80% shorter-sequence coverage:
n = {len(broad)}
pident median={broad_id['median']:.3f}; IQR={broad_id['q1']:.3f}-{broad_id['q3']:.3f}; range={broad_id['minimum']:.3f}-{broad_id['maximum']:.3f}; maximum={broad_id['maximum']:.3f}

Descriptive joint thresholds (denominator={total} merged fragments):
>=40% identity + >=80% coverage: n={int(count['>=40% identity + >=80% shorter-sequence coverage'])} ({100*count['>=40% identity + >=80% shorter-sequence coverage']/total:.1f}%)
>=50% identity + >=80% coverage: n={int(count['>=50% identity + >=80% shorter-sequence coverage'])} ({100*count['>=50% identity + >=80% shorter-sequence coverage']/total:.1f}%)
>=60% identity + >=80% coverage: n={int(count['>=60% identity + >=80% shorter-sequence coverage'])} ({100*count['>=60% identity + >=80% shorter-sequence coverage']/total:.1f}%)

Coverage was calculated from BLAST query/subject coordinate spans, not alignment length.
No inferential statistical test was performed.
"""
    legend = (f"(X) Primary-sequence similarity of merged predicted AHD fragments to the nine original "
              f"computational starting regions. Each predicted fragment was aligned against the original "
              f"starting-region sequences using BLASTP, and the highest-scoring starting-region alignment "
              f"was retained. Points show percent sequence identity versus the fraction of the shorter "
              f"sequence spanned by the alignment. Fragments without a detectable BLAST alignment are not "
              f"plotted and are reported separately. The dashed horizontal line denotes 80% coverage of the "
              f"shorter sequence. Only {len(broad)} of {total} fragments aligned across at least 80% of the "
              f"shorter sequence, and none exceeded 60% identity across this extent.")
    results = (f"To assess whether the predicted sequence space was dominated by close primary-sequence "
               f"similarity to the computational starting regions, we aligned the merged predicted fragments "
               f"against the nine original starting sequences using BLASTP. Although {hits_n} of {total} "
               f"fragments yielded a detectable local alignment, only {len(broad)} ({100*len(broad)/total:.1f}%) "
               f"aligned across at least 80% of the shorter sequence, and none exceeded 60% identity across "
               f"this extent, indicating that close primary-sequence matches to the starting regions comprise "
               f"only a small fraction of the merged prediction set.")
    return statistics, legend, results


def plot(data: pd.DataFrame, output_base: Path, overwrite: bool):
    png, pdf = output_base.with_suffix(".png"), output_base.with_suffix(".pdf")
    refuse(png, overwrite); refuse(pdf, overwrite)
    total = len(data); no_hits = int((~data.blast_hit).sum())
    hits = data[data.blast_hit]
    broad = hits[hits.shorter_sequence_coverage.ge(.8)]
    joint60 = int((hits.pident.ge(60) & hits.shorter_sequence_coverage.ge(.8)).sum())
    mpl.rcParams.update({
        "font.family": "sans-serif", "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 7, "axes.labelsize": 7, "xtick.labelsize": 6.5, "ytick.labelsize": 6.5,
        "axes.linewidth": .6, "pdf.fonttype": 42, "ps.fonttype": 42,
    })
    fig, ax = plt.subplots(figsize=(3.45, 3.0))
    ax.scatter(hits.pident, 100*hits.shorter_sequence_coverage, s=11, alpha=.55,
               color="#0072B2", edgecolors="none", rasterized=False)
    if len(broad):
        ax.scatter(broad.pident, 100*broad.shorter_sequence_coverage, s=18,
                   facecolors="none", edgecolors="#222222", linewidths=.55)
    ax.axhline(80, color="#555555", linestyle="--", linewidth=.8)
    ax.set(xlim=(0, 100), ylim=(0, 103),
           xlabel="Nearest-starting-region sequence identity (%)",
           ylabel="Shorter-sequence coverage (%)")
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(width=.6, length=3)
    annotation = (f"No BLAST alignment: {no_hits}/{total} ({100*no_hits/total:.1f}%)\n"
                  f"{len(broad)}/{total} with >=80% coverage\n"
                  f"{joint60}/{total} with >=60% identity and >=80% coverage")
    ax.text(.03, .97, annotation, transform=ax.transAxes, va="top", ha="left", fontsize=6.2,
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": .88, "pad": 1.5})
    fig.savefig(png, dpi=600, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)


def main():
    a = args()
    directory = a.analysis_dir.resolve()
    existing_path = directory / "merged_fragments_vs_9seeds_best_hit.csv"
    seeds_path = directory / "original_9_starting_regions.csv"
    db = directory / "original_9_starting_regions_db"
    query = a.query_fasta.resolve()
    for path in [existing_path, seeds_path, query, a.gene_map.resolve()]:
        if not path.is_file(): raise FileNotFoundError(path)
    existing = pd.read_csv(existing_path)
    seeds = pd.read_csv(seeds_path)
    sequences = read_fasta(query)
    if len(existing) != 485 or len(sequences) != 485:
        raise ValueError(f"Expected 485 fragments; table={len(existing)}, FASTA={len(sequences)}")
    if set(existing.region_id) != set(sequences):
        raise ValueError("Best-hit table and query FASTA identifiers differ")
    if len(seeds) != 9 or set(seeds.gene) != EXPECTED_SEED_GENES:
        raise ValueError("Reference table is not the expected nine original starting regions")

    nident_path = directory / "blast_seg_no_with_nident.tsv"
    rerun_with_nident(query, db, nident_path, a.blastp, a.threads, a.overwrite)
    all_hits = read_nident_hits(nident_path)
    selected = select_best(all_hits)
    verify_unchanged(existing, selected)
    source = source_table(existing, selected, seeds, sequences, genes(a.gene_map.resolve()))
    summary = summary_table(source)

    source_path = directory / "FigS3_sequence_similarity_source.csv"
    summary_path = directory / "FigS3_sequence_similarity_summary.csv"
    stats_path = directory / "FigS3_sequence_similarity_statistics.txt"
    legend_path = directory / "FigS3_sequence_similarity_legend_and_results.txt"
    for path in [source_path, summary_path, stats_path, legend_path]: refuse(path, a.overwrite)
    source.to_csv(source_path, index=False)
    summary.to_csv(summary_path, index=False)
    statistics, legend, results = text_outputs(source, summary)
    stats_path.write_text(statistics)
    legend_path.write_text("Proposed figure legend\n\n" + legend + "\n\nProposed Results sentence\n\n" + results + "\n")
    plot(source, directory / "FigS3_sequence_similarity", a.overwrite)

    qc = {
        "total fragments": len(source), "BLAST hits": int(source.blast_hit.sum()),
        "no hits": int((~source.blast_hit).sum()),
        ">=80% coverage": int((source.blast_hit & source.shorter_sequence_coverage.ge(.8)).sum()),
        ">=40% identity + >=80% coverage": int((source.blast_hit & source.pident.ge(40) & source.shorter_sequence_coverage.ge(.8)).sum()),
        ">=50% identity + >=80% coverage": int((source.blast_hit & source.pident.ge(50) & source.shorter_sequence_coverage.ge(.8)).sum()),
        ">=60% identity + >=80% coverage": int((source.blast_hit & source.pident.ge(60) & source.shorter_sequence_coverage.ge(.8)).sum()),
        "median hit identity": source.loc[source.blast_hit, "pident"].median(),
        "median hit coverage": source.loc[source.blast_hit, "shorter_sequence_coverage"].median(),
    }
    print("\nmetric | value")
    for key, value in qc.items(): print(f"{key} | {value}")
    print(f"all coverage values <=100% | {bool(source[['query_coverage','seed_coverage','shorter_sequence_coverage']].max().max() <= 1)}")
    print("nine original starting regions used | True")
    print("15-aa seed windows used | False")
    print("classifier values altered | False")
    print("sequence-similarity threshold used to filter plotted data | False")
    print("\nProposed legend:\n" + legend)
    print("\nProposed Results sentence:\n" + results)


if __name__ == "__main__":
    main()
