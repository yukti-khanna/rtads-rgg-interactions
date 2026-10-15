# Final Figure 1 / Figure S2 analysis pipeline

Reproducible, source-audited pipeline for the analytical panels in manuscript
Figure 1, Supplementary Figure S2, and Supplementary Tables S2–S4. It excludes
the Figure 1D graphical workflow, manuscript Figure 2, and Supplementary Figure
S3.

## Source of truth

The two CSP-containing workbooks are **not interchangeable**.

| Result | Authoritative input | Analysis retained |
|---|---|---|
| Figure 1A XRGG/CSP matrix | `inputs/tables/xrgg csps.xlsx` | Mean of numeric per-residue CSP values on rows whose residue ID begins with `@`. FOXM1 RRGG was measured at 0.58 ligand equivalents; FOXM1 KRGG/LRGG were measured at 1.00 equivalent and are scaled to 0.58 for plotting. Colour is normalized within each TF. |
| Figure 1B reference Kd | `inputs/tables/xrgg_AHD_parent_master.csv` | Curated reference Kd and uncertainty in the NMR master. |
| Figure S2B WT/shuffle analysis | `inputs/tables/xrgg_AHD_parent_master.csv` | WT and shuffle measurements in the NMR master. |
| Figure 1C | `fig1C_corrected/raw_feature_values.csv` | Four already selected raw features. Hedges' g remains window-level; CIs bootstrap parent regions (10,000 iterations). |
| Figure 1E and S2D–F | cached external-AD recurrence tables, or explicit rerun | 168-model ensemble. Final catalogue uses 168/168 positive windows before merging. 85% is sensitivity only. |
| Figure S2C | cached nine-seed BLAST audit, or explicit rerun | 485 merged predicted fragments versus nine full starting regions; AXIN1 excluded. |

### Why the curated NMR workbook differed

`Table_S2_Figure1_NMR_CURATED.xlsx` is a historical template/audit record. Its
qualitative peptide-call sheet was marked draft, and its old FOXM1 raw sheet is
an incomplete/different subset that does not reproduce Figure 1A. Neither is
the numerical source for Figure 1A.

The pipeline creates a validated Table S2 containing:

- `S2A_CSP_matrix_FINAL`, calculated from the full raw CSP workbook;
- `S2G_CSP_raw_EXACT`, a rectangular import of the full workbook;
- `S2H_NMR_master_EXACT`, the NMR master used for Kd and WT/shuffle results;
- a source-of-truth notice in the README sheet.

Historical sheets remain but are renamed as historical audit material. Missing
TF–peptide measurements remain missing; absent data are not converted into
qualitative calls.

## Requirements

- Python 3.10+
- NumPy, pandas, SciPy, Matplotlib, Pillow, and openpyxl
- BLAST+ only when `--rerun-blast` is selected

No SLURM job is needed for the standard pipeline. Run it from `fig2` in a
normal terminal. The optional external-AD rerun reads a multi-million-window
universe and may be better suited to a compute node.

## Quick start

```bash
cd /path/to/fig2
./run_final_manuscript_pipeline.sh --analysis-mode rerun
```

This reruns the light deterministic analyses (CSP/Table S2 audit and Figure 1C
parent-cluster bootstrap), then creates:

```text
outputs/final_manuscript_package/
```

It does **not** retrain classifiers, rescore candidates, recreate matched
controls, or change feature selection.

## Analysis modes

Rerun the validated analyses:

```bash
./run_final_manuscript_pipeline.sh --analysis-mode rerun
```

If that output already contains an analysis cache, use a new output directory
or explicitly replace only the cache:

```bash
./run_final_manuscript_pipeline.sh --analysis-mode rerun --overwrite-analysis
```

Reuse the validated cache without recalculating it:

```bash
./run_final_manuscript_pipeline.sh --analysis-mode reuse
```

Reuse mode never falls back to the historical CSP sheets. It stops if the
validated cache is missing.

Other useful forms:

```bash
./run_final_manuscript_pipeline.sh --analysis-mode rerun --output-dir outputs/final_package_fresh
./run_final_manuscript_pipeline.sh --analysis-mode rerun --tables-only
./run_final_manuscript_pipeline.sh --analysis-mode rerun --figures-only
```

## Optional heavy analyses

These are opt-in and are not run by the standard command.

Rerun the nine-seed BLAST audit:

```bash
./run_final_manuscript_pipeline.sh \
  --analysis-mode rerun \
  --rerun-blast \
  --threads 8
```

The query is `inputs/fasta/100_all_at2_frags_dis.fasta` (485 merged regions),
and the database contains the verified nine full starting regions, not 15-aa
seed windows.

Rerun external-AD recurrence:

```bash
./run_final_manuscript_pipeline.sh \
  --analysis-mode rerun \
  --rerun-external-ad \
  --candidate-window-fasta /absolute/path/to/reference_proteome_dis_and_tfs_15aa.fasta \
  --window-votes /absolute/path/to/candidate_vote_fractions_reconstructed.csv \
  --candidate-region-table /absolute/path/to/biological_candidate_regions.csv
```

The complete candidate-universe FASTA is explicit so the pipeline cannot
silently substitute a predicted-only FASTA. This uses reconstructed 168-model
vote fractions and sets the window-selection threshold to `1.0` (168/168).
Intermediate recurrence thresholds, including 85%, remain sensitivity results.

## Outputs

```text
outputs/final_manuscript_package/
├── analysis/
│   ├── Fig1C_cluster_bootstrap_statistics.csv
│   ├── Table_S2_Figure1_NMR_VALIDATED.xlsx
│   └── Table_S2_source_audit.json
├── figures/
│   ├── Figure1/
│   └── FigureS2/
├── supplementary_tables/
│   ├── Table_S2_Figure1_NMR_FINAL.xlsx
│   ├── Table_S3_Figure1C_Features_FINAL.xlsx
│   └── Table_S4_rAHD_Classifier_Benchmark_BLAST_FINAL.xlsx
└── manifest.json
```

Optional heavy-analysis results are kept below `analysis/`.

## Interpretation safeguards

- `rAHD` means regulatory acidic-hydrophobic disordered region.
- The final catalogue derives from 100% window-level consensus before merging
  and contains 230 regions in 190 TF proteins.
- The 100% recurrence point is the window-selection criterion; it is not
  numerically identical to final-catalogue overlap.
- The ≥85%-support set is a sensitivity set, not the final catalogue.
- Protein and region counts are not interchanged.

## Figure S2A limitation

The editable spectra montage was not supplied. Figure S2A is preserved as a
frozen crop of the manuscript reference; the analytical S2 panels are rebuilt.

## Cleanup

Nothing is deleted or moved. Inspect the figures, workbooks, source-audit JSON,
and `manifest.json` before manually archiving superseded folders. See
`KEEP_FOR_FINAL_PIPELINE.md` for the minimum retained source set.

Preview the prepared conservative cleanup list with:

```bash
python scripts/cleanup_final_branch.py
```

After validating the final package, apply exactly that list with:

```bash
python scripts/cleanup_final_branch.py --apply
```

The cleaner never selects raw inputs, final pipeline inputs, source scripts,
tracked external-dataset BLAST mappings, or `outputs/final_manuscript_package`.
Before removing the old reviewer directory, it moves the two large optional
external-recurrence inputs into `final_manuscript_pipeline/local_large_inputs/`.
That directory is intentionally Git-ignored; pass those paths to the optional
rerun command if needed.
