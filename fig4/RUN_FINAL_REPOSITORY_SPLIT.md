# Local migration: Figure 3, S4 and S5

The migration script is prepared, not executed. It does not run simulations,
recalculate B22, or rebuild interaction energies. It validates the approved
tables, exports frozen sources, packages convergence/sampling sources in the
analysis repository, and exports exact approved-image raster references.

## Branches

Create a separate branch in each repository before executing:

```bash
git -C /Users/yuktikhanna/rtads-rgg-interactions switch -c codex/figure3-s4-s5-organization
git -C /Users/yuktikhanna/idr-interaction-analysis switch -c codex/convergence-sampling-organization
```

Existing uncommitted changes are not protected merely by creating a branch.
Review `git status` in both repositories first. The migration separately backs
up overwritten documentation and every explicitly removed file.

## Preview

```bash
cd /Users/yuktikhanna/rtads-rgg-interactions/fig4
conda activate calvados
python prepare_final_repository_split.py --backup-dir /Users/yuktikhanna/figure_analysis_migration_backups
```

## Execute with narrow cleanup

```bash
python prepare_final_repository_split.py \
  --backup-dir /Users/yuktikhanna/figure_analysis_migration_backups \
  --execute --cleanup
```

The exact cleanup allowlist is printed by the preview. Only those superseded
files are removed, after checksum-verified backup. All other historical code,
data, caches, and outputs remain. Do not delete entire reviewer_analysis or
task2/task3 folders: they still contain dependencies and authoritative sources.

## What is and is not produced

- Approved combined figures and individually cropped panel references, PNG/PDF.
- Correct current numbering: Figure 3 A–D; S4 A–H; S5 A–E.
- Tables S6/S7 copied intact; sheet-level CSV exports with historical notes retained.
- A file/checksum migration manifest and cleanup report.
- A fig4 README and analysis-source package README.
- No editable vector replotting: raster-reference exports are labelled accordingly.
- Uses the existing analysis-repository layout: inputs/convergence_sampling,
  scripts/verify_convergence_sampling_sources.py, config/convergence_sampling.json,
  results/b22_convergence and results/sampling. The Snakefile gains optional
  convergence targets; config.yaml is unchanged. B22 recalculation is guarded off by default.
- The old analysis-repository fig4/task2_b22_convergence bundle is copied to
  results/b22_convergence/legacy_bundle before cleanup removes its old location.
  Original scripts are also retained under scripts/historical, not enabled automatically.
- The current analysis README is preserved and extended with an idempotent section.

After migration, verify the installed sources (no B22 recalculation):

```bash
cd /Users/yuktikhanna/idr-interaction-analysis
python scripts/verify_convergence_sampling_sources.py --config config/convergence_sampling.json
```

Or run convergence separately via Snakemake:

```bash
snakemake --cores 1 b22_convergence --config include_b22_convergence=false
```

Or include it after the main workflow:

```bash
snakemake --cores 1 --config include_b22_convergence=true
```

Preview either command first by adding `-n`. No convergence target reads RDFs
or trajectories. Combined mode retains the main pipeline dependencies but fails
explicitly if the B22-producing rule would execute, rather than recalculating B22.
The installer is also available separately, if migration was already completed:

```bash
python /Users/yuktikhanna/rtads-rgg-interactions/fig4/scripts/install_convergence_snakemake.py \
  --repo /Users/yuktikhanna/idr-interaction-analysis \
  --backup-dir /Users/yuktikhanna/figure_analysis_migration_backups --execute
```

Inspect all crop boundaries after execution. 600-dpi metadata cannot add detail
to the approved input PNGs. Historical S6B/S6F notes are documented rather than
blindly renumbering scientific source workbooks or inventing a new S5 panel.
