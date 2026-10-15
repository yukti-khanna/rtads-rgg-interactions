# LEF1 context and reviewer analyses

All outputs stay in `fig6/revision_analysis/`. Other figure folders, archived simulations, reference repositories and manuscripts are read-only. The curated `fig6_parent_master.csv` is never rebuilt from Excel by this runner. Raw outcome labels are ignored. Numerical Kd values and curated outcomes remain authoritative, including p53–G3BP1 and FOXO4–G3BP1.

## Run locally

From Terminal:

```bash
cd /Users/yuktikhanna/rtads-rgg-interactions/fig6
conda activate calvados
export MPLCONFIGDIR=/private/tmp/fig6-mpl
export PYTHONDONTWRITEBYTECODE=1
python verify_revision_analysis.py
python run_revision_analysis.py --stage all
python verify_revision_analysis.py --outputs
```

Or run stages independently:

```bash
python run_revision_analysis.py --stage inventory
python run_revision_analysis.py --stage plots
python run_revision_analysis.py --stage trajectories
python run_revision_analysis.py --stage report
```

`plots` produces the expanded main panels, ROC and threshold analyses, grouped-bootstrap statistics, synthetic-repeat reference panel, all-LEF1 context panel, matched AHD score comparisons, natural-partner WT–shuffle panel and combined intensity correlation. It performs no trajectory energy calculation, model fitting or B22 recalculation. It uses 2,000 cluster resamples per analysis; it is not an instantaneous plotting-only step.

`trajectories` streams all supplied sliced frames of eligible WT pairs against the nine LEF1 partners. Exact-sequence topology and serialized OpenMM System files are required. The supplied combined DCDs contain coordinate-transformed frames, so raw-job replica boundaries cannot be recovered by exact coordinate matching. The workflow therefore reports sensitivity across ten contiguous blocks and leaves replica SEM unavailable; these blocks are not independent replicas. Rerunning reuses completed caches if input and analysis hashes match. No independent-frame confidence intervals are reported.

The currently discovered simulation root is `/Users/yuktikhanna/calvados/fig6/run_validated_final`. Additional archived roots can be supplied without editing code:

```bash
python run_revision_analysis.py --stage trajectories --run-root /absolute/path/to/additional/run_root
```

Expected layout is `test_PAIR/298k_20.0nm/job_*/`, containing `top.pdb`, the corresponding `PAIR_20.0nm_job_N.xml`, and original DCDs. Use `--run-root` for the inventory stage too when auditing additional sources. Root discovery does not infer identities from atom counts, score agreement or filenames alone.

## Results

- `paper_figures/`: expanded outcome and fitted-Kd panels, formal audited-subset prediction/simulation accuracy and main overview.
- `supplementary_figures/`: paired predicted/simulated intensity correlation.
- `figures/`: ROC, thresholds, LEF1 reference/context, matched AHDs, WT–shuffle, and eligible trajectory/map panels. Each has PDF, SVG, 300-dpi PNG, plotted CSV and caption; the existing paper exporter uses 600 dpi.
- `tables/`: AUROC and grouped intervals, threshold held-out predictions, quantitative rank concordance, panel membership, PPI records/assay categories, boundary checks, sequence descriptors, matched trajectory differences and source hashes.
- `trajectories/inventory.csv`: missing, unresolved, ready or completed status for every eligible WT pair.
- `trajectories/cache/PAIR/`: exact provenance, frame summaries, replica means, contiguous-block sensitivity, signed residue/class energy maps and comparisons with Figure 4 energy functions.
- `REVIEWER_RESPONSE_DRAFT.md`: response text and generated statistics, with outstanding experimental work distinguished.
- `revision_overview.png`: compact supplementary overview; inspect individual exported figures at publication size as well.

## Interpretation and validation

The synthetic RRGG reference set stays separate from natural-partner Kd correlations. CSP magnitude and Kd are separate observables. NB does not imply zero attraction or an infinite Kd. No ordered NB/LB/Kd hierarchy is imposed. Thresholds are exploratory and specific to the frozen log10(-B22) scale. AHD-parent and RGG-parent bootstrap analyses are separate sensitivities, not a joint two-way clustered confidence interval.

Figure 4 potentials are reused with exact serialized per-particle charges and parameters, including terminal charges, rather than guessed residue-table values. Energies are signed kJ/mol; PBC distances are converted from Angstrom to nm. Contact cutoffs are 0.6, 0.8 and 1.0 nm. Contact-conditioned averages mean frames with at least one interchain residue contact below 0.8 nm. Attractive and repulsive signed total contributions are kept separately. They are descriptive energy components, not additive contributions to B22 or proof of binding mechanism.

The validated RDFs can be used later as an independent encounter-frequency cross-check. They are not required for the present energy/contact calculation and are not substituted for trajectory-derived contact occupancies.

Experimental paired deltas are not calculated without paired measurements and confirmed condition equivalence. Top-titration bound fractions and spectral-arrow verification still require lab information. PPI corroboration is at protein level and cannot be attributed to an experimental fragment or shuffled sequence.

## Preparation status

The inventory stage was run before the user requested local execution. Plotting, bootstrap analyses, trajectory calculations and the synthetic verification script have been prepared but have not been run by the agent. Existing figures outside this revision output are from earlier work; they are not evidence that this new workflow passed verification. Final scientific and visual checks are for the local run.
