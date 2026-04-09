
# Figure 2 organization

This package assumes the following layout:

fig2/
├── inputs/
│   ├── fasta/
│   └── tables/
├── outputs/
│   ├── main/
│   ├── supplement/
│   ├── blast/
│   └── cache/
└── scripts/

Place raw FASTA files in `inputs/fasta/` and raw feature tables in `inputs/tables/`.
Generated BLAST outputs and filtered same-protein tables go to `outputs/blast/`.
Final main-figure panels go to `outputs/main/`.
Final supplementary panels go to `outputs/supplement/`.
Intermediate cached tables go to `outputs/cache/`.

## Scripts
- `plot_fig2_feature_and_overlap_panels.py`
  - main Fig. 2B seed-vs-random feature boxplots
  - supplement predicted-vs-random feature boxplots
  - detailed supplementary overlap plot
  - main Fig. 2D support-summary plot
- `plot_figS2_rtad_midpoint_distribution.py`
  - supplementary midpoint-distribution panel only
- `blast_predicted_rtads_vs_ad_catalogs.py`
- `blast_gsl_catalog_vs_ad_catalogs.py`
- `blast_kotha_hc_vs_soto_catalog.py`
- `blast_staller2022_vs_ad_catalogs.py`

All plotting scripts avoid titles by default. Add `--title "..."` only if you explicitly want one.
