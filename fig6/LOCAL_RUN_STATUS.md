# Local execution handoff — 2026-10-01

## Latest master-table preparation

The new master-based workflow is prepared in `run_validation.py`, `validation_master.py`, `validation_statistics.py`, and `plot_master_validation.py`, with `validation_sources.json` and isolated fixtures in `verify_master_workflow.py`. It has not been executed for this request. Run `python run_validation.py --verify` locally; replacement experiments use `--experiments path/to/expanded.csv`. See `README_validation_master.md` for exact interpreter commands and mappings. The new run will create `updated_validation/validation_master.csv` and `updated_validation/paper_figures/`, without RDF/B22 recalculation or historical model recovery.

## Previous execution status

Your latest instruction is to run the workflow yourself locally. No further analysis, tests or plotting are being executed by the agent.

The previous run completed the eight numerical checks, prediction inventory, B22 calculations, expanded statistics, historical model recovery, simulation compatibility audit and complete-set diagnostics. Plot generation had started when the runner was stopped. Its plotting subprocess subsequently exited. Final report generation, expanded export verification and full visual inspection were not completed by that runner.

Generated outputs currently present are provisional. The scripts include the NumPy trapezoidal-integration compatibility fix and explicit historical/new sequence-length column names to avoid pandas merge suffix collisions. Reference repositories remain unchanged.

Run the entire workflow to regenerate a consistent set:

```bash
cd /Users/yuktikhanna/rtads-rgg-interactions/fig6
PYTHONDONTWRITEBYTECODE=1 MPLCONFIGDIR=/private/tmp/fig6-mpl \
  /Users/yuktikhanna/miniconda3/envs/calvados/bin/python3 run_validation_workflow.py
```

Inspect the resulting PNGs after execution. Automated checks do not certify visual layout. The export manifest intentionally marks visual inspection as awaiting inspection.
