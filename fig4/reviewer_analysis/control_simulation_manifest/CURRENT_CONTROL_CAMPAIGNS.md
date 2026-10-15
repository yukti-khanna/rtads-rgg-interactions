# Current final design: two separate NEW simulation campaigns

`simulation_pair_names_400.txt` is an immutable INPUT exclusion list, not a current output.
Historical `simulation_manifest_400.csv` and `simulation_sequences_400.fasta` describe the superseded campaign. Do not submit those again.

Run from `fig4`, in order:

```sh
python 01_make_control_shuffles.py
python 02_build_control_pair_universe.py
python 03_select_new_shuffle_400.py
python 04_select_new_basic_controls.py
```

All selection scripts support `--existing-pairs`, `--input-dir`, `--output-dir`, and `--seed`; the default exclusion is the protected historical pair list above. If exclusions change, rebuild Script 2 before running either selection script. Script 1 retains and shuffles the entire native eligible pool (34 native candidates plus three external ADs); it does not select the final 13 native parents. Script 3 selects the 13 native parents and five common RGG contexts by uniform run-status-only rejection sampling after candidate exclusions.

The shuffle campaign has 400 NEW systems: 320 TAD-order perturbations and 80 RGG-order perturbations. Its only submission files are `shuffle_pairs_400_new.txt`, `shuffle_sequences_400_new.fasta`, and `shuffle_manifest_400_new.csv`. The FASTA contains 90 identifiers. Missing WT references are listed separately in `shuffle_missing_WT_reference_pairs.txt` and are NOT included in the 400.

The default basic-control campaign has 200 NEW WT-only systems against 50 shared RGG regions from distinct proteins. Its submission files are `basic_control_pairs_200_new.txt`, `basic_control_sequences_200_new.fasta`, and `basic_control_manifest_200_new.csv`. The FASTA contains 54 identifiers.

For the alternative 100-system basic campaign, run `python 04_select_new_basic_controls.py --n-systems 100`; it writes distinct `_100_` outputs and uses 25 common RGG proteins (29 FASTA identifiers). This is an alternative campaign size, not an additional campaign to submit alongside the 200-system version.

Global seed: 20260906. Both new pair sets have zero overlap with the input exclusion list. All exclusions are exact-name matches based only on availability; no outcomes or feature rankings are used. No simulations are launched by this workflow.

Current final audit: `control_selection_audit_final.json`. Classification of the historical 400: `existing_400_pair_classification.csv` (240 acidic/native shuffle, 20 basic WT, 60 unwanted basic-shuffle, 80 other WT references). Every historical name is excluded, including unwanted basic shuffles. The targets are 400 NEW shuffle systems and 200 NEW WT-only basic systems, not combined historical-plus-new totals.

Current reports: `SCRIPT1_SHUFFLE_GENERATION_REPORT.md`, `SCRIPT3_NEW_SHUFFLE_400_REPORT.md`, and `SCRIPT4_NEW_BASIC_200_REPORT.md`. Older report/audit names describe superseded versions; use the current filenames above.

Regression tests: `python -m unittest test_control_campaign_workflow.py -v`.
