# Revised Figure 5 Supplementary Data

This package builder creates the revised Figure 5 supplementary-data workbooks from existing analysis outputs. It does not retrain models, regenerate simulations, or change upstream results.

Run from the Figure 5 workspace:

```bash
cd /Users/yuktikhanna/rtads-rgg-interactions/fig5
python scripts/build_revised_fig5_supplementary_data.py \
  --out /Users/yuktikhanna/rtads-rgg-interactions/fig5/outputs/revised_fig5_supplementary_data
```

Outputs:

- `outputs/revised_fig5_supplementary_data/Table_S6_revised.xlsx`
- `outputs/revised_fig5_supplementary_data/Table_S9_reviewer_controls.xlsx`
- `outputs/revised_fig5_supplementary_data/csv/Table_S6_revised/*.csv`
- `outputs/revised_fig5_supplementary_data/csv/Table_S9_reviewer_controls/*.csv`
- `outputs/revised_fig5_supplementary_data/supplementary_data_audit.txt`

If any upstream source moved, create a JSON file with only the paths you want to override and pass it with `--paths-json`.

Example:

```json
{
  "shuffle_effects": "/absolute/path/to/shuffle_nonbasic_matched_effects.csv",
  "shuffle_noise": "/absolute/path/to/shuffle_effect_vs_replica_noise.csv",
  "shuffle_prediction_delta": "/absolute/path/to/shuffle_prediction_delta.csv"
}
```

Then run:

```bash
python scripts/build_revised_fig5_supplementary_data.py \
  --paths-json /absolute/path/to/source_paths.json \
  --out /Users/yuktikhanna/rtads-rgg-interactions/fig5/outputs/revised_fig5_supplementary_data
```

Python requirements:

```bash
pip install pandas numpy openpyxl
```

The script writes all new outputs under `/Users/yuktikhanna/rtads-rgg-interactions/fig5/outputs/revised_fig5_supplementary_data`.
