Part II-A S8 frozen-model prediction workflow

Step 1: prepare/reconcile input feature CSVs. This does not run the model.

python /Users/yuktikhanna/rtads-rgg-interactions/fig5/scripts/prepare_S8_final_prediction_inputs.py

Step 2: generate frozen-model prediction tables from those prepared inputs.

python /Users/yuktikhanna/rtads-rgg-interactions/fig5/scripts/part_IIA_generate_S8_frozen_predictions.py

Optional if the fig5 folder is elsewhere:

python /path/to/fig5/scripts/prepare_S8_final_prediction_inputs.py
python /path/to/fig5/scripts/part_IIA_generate_S8_frozen_predictions.py --fig5-root /path/to/fig5

Prepared inputs are written to:
/Users/yuktikhanna/rtads-rgg-interactions/fig5/controls/final_prediction_inputs/

Predictions are written to:
/Users/yuktikhanna/rtads-rgg-interactions/fig5/controls/final_predictions/

The prediction script intentionally stops before any S8 plotting if an S8 dataset cannot be reconciled to the exact frozen 13-feature representation. At the moment, non-dataset WT, shuffle WT parent, shuffle, and basic-control inputs are ready. Histidine pH target-only files are present, but pH-specific model-ready feature inputs are missing/ambiguous.
