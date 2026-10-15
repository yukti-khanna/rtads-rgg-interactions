# Extended S8 control analysis cluster run package

Run from the mirrored `fig5` directory after activating the cluster environment that has pandas, scipy, scikit-learn, joblib, TensorFlow/Keras, and the dependencies used by the existing Figure 5 scripts.

```bash
bash s8_extended_controls/run_on_cluster.sh
```

If the RDFs are not in one of the default project-relative locations, pass the RDF directory to the first script through the wrapper:

```bash
bash s8_extended_controls/run_on_cluster.sh --rdf-dir ../fig4/controls_rdfs
```

Default RDF locations checked by `prepare_new_s8_control_inputs.py` are:

1. `fig5/controls/rdfs`
2. `fig5/fig5_final_model_validation_local/inputs/controls/rdfs`
3. `../fig4/controls_rdfs`

The package reuses:

- RDF-to-B22 code: `fig5_final_model_validation_local/scripts/rdf_b22_functions.py`
- Feature generation: `fig5_final_model_validation_local/scripts/historical_patch_get_features.py`
- Frozen predictor artifacts: `outputs/final_hybrid_model/frozen_model/`
- Existing accepted S8/S7 control predictions under `controls/final_predictions/`

Outputs are written to `s8_extended_controls/outputs/`.

The prediction script first reruns the frozen predictor on existing accepted control rows and requires numerical identity before predicting new rows. If the cluster model artifacts are the authoritative accepted artifacts, this should pass. If it fails, the script stops to avoid mixed-provenance predictions.

Current local preflight found one missing RDF by exact name in the local copy:

`RGG_Q13595_4_1_WT__TAD_Q9UJU2_174_256_basic_AD_control_WT`

If this file exists on the cluster, the run should proceed. If not, the preparation script stops unless `--allow-missing-rdf` is passed, which is not recommended for the final analysis.
