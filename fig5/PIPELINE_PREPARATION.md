# Manuscript pipeline preparation

Run from the fig5 root, or use an absolute script path:

    python scripts/build_manuscript_outputs.py --check-only
    python scripts/build_manuscript_outputs.py

The first command validates saved predictor-export checksums. The second builds numeric panels and Tables S8/S9/S10 under outputs/manuscript. No training, feature calculation, model loading, prediction or RDF integration is performed. Display text uses rAHD; internal identifiers remain compatible.

Sources in inputs/ are snapshots of accepted workbooks and corrected CSVs. provenance/manuscript_input_manifest.json records original workbook locations. inputs/predictor/export_manifest.json matches the predictor repository's outputs/manuscript_export manifest. Controls remain only in fig5. Histidine prediction sources are retained under inputs/controls and exported separately to analyses, not manuscript tables.

Figure 4 numeric panels B/C/D, S6 control/residual panels and S7 FINCHES panels are scripted. Figure 4's manually assembled workflow panel A is not recreated automatically; the existing editable descriptor-selection artwork is preserved in inputs/predictor/artwork. Illustrator assembly remains separate. A full figure layout matching the supplied PDF has not been automatically rebuilt.

Preparation validation: Python syntax and predictor input hashes only. The complete figure/table build has NOT been executed or visually checked, as requested. The server model has NOT been loaded or substituted. No source files or trial directories were deleted.

## Unresolved provenance

1. Accepted server-model hashes differ from the local frozen model. Install unchanged server artifacts in idr-interaction-predict/models/final_frozen before inference.
2. prediction_pipeline/README.md records a global-residue-table discrepancy between training-consistent sequence features and older accepted controls. Current figures reproduce the stored control predictions, not a newly verified sequence-to-prediction calculation. This needs explicit resolution before claiming complete model consistency.
3. Archived F2 development references an absent original CV-fold file; from-scratch descriptor selection is not certified. Historical scripts are preserved, not silently repaired with different folds.

Cleanup is deferred until the build is executed, inspected and the source/dependency audit is complete.
