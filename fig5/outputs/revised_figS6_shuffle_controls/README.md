# Updated Figure S6 shuffle controls

230 AHD shuffles (160 original + 70 new), 115 RG/RGG shuffles (80 original + 35 new), and 115 unique WT parents.

- FigS6B_shuffle_absolute_prediction: absolute simulated versus predicted shuffled-pair strength, using the previously requested primary validation view.
- FigS6B_shuffle_delta_diagnostic: updated version of the old B difference-versus-difference view; retained as a secondary diagnostic, not an additional primary panel.
- FigS6C_shuffle_delta_uncertainty: existing extended-control uncertainty analysis re-exported, with identical statistics. Its axes follow the existing uncertainty script (predicted delta on x, simulated delta on y).

PDFs are vector; PNGs are exported at 600 dpi. Existing plotting functions were reused from s8_extended_controls/S8B_absolute_shuffle_prediction_build.py and scripts/run_predictor_shuffle_uncertainty.py. Original/new/combined statistics and audited source tables retain their historical S8 filenames in this directory.

The cached uncertainty analysis already included all 345 shuffles. Pair IDs, WT parents, targets and predictions were checked against the latest source before reusing its 20,000-resample results. The displayed interval propagates finite-replica simulation uncertainty conditional on these pairs; it is not a population confidence interval.

No simulations, RDF/B22 calculations, feature generation, model inference or fitting were run. Other panels and the manuscript DOCX were not modified.
