# Final Figure 5 package

Run `python build_final_figure_package.py` from `fig6/` to regenerate the analysis, main Figure 5, Supplementary Figures S10-S11, source-data CSVs and the single `Supplementary_Table_S11.xlsx` workbook. Add `--with-trajectories` only when the trajectory calculations need to be rebuilt; the default reuses the validated cached trajectory summaries and residue maps.

The curated `fig6_parent_master.csv` is the experimental source of truth. The build never calls the Excel importer, retrains the predictor or edits manuscript files. Panel 5A remains reserved for author-supplied spectra and the fitted curve. Panel 5B uses its established renderer unchanged.

Supplementary Table S11 uses descriptive sheets for the NMR dataset, validation statistics, AUROC and concordance, threshold sensitivity, intensity ratios, shuffle comparisons, LEF1 context, highest-titration bound-fraction fields and the data dictionary. Detailed residue-pair maps remain available as an accompanying CSV.
