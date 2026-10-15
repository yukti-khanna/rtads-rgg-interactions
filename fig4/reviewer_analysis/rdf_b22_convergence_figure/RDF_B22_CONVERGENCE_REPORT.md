# RDF/B₂₂ convergence report

This package reads the completed `reviewer_analysis/fig4_task2_b22_convergence` tables and historical RDF pickles. It does not run CALVADOS or recalculate RDF/B₂₂ quantities.

The script exports the authoritative replica-level, replica-number, and 15-pair trajectory tables unchanged, and calculates representative-pair RDF RMSE values relative to each five-replica mean. It also verifies the existing five-replica B₂₂ statistics through the imported tables. Run the script locally to populate the exact numerical report and figure files.

## RESULTS_SENTENCES

The five independent RDF replicas for the representative strong and weak interactions are displayed together with their five-replica means, providing a direct descriptive view of reproducibility across replicas. Between-replica B₂₂ variability is summarized across all 917 training-set pairs, and the absolute deviation from the five-replica estimate decreases as additional replicas are included. The existing 80–100% trajectory comparison for the prespecified weak, intermediate, and strong pairs quantifies the small late-trajectory changes without claiming mathematical convergence.
