# Main Figure 4 Provenance

## A

Source files: `/Users/yuktikhanna/rtads-rgg-interactions/fig4/out_fig4/Figure4_panel_a.png` and `/Users/yuktikhanna/rtads-rgg-interactions/fig4/out_fig4/Figure4_panel_a.pdf`. Historical script/cache: `fig4.py` and `out_fig4/cache/Fig4a_B22_values.csv`. Values changed: no. Status: scientifically unchanged.

## B

Exact pair IDs: `O60506_3_1__A8MYZ6_430_473` for the strong/interacting example and `O00144_1_1_P49910_193_208` for the weak/non-interacting example. RDF directory: `/Users/yuktikhanna/calvados/test_rdfs`. Exact historical replicas used: `job_0`-`job_4`, corresponding to `run_1`-`run_5` in the legacy RDF pickles. Additional runs, where present, were excluded. The central RDF curve is the five-replica mean, and the shaded band is between-replica SD. B22 uncertainty was computed from the same five stored pH 7.0 RDF replicates using the existing `rdf_b22_functions.py` correction routine. Pair provenance table: `Fig4B_pair_provenance.csv`.

## C

Task 5 source files: `sequence_oriented_observed_matrix.csv`, `sequence_oriented_expected_matrix.csv`, `sequence_oriented_log2_enrichment_matrix.csv`, `sequence_oriented_bootstrap.csv`, and `orientation_audit_final.csv`. Orientation method: sequence-oriented rows are RGG residue classes and columns are R-TAD residue classes. The Acidic-RGG x Acidic-R-TAD cell is shown as a dash because the observed attractive contribution is zero while the expected value is positive, giving unregularized enrichment of negative infinity. The historical pseudocount-like `-20.87` display value is not used.

## D

Task 3 source files: `fig4_917_sampling_membership.csv`, `heldout_23x78_features.csv`, `electrostatic_correlations_917_vs_heldout.csv`, and `heldout_clustered_bootstrap_correlations.csv`. The original set is the feature-guided 917-pair simulation panel. The independent set is the sequence-exclusive 23 TAD x 78 RGG complete cross-product, 1,794 simulated pairs, with no TAD or RGG sequence overlap with the original 917. Correlations for net-charge product: 917 Pearson r=0.874780, Spearman rho=0.955937; independent Pearson r=0.924277, Spearman rho=0.922815. Independent pair-bootstrap Pearson CI=0.914-0.934; sequence-cluster Pearson CI=0.898-0.948.
