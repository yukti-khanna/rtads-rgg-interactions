# B22 reproducibility/convergence report

Inputs were read from the completed Figure 4 convergence/reproducibility outputs. No simulations were run, no RDFs or B22 values were recalculated from raw trajectories, and no manuscript files were edited.

## Panel A

Five fixed representative AHD-RG/RGG pairs were ordered from weakest to strongest five-replica mean B22. The weak FZD9 RG/RGG-ZNF165 AHD and strong SYNCRIP RG/RGG-FOXO6 AHD pairs are retained from original Figure 4B. The three additional fixed representatives are FZD9 RG/RGG-SOX11 AHD, COL7A1 RG/RGG-ZNF165 AHD, and NCL RG/RGG-LEF1 AHD.

   pair_label                     pair_id     category      mean_B22         SD         SE                                            selection_reason                                                          source
  FZD9-ZNF165  O00144_1_1__P49910_193_208         weak   -424.958483 211.956612  94.789878                                     original Figure 4B pair                                             per_replica_b22.csv
   FZD9-SOX11  O00144_1_1__P35716_353_367         weak   -538.676570 197.994325  88.545754 fixed user-specified additional Panel A representative pair                                             per_replica_b22.csv
COL7A1-ZNF165 Q02388_5_13__P49910_193_208 intermediate  -7297.122748 189.428422  84.714966 fixed user-specified additional Panel A representative pair                                             per_replica_b22.csv
     NCL-LEF1    P19338_5_1__Q9UJU2_14_65 intermediate  -8780.879469 307.814740 137.658936 fixed user-specified additional Panel A representative pair                                             per_replica_b22.csv
SYNCRIP-FOXO6  O60506_3_1__A8MYZ6_430_473       strong -20730.525420 762.115916 340.828599                                     original Figure 4B pair existing final_fig4_revision load_rdfs on historical RDF pickle

The strongest and weakest representative pairs remain clearly separated across all individual replicas.

## Panel B

Exact summary values for absolute deviation from the five-replica estimate:

 k  n_combinations  median_absolute_deviation  absolute_deviation_p05  absolute_deviation_p95  fraction_within_5pct  fraction_within_10pct  fraction_within_20pct
 1            4585                 121.073486               11.599596              386.394029              0.244275               0.441439               0.710578
 2            9170                  72.685436                6.777926              244.109413              0.368593               0.634351               0.883751
 3            9170                  48.456957                4.518617              162.739608              0.514177               0.794111               0.963359
 4            4585                  30.268371                2.899899               96.598507              0.710578               0.928244               0.994766
 5             917                   0.000000                0.000000                0.000000              1.000000               1.000000               1.000000

B22 estimates progressively stabilised as additional replicas were included across the training set.

## Panel C

Exact 80->100% trajectory-length summary values for the original 15 representative pairs:

      regime  median_relative_change_80_100  n_pairs  n_below_5pct  n_below_10pct
intermediate                       0.010859        5             5              5
      strong                       0.032567        5             4              5
        weak                       0.014475        5             4              5

Across all 15 pairs, 13/15 changed by <5% and 15/15 changed by <10% between 80% and 100% of the trajectory.

Extending trajectories from 80% to 100% produced only small changes in B22 for representative weak, intermediate and strong pairs.
