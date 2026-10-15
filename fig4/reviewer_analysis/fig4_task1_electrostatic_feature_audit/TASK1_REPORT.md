# Figure 4 electrostatic-feature audit (Reviewer 1, R1-11/R1-12)

## 1. Question

This audit tests whether the Figure 4 implementation's `Opposite_Charge_Number` (OCN) is just the negative product of the two segment net charges, whether normalized OCN is only length-normalized OCN, whether `Charge_Complementarity` is mathematically distinct, and whether the reported OCN–interaction-strength relationship is reproducible in the exact Figure 4D population.

## 2. Inputs

- `/Users/yuktikhanna/rtads-rgg-interactions/fig4/get_features_all.py` (definition audit only)
- `/Users/yuktikhanna/rtads-rgg-interactions/fig4/sim_pairs_added_features_data_270525.csv` (master pair-level features)
- `/Users/yuktikhanna/rtads-rgg-interactions/fig4/out_fig4/cache/Fig4d_scatter_table.csv` (authoritative Figure 4D pair list, OCN, and `B22_corr1_log10`)
- `/Users/yuktikhanna/rtads-rgg-interactions/fig4/residues.csv` (inspected as the residue-charge source; independent pair-level checks use stored charges and lengths)

The master table has 1039 rows and 1039 unique names. The Figure 4D cache has 917 rows and 917 unique names. All 917 cached names mapped uniquely and completely: **yes**.

## 3. Implemented definitions

The executable code in `get_features_all.py` defines:

- Total RGG and TAD charges as sums of residue-property `q` values (lines 270–276): \(Q_{RGG}=\sum_i q_i\), \(Q_{TAD}=\sum_j q_j\). In `residues.csv`, R/K have +1, D/E have -1, and the other listed residues have 0.
- Segment NCPR (lines 283–288): \(NCPR_{RGG}=Q_{RGG}/L_{RGG}\), \(NCPR_{TAD}=Q_{TAD}/L_{TAD}\), and `Average_NCPR` is their arithmetic mean.
- OCN in `opp_charges` (lines 190–214): \((p_R n_T+n_R p_T)-(p_R p_T+n_R n_T)\), where \(p\) and \(n\) are the magnitudes of positive and negative charge counts.
- Charge complementarity (lines 154–157): \(C=|Q_R+Q_T|-\tfrac12(|Q_R|+|Q_T|)\).
- `Total_Length` (line 412): \(L_R+L_T\).
- `Normalized_OCN` (line 417): \(OCN/(L_R+L_T)\).

## 4. Algebraic result

With \(Q_R=p_R-n_R\) and \(Q_T=p_T-n_T\),

\[
-Q_RQ_T=-(p_R-n_R)(p_T-n_T)
=-p_Rp_T+p_Rn_T+n_Rp_T-n_Rn_T
=(p_Rn_T+n_Rp_T)-(p_Rp_T+n_Rn_T)=OCN.
\]

Thus the implemented OCN is **exactly algebraically equivalent** to \(-Q_{RGG}Q_{TAD}\); this conclusion does not depend on an observed correlation.

## 5. Numerical verification

For the exact 917 Figure 4D pairs, the maximum absolute `OCN - (-Q_RGG*Q_TAD)` difference was **0**, the mean absolute difference was **0**, and **0** rows differed at `np.isclose(atol=1e-12, rtol=1e-12)`. The Pearson and Spearman coefficients between the two columns were both **0.99999999999999967** and **1**, respectively.

Normalized OCN status: **A stored column was found in 4 CSV files (1240 row occurrences, including duplicated supplementary copies) and compared with OCN/Total_Length; max absolute difference=2.2204460492503131e-16, mean absolute difference=2.3267678106811495e-17, non-equal count=0 at the stated tolerance. The value was independently reconstructed for all 917 primary pairs because the 1,039-row master table lacks the column.** Its correlation with OCN was Pearson **0.862419** and Spearman **0.968641**. This is normalization-derived dependence, not algebraic identity unless total length is constant.

## 6. Charge-complementarity comparison

The independent formula reproduced stored `Charge_Complementarity`: maximum absolute error **0**, mean absolute error **0**, and **0** non-equal rows at the stated tolerance.

Charge complementarity is a piecewise-linear function of the two net charges; OCN is their negative product. They are therefore mathematically distinct. In these 917 pairs, OCN versus charge complementarity had Pearson r = **-0.452199** (p = 2.04524e-47) and Spearman rho = **-0.580406** (p = 1.09052e-83). Normalized OCN versus charge complementarity had Pearson r = **-0.62625** and Spearman rho = **-0.657345**.

The full Pearson and Spearman matrices distinguish ordinary statistical correlations from the exact OCN/net-charge-product identity. OCN correlations with TAD charge were Pearson **-0.468904** and Spearman **-0.491666**; with RGG charge they were Pearson **0.72849** and Spearman **0.840532**.

## 7. Figure 4D reproduction

Without outlier removal, OCN versus `B22_corr1_log10` for n = **917** gave Pearson r = **0.874779963** (p = 3.46423419e-290), Spearman rho = **0.955936671** (p = 0), slope = **0.0129049109**, intercept = **2.78584748**, and R² = **0.765239983**. Replacing OCN with \(-Q_RQ_T\) produced numerically identical results: **yes**.

## 8. Reviewer-relevant conclusion

The implemented OCN is the negative product of the segment net charges. Normalized OCN is the same value divided by total sequence length. Charge complementarity is a distinct, related net-charge descriptor, not OCN under another name. None of these three definitions uses residue positions or charge ordering; accordingly, none is a true charge-patterning or residue-pairing metric. The observed correlations describe this dataset and should not be mistaken for additional algebraic identities.

## 9. Implications for subsequent analysis

Electrostatic descriptors shown here to be exactly redundant or deterministically derived should not later be interpreted as independent mechanistic discoveries. Model-feature ablation or retraining belongs to the later Figure 5 task and was not performed here.
