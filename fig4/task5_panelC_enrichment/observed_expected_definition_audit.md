# Observed and expected definitions

The actual rtads fig4.py calls per_sim_AA_pair_matrices from the historical helper. The supplied IDR scripts/fig4.py is a different implementation: it requires avg_total_per_pair and would skip these legacy PKLs. It cannot reproduce the cached panel with these files.

For system s and residue positions u,v, a_suv=max(-(mean AH_suv + mean Yukawa_suv),0). Clipping occurs AFTER trajectory averaging, not on each frame. For amino-acid identities a,b, M_sab=sum(a_suv)/N_sab when that identity-pair opportunity exists, otherwise zero. O_ij=(1/917) sum_s sum_(a in class i,b in class j) M_sab. This is a sum of amino-acid-pair means, not a direct sum of every residue-pair energy and not a contact frequency. Missing amino-acid identity pairs contribute zero in the historical calculation.

E_ij=(sum_j O_ij)(sum_i O_ij)/sum_ij O_ij. This tests independence of the attractive-energy matrix marginals, not random contact expectation from sequence composition. L_ij=log2((O_ij+1e-9)/(E_ij+1e-9)). Colors are clipped at ±2; the original annotations are not clipped.

Classes: Basic R/K/H; Acidic D/E; Aromatic F/Y/W; Other remaining canonical residues. Historical orientation uses R/G fraction >=0.6; ambiguous cases fall back to res1 as RGG. Sequence matches are audited independently.

Support representation: no opportunities => not estimable; opportunities present and observed=0 => zero_observed with -Inf for E>0; positive observed and expected => estimated, with support counts and bootstrap intervals. No arbitrary minimum sample cutoff is imposed. Candidate B normalizes O by its total, preserving the historical weighting; it is not a contact probability or fraction of raw physical energy.
