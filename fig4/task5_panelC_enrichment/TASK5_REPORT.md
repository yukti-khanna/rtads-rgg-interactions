# Task 5 Panel C enrichment audit

Authoritative pairs: 917. Energy files mapped: 917. Stored n_jobs distribution: {'None': 917}. Exactly-five confirmed: 0. Additional-job confirmed: 0. Missing metadata means unknown, not zero jobs; the stored ensemble cannot be certified as historical five jobs. The legacy save_energies_pp function saves no job-count metadata. No trajectories were regenerated or read.

Exact acidic–acidic log2 value: -20.873768929335252. Observed: 0. Expected: 0.0019214555592987619. Maximum absolute difference from original cached matrix: 3.5527136788005009e-15. This reproduces the actual rtads implementation, which differs from the IDR script specified in the brief; see the definition audit.

Acidic–acidic opportunities occur in 656 systems, covering 8131 residue pairs. Its unregularized enrichment is -inf. The finite depletion is set by the pseudocount; opportunities exist but no net attractive contribution survives the historical transformation. This does not imply no contacts or no transient attractive frames.

Basic-RGG × acidic-TAD enrichment: 1.2984525379799186; bootstrap 95% CI [1.17578, 1.44034]. Enrichment rank: 2/16. Observed-statistic rank: 1/16. These ranks answer different questions. Positive association alone does not establish this cell as highest enrichment. Pseudocount dependence is tabulated without selecting a visually convenient value.

Orientation metadata: {'reversed': 480, 'correct': 437}. Historical axes correct by sequence matching: 502/917. No silent sequence flipping was applied to the reproduction.

Bootstrap: 10,000 simulation resamples, seed 20260910; expected marginals recomputed each draw. Extended-real quantiles use nearest empirical order statistics, preserving -Inf. This quantifies system resampling, not trajectory/job uncertainty or independence of systems sharing sequences.

Recommended representation: Candidate A, accurately labelled attractive-energy enrichment, with zero-observed cells explicitly flagged and clipped colors disclosed. Candidate B helps readers see the normalized underlying statistic without pseudocounts, but retains its amino-acid-pair weighting. Preserve the original Figure 4 pending scientific review. Histories with unknown job counts require an external manifest to establish ensemble consistency.

No residue-level contact-frequency data were found locally for the exact 917 population; the existing Figure 4C statistic is derived from attractive interaction energies calculated from simulation DCD trajectories. Contact scripts and unrelated single-system contact plots exist. The inspected calvados/run/analysis/contact_results_20nm.pkl contains lists of event durations keyed by whole simulation pairs, not residue-pair count matrices. scripts/contacts_analysis.py computes these duration lists from distance traces (default cutoff 10 Angstrom); the stored lists cannot reconstruct residue-class contact enrichment or establish the actual cutoff used.

Output directory: /Users/yuktikhanna/rtads-rgg-interactions/fig4/task5_panelC_enrichment

## Sequence-verified candidate correction

Candidates A and B use sequence-verified axes, transposing only systems whose exact sequence matches establish reversal of the historical axes. Exact-reproduction tables above remain unchanged. The original orientation is incorrect in 415 systems; therefore the original directional biological interpretation cannot be accepted directly.

Corrected basic-RGG × acidic-TAD log2 enrichment: 0.786421026, 95% bootstrap CI [0.747588904, 0.826974528]. Observed-statistic rank 1/16; enrichment rank 2/16. Positive bootstrap fraction 1. Use these corrected estimates when evaluating the directional conclusion. Candidate axes are corrected, not merely relabelled.

## Final recommendation for Figure 4C

1. Remove the −20.87 annotation from the revised panel. Keep it only in the historical reproduction. It is the pseudocount-dependent finite representation of zero observed attractive contribution with positive expected contribution; unregularized enrichment is −Inf. Display “zero observed” explicitly.
2. Call the panel **residue-class attractive interaction-energy enrichment**, not contact enrichment. Expected values derive from the energy-statistic marginals, not sequence-only contact frequencies.
3. Replace the heuristic orientation. Exact sequence matches establish 415/917 incorrect historical assignments; 480 refers only to reversed PKL storage order.
4. Use **sequence-oriented, support-aware Candidate A** (figures/candidate_A.pdf) for the revised Figure 4C, with the statistic, weighting, zero-observed treatment and ±2 color clipping explained in its caption. Candidate B remains a diagnostic of the normalized underlying quantity. Disclose unknown historical job counts.
5. Preferential Basic-RGG × Acidic-TAD contributions remain supported: observed=0.013618873588792806, expected=0.0078959587489047765, log2 enrichment=0.78642102578337614; stored 10,000-resample bootstrap CI [0.74758890387759724, 0.82697452840330743], positive fraction=1.0. This cell ranks 1/16 by observed interaction statistic and 2/16 by enrichment. Do not describe it as the highest enrichment. This supports an energy-based preference, not a measured contact-frequency claim.

Cleanup verification independently reread stored energy arrays and exact sequences, reproduced the corrected 4×4 and historical 20×20 matrices, and checked the saved bootstrap estimate against the verified central value. The existing bootstrap was not rerun. All historical matrices, candidate figures and prior analysis outputs were preserved.
