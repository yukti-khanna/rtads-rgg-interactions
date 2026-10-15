# Orientation count reconciliation

“Reversed” in orientation_audit.csv describes the **stored PKL order**: res1 is the TAD sequence and res2 is the RGG sequence, established by exact full-sequence identity against authoritative pair metadata and FASTA. This occurs in 480/917 files. The other 437 store res1=RGG and res2=TAD. Stored order alone is not an analysis error.

“Incorrect historical orientation” describes the **axes after the historical heuristic**. The heuristic chooses res2 as RGG only when its R/G fraction is at least 0.6 and res1's is below 0.6; otherwise it chooses res1. The historical assignment disagrees with the exact sequence assignment in 415/917 systems. Thus 502 historical assignments are correct.

Cross-tabulation (rows: sequence-verified RGG side; columns: historical RGG side):

historical_RGG_side  res1  res2
corrected_RGG_side             
res1                  437     0
res2                  415    65

The off-diagonal counts are errors. Quote **415/917 historical Figure 4C axis assignments were incorrect** when discussing the panel problem. Quote 480 only when describing PKL storage order. Corrected assignments use the rgg/tad IDs in the exact 917 pair definitions and full-sequence matching, never the R/G-content heuristic. All 917 matches are unambiguous.
