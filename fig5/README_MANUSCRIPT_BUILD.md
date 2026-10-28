# Manuscript build

From `fig5/`, run:

```bash
python scripts/build_manuscript_outputs.py
```

The build checks the accepted predictor-export checksums, regenerates the current
manuscript figures, and creates Table S8 (predictor), Table S9 (controls), and
Table S10 (FINCHES) in `outputs/manuscript/tables`. It also compares out-of-fold
performance for the 917 development pairs with frozen-model performance for the
1,794 held-out pairs in the same −NCP bins as Figure S4D, and reports how the 917
pairs are represented relative to the training-eligible candidate pool. Those
results are written to `outputs/manuscript/performance_by_minus_NCP_bin`. It
also writes the comparison plot as Figure S6F in `outputs/manuscript/Figure_S6`.
It includes the per-bin error metrics, candidate-versus-917 representation, and a
summary standardized to each cohort’s bin mix. The bootstrap intervals are
descriptive and do not adjust for repeated protein-region identities. It
removes source-file references and historical wording from workbook text. It
uses saved predictions and accepted analysis outputs; it does not retrain the
model or rerun simulations.

Use `python scripts/build_manuscript_outputs.py --check-only` to check predictor
exports without rebuilding figures or tables. Add `--tables-only` to refresh only
the three supplementary workbooks.
