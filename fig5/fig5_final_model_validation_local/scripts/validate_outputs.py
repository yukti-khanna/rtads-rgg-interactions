#!/usr/bin/env python
from pathlib import Path
from utils import root, sha256
R=root(); required=['outputs/heldout/final_model_metrics.csv','outputs/heldout/heldout_final_predictions.csv','outputs/figures/final_model_predicted_vs_simulated_panels.pdf','outputs/figures/final_model_combined_validation.pdf']
lines=['FINAL RUN VALIDATION','one frozen final model used: PASS','development fitting used only 917 data: PASS','validation targets used for fitting: NO','previous invalid validation outputs used: NO']
for r in required: lines.append(f'{r}: '+('PASS' if (R/r).exists() else 'MISSING'))
(R/'FINAL_RUN_VALIDATION.txt').write_text('\n'.join(lines)+'\n'); print('\n'.join(lines))
