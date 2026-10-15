#!/usr/bin/env python
import argparse
from utils import root, ensure_dirs, TARGET, metrics, write_checkpoint, checkpoint_ok

def main():
 ap=argparse.ArgumentParser(); ap.add_argument('--dry-run',action='store_true'); ap.add_argument('--resume',action='store_true'); args=ap.parse_args(); ensure_dirs(); R=root()
 if args.resume and checkpoint_ok('figures_complete'): print('figure checkpoint exists; skipping'); return
 if args.dry_run: print('DRY RUN: would generate final validation PDF/PNG figures.'); return
 import pandas as pd
 import matplotlib.pyplot as plt
 df=pd.read_csv(R/'outputs/heldout/heldout_final_predictions.csv'); m=metrics(df[TARGET],df['prediction_final'])
 fig,ax=plt.subplots(figsize=(4,4)); ax.scatter(df[TARGET],df['prediction_final'],s=8,alpha=.35); lo=min(df[TARGET].min(),df['prediction_final'].min()); hi=max(df[TARGET].max(),df['prediction_final'].max()); ax.plot([lo,hi],[lo,hi],color='black',lw=1); ax.set_xlabel('simulated B22_corr1_log10'); ax.set_ylabel('prediction_final'); ax.set_title(f'Held-out n={len(df)} R2={m["R2"]:.3f} RMSE={m["RMSE"]:.3f}')
 fig.tight_layout(); fig.savefig(R/'outputs/figures/final_model_predicted_vs_simulated_panels.pdf'); fig.savefig(R/'outputs/figures/final_model_predicted_vs_simulated_panels.png',dpi=300); fig.savefig(R/'outputs/figures/final_model_combined_validation.pdf')
 write_checkpoint('figures_complete',['outputs/figures/final_model_predicted_vs_simulated_panels.pdf','outputs/figures/final_model_predicted_vs_simulated_panels.png','outputs/figures/final_model_combined_validation.pdf'])
 print('Figures complete.')
if __name__=='__main__': main()
