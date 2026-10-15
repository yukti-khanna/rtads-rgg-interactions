#!/usr/bin/env python
import argparse, pandas as pd, json, numpy as np
from utils import root, ensure_dirs, TARGET, metrics, write_checkpoint, checkpoint_ok

def main():
 ap=argparse.ArgumentParser(); ap.add_argument('--dry-run',action='store_true'); ap.add_argument('--resume',action='store_true'); args=ap.parse_args(); ensure_dirs(); R=root()
 if args.resume and checkpoint_ok('analysis_complete'): print('analysis checkpoint exists; skipping'); return
 if args.dry_run: print('DRY RUN: would calculate final-model metrics and delta analyses; no competing model tables.'); return
 df=pd.read_csv(R/'outputs/heldout/heldout_final_predictions.csv')
 rows=[]; rows.append({'dataset':'heldout_all',**metrics(df[TARGET],df['prediction_final'])})
 hi=df[df['Opposite_Charge_Number']>=40]; rows.append({'dataset':'heldout_OCN_ge_40',**metrics(hi[TARGET],hi['prediction_final'])})
 pd.DataFrame(rows).to_csv(R/'outputs/heldout/final_model_metrics.csv',index=False)
 for sub,fn,group in [('histidine','histidine_final_predictions.csv',None),('controls','control_final_predictions.csv','condition')]:
  p=R/'outputs'/sub/fn
  if p.exists():
   d=pd.read_csv(p); rr=[{'dataset':sub+'_all',**metrics(d[TARGET],d['prediction_final'])}]
   if group and group in d.columns:
    for g,dd in d.groupby(group): rr.append({'dataset':sub+'_'+str(g),**metrics(dd[TARGET],dd['prediction_final'])})
   pd.DataFrame(rr).to_csv(R/'outputs'/sub/'final_model_metrics.csv',index=False)
 (R/'outputs/summary/final_validation_summary.txt').write_text('Final validation used one frozen F2-P8 + historical residual model. Validation data were used only for prediction/evaluation.\n')
 write_checkpoint('analysis_complete',['outputs/heldout/final_model_metrics.csv','outputs/histidine/final_model_metrics.csv','outputs/controls/final_model_metrics.csv','outputs/summary/final_validation_summary.txt'])
 print('Analysis complete.')
if __name__=='__main__': main()
