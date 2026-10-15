#!/usr/bin/env python
import argparse, json, numpy as np, pandas as pd
from pathlib import Path
from utils import root, ensure_dirs, FEATURES, TARGET, metrics, write_checkpoint, checkpoint_ok

def load_residual(R):
 cal=json.loads((R/'artifacts/final_model/residual_calibration.json').read_text())
 if (R/'artifacts/final_model/residual_model.keras').exists():
  import tensorflow as tf
  m=tf.keras.models.load_model(R/'artifacts/final_model/residual_model.keras')
  return cal, lambda X: m.predict(X,verbose=0).ravel()
 import joblib
 reg=joblib.load(R/'artifacts/final_model/residual_model_sklearn_fallback.pkl')
 return cal, lambda X: reg.predict(X)
def predict_file(infile,outfile):
 R=root(); df=pd.read_csv(R/infile); order=json.loads((R/'artifacts/final_model/feature_order.json').read_text())
 if order!=FEATURES: raise SystemExit('Feature order mismatch')
 import joblib
 sc=joblib.load(R/'artifacts/final_model/base_scaler.pkl'); models=joblib.load(R/'artifacts/final_model/base_model.pkl'); cal,rf=load_residual(R)
 X=sc.transform(df[FEATURES].to_numpy(float)); base=np.mean([m.predict(X) for m in models],axis=0); raw=rf(X)
 gate=1/(1+np.exp(-(df['Opposite_Charge_Number'].to_numpy(float)-cal['gate_center'])/cal['gate_scale']))
 corr=np.clip(cal['lambda_hat']*gate*raw,-cal['cap'],cal['cap']); final=base+corr
 if cal['lambda_hat']!=0 and np.allclose(base,final): raise SystemExit('Hybrid predictions equal base despite nonzero lambda')
 out=df.copy(); out['prediction_base_F2P8']=base; out['residual_raw']=raw; out['residual_gate']=gate; out['residual_lambda_hat']=cal['lambda_hat']; out['residual_applied_correction']=corr; out['prediction_final']=final
 out.to_csv(R/outfile,index=False)
def main():
 ap=argparse.ArgumentParser(); ap.add_argument('--dry-run',action='store_true'); ap.add_argument('--resume',action='store_true'); args=ap.parse_args(); ensure_dirs(); R=root()
 if args.resume and checkpoint_ok('predictions_complete'): print('prediction checkpoint exists; skipping'); return
 if args.dry_run: print('DRY RUN: would load one frozen final model and predict validation sets only.'); return
 predict_file('outputs/heldout/heldout_features.csv','outputs/heldout/heldout_final_predictions.csv')
 predict_file('outputs/histidine/histidine_features.csv','outputs/histidine/histidine_final_predictions.csv')
 predict_file('outputs/controls/control_features.csv','outputs/controls/control_final_predictions.csv')
 write_checkpoint('predictions_complete',['outputs/heldout/heldout_final_predictions.csv','outputs/histidine/histidine_final_predictions.csv','outputs/controls/control_final_predictions.csv'])
 print('Predictions written.')
if __name__=='__main__': main()
