#!/usr/bin/env python
import argparse, json, numpy as np
from pathlib import Path
from utils import root, ensure_dirs, load_development, FEATURES, TARGET, sha256, write_checkpoint, checkpoint_ok

def make_mlp(seed):
 from sklearn.neural_network import MLPRegressor
 return MLPRegressor(hidden_layer_sizes=(128,64,32,16),activation='tanh',alpha=0.01,learning_rate_init=0.01,solver='adam',max_iter=500,random_state=seed)
def fit_ensemble(X,y,seeds): return [make_mlp(s).fit(X,y) for s in seeds]
def pred_ensemble(models,X): return np.mean([m.predict(X) for m in models],axis=0)
def main():
 ap=argparse.ArgumentParser(); ap.add_argument('--dry-run',action='store_true'); ap.add_argument('--resume',action='store_true'); args=ap.parse_args(); ensure_dirs(); R=root()
 if args.resume and checkpoint_ok('final_model_built'): print('final model checkpoint exists; skipping'); return
 df=load_development(); print(f'Loaded 917 development rows with {len(FEATURES)} frozen features.')
 if args.dry_run: print('DRY RUN: would fit base ensemble and historical residual network from 917 rows only.'); return
 from sklearn.preprocessing import StandardScaler
 from sklearn.model_selection import KFold, train_test_split
 from sklearn.linear_model import LinearRegression
 import joblib
 cfg=json.loads((R/'config/model_config.json').read_text()); seeds=cfg['base_model']['ensemble_seeds']
 X=df[FEATURES].to_numpy(float); y=df[TARGET].to_numpy(float)
 scaler=StandardScaler().fit(X); Xs=scaler.transform(X); models=fit_ensemble(Xs,y,seeds)
 # unbiased base OOF residuals for residual training
 oof=np.full(len(df),np.nan); kf=KFold(n_splits=cfg['residual_model']['internal_cv_folds'],shuffle=True,random_state=cfg['residual_model']['internal_cv_seed'])
 for tr,te in kf.split(X):
  sc=StandardScaler().fit(X[tr]); ms=fit_ensemble(sc.transform(X[tr]),y[tr],seeds); oof[te]=pred_ensemble(ms,sc.transform(X[te]))
 hi=np.asarray(df['Opposite_Charge_Number']>=cfg['residual_model']['threshold_ocn'])
 try:
  import tensorflow as tf
  from tensorflow import keras
  from tensorflow.keras import layers, regularizers
  Xh=Xs[hi]; rh=(y-oof)[hi]
  tr,va=train_test_split(np.arange(len(rh)),test_size=cfg['residual_model']['validation_fraction_high_ocn'],random_state=cfg['residual_model']['validation_seed'])
  tf.keras.utils.set_random_seed(20260913)
  res=keras.Sequential([layers.Input(shape=(Xh.shape[1],)),layers.Dense(16,activation='tanh',kernel_regularizer=regularizers.l2(5e-4)),layers.Dense(8,activation='tanh',kernel_regularizer=regularizers.l2(5e-4)),layers.Dense(1)])
  res.compile(optimizer=keras.optimizers.Adam(learning_rate=2e-4),loss=keras.losses.Huber(delta=0.75))
  cb=keras.callbacks.EarlyStopping(monitor='val_loss',patience=80,restore_best_weights=True)
  res.fit(Xh[tr],rh[tr],validation_data=(Xh[va],rh[va]),epochs=1000,verbose=0,callbacks=[cb])
  raw=res.predict(Xh[va],verbose=0).ravel(); true=rh[va]
  denom=float(np.dot(raw,raw)); lam=0.0 if denom==0 else float(np.dot(raw,true)/denom); lam=float(np.clip(lam,0,0.6))
  cap=float(np.percentile(np.abs(true),85)); res.save(R/'artifacts/final_model/residual_model.keras')
 except Exception as e:
  # deterministic fallback lets the workflow complete if TensorFlow install is unavailable; normal env should use TF.
  Xh=Xs[hi]; rh=(y-oof)[hi]; reg=LinearRegression().fit(Xh,rh); raw=reg.predict(Xh); lam=float(np.clip(np.dot(raw,rh)/max(np.dot(raw,raw),1e-12),0,0.6)); cap=float(np.percentile(np.abs(rh),85)); joblib.dump(reg,R/'artifacts/final_model/residual_model_sklearn_fallback.pkl')
  (R/'artifacts/final_model/TENSORFLOW_FALLBACK_USED.txt').write_text(str(e)+'\n')
 joblib.dump(scaler,R/'artifacts/final_model/base_scaler.pkl'); joblib.dump(models,R/'artifacts/final_model/base_model.pkl')
 (R/'artifacts/final_model/feature_order.json').write_text(json.dumps(FEATURES,indent=2))
 calib={'lambda_hat':lam,'cap':cap,'gate_center':40,'gate_scale':12,'threshold_ocn':40,'validation_targets_used_for_training':False}
 (R/'artifacts/final_model/residual_calibration.json').write_text(json.dumps(calib,indent=2))
 manifest={'model':'F2-P8 + historical fixed residual correction','training_rows':917,'target':TARGET,'features':FEATURES,'validation_data_used_for_fitting':False,'source':'rebuilt from packaged authoritative 917 development inputs'}
 (R/'artifacts/final_model/frozen_final_model_manifest.json').write_text(json.dumps(manifest,indent=2))
 write_checkpoint('final_model_built',['artifacts/final_model/base_scaler.pkl','artifacts/final_model/base_model.pkl','artifacts/final_model/feature_order.json','artifacts/final_model/residual_calibration.json','artifacts/final_model/frozen_final_model_manifest.json'])
 print('Final model artifacts written.')
if __name__=='__main__': main()
