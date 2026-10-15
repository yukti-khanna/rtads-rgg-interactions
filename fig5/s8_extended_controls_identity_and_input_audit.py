from pathlib import Path
import pandas as pd, numpy as np, json, hashlib, joblib, re, sys
ROOT=Path('/Users/yuktikhanna/rtads-rgg-interactions/fig5')
OUT=ROOT/'s8_extended_controls'; OUT.mkdir(exist_ok=True)
ATT=Path('/Users/yuktikhanna/rtads-rgg-interactions/fig4/reviewer_analysis/control_simulation_manifest')
RDFDIRS=[ROOT/'controls/rdfs', ROOT/'fig5_final_model_validation_local/inputs/controls/rdfs', Path('/Users/yuktikhanna/rtads-rgg-interactions/fig4/controls_rdfs')]
FROZEN=ROOT/'outputs/final_hybrid_model/frozen_model'
FEATURES=json.loads((FROZEN/'MODEL_MANIFEST.json').read_text())['feature_order']

def sha(p):
 h=hashlib.sha256();
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''): h.update(b)
 return h.hexdigest()

def read_names(fn): return [x.strip() for x in (ATT/fn).read_text().splitlines() if x.strip()]
def parse_rdf_name(p):
 s=p.stem
 if s.startswith('rdfs_'): s=s[5:]
 if s.endswith('_20nm'): s=s[:-5]
 return s

def classify(name):
 left,right=name.split('__')
 if '_SHUF' in left: return 'RGG_shuffle'
 if '_SHUF' in right: return 'TAD_shuffle'
 if 'basic_AD_control' in right: return 'basic_WT'
 if 'external_AD_core' in right: return 'external_AD_core_WT'
 return 'WT'

def parent_for_shuffle(name):
 left,right=name.split('__')
 left_parent=re.sub(r'_SHUF\d+', '_WT', left)
 right_parent=re.sub(r'_SHUF\d+', '_WT', right)
 return f'{left_parent}__{right_parent}'

def soft_gate(x,c,s): return 1/(1+np.exp(-((np.asarray(x,float)-c)/s)))
def predict(df):
 import tensorflow as tf
 manifest=json.loads((FROZEN/'MODEL_MANIFEST.json').read_text())
 base=joblib.load(FROZEN/'base_model_ensemble.joblib')
 prep=joblib.load(FROZEN/'residual_preprocessor.joblib')
 residual=tf.keras.models.load_model(FROZEN/'residual_nn.keras')
 X=df[manifest['feature_order']].to_numpy(float)
 bp=np.mean([m.predict(base['scaler'].transform(X)) for m in base['models']],axis=0)
 raw=residual.predict(prep['scaler'].transform(X),verbose=0).reshape(-1)
 minus=df[manifest['minus_NCP_feature']].to_numpy(float)
 gate=soft_gate(minus,manifest['gate_center'],manifest['gate_scale'])
 app=manifest['lambda_final']*gate*np.clip(raw,-manifest['residual_cap'],manifest['residual_cap'])
 return pd.DataFrame({'name':df['name'].astype(str),'base_prediction_new':bp,'raw_residual_prediction_new':raw,'gate_value_new':gate,'applied_correction_new':app,'final_hybrid_prediction_new':bp+app})

new_dataset=read_names('dataset_new_pairs_to_run_140.txt')
new_non=read_names('nondataset_new_pairs_to_run_140.txt')
all_new=new_dataset+new_non
rdf_map={}
for d in RDFDIRS:
 if d.exists():
  for p in d.glob('*.pkl'):
   n=parse_rdf_name(p)
   rdf_map.setdefault(n,str(p))
existing_files=['nonbasic_WT_final_predictions.csv','shuffle_WT_parent_final_predictions.csv','nonbasic_shuffle_final_predictions.csv','histidine_pH6p5_final_predictions.csv','basic_control_final_predictions_ALL80.csv']
existing=[]
for f in existing_files:
 df=pd.read_csv(ROOT/'controls/final_predictions'/f)
 df['existing_source_file']=f
 existing.append(df)
existing=pd.concat(existing,ignore_index=True,sort=False)
# audit CSV
rows=[]
for group,names in [('new_dataset_shuffle_manifest',new_dataset),('new_nondataset_wt_manifest',new_non)]:
 for n in names:
  rows.append({'input_group':group,'name':n,'classification':classify(n),'parent_WT_pair_id':parent_for_shuffle(n) if classify(n) in ['RGG_shuffle','TAD_shuffle'] else n,'rdf_found':n in rdf_map,'rdf_path':rdf_map.get(n,''),'already_in_existing_S8':n in set(existing['name'].astype(str))})
audit=pd.DataFrame(rows)
audit.to_csv(OUT/'new_controls_input_audit.csv',index=False)
# sequence audit quick
seqs={}
for fasta in ['dataset_new_sequences_to_run.fasta','nondataset_new_sequences_to_run.fasta']:
 key=None
 for line in (ATT/fasta).read_text().splitlines():
  if line.startswith('>'):
   key=line[1:].strip().split()[0]; seqs[key]=''
  elif key: seqs[key]+=line.strip()
seq_rows=[]
for n in all_new:
 left,right=n.split('__')
 seq_rows.append({'name':n,'left_sequence_id':left,'right_sequence_id':right,'left_found':left in seqs,'right_found':right in seqs})
pd.DataFrame(seq_rows).to_csv(OUT/'new_controls_sequence_audit.csv',index=False)
# identity check
identity_lines=[]
identity_lines.append('S8 extension preflight identity and input audit')
identity_lines.append('Model artifacts checked:')
for fn in ['MODEL_MANIFEST.json','base_model_ensemble.joblib','residual_preprocessor.joblib','residual_nn.keras','predict_final_hybrid.py']:
 p=FROZEN/fn
 identity_lines.append(f'{fn}\t{p.exists()}\t{sha(p) if p.exists() else ""}')
# use existing rows with all features and predictions
check=existing.dropna(subset=FEATURES+['base_prediction','raw_residual_prediction','gate_value','applied_correction','final_hybrid_prediction']).copy()
# WT parents are intentionally present in more than one existing control file; use unique rows for model identity.
check=check.drop_duplicates(subset=['name'])
try:
 pred=predict(check[['name']+FEATURES])
 m=check.merge(pred,on='name',how='inner',validate='one_to_one')
 diffs={
  'base_prediction':float(np.max(np.abs(m['base_prediction_new']-m['base_prediction']))),
  'raw_residual_prediction':float(np.max(np.abs(m['raw_residual_prediction_new']-m['raw_residual_prediction']))),
  'gate_value':float(np.max(np.abs(m['gate_value_new']-m['gate_value']))),
  'applied_correction':float(np.max(np.abs(m['applied_correction_new']-m['applied_correction']))),
  'final_hybrid_prediction':float(np.max(np.abs(m['final_hybrid_prediction_new']-m['final_hybrid_prediction']))),
 }
 identity_lines.append(f'identity_rows_compared={len(m)}')
 identity_lines.append('identity_max_abs_diffs='+json.dumps(diffs,sort_keys=True))
 identity_lines.append('identity_pass_1e-10='+str(all(v<=1e-10 for v in diffs.values())))
except Exception as e:
 identity_lines.append('identity_check_error='+repr(e))
# summarize inputs
identity_lines += [
 f'new dataset manifest rows={len(new_dataset)} unique={len(set(new_dataset))}',
 f'new nondataset manifest rows={len(new_non)} unique={len(set(new_non))}',
 f'new supplied total={len(all_new)} unique={len(set(all_new))}',
 f'RDF found total={int(audit.rdf_found.sum())} missing={int((~audit.rdf_found).sum())}',
 'missing RDF IDs='+json.dumps(audit.loc[~audit.rdf_found,'name'].tolist()),
 f'already duplicated in existing S8={int(audit.already_in_existing_S8.sum())}',
 f'all sequences found rows={int((pd.DataFrame(seq_rows)[["left_found","right_found"]].all(axis=1)).sum())} of {len(seq_rows)}',
 'classification counts='+json.dumps(audit.classification.value_counts().to_dict(),sort_keys=True),
]
(OUT/'PREFLIGHT_STOP_AUDIT.txt').write_text('\n'.join(identity_lines)+'\n')
print('\n'.join(identity_lines))
