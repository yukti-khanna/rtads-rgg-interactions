#!/usr/bin/env python
import argparse, json, re, pickle, sys
from pathlib import Path
import numpy as np, pandas as pd
from utils import root, ensure_dirs, derive_means, read_csv, require_features, FEATURES, TARGET, write_checkpoint, checkpoint_ok, parse_pair_from_rdf_stem, config, resolve_input_path

def merge_heldout():
 base=derive_means(read_csv('inputs/heldout/taken_out_preds_with_hybrid_021125_sel_cols.csv')); patch=read_csv('inputs/heldout/features_heldout_sequence_exclusive.csv')
 if len(base)!=1794 or len(patch)!=1794: raise SystemExit('Heldout row count must be 1794')
 cols=['opp_charge_std','scd_patch1_std','scd_patch2_std','rg_fraction_patch1_std','aromatic_fraction_patch2_std','shd_patch1_std','shd_patch2_std','ah_energy_std']
 df=base.merge(patch[['name']+cols],on='name',how='left',validate='one_to_one'); require_features(df); return df
def b22_from_rdf_dir(rdf_dir, out_name):
 R=root(); sys.path.insert(0,str(R/'scripts'))
 rdf_path=resolve_input_path(rdf_dir)
 if not rdf_path.exists(): raise SystemExit(f'RDF directory not found: {rdf_path}')
 import pandas as pd
 from rdf_b22_functions import corrected_rdfs_b22
 rows=[]
 for f in sorted(rdf_path.glob('*.pkl')):
  try:
   df=pd.read_pickle(f); _, b22s, _=corrected_rdfs_b22(df, 7.5, box_size=20, delta=2, rc_nm=4.0)
   vals=[]
   for v in b22s.values():
    if 'b22_corr1' in v: vals.append(v['b22_corr1'])
    elif 'Corrected_B22_Hummer' in v: vals.append(v['Corrected_B22_Hummer'])
   vals=[float(x) for x in vals if np.isfinite(float(x))]
   if vals:
    m=float(np.mean(vals)); rows.append({'name':parse_pair_from_rdf_stem(f.stem),'Corrected_B22_Hummer':m,'B22_corr1_log10':np.log10(-m) if m<0 else np.nan,'rdf_file':str(f)})
  except Exception as e:
   rows.append({'name':parse_pair_from_rdf_stem(f.stem),'Corrected_B22_Hummer':np.nan,'B22_corr1_log10':np.nan,'rdf_file':str(f),'rdf_error':str(e)})
 out=R/out_name
 dfout=pd.DataFrame(rows, columns=['name','Corrected_B22_Hummer','B22_corr1_log10','rdf_file','rdf_error'])
 dfout.to_csv(out,index=False)
 good=dfout[np.isfinite(dfout['B22_corr1_log10'])].copy()
 if good.empty: raise SystemExit(f'No valid B22_corr1 values could be extracted from {rdf_dir}; see {out_name}')
 return good

def read_fasta(path):
 d={}; key=None
 for line in open(path):
  line=line.strip()
  if not line: continue
  if line.startswith('>'):
   key=line[1:].split()[0]; d[key]=''
  elif key: d[key]+=line
 return d
def write_fasta(records,path):
 with open(path,'w') as f:
  for k,v in records.items(): f.write(f'>{k}\n{v}\n')
def run_historical_patch(names, fasta, out_csv):
 import subprocess, sys
 R=root(); tmp=R/'outputs/_feature_work'; tmp.mkdir(parents=True,exist_ok=True)
 names_file=tmp/(out_csv.replace('.csv','_names.txt'))
 names_file.write_text('\n'.join(names)+'\n')
 cmd=[sys.executable,str(R/'scripts/historical_patch_get_features.py'),'--names_file',str(names_file),'--residue_csv',str(R/'inputs/histidine/metadata/residues.csv'),'--fasta_file',str(fasta),'--out_folder',str(tmp),'--out_csv',out_csv]
 subprocess.run(cmd,check=True)
 return pd.read_csv(tmp/out_csv)
def build_control_fasta_and_names():
 R=root(); man=pd.read_csv(R/'inputs/controls/manifest/simulation_manifest_400.csv')
 records={}; names=[]
 for _,r in man.iterrows():
  records[str(r.TAD_variant_id)]=str(r.TAD_sequence); records[str(r.RGG_variant_id)]=str(r.RGG_sequence)
  names.append(f'{r.RGG_variant_id}__{r.TAD_variant_id}')
 fasta=R/'outputs/controls/control_sequences_for_features.fasta'; write_fasta(records,fasta)
 mapdf=man.copy(); mapdf['feature_name']=[f'{r.RGG_variant_id}__{r.TAD_variant_id}' for _,r in man.iterrows()]
 return fasta,names,mapdf
def prepare_special_features(kind,b22):
 R=root(); cols=['name','B22_corr1_log10']
 if kind=='histidine':
  fasta=R/'inputs/histidine/metadata/all_tads_rggs_dis.fasta'; seqs=read_fasta(fasta); names=[]
  for n in b22['name']:
   if '__' in n:
    a,b=n.split('__',1)
    if a in seqs and b in seqs: names.append(n)
  feats=run_historical_patch(names,fasta,'histidine_generated_features.csv')
  out=b22.merge(feats,on='name',how='inner',validate='one_to_one')
  out=derive_means(out); require_features(out)
  out.to_csv(R/'outputs/histidine/histidine_features.csv',index=False)
  return 'outputs/histidine/histidine_features.csv'
 if kind=='controls':
  fasta,names,mapdf=build_control_fasta_and_names(); feats=run_historical_patch(names,fasta,'control_generated_features.csv')
  rev=mapdf[['name','feature_name','condition','TAD_variant_type','RGG_variant_type','TAD_parent_id','RGG_parent_id']]
  feats=feats.merge(rev,left_on='name',right_on='feature_name',how='left',validate='one_to_one')
  feats=feats.drop(columns=['name']).rename(columns={'name_y':'name'}) if 'name_y' in feats.columns else feats
  feats['name']=feats['name'] if 'name' in feats else feats['feature_name']
  out=b22.merge(feats,on='name',how='inner',validate='one_to_one') if set(b22['name']).issubset(set(feats['name'])) else feats.merge(b22,left_on='feature_name',right_on='name',how='inner',validate='one_to_one')
  if 'name_x' in out.columns: out=out.rename(columns={'name_x':'feature_name','name_y':'name'})
  out=derive_means(out); require_features(out)
  out.to_csv(R/'outputs/controls/control_features.csv',index=False)
  return 'outputs/controls/control_features.csv'

def main():
 ap=argparse.ArgumentParser(); ap.add_argument('--dry-run',action='store_true'); ap.add_argument('--resume',action='store_true'); ap.add_argument('--histidine-rdf-dir'); ap.add_argument('--control-rdf-dir'); args=ap.parse_args(); ensure_dirs(); R=root(); cfg=config(); hist_rdf=args.histidine_rdf_dir or cfg.get('external_rdf_paths',{}).get('histidine_rdfs',''); ctrl_rdf=args.control_rdf_dir or cfg.get('external_rdf_paths',{}).get('control_rdfs','')
 if args.resume and checkpoint_ok('features_prepared'): print('feature checkpoint exists; skipping'); return
 held=merge_heldout(); print(f'Prepared heldout features: {len(held)} rows')
 if args.dry_run:
  hdir=resolve_input_path(hist_rdf) if hist_rdf and not hist_rdf.startswith('SET_ME') else None; cdir=resolve_input_path(ctrl_rdf) if ctrl_rdf and not ctrl_rdf.startswith('SET_ME') else None
  print(f'DRY RUN: histidine RDF path={hdir}, files={len(list(hdir.glob("*.pkl"))) if hdir and hdir.exists() else 0}; control RDF path={cdir}, files={len(list(cdir.glob("*.pkl"))) if cdir and cdir.exists() else 0}')
  print('DRY RUN: would compute histidine/control B22 from packaged RDF PKLs and derive missing frozen features from packaged sequences/manifests.')
  return
 held.to_csv(R/'outputs/heldout/heldout_features.csv',index=False)
 hb=b22_from_rdf_dir(hist_rdf,'outputs/histidine/histidine_b22_from_original_rdfs.csv')
 cb=b22_from_rdf_dir(ctrl_rdf,'outputs/controls/control_b22_from_original_rdfs.csv')
 hf=prepare_special_features('histidine', hb)
 cf=prepare_special_features('controls', cb)
 (R/'outputs/histidine/histidine_feature_provenance.txt').write_text('Histidine targets are computed from packaged original histidine RDF PKLs. If all 13 frozen features are absent for a histidine pair, extend this stage using the packaged FASTA/metadata and historical_patch_get_features.py; no validation fitting/tuning is done here.\n')
 (R/'outputs/controls/control_feature_provenance.txt').write_text('Control targets are computed from packaged original control RDF PKLs and manifest-defined sequences/classes. Exact shuffles/basic/reference sequences are copied under inputs/controls/manifest.\n')
 write_checkpoint('features_prepared',['outputs/heldout/heldout_features.csv','outputs/histidine/histidine_b22_from_original_rdfs.csv','outputs/controls/control_b22_from_original_rdfs.csv','outputs/histidine/histidine_features.csv','outputs/controls/control_features.csv'])
 print('Validation source features/targets prepared.')
if __name__=='__main__': main()
