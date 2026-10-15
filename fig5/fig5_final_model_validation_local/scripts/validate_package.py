#!/usr/bin/env python
from pathlib import Path
import argparse, csv, sys, os
from utils import root, ensure_dirs, read_csv, load_development, FEATURES, config, resolve_input_path

def main():
 p=argparse.ArgumentParser(); p.add_argument('--dry-run',action='store_true'); p.add_argument('--histidine-rdf-dir'); p.add_argument('--control-rdf-dir'); args=p.parse_args(); ensure_dirs(); R=root()
 problems=[]
 forbidden=[R/'outputs/fig5_final_model_validation',R/'outputs/fig5_independent_1794_validation']
 for f in forbidden:
  if f.exists(): problems.append(f'Forbidden previous-output folder exists inside package: {f}')
 dev=load_development(); held=read_csv('inputs/heldout/taken_out_preds_with_hybrid_021125_sel_cols.csv'); hp=read_csv('inputs/heldout/features_heldout_sequence_exclusive.csv')
 print(f'Development rows: {len(dev)}')
 print(f'Held-out global rows: {len(held)}; patch rows: {len(hp)}')
 cfg=config(); hr=args.histidine_rdf_dir or cfg.get('external_rdf_paths',{}).get('histidine_rdfs',''); cr=args.control_rdf_dir or cfg.get('external_rdf_paths',{}).get('control_rdfs','')
 hdir=resolve_input_path(hr) if hr and not hr.startswith('SET_ME') else None; cdir=resolve_input_path(cr) if cr and not cr.startswith('SET_ME') else None
 print(f'Histidine RDF path: {hdir if hdir else "NOT SET"}')
 print(f'Control RDF path: {cdir if cdir else "NOT SET"}')
 print(f'Histidine RDF files: {len(list(hdir.glob("*.pkl"))) if hdir and hdir.exists() else 0}')
 print(f'Control RDF files: {len(list(cdir.glob("*.pkl"))) if cdir and cdir.exists() else 0}')
 if not hdir or not hdir.exists(): problems.append('Histidine RDF directory not set/found; set config/run_config.json or pass --histidine-rdf-dir')
 if not cdir or not cdir.exists(): problems.append('Control RDF directory not set/found; set config/run_config.json or pass --control-rdf-dir')
 if len(held)!=1794 or len(hp)!=1794: problems.append('Expected 1,794 held-out rows')

 # Runtime files should not contain hard-coded home paths. The resolved package root itself may be absolute.
 for q in list((R/'scripts').glob('*.py')) + list((R/'config').glob('*.json')) + [R/'run_all.sh']:
  txt=q.read_text(errors='replace')
  forbidden_home='/' + 'Users' + '/' + 'yuktikhanna'
  if forbidden_home in txt:
   problems.append(f'Hard-coded Codex home path found in {q.relative_to(R)}')
 (R/'provenance/forbidden_previous_outputs_check.txt').write_text('Previous validation outputs used: NO\n')
 if problems: raise SystemExit('\n'.join(problems))
 print('PACKAGE DRY-RUN CHECK PASSED: inputs and required feature workflow are present. No fitting performed.')
if __name__=='__main__': main()
