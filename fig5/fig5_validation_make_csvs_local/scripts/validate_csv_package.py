#!/usr/bin/env python
import argparse
from pathlib import Path
from utils import root, load_development, read_csv, config, resolve_input_path

def main():
 ap=argparse.ArgumentParser(); ap.add_argument('--dry-run',action='store_true'); ap.add_argument('--histidine-rdf-dir'); ap.add_argument('--control-rdf-dir'); args=ap.parse_args(); R=root(); cfg=config()
 dev=load_development(); held=read_csv('inputs/heldout/taken_out_preds_with_hybrid_021125_sel_cols.csv'); hp=read_csv('inputs/heldout/features_heldout_sequence_exclusive.csv')
 hr=args.histidine_rdf_dir or cfg['external_rdf_paths']['histidine_rdfs']; cr=args.control_rdf_dir or cfg['external_rdf_paths']['control_rdfs']
 hdir=resolve_input_path(hr) if hr and not hr.startswith('SET_ME') else None; cdir=resolve_input_path(cr) if cr and not cr.startswith('SET_ME') else None
 print(f'Development rows: {len(dev)}')
 print(f'Held-out global rows: {len(held)}; patch rows: {len(hp)}')
 print(f'Histidine RDF path: {hdir if hdir else "NOT SET"}; files={len(list(hdir.glob("*.pkl"))) if hdir and hdir.exists() else 0}')
 print(f'Control RDF path: {cdir if cdir else "NOT SET"}; files={len(list(cdir.glob("*.pkl"))) if cdir and cdir.exists() else 0}')
 if len(dev)!=917 or len(held)!=1794 or len(hp)!=1794: raise SystemExit('row-count validation failed')
 if not hdir or not hdir.exists(): raise SystemExit('set histidine RDF path')
 if not cdir or not cdir.exists(): raise SystemExit('set control RDF path')
 print('CSV package validation passed.')
if __name__=='__main__': main()
