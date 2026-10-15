#!/usr/bin/env python3
"""Build manuscript panels and tables from saved source data; never infer/train."""
from pathlib import Path
import argparse, importlib.util, json, hashlib, re
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'outputs/manuscript'

def module(name):
    p=ROOT/'scripts/manuscript_helpers'/name
    spec=importlib.util.spec_from_file_location(p.stem,p);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m

def label(value):
    return re.sub(r'(?<![\w])(?:AHD|TAD)(?![\w])','rAHD',value) if isinstance(value,str) else value

def check():
    errors=[]
    manifest=json.loads((ROOT/'inputs/predictor/export_manifest.json').read_text())
    for item in manifest:
        p=ROOT/'inputs/predictor'/item['path']
        if not p.exists() or hashlib.sha256(p.read_bytes()).hexdigest()!=item['sha256']:errors.append(str(p))
    if errors:raise RuntimeError('Input checksum mismatch: '+str(errors))
    print('Predictor export checksums passed. No models loaded.')

def figures():
    import pandas as pd
    import matplotlib.pyplot as plt
    # Display-only substitution, leaving stored feature identifiers unchanged.
    from matplotlib.text import Text
    original=Text.set_text
    def set_text(self,s):return original(self,label(s))
    Text.set_text=set_text
    for name in ['Figure_4','Figure_S6','Figure_S7']:(OUT/name).mkdir(parents=True,exist_ok=True)
    m=module('update_final_figure_panels_from_predictions.py')
    m.make_main();m.make_s5()
    for old,new in [('Fig5D_development_OOF','Figure4B_development'),('Fig5E_heldout_1794','Figure4C_independent'),('Fig5F_full_landscape','Figure4D_landscape')]:
        for ext in ['pdf','svg','png']:(m.MAIN/f'{old}.{ext}').replace(m.MAIN/f'{new}.{ext}')
    # Keep only the accepted residual-correction panel in the manuscript set.
    for p in m.SUPP.glob('FigS5*'):
        if p.stem=='FigS5D_applied_residual_correction':p.replace(p.with_name('FigureS6A_residual_correction'+p.suffix))
        else:
            extra=OUT/'diagnostics/residual';extra.mkdir(parents=True,exist_ok=True);p.replace(extra/p.name)
    c=module('make_extended_s8_outputs.py')
    for src,stem in [('S8A_extended_source.csv','S8A_non_dataset_WT_extended'),('S8D_extended_source.csv','S8D_basic_controls_extended')]:
        df=pd.read_csv(ROOT/'inputs/controls'/src);valid=df.B22_corr1_log10.notna()
        c.plot_scatter(df[valid],stem,'' if valid.all() else f'{valid.sum()}/{len(df)} controls have B22 < 0')
        target='FigureS6D_nonbasic_WT' if 'S8A' in src else 'FigureS6E_basic_controls'
        for ext in ['pdf','svg','png']:(c.PANEL/f'{stem}.{ext}').replace(c.PANEL/f'{target}.{ext}')
    spec=importlib.util.spec_from_file_location('shuffle_export',ROOT/'scripts/export_updated_S6_shuffle_controls.py');b=importlib.util.module_from_spec(spec);spec.loader.exec_module(b)
    b.OUT=OUT/'Figure_S6'
    import shutil
    shutil.copy2(ROOT/'inputs/controls/S8B_shuffle_source_audited.csv',b.OUT/'S8B_shuffle_source_audited.csv')
    b.plot_absolute_b()
    u=module('run_predictor_shuffle_uncertainty.py')
    data=pd.read_csv(ROOT/'inputs/controls/predictor_shuffle_replica_data.csv');current=pd.read_csv(ROOT/'inputs/controls/current_delta_analysis.csv');summary=pd.read_csv(ROOT/'inputs/controls/predictor_delta_uncertainty_summary.csv')
    fig,axs=plt.subplots(1,2,figsize=(5.1,2.5))
    for ax,g in zip(axs,u.GROUPS):u.draw_panel(ax,g,'',data,current,summary)
    fig.subplots_adjust(left=.12,right=.98,bottom=.20,top=.87,wspace=.45)
    for ext in ['pdf','svg','png']:fig.savefig(b.OUT/f'FigureS6C_delta_uncertainty.{ext}',dpi=600)
    plt.close(fig)
    f=module('make_FigS6_FINCHES_scaled_panels.py')
    for src,fun in [('S6A FINCHES simulation.csv',f.save_panel_a),('S6B FINCHES landscape.csv',f.save_panel_b)]:
        df=pd.read_csv(ROOT/'inputs/finches'/src).rename(columns={'scaled FINCHES value':'eps_scaled','scaled interaction value':'y_scaled','scaled final prediction':'y_scaled'})
        fun(df,f.pearson(df.eps_scaled,df.y_scaled))
    for p in f.OUTDIR.glob('FigS6*'):p.replace(p.with_name(p.name.replace('FigS6','FigureS7')))
    Text.set_text=original

def tables():
    # User-run portable workbook writer; no model calculation.
    import pandas as pd
    from openpyxl import Workbook
    from openpyxl.styles import Font
    target=OUT/'tables';target.mkdir(parents=True,exist_ok=True)
    for kind,num in [('predictor',8),('controls',9),('finches',10)]:
        w=Workbook();w.remove(w.active)
        note=w.create_sheet('Build provenance');note.append(['Item','Value']);note.append(['Terminology','rAHD; internal historical feature identifiers preserved']);note.append(['Controls caveat','Stored accepted predictions; historical global residue-table discrepancy documented in PIPELINE_PREPARATION.md']);note.append(['Histidine','Retained separately; excluded from these tables'])
        files=sorted((ROOT/'inputs'/kind).glob('*.csv'))
        if kind=='controls':
            names=['S8A_extended_source.csv','S8B_shuffle_source_audited.csv','S8D_extended_source.csv','S8B_absolute_metrics_original_new_combined.csv','S8B_delta_metrics_original_new_combined.csv','predictor_delta_uncertainty_summary.csv']
            files=[ROOT/'inputs/controls'/n for n in names]
        for p in files:
            if 'histidine' in p.name.lower() or 'S8C' in p.name:continue
            df=pd.read_csv(p)
            if kind=='controls' and p.name=='S8B_shuffle_source_audited.csv':
                features=pd.read_csv(ROOT/'inputs/controls/S8B_extended_source.csv')
                extra=[c for c in features.columns if c not in df.columns and c!='name']
                df=df.merge(features[['name']+extra],left_on='shuffle_pair_id',right_on='name',how='left',validate='one_to_one')
            name=label(p.stem).replace('Fig5','Fig4').replace('FigS5','FigS6')[:31]
            if name in w.sheetnames:raise ValueError(f'Duplicate sheet: {name}')
            s=w.create_sheet(name);s.append([label(x) for x in df.columns])
            for row in df.itertuples(index=False,name=None):s.append([None if pd.isna(x) else label(x) for x in row])
            s.freeze_panes='A2';s.auto_filter.ref=s.dimensions
            for cell in s[1]:cell.font=Font(bold=True)
        w.save(target/f'Table_S{num}_{kind}.xlsx')
    # Histidine is retained separately, never added to a manuscript table.
    hist=ROOT/'inputs/controls/histidine_pH6p5_final_predictions.csv'
    if hist.exists():
        import shutil
        dest=OUT/'analyses/histidine';dest.mkdir(parents=True,exist_ok=True);shutil.copy2(hist,dest/hist.name)

def main():
    p=argparse.ArgumentParser();p.add_argument('--check-only',action='store_true');a=p.parse_args();check()
    if not a.check_only:figures();tables()
if __name__=='__main__':main()
