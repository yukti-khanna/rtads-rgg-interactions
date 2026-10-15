#!/usr/bin/env python3
"""Install existing convergence sources and targets; never execute analysis."""
import datetime
import json
import shutil
from pathlib import Path
import pandas as pd
from install_convergence_snakemake import RULES, updated_snake

def main():
    here=Path(__file__).resolve().parent
    repo=Path('/Users/yuktikhanna/idr-interaction-analysis')
    tables=Path('/Users/yuktikhanna/writing/rbp-tfs/cell_reports')
    names={'S6':'Table_S6_AHD_RG_RGG_simulation_dataset_and_sampling_provenance.xlsx',
           'S7':'Table_S7_simulation_convergence_shuffle_controls_interaction_energy.xlsx'}
    frames={s:pd.read_excel(tables/names[s[:2]],sheet_name=s,header=3)
            for s in ['S6C','S6D','S6E','S7A','S7B']}
    snake=updated_snake((repo/'Snakefile').read_text())
    backup=repo/'provenance'/'installation_backups'/datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    backup.mkdir(parents=True,exist_ok=False)
    written=[]
    def preserve(path):
        if path.exists():
            target=backup/path.relative_to(repo)
            target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(path,target)
        path.parent.mkdir(parents=True,exist_ok=True)
        written.append(str(path))
    source=repo/'inputs/convergence_sampling/frozen_sources'
    for sheet,frame in frames.items():
        path=source/f'Table_{sheet}.csv';preserve(path)
        frame.loc[:,~frame.columns.astype(str).str.startswith('Unnamed')].to_csv(path,index=False)
    path=repo/'scripts/verify_convergence_sampling_sources.py';preserve(path)
    shutil.copy2(here/path.name,path)
    config=dict(frozen_sources='inputs/convergence_sampling/frozen_sources',
                convergence_results='results/b22_convergence/verified',
                sampling_results='results/sampling/verified')
    for path,text in [(repo/'config/convergence_sampling.json',json.dumps(config,indent=2)),
                      (repo/'workflow/convergence.smk',RULES),(repo/'Snakefile',snake)]:
        preserve(path);path.write_text(text)
    readme=repo/'README.md';text=readme.read_text()
    begin='<!-- BEGIN EXISTING B22 CONVERGENCE -->';end='<!-- END EXISTING B22 CONVERGENCE -->'
    if begin in text:
        before,tail=text.split(begin,1)
        if end not in tail: raise ValueError('Unclosed README section.')
        text=before+tail.split(end,1)[1]
    addition='''
## Existing-B22 convergence and sampling summaries

Run only convergence verification/export (no RDF extraction or B22 recalculation):

```bash
snakemake --cores 1 b22_convergence --config include_b22_convergence=false
```

Or include it after the main workflow:

```bash
snakemake --cores 1 --config include_b22_convergence=true
```

Frozen established tables are under inputs/convergence_sampling/frozen_sources;
outputs are under results/b22_convergence/verified and results/sampling/verified.
This target verifies established all-combination and trajectory-convergence
summaries; it does not recompute convergence from trajectories or replica values.
The main B22-producing rule is guarded against recalculation by default. If it
is scheduled, restore established outputs instead of enabling recalculation.
No simulations or analysis are run by installation. Figure plotting stays in
rtads-rgg-interactions/fig4 (current manuscript Figure 3, S4, S5).
'''
    preserve(readme);readme.write_text(text.rstrip()+'\n\n'+begin+'\n'+addition+end+'\n')
    print('INSTALLED:\n'+'\n'.join(written));print('BACKUP:',backup)
    print('No analysis, Snakemake jobs, simulation or B22 calculation executed.')

if __name__=='__main__':main()
