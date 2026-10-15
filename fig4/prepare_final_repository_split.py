#!/usr/bin/env python3
"""Local-only migration of approved Figure 3/S4/S5 assets and analysis sources.

Default: print plan only. --execute writes files. --cleanup additionally removes
only the exact superseded files listed below, after an external backup.
Approved PNGs are preserved and extracted as RASTER panel references, not
misrepresented as vector reconstructions of the original plotting scripts.
No simulations, trajectory processing or B22 computations are run.
"""
import argparse
import datetime
import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
OLD = [
    'reviewer_analysis/training_selection_provenance/build_final_s4_v4.py',
    'reviewer_analysis/training_selection_provenance/revised_Figure_S4_A-H_proposed_v4.pdf',
    'reviewer_analysis/training_selection_provenance/revised_Figure_S4_A-H_proposed_v4.png',
    'reviewer_analysis/rdf_b22_convergence_figure/plot_rdf_b22_convergence.py',
    'reviewer_analysis/final_fig4_revision/build_Fig4D_final_restored.py',
]
TABLES = {
    'S6': 'Table_S6_AHD_RG_RGG_simulation_dataset_and_sampling_provenance.xlsx',
    'S7': 'Table_S7_simulation_convergence_shuffle_controls_interaction_energy.xlsx',
}
NUMERICAL_SOURCES = [
    'task2_b22_convergence/replicate_count_convergence.csv',
    'task2_b22_convergence/replicate_count_convergence_summary.csv',
    'task2_b22_convergence/convergence_pair_selection.csv',
    'task2_b22_convergence/time_convergence_summary.csv',
    'task2_b22_convergence/time_convergence_per_replica.csv',
    'task2_b22_convergence/per_replica_b22.csv',
    'reviewer_analysis/training_selection_provenance/training_selection_stage_counts.csv',
    'reviewer_analysis/training_selection_provenance/training_selection_pair_provenance.csv',
    'reviewer_analysis/training_selection_provenance/TRAINING_SET_SELECTION_PROVENANCE.md',
    'reviewer_analysis/training_selection_provenance/B22_distribution_stability_by_stage.csv',
    'reviewer_analysis/training_selection_provenance/ncp_sampling_bin_audit.csv',
]

def digest(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--analysis-repo', type=Path, default=Path('/Users/yuktikhanna/idr-interaction-analysis'))
    ap.add_argument('--tables-dir', type=Path, default=Path('/Users/yuktikhanna/writing/rbp-tfs/cell_reports'))
    ap.add_argument('--figures-dir', type=Path, default=Path('/Users/yuktikhanna/Downloads/Ai-and-PDF 3'))
    ap.add_argument('--backup-dir', type=Path, required=True, help='Backup directory OUTSIDE both repositories.')
    ap.add_argument('--execute', action='store_true')
    ap.add_argument('--cleanup', action='store_true')
    args = ap.parse_args()
    analysis = args.analysis_repo.resolve(strict=True)
    backup = args.backup_dir.resolve()
    if any(backup == p or p in backup.parents for p in [ROOT, analysis]):
        ap.error('Backup must be outside both repositories.')
    required = [args.tables_dir/n for n in TABLES.values()] + [args.figures_dir/f'Figure-{n}.png' for n in ['3','S4','S5']]
    missing = [str(p) for p in required if not p.is_file()]
    if missing:
        raise FileNotFoundError('\n'.join(missing))
    print('Figure repository:', ROOT, '\nAnalysis repository:', analysis)
    print('Approved numbering: Figure 3 A-D; S4 A-H; S5 A-E. No S4I, S5F or S6 figure added.')
    print('Plan: export table sources, preserve approved figures, extract raster panel references,')
    print('copy numerical provenance into analysis repository; preserve figure-specific analysis in fig4.')
    obsolete = [ROOT/p for p in OLD if (ROOT/p).is_file()]
    print('Exact cleanup candidates:\n'+'\n'.join(str(p) for p in obsolete))
    if not args.execute:
        print('PLAN ONLY. Add --execute to write. Add --cleanup only when ready to remove the listed files.')
        return
    import numpy as np
    import pandas as pd
    from PIL import Image
    sheets = {}
    for key,name in TABLES.items():
        x = pd.ExcelFile(args.tables_dir/name)
        sheets.update({s: pd.read_excel(x,s,header=3) for s in x.sheet_names})
    # Mandatory numerical gates, before exporting or deleting anything.
    train = sheets['S6C']; held = sheets['S6D']; shuf = sheets['S7C']
    if len(train)!=917 or train.pair_id.nunique()!=917 or len(held)!=1794:
        raise ValueError('Training/held-out source counts do not match the approved tables.')
    if len(shuf)!=300 or shuf.WT_pair_id.nunique()!=100 or shuf.shuffle_type.value_counts().to_dict()!={'AHD shuffle':200,'RG/RGG shuffle':100}:
        raise ValueError('Shuffle count mismatch.')
    np.testing.assert_allclose(train[[f'B22_rep{i}' for i in range(1,6)]].mean(axis=1), train.B22_mean_5rep, atol=1e-9)
    conv = sheets['S7A'].dropna(subset=['k']).sort_values('k')
    np.testing.assert_allclose(conv.median_absolute_deviation, [121.073486,72.685436,48.456957,30.268371,0], atol=1e-6)
    np.testing.assert_allclose(conv[['fraction_within_5pct','fraction_within_10pct','fraction_within_20pct']],
        [[.244275,.441439,.710578],[.368593,.634351,.883751],[.514177,.794111,.963359],[.710578,.928244,.994766],[1,1,1]], atol=1e-6)
    energy = sheets['S7E'].dropna(subset=['RG_RGG_class','AHD_class'])
    np.testing.assert_allclose(energy.mean_percentage.sum(),100,atol=1e-9)
    times = sheets['S7B'].dropna(subset=['pair_id']).pivot(index=['pair_id','regime'],columns='trajectory_fraction',values='mean')
    changes = 100*abs(times[.8]-times[1.])/abs(times[1.])
    if len(times)!=15 or sum(changes<5)!=13 or sum(changes<10)!=15:
        raise ValueError('Trajectory convergence verification failed.')
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    backup = backup/timestamp
    backup.mkdir(parents=True,exist_ok=False)
    manifest = []
    def put(src,dst):
        src,dst = Path(src),Path(dst)
        if dst.exists():
            if digest(src)==digest(dst): return
            old = backup/'overwritten'/dst.relative_to(dst.anchor)
            old.parent.mkdir(parents=True,exist_ok=True); shutil.copy2(dst,old)
        dst.parent.mkdir(parents=True,exist_ok=True); shutil.copy2(src,dst)
        manifest.append(dict(source=str(src),destination=str(dst),sha256=digest(dst)))
    sources = ROOT/'inputs'/'source_data'/'approved_figure3_S4_S5'
    outputs = ROOT/'outputs'/'approved_figure3_S4_S5'
    provenance = ROOT/'provenance'/'approved_figure3_S4_S5'
    package = analysis/'inputs'/'convergence_sampling'
    for p in [sources,outputs,provenance,package]: p.mkdir(parents=True,exist_ok=True)
    # Full workbooks retain formatting, notes, side-by-side tables and numbering.
    for name in TABLES.values(): put(args.tables_dir/name,ROOT/'inputs'/'supplementary_tables'/name)
    for s,d in sheets.items():
        d = d.loc[:,~d.columns.astype(str).str.startswith('Unnamed')]
        dest = sources/f'Table_{s}.csv'
        if dest.exists(): put(dest,backup/'prior_exports'/dest.name)
        d.to_csv(dest,index=False)
        manifest.append(dict(source=f'{TABLES[s[:2]]}::{s}',destination=str(dest),sha256=digest(dest)))
        if s.startswith('S6') or s in ['S7A','S7B']:
            put(dest,package/'frozen_sources'/dest.name)
    for relative in NUMERICAL_SOURCES:
        src = ROOT/relative
        if src.is_file():
            category = 'b22_convergence' if relative.startswith('task2_b22_convergence/') else 'sampling'
            put(src,analysis/'results'/category/'historical_sources'/src.name)
            put(src,sources/'numerical_provenance'/relative)
    put(ROOT/'scripts'/'verify_convergence_sampling_sources.py',analysis/'scripts'/'verify_convergence_sampling_sources.py')
    legacy_bundle = analysis/'fig4'/'task2_b22_convergence'
    relocated = []
    if legacy_bundle.is_dir():
        for src in sorted(legacy_bundle.rglob('*')):
            if not src.is_file() or src.is_symlink(): continue
            dest = analysis/'results'/'b22_convergence'/'legacy_bundle'/src.relative_to(legacy_bundle)
            put(src,dest)
            relocated.append((src,dest))
        for name in ['run_task2_b22_convergence.py','add_task2_cleanup_outputs.py']:
            if (legacy_bundle/name).is_file():
                put(legacy_bundle/name,analysis/'scripts'/'historical'/name)
    # Keep old plotting entry points as traceable, non-default source snapshots.
    for relative in [
        'reviewer_analysis/final_fig4_revision/build_final_fig4_revision.py',
        'reviewer_analysis/rdf_b22_convergence_figure/plot_rdf_b22_convergence_v2.py',
        'reviewer_analysis/shuffle_replica_uncertainty/run_shuffle_replica_uncertainty.py',
        'reviewer_analysis/shuffle_replica_uncertainty/global_null_analysis/run_global_shuffle_null.py',
        'reviewer_analysis/fig3c_length_normalised_energy/build_panelC_options.py',
        'reviewer_analysis/fig3c_length_normalised_energy/build_residue_level_panelC.py']:
        if (ROOT/relative).is_file(): put(ROOT/relative,provenance/'historical_plotting_code'/relative)
    # Exact visual reference exports. Crops include the supplied panel letters.
    boxes = {
        '3': {'A':(0,0,.51,.47),'B':(.51,0,1,.47),'C':(0,.47,.51,1),'D':(.51,.47,1,1)},
        'S4': {letter:(col*.5,row*.25,(col+1)*.5,(row+1)*.25) for row in range(4) for col,letter in enumerate('ABCDEFGH'[row*2:row*2+2])},
        'S5': {'A':(0,0,.51,.28),'B':(.51,0,1,.28),'C':(0,.28,.51,.56),'D':(.51,.28,1,.56),'E':(0,.56,1,1)},
    }
    for number,panels in boxes.items():
        original = args.figures_dir/f'Figure-{number}.png'
        put(original,sources/'approved_visual_references'/original.name)
        with Image.open(original) as image:
            image = image.convert('RGB'); w,h = image.size
            image.save(outputs/f'Figure{number}_approved_reference.png',dpi=(600,600))
            image.save(outputs/f'Figure{number}_approved_reference.pdf',resolution=600)
            for letter,box in panels.items():
                panel = image.crop(tuple(round(v*d) for v,d in zip(box,(w,h,w,h))))
                stem = f'Fig{number}{letter}_raster_reference'
                panel.save(outputs/f'{stem}.png',dpi=(600,600))
                panel.save(outputs/f'{stem}.pdf',resolution=600)
    readme = '''# Figure 3, Supplementary Figures S4 and S5

The historical directory name is `fig4`; current manuscript numbering is Figure 3, S4 and S5.
Approved numerical sources are in inputs/source_data/approved_figure3_S4_S5.
Tables S6/S7 are in inputs/supplementary_tables. Figure-specific shuffle and energy
analysis remains under reviewer_analysis. Reusable convergence and sampling sources
are exported to the existing inputs/, scripts/ and results/ layout in idr-interaction-analysis.

Run prepare_final_repository_split.py --help for local migration options.
Outputs labelled raster_reference are extractions of the approved figure images,
not editable vector reconstructions. Their 600-dpi metadata does not increase the
resolution of the supplied pixels. Final figure assembly/layout is inherited exactly
from the approved images (including S4's 4-row × 2-column layout).

Figure 3: A log-strength count histogram; B two RDF examples; C equal-system
fractional attractive-energy class contributions; D −NCP correlation coloured by mean λ.
S4: A–C pilot feature relationships; D bin coverage; E retrospective distribution
stability; F–H −NCP, length and mean-λ distributions.
S5: A five fixed examples; B all replica combinations; C original 15 time-convergence
pairs; D 300 shuffles/100 parents; E bootstrap uncertainty and multiplicity summaries.
Do not substitute old class-enrichment or sparse-trajectory energy outputs.

Table S7 contains historical S6B/S6F references; these mean current S5B and the
supporting shuffle-null analysis, respectively. The global null remains supporting
analysis and is not automatically added as a new figure panel.
'''
    def write_safe(path,text):
        if path.exists(): put(path,backup/'previous_documentation'/path.relative_to(path.anchor))
        path.write_text(text)
    write_safe(ROOT/'README.md',readme)
    write_safe(package/'README.md','''# Convergence and sampling inputs

Frozen numerical sources supporting current Figure 3, S4 and S5 are packaged here.
No trajectories, simulations or B22 values are recomputed by the migration.
frozen_sources contains Table S6 and S7A/B snapshots. Detailed historical outputs
are under results/b22_convergence and results/sampling. Figure plotting, shuffle controls and
interaction-energy decomposition remain in the manuscript figure repository.
Run scripts/verify_convergence_sampling_sources.py from the repository root to
verify and export summaries without recalculating B22. Historical trajectory
analysis scripts are retained under scripts/historical and are not run automatically.
''')
    write_safe(analysis/'config'/'convergence_sampling.json',json.dumps(dict(
        frozen_sources='inputs/convergence_sampling/frozen_sources',
        convergence_results='results/b22_convergence/verified',
        sampling_results='results/sampling/verified',
        figure_repository=str(ROOT),approved_figures=['3','S4','S5']),indent=2))
    subprocess.run([sys.executable,str(ROOT/'scripts'/'install_convergence_snakemake.py'),
        '--repo',str(analysis),'--backup-dir',str(backup),'--execute'],check=True)
    begin='<!-- BEGIN CONVERGENCE SAMPLING -->'
    end='<!-- END CONVERGENCE SAMPLING -->'
    existing_readme=(analysis/'README.md').read_text()
    if begin in existing_readme:
        before,tail=existing_readme.split(begin,1)
        if end not in tail: raise ValueError('Unclosed README migration section; refusing overwrite.')
        _,after=tail.split(end,1); existing_readme=before+after
    addition='''
## Convergence and sampling analyses

The existing main Snakemake workflow is extended by workflow/convergence.smk;
config/config.yaml is preserved. Convergence is opt-in and reads existing tables.
Frozen inputs for the manuscript's Figure 3, S4 and S5 are stored under
inputs/convergence_sampling/frozen_sources. Detailed convergence results live in
results/b22_convergence; sampling provenance and stability results in results/sampling.
The old fig4/task2_b22_convergence bundle is preserved under
results/b22_convergence/legacy_bundle; its original analysis entry points are
retained under scripts/historical for provenance, not as default commands.

To verify established numerical outputs and export current summaries:

```bash
python scripts/verify_convergence_sampling_sources.py --config config/convergence_sampling.json
```

Separate Snakemake target:

```bash
snakemake --cores 1 b22_convergence --config include_b22_convergence=false
```

Include convergence after the main workflow:

```bash
snakemake --cores 1 --config include_b22_convergence=true
```

The combined mode waits for the main outputs before verifying convergence.
reuse_existing_b22 defaults to true: if compute_b22 would be scheduled, it fails
instead of recalculating B22. Restore the established output rather than enabling
recalculation during this migration. Standalone convergence has no RDF/B22 producer dependencies.

This command does not run simulations or recalculate B22. It checks all-combination
replica convergence, the original 15 trajectory-convergence pairs, dataset counts,
and retained sampling cohorts. Historical full recomputation remains a separate
operation requiring its original inputs and flags. Manuscript plotting, figure sources, shuffle-specific analyses
and energy decomposition remain in rtads-rgg-interactions/fig4.
'''
    write_safe(analysis/'README.md',existing_readme.rstrip()+'\n\n'+begin+'\n'+addition+end+'\n')
    write_safe(provenance/'migration_manifest.json',json.dumps(manifest,indent=2))
    # Cleanup is deliberately narrow; old cache/data folders are not deleted.
    deleted = []
    if args.cleanup:
        for src in obsolete:
            dest = backup/'removed_from_fig4'/src.relative_to(ROOT)
            dest.parent.mkdir(parents=True,exist_ok=True); shutil.copy2(src,dest)
            if digest(src)!=digest(dest): raise ValueError(f'Backup mismatch: {src}')
            src.unlink(); deleted.append(str(src))
        for src,dest in relocated:
            if digest(src)!=digest(dest): raise ValueError(f'Relocated file mismatch: {src}')
            saved=backup/'removed_from_analysis'/src.relative_to(analysis)
            saved.parent.mkdir(parents=True,exist_ok=True); shutil.copy2(src,saved)
            if digest(src)!=digest(saved): raise ValueError(f'Backup mismatch: {src}')
            src.unlink(); deleted.append(str(src))
        # Remove only now-empty directories within the precisely resolved bundle.
        if legacy_bundle.is_dir():
            for folder in sorted([p for p in legacy_bundle.rglob('*') if p.is_dir()],key=lambda p:len(p.parts),reverse=True):
                if not any(folder.iterdir()): folder.rmdir()
            if not any(legacy_bundle.iterdir()): legacy_bundle.rmdir()
    write_safe(provenance/'cleanup_report.json',json.dumps(dict(backup=str(backup),removed=deleted,
        retained='All other historical files retained; no recursive deletion performed.'),indent=2))
    print('VERIFIED: 917 training, 1794 held-out, 300 shuffles/100 parents; authoritative convergence summaries.')
    print('Outputs:',outputs,'\nSources:',sources,'\nAnalysis package:',package,'\nBackup:',backup)
    print('Removed:',len(deleted),'explicit superseded files. No simulations/B22/trajectory calculations run.')
    print('NEXT: inspect raster crop boundaries; run scripts/verify_convergence_sampling_sources.py in the analysis repository.')

if __name__=='__main__': main()
