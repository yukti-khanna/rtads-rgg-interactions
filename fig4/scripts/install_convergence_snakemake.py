#!/usr/bin/env python3
"""Install optional convergence targets. Default is preview; never runs Snakemake."""
import argparse
import datetime
from pathlib import Path
import shutil

RULES = '''# Existing-B22 convergence only: never depends on compute_b22 or RDF extraction.
def convergence_flag(value):
    if isinstance(value, bool): return value
    if str(value).lower() in ("true", "1", "yes"): return True
    if str(value).lower() in ("false", "0", "no"): return False
    raise ValueError("include_b22_convergence must be true or false")

CONVERGENCE_WITH_MAIN = convergence_flag(config.get("include_b22_convergence", False))
CONVERGENCE_REPORT = "results/b22_convergence/verified/verification_report.json"
CONVERGENCE_MAIN_INPUTS = [
    config["outputs"]["prepared_pairs_csv"],
    "results/prepared/calvados_inputs.done",
    config["outputs"]["rdf_summary_csv"],
    config["outputs"]["b22_csv"],
    config["outputs"]["features_csv"],
    config["outputs"]["features_with_b22_csv"],
    "results/figures/feature_vs_b22",
] if CONVERGENCE_WITH_MAIN else []

rule b22_convergence:
    input:
        CONVERGENCE_REPORT

rule verify_existing_b22_convergence:
    input:
        sources=expand("inputs/convergence_sampling/frozen_sources/Table_{sheet}.csv",
                       sheet=["S6C", "S6D", "S6E", "S7A", "S7B"]),
        configuration="config/convergence_sampling.json",
        script="scripts/verify_convergence_sampling_sources.py",
        main=CONVERGENCE_MAIN_INPUTS
    output:
        report=CONVERGENCE_REPORT,
        replica="results/b22_convergence/verified/replicate_count_convergence_summary.csv",
        trajectory="results/b22_convergence/verified/trajectory_final_change.csv",
        sampling="results/sampling/verified/B22_distribution_stability_by_stage.csv"
    shell:
        "python {input.script:q} --config {input.configuration:q}"
'''

def updated_snake(text):
    include='include: "workflow/convergence.smk"'
    if include not in text:
        anchor='configfile: "config/config.yaml"'
        if text.count(anchor)!=1: raise ValueError('Unexpected Snakefile config header.')
        text=text.replace(anchor,anchor+'\n\n'+include,1)
    # Keep rule all as the default despite the earlier included target rules.
    anchor='rule all:\n    input:\n'
    if anchor not in text and 'rule all:\n    default_target: True\n    input:\n' not in text:
        raise ValueError('Cannot locate rule all input block.')
    if 'default_target: True' not in text.split('rule core_inputs:',1)[0]:
        text=text.replace(anchor,'rule all:\n    default_target: True\n    input:\n',1)
    anchor='rule all:\n    default_target: True\n    input:\n'
    extra='        ([CONVERGENCE_REPORT] if CONVERGENCE_WITH_MAIN else []),\n'
    if extra not in text: text=text.replace(anchor,anchor+extra,1)
    # Guard B22 recalculation by default. If an established output is missing or
    # stale, fail explicitly rather than silently recomputing it.
    start=text.index('rule compute_b22:')
    finish=text.index('\nrule ',start+1)
    block=text[start:finish]
    if 'reuse_existing_b22' not in block:
        block=block.replace('    params:\n','    params:\n        reuse_existing=lambda wildcards: str(convergence_flag(config.get("reuse_existing_b22", True))),\n',1)
        shell='        python scripts/compute_b22.py'
        if shell not in block: raise ValueError('Unexpected compute_b22 shell command.')
        block=block.replace(shell,'''        if [ "{params.reuse_existing}" = "True" ]; then
          echo "B22 recalculation disabled. Restore the established B22 output, or explicitly set reuse_existing_b22=false for a new analysis." >&2
          exit 1
        fi
'''+shell,1)
        text=text[:start]+block+text[finish:]
    return text

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--repo',type=Path,default=Path('/Users/yuktikhanna/idr-interaction-analysis'))
    p.add_argument('--backup-dir',type=Path,required=True)
    p.add_argument('--execute',action='store_true')
    a=p.parse_args(); repo=a.repo.resolve(strict=True)
    snake=repo/'Snakefile'; content=updated_snake(snake.read_text())
    print('Will update:',snake,'\nWill install:',repo/'workflow/convergence.smk')
    print('Separate: snakemake --cores 1 b22_convergence --config include_b22_convergence=false')
    print('Together: snakemake --cores 1 --config include_b22_convergence=true')
    if not a.execute:
        print('PREVIEW ONLY; nothing written or run.'); return
    base=a.backup_dir.resolve()
    if base==repo or repo in base.parents: raise ValueError('Backup must be outside repository.')
    backup=base/datetime.datetime.now().strftime('snakemake_%Y%m%d_%H%M%S')
    backup.mkdir(parents=True,exist_ok=False)
    rules=repo/'workflow/convergence.smk'
    for path in [snake,rules]:
        if path.is_file():
            dest=backup/path.relative_to(repo);dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(path,dest)
    rules.parent.mkdir(parents=True,exist_ok=True); rules.write_text(RULES)
    snake.write_text(content)
    print('Installed. No Snakemake execution or B22 calculation performed. Backup:',backup)

if __name__=='__main__': main()
