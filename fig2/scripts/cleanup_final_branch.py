#!/usr/bin/env python3
"""Preview or remove superseded local Figure 2 development artifacts.

The default is a dry run.  Deletion is restricted to the explicit relative
paths below and cannot escape the fig2 directory.  Raw inputs, final pipeline
inputs, source scripts, tracked BLAST mapping tables, and the final package are
never selected.
"""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
LARGE_LOCAL_INPUTS = {
    "outputs/reviewer_family_bias/prediction_universe_bias/candidate_vote_fractions_reconstructed.csv":
        "final_manuscript_pipeline/local_large_inputs/candidate_vote_fractions_reconstructed.csv",
    "outputs/reviewer_family_bias/prediction_universe_bias_region_level/biological_candidate_regions.csv":
        "final_manuscript_pipeline/local_large_inputs/biological_candidate_regions.csv",
}
OBSOLETE = [
    "FIGURE2_REVIEWER_ANALYSIS_REPORT.md",
    "Fig1C_cluster_bootstrap_statistics.csv",
    "README_RUN_ME.md",
    "README_RUN_YOURSELF.md",
    "README_computational_figure2_revision.md",
    "RUN_MERGED_FRAGMENTS_VS_9SEEDS.md",
    "RUN_NEAREST_SEED_BLAST_AUDIT.md",
    "RUN_REVIEWER_ANALYSIS.md",
    "feature_sets.zip",
    "models_all.zip",
    "figure2_computational_revision_run_yourself.zip",
    "fig1C_boxplot_final",
    "tmp",
    "xrgg_figs",
    "xrgg_panel_source_data",
    "xrgg_panels",
    "outputs/ahd_correspondence_comparison_111026",
    "outputs/cache",
    "outputs/fig1",
    "outputs/main",
    "outputs/reviewer_effect_sizes",
    "outputs/reviewer_external_AD_recovery",
    "outputs/reviewer_family_bias",
    "outputs/revised_computational_figure2",
    "outputs/revised_computational_figure2_previous_140_draft_20260914",
    "outputs/supplement",
]


def safe(path: Path) -> Path:
    resolved = path.resolve()
    if resolved == ROOT or ROOT not in resolved.parents:
        raise ValueError(f"Refusing unsafe target: {resolved}")
    return resolved


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--apply", action="store_true", help="Actually remove the listed paths")
    args = parser.parse_args()
    existing = [safe(ROOT / item) for item in OBSOLETE if (ROOT / item).exists()]
    print("Mode:", "DELETE" if args.apply else "DRY RUN")
    print("The following superseded paths are selected:")
    for path in existing:
        print(" ", path.relative_to(ROOT))
    preserved = [(safe(ROOT / source), safe(ROOT / destination))
                 for source, destination in LARGE_LOCAL_INPUTS.items()
                 if (ROOT / source).is_file()]
    if preserved:
        print("\nLarge optional rerun inputs will be preserved locally (not for Git):")
        for source, destination in preserved:
            print(f"  {source.relative_to(ROOT)} -> {destination.relative_to(ROOT)}")
    if not args.apply:
        print("\nNothing was removed. Re-run with --apply after validating the final package.")
        return
    for source, destination in preserved:
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists():
            raise FileExistsError(f"Refusing to replace preserved input: {destination}")
        shutil.move(source, destination)
    for path in existing:
        if path.is_dir() and not path.is_symlink():
            shutil.rmtree(path)
        else:
            path.unlink()
    print(f"\nRemoved {len(existing)} explicitly listed paths.")


if __name__ == "__main__":
    main()
