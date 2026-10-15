#!/usr/bin/env python3
"""Write a build-status audit for revised Figure 5, Figure S5, and draft S8."""
from __future__ import annotations

from fig5_revision_plot_utils import ROOT, SUPP_OUT, ensure_dir


MAIN_SOURCES = {
    "A": "schematic drawn by scripts/make_revised_Fig5.py",
    "B": ROOT / "reviewer_feature_family_decomposition_training_only" / "cv_metrics_by_repeat.csv",
    "C": ROOT / "reviewer_f2_patch_context_training_only" / "cv_metrics_by_repeat.csv",
    "D": ROOT / "outputs" / "fig5_tensorflow_residual_final_base" / "outer_oof_predictions_F2_P8.csv",
    "E": ROOT / "fig5_validation_run_model_local" / "outputs" / "heldout" / "heldout_final_predictions.csv",
    "F": ROOT / "unsim_preds_with_hybrid_021125_sel_cols.csv",
}

S5_SOURCES = {
    "A": ROOT / "outputs" / "fig5_tensorflow_residual_final_base" / "outer_oof_predictions_F2_P8.csv",
    "B": ROOT / "outputs" / "fig5_tensorflow_residual_final_base" / "base_hybrid_paired_differences.csv",
    "C": "schematic drawn by scripts/make_revised_FigS5.py",
    "D": ROOT / "outputs" / "fig5_tensorflow_residual_final_base" / "fold_residual_diagnostics.csv",
    "E": ROOT / "outputs" / "fig5_high_OCN_tail_analysis" / "tail_paired_model_comparisons.csv",
    "F": "schematic drawn by scripts/make_revised_FigS5.py",
}

S8_SOURCES = {
    "D": ROOT / "outputs" / "fig5_histidine_pH_audit" / "histidine_matched_b22.csv",
    "E": ROOT / "outputs" / "fig5_histidine_pH_audit" / "histidine_matched_b22.csv",
}


def exists_text(source) -> str:
    if isinstance(source, str):
        return source
    return f"{source} exists={source.exists()}"


def main():
    out = ensure_dir(SUPP_OUT)
    lines = ["Figure 5 figure build status", ""]
    lines.append("Main Figure 5")
    for panel, source in MAIN_SOURCES.items():
        status = "PASS" if isinstance(source, str) or source.exists() else "MISSING"
        lines.append(f"{panel} {status} - {exists_text(source)}")
    lines.append("")
    lines.append("Supplementary S5")
    for panel, source in S5_SOURCES.items():
        status = "PASS" if isinstance(source, str) or source.exists() else "MISSING"
        lines.append(f"{panel} {status} - {exists_text(source)}")
    lines.append("")
    lines.append("Supplementary S8")
    lines.append("A PENDING - running simulations")
    lines.append("B PENDING - running simulations")
    lines.append("C PENDING - running simulations")
    for panel, source in S8_SOURCES.items():
        status = "PASS" if source.exists() else "MISSING"
        lines.append(f"{panel} {status} - {source} exists={source.exists()}")
    lines.append("F PENDING - running simulations")
    lines.append("")
    lines.append("Completed-panel source files are listed above. Preliminary non-dataset WT, shuffle, and basic-region control outputs are intentionally not used.")
    lines.append("")
    lines.append("Scatter-panel data/style separation")
    lines.append("Fig5D | data source: revised OOF data | plot-style source: old manuscript scatter function in /Users/yuktikhanna/rtads-rgg-interactions/fig5/fig5.py")
    lines.append("Fig5E | data source: revised frozen 1794 data | plot-style source: old manuscript scatter function in /Users/yuktikhanna/rtads-rgg-interactions/fig5/fig5.py")
    lines.append("Fig5F | data source: revised full landscape | plot-style source: old manuscript landscape/scatter function in /Users/yuktikhanna/rtads-rgg-interactions/fig5/fig5.py")
    lines.append("FigS5A | data source: revised OOF residual data | plot-style source: old manuscript scatter aesthetics in /Users/yuktikhanna/rtads-rgg-interactions/fig5/fig5.py")
    lines.append("FigS8D | data source: histidine pH audit | plot-style source: old manuscript scatter function in /Users/yuktikhanna/rtads-rgg-interactions/fig5/fig5.py")
    lines.append("FigS8E | data source: frozen pH6.5 validation | plot-style source: old manuscript scatter function in /Users/yuktikhanna/rtads-rgg-interactions/fig5/fig5.py")
    (out / "Figure5_figure_build_status.txt").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
