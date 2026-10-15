#!/usr/bin/env python3
"""Regenerate the complete Figure 5 revision package from the curated master CSV."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
MASTER = ROOT / "fig6_parent_master.csv"
OUT = ROOT / "final_figure_package"
BOUND_COLUMNS = [
    "highest_ligand_equivalents",
    "fraction_bound_highest_ligand",
    "percent_bound_highest_ligand",
]
ALLOWED_OUTCOMES = {"Kd_fit", "LB", "NB", "Kd_pending"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def numeric(frame: pd.DataFrame, column: str) -> pd.Series:
    values = pd.to_numeric(frame[column], errors="coerce")
    invalid = frame[column].notna() & frame[column].astype(str).str.strip().ne("") & values.isna()
    if invalid.any():
        pairs = frame.loc[invalid, "pair_id"].tolist()
        raise ValueError(f"{column} contains non-numeric values for: {pairs}")
    return values


def validate_master() -> pd.DataFrame:
    frame = pd.read_csv(MASTER)
    required = {
        "pair_id", "Outcome", "Kd_uM", "Kd_uM_err", "Pred_log10_negB22",
        "Sim_log10_negB22", "experiment_status", *BOUND_COLUMNS,
    }
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"fig6_parent_master.csv is missing required columns: {missing}")
    if frame.pair_id.isna().any() or frame.pair_id.astype(str).str.strip().eq("").any():
        raise ValueError("Every master row requires a non-empty pair_id")
    duplicates = frame.loc[frame.pair_id.duplicated(keep=False), "pair_id"].tolist()
    if duplicates:
        raise ValueError(f"Duplicate pair_id values: {duplicates}")

    outcomes = set(frame.Outcome.dropna().astype(str))
    invalid_outcomes = sorted(outcomes.difference(ALLOWED_OUTCOMES))
    if invalid_outcomes:
        raise ValueError(f"Unsupported curated Outcome values: {invalid_outcomes}")
    if frame.Outcome.isna().any():
        pairs = frame.loc[frame.Outcome.isna(), "pair_id"].tolist()
        raise ValueError(f"Outcome is missing for: {pairs}")

    kd = numeric(frame, "Kd_uM")
    kd_error = numeric(frame, "Kd_uM_err")
    fitted = frame.Outcome.eq("Kd_fit")
    invalid_fit = fitted & (~np.isfinite(kd) | kd.le(0))
    if invalid_fit.any():
        raise ValueError(f"Kd_fit rows require a positive Kd_uM: {frame.loc[invalid_fit, 'pair_id'].tolist()}")
    contradictory = ~fitted & kd.notna()
    if contradictory.any():
        raise ValueError(f"Only Kd_fit rows may contain Kd_uM: {frame.loc[contradictory, 'pair_id'].tolist()}")
    invalid_error = kd_error.notna() & kd_error.lt(0)
    if invalid_error.any():
        raise ValueError(f"Kd_uM_err cannot be negative: {frame.loc[invalid_error, 'pair_id'].tolist()}")

    for column in ["Pred_log10_negB22", "Sim_log10_negB22"]:
        values = numeric(frame, column)
        missing_scores = ~np.isfinite(values)
        if missing_scores.any():
            raise ValueError(f"{column} is required for every master row: {frame.loc[missing_scores, 'pair_id'].tolist()}")

    equivalents = numeric(frame, "highest_ligand_equivalents")
    fraction = numeric(frame, "fraction_bound_highest_ligand")
    percent = numeric(frame, "percent_bound_highest_ligand")
    bad_eq = equivalents.notna() & equivalents.le(0)
    bad_fraction = fraction.notna() & ~fraction.between(0, 1)
    bad_percent = percent.notna() & ~percent.between(0, 100)
    if bad_eq.any():
        raise ValueError(f"highest_ligand_equivalents must be positive: {frame.loc[bad_eq, 'pair_id'].tolist()}")
    if bad_fraction.any():
        raise ValueError(f"fraction_bound_highest_ligand must be between 0 and 1: {frame.loc[bad_fraction, 'pair_id'].tolist()}")
    if bad_percent.any():
        raise ValueError(f"percent_bound_highest_ligand must be between 0 and 100: {frame.loc[bad_percent, 'pair_id'].tolist()}")
    without_equivalent = (fraction.notna() | percent.notna()) & equivalents.isna()
    if without_equivalent.any():
        raise ValueError(f"Bound fractions require highest_ligand_equivalents: {frame.loc[without_equivalent, 'pair_id'].tolist()}")
    both = fraction.notna() & percent.notna()
    inconsistent = both & ~np.isclose(percent, 100 * fraction, rtol=0, atol=0.05)
    if inconsistent.any():
        raise ValueError(f"Percent bound must equal 100 × fraction bound: {frame.loc[inconsistent, 'pair_id'].tolist()}")
    return frame


def run(command: list[str], label: str, env: dict[str, str]) -> None:
    print(f"[{label}]", flush=True)
    subprocess.run(command, cwd=ROOT, env=env, check=True)


def node_runtime() -> tuple[Path, Path]:
    bundle = Path.home() / ".cache/codex-runtimes/codex-primary-runtime/dependencies/node"
    node = bundle / "bin/node"
    modules = bundle / "node_modules"
    if node.exists() and modules.exists():
        return node, modules
    executable = shutil.which("node")
    configured_modules = os.environ.get("FIG6_ARTIFACT_TOOL_NODE_MODULES")
    if executable and configured_modules and Path(configured_modules).exists():
        return Path(executable), Path(configured_modules)
    raise RuntimeError(
        "The Table S11 builder requires Node and @oai/artifact-tool. "
        "Use the Codex workspace runtime or set FIG6_ARTIFACT_TOOL_NODE_MODULES to its node_modules directory."
    )


def build_workbook(env: dict[str, str]) -> None:
    node, modules = node_runtime()
    with tempfile.TemporaryDirectory(prefix="fig6-table-s10-") as temp_name:
        temp = Path(temp_name)
        (temp / "node_modules").symlink_to(modules, target_is_directory=True)
        builder = temp / "build_supplementary_tables.mjs"
        shutil.copy2(ROOT / "build_supplementary_tables.mjs", builder)
        run([str(node), str(builder), str(OUT)], "Building Supplementary Table S11", env)


def validate_outputs(master: pd.DataFrame) -> None:
    required = [
        OUT / "Supplementary_Table_S11.xlsx",
        OUT / "Supplementary_Table_S11.xlsx.inspect.ndjson",
        OUT / "Supplementary_Table_S11_preview.png",
        OUT / "FIGURE_STORYLINE.md",
        OUT / "CAPTIONS.md",
    ]
    for stem in ["Figure5_with_panelA_placeholder", "FigureS10_simulation_and_intensity", "FigureS11_shuffle_and_LEF1_context"]:
        required.extend(OUT / "figures" / f"{stem}.{ext}" for ext in ["pdf", "svg", "png"])
    missing = [str(path.relative_to(ROOT)) for path in required if not path.exists() or path.stat().st_size == 0]
    if missing:
        raise RuntimeError(f"Final package is missing required outputs: {missing}")

    expected = set(master.loc[master.Outcome.isin(["NB", "LB", "Kd_fit"]), "pair_id"])
    panel_b = pd.read_csv(OUT / "source_data/Fig5B_scores_by_outcome.csv")
    if set(panel_b.pair_id) != expected:
        raise RuntimeError("Panel B membership does not match all eligible curated master rows")
    nmr = pd.read_csv(OUT / "tables/Table_S11_NMR_outcomes_and_Kd.csv")
    if set(nmr.pair_id) != set(master.pair_id):
        raise RuntimeError("Supplementary Table S11 NMR sheet does not contain every curated master row")
    fitted = set(master.loc[master.Outcome.eq("Kd_fit"), "pair_id"])
    panel_d = pd.read_csv(OUT / "source_data/Fig5D_fitted_affinities.csv")
    if set(panel_d.pair_id) != fitted:
        raise RuntimeError("Panel D membership does not match all fitted Kd rows")

    inspect_text = (OUT / "Supplementary_Table_S11.xlsx.inspect.ndjson").read_text()
    expected_sheets = [
        "NMR outcomes and Kd", "Validation statistics", "AUROC and concordance",
        "Threshold sensitivity", "Intensity ratios", "Shuffle comparisons",
        "LEF1 context", "Bound-fraction guide", "Data dictionary", "pH simulation comparison", "pH predictor comparison",
    ]
    absent = [name for name in expected_sheets if name not in inspect_text]
    if absent:
        raise RuntimeError(f"Supplementary Table S11 is missing sheets: {absent}")


def remove_superseded_workbook_files() -> None:
    for name in [
        "Supplementary_Tables_S10-S15.xlsx",
        "Supplementary_Tables_S10-S15.xlsx.inspect.ndjson",
        "Supplementary_Tables_preview.png",
    ]:
        path = OUT / name
        if path.exists():
            path.unlink()


def write_manifest(master_hash: str, with_trajectories: bool) -> None:
    report = {
        "master_csv": str(MASTER),
        "master_sha256": master_hash,
        "with_trajectories": with_trajectories,
        "python": sys.executable,
        "manifest": "PACKAGE_MANIFEST.csv",
    }
    (OUT / "BUILD_REPORT.json").write_text(json.dumps(report, indent=2) + "\n")
    records = []
    for path in sorted(OUT.rglob("*")):
        if path.is_file() and path.name != "PACKAGE_MANIFEST.csv":
            records.append({"path": str(path.relative_to(OUT)), "bytes": path.stat().st_size, "sha256": sha256(path)})
    pd.DataFrame(records).to_csv(OUT / "PACKAGE_MANIFEST.csv", index=False)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--with-trajectories", action="store_true", help="Recompute trajectory analyses instead of reusing validated caches")
    args = parser.parse_args()

    os.chdir(ROOT)
    env = os.environ.copy()
    env.setdefault("MPLCONFIGDIR", str(ROOT / ".mplconfig"))
    master = validate_master()
    master_hash = sha256(MASTER)
    python = sys.executable

    run([python, "run_revision_analysis.py", "--stage", "inventory"], "Auditing inputs", env)
    run([python, "run_revision_analysis.py", "--stage", "plots"], "Regenerating analyses and source figures", env)
    if args.with_trajectories:
        run([python, "run_revision_analysis.py", "--stage", "trajectories"], "Recomputing trajectory analyses", env)
    run([python, "run_revision_analysis.py", "--stage", "report"], "Refreshing analysis report", env)
    run([python, "prepare_final_figure_package.py"], "Assembling publication figures and tables", env)
    build_workbook(env)
    run([python, "verify_revision_analysis.py", "--outputs"], "Running numerical and output checks", env)
    validate_outputs(master)
    remove_superseded_workbook_files()
    write_manifest(master_hash, args.with_trajectories)
    if sha256(MASTER) != master_hash:
        raise RuntimeError("The curated master CSV changed during the build")
    print(f"Completed final package: {OUT}", flush=True)


if __name__ == "__main__":
    main()
