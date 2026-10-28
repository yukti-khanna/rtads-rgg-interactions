#!/usr/bin/env python3
"""
Figure S6 (scaled): FINCHES epsilon vs CALVADOS simulation/final model outputs.

This update preserves the original FINCHES sign/orientation and scaling logic:
    eps_scaled = (eps_max - epsilon) / (eps_max - eps_min)
so that more negative/more attractive epsilon maps closer to 1. The y axis is
scaled with one common min/max across Panel S6A's definitive 917 simulated
values and Panel S6B's definitive full-landscape final_hybrid_prediction values.

FINCHES epsilon values are read from the pre-existing FINCHES output matrix and
are not recomputed. Figure 5 model predictions are read from the finalized fig5
source tables and are not regenerated.
"""
from __future__ import annotations

from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

FIGSIZE = (3.2, 2.6)
DPI = 600


def find_repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = find_repo_root()
FIG5_ROOT = REPO_ROOT / "fig5"
OUTDIR = FIG5_ROOT / "outputs/manuscript/Figure_S7"
OUTDIR.mkdir(parents=True, exist_ok=True)

# Keep the existing FINCHES source convention without hard-coding /Users/...
EPS_CANDIDATES = [
    SCRIPT_DIR / "epsilon_values_cf.csv",
    SCRIPT_DIR / "cf" / "epsilon_values_cf.csv",
    REPO_ROOT / "finches" / "cf" / "epsilon_values_cf.csv",
    Path.home() / "finches" / "cf" / "epsilon_values_cf.csv",
]

DEV_CSV = FIG5_ROOT / "outputs" / "final_hybrid_model" / "final_predictions" / "development_917_OOF_final_hybrid.csv"
TEST_CSV = FIG5_ROOT / "outputs" / "final_hybrid_model" / "final_predictions" / "test_1794_final_hybrid_predictions.csv"
FULL_CSV = FIG5_ROOT / "outputs" / "final_hybrid_model" / "final_predictions" / "full_landscape_final_hybrid_predictions.csv"

plt.rcParams.update({
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "font.size": 8,
    "axes.labelsize": 8,
    "axes.titlesize": 8,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
})


def pearson(x, y) -> float:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    m = np.isfinite(x) & np.isfinite(y)
    if m.sum() < 3:
        return float("nan")
    return float(np.corrcoef(x[m], y[m])[0, 1])


def spearman(x, y) -> float:
    x = pd.Series(x, dtype="float64")
    y = pd.Series(y, dtype="float64")
    m = np.isfinite(x) & np.isfinite(y)
    if int(m.sum()) < 3:
        return float("nan")
    return float(x[m].corr(y[m], method="spearman"))


def minmax(x, x_min, x_max):
    denom = x_max - x_min
    if denom == 0:
        return np.zeros_like(np.asarray(x, dtype=float), dtype=float)
    return (np.asarray(x, dtype=float) - x_min) / denom


def resolve_epsilon_source() -> Path:
    for path in EPS_CANDIDATES:
        if path.exists():
            return path
    raise FileNotFoundError("Could not find pre-existing FINCHES epsilon_values_cf.csv in expected locations: " + "; ".join(map(str, EPS_CANDIDATES)))


def load_epsilon_long(path: Path) -> tuple[pd.DataFrame, dict]:
    eps_df = pd.read_csv(path, index_col=0)
    eps_df.index = eps_df.index.astype(str).str.strip()
    eps_df.columns = eps_df.columns.astype(str).str.strip()
    eps_long = eps_df.stack(dropna=False).reset_index()
    eps_long.columns = ["rgg", "tad", "epsilon"]
    eps_long["rgg"] = eps_long["rgg"].astype("string").str.strip()
    eps_long["tad"] = eps_long["tad"].astype("string").str.strip()
    eps_long["epsilon"] = pd.to_numeric(eps_long["epsilon"], errors="coerce")
    eps_long = eps_long.dropna(subset=["epsilon"]).copy()
    eps_long["name"] = eps_long["rgg"].astype(str) + "__" + eps_long["tad"].astype(str)
    dup = eps_long["name"].duplicated(keep=False)
    vals = eps_long["epsilon"].to_numpy(float)
    info = {
        "source_filename": str(path),
        "pair_id_column": "constructed name = rgg + '__' + tad from matrix row/column labels",
        "epsilon_column": "epsilon",
        "total_rows": int(len(eps_long)),
        "unique_pair_ids": int(eps_long["name"].nunique()),
        "duplicate_pair_ids": int(dup.sum()),
        "epsilon_min": float(np.nanmin(vals)),
        "epsilon_max": float(np.nanmax(vals)),
    }
    if dup.any():
        raise RuntimeError("Duplicate FINCHES pair IDs in epsilon table")
    return eps_long, info


def add_pair_parts(df: pd.DataFrame, pair_col: str) -> pd.DataFrame:
    out = df.copy()
    split = out[pair_col].astype("string").str.split("__", n=1, expand=True)
    out["rgg"] = split[0].str.strip()
    out["tad"] = split[1].str.strip()
    out["name"] = out[pair_col].astype(str)
    return out


def reconcile(left_ids, right_ids):
    left = set(map(str, left_ids))
    right = set(map(str, right_ids))
    return {
        "left_n": len(left),
        "right_n": len(right),
        "matched_n": len(left & right),
        "left_only": sorted(left - right),
        "right_only": sorted(right - left),
    }


def exact_merge_pairs(data: pd.DataFrame, eps_long: pd.DataFrame, label: str) -> tuple[pd.DataFrame, dict]:
    rec = reconcile(data["name"], eps_long["name"])
    if rec["left_only"]:
        raise RuntimeError(f"{label}: {len(rec['left_only'])} data IDs missing from FINCHES epsilon table; examples={rec['left_only'][:20]}")
    merged = data.merge(eps_long[["name", "rgg", "tad", "epsilon"]], on="name", how="left", validate="one_to_one")
    if merged["epsilon"].isna().any():
        raise RuntimeError(f"{label}: missing epsilon after exact merge")
    return merged, rec


def save_panel_a(train_m: pd.DataFrame, r_train: float):
    fig = plt.figure(figsize=FIGSIZE)
    ax = plt.gca()
    ax.scatter(train_m["eps_scaled"], train_m["y_scaled"], s=10, alpha=0.6)
    ax.set_xlabel("FINCHES ε (scaled; more attractive → 1)")
    ax.set_ylabel(r"$\log_{10}(-B_{22})$ (scaled)")
    ax.set_title(f"Training (n={len(train_m):,}); r={r_train:.2f}")
    x = train_m["eps_scaled"].to_numpy()
    y = train_m["y_scaled"].to_numpy()
    m = np.isfinite(x) & np.isfinite(y)
    if m.sum() >= 3:
        coef = np.polyfit(x[m], y[m], 1)
        xx = np.linspace(np.nanmin(x[m]), np.nanmax(x[m]), 200)
        yy = coef[0] * xx + coef[1]
        ax.plot(xx, yy, linewidth=1.2)
    plt.tight_layout()
    fig.savefig(OUTDIR / "FigS6A_scaled_FINCHES_vs_simulated_training.png", dpi=DPI)
    fig.savefig(OUTDIR / "FigS6A_scaled_FINCHES_vs_simulated_training.pdf")
    fig.savefig(OUTDIR / "FigS6A_scaled_FINCHES_vs_simulated_training.svg")
    plt.close(fig)


def save_panel_b(unsim_m: pd.DataFrame, r_unsim: float):
    fig = plt.figure(figsize=FIGSIZE)
    ax = plt.gca()
    hb = ax.hexbin(unsim_m["eps_scaled"], unsim_m["y_scaled"], gridsize=55, mincnt=1)
    ax.set_xlabel("FINCHES ε (scaled; more attractive → 1)")
    ax.set_ylabel(r"$\log_{10}(-B_{22})$ (scaled, final prediction)")
    ax.set_title(f"All predicted pairs (n={len(unsim_m):,}); r={r_unsim:.2f}")
    cb = plt.colorbar(hb, ax=ax)
    cb.set_label("Pair density")
    plt.tight_layout()
    fig.savefig(OUTDIR / "FigS6B_scaled_FINCHES_vs_predicted_allpairs.png", dpi=DPI)
    fig.savefig(OUTDIR / "FigS6B_scaled_FINCHES_vs_predicted_allpairs.pdf")
    fig.savefig(OUTDIR / "FigS6B_scaled_FINCHES_vs_predicted_allpairs.svg")
    plt.close(fig)


def main():
    eps_path = resolve_epsilon_source()
    eps_long, eps_info = load_epsilon_long(eps_path)

    dev = pd.read_csv(DEV_CSV, usecols=["pair_id", "simulated_target"])
    dev = dev.rename(columns={"pair_id": "name", "simulated_target": "B22_corr1_log10"})
    if len(dev) != 917 or dev["name"].nunique() != 917:
        raise RuntimeError("Development table is not 917 unique pairs")

    full = pd.read_csv(FULL_CSV, usecols=["pair_id", "final_hybrid_prediction"])
    full = full.rename(columns={"pair_id": "name"})
    if full["name"].duplicated().any():
        raise RuntimeError("Full-landscape prediction table has duplicate pair IDs")

    train_m, dev_rec = exact_merge_pairs(dev, eps_long, "development_917")
    full_rec = reconcile(eps_long["name"], full["name"])
    if full_rec["right_only"]:
        details = {
            "FINCHES_total_rows": eps_info["total_rows"],
            "FINCHES_unique_pair_ids": eps_info["unique_pair_ids"],
            "full_landscape_total_rows": int(len(full)),
            "full_landscape_unique_pair_ids": int(full["name"].nunique()),
            "matched_ids": full_rec["matched_n"],
            "FINCHES_only_examples": full_rec["left_only"][:50],
            "prediction_only_examples": full_rec["right_only"][:50],
        }
        raise RuntimeError("FINCHES/full-landscape exact ID reconciliation failed because prediction IDs are missing from FINCHES: " + json.dumps(details, indent=2))
    # The pre-existing FINCHES matrix is a larger Cartesian table. All final
    # full-landscape prediction IDs are present exactly; use that exact subset
    # for Panel S6B and report FINCHES-only extras in the audit.
    unsim_m = full.merge(eps_long[["name", "rgg", "tad", "epsilon"]], on="name", how="left", validate="one_to_one")

    eps_min, eps_max = eps_info["epsilon_min"], eps_info["epsilon_max"]
    y_min = float(min(train_m["B22_corr1_log10"].min(), unsim_m["final_hybrid_prediction"].min()))
    y_max = float(max(train_m["B22_corr1_log10"].max(), unsim_m["final_hybrid_prediction"].max()))

    train_m["eps_scaled"] = (eps_max - train_m["epsilon"]) / (eps_max - eps_min)
    unsim_m["eps_scaled"] = (eps_max - unsim_m["epsilon"]) / (eps_max - eps_min)
    train_m["y_scaled"] = minmax(train_m["B22_corr1_log10"], y_min, y_max)
    unsim_m["y_scaled"] = minmax(unsim_m["final_hybrid_prediction"], y_min, y_max)

    train_m.to_csv(OUTDIR / "FigS6A_training_scaled_merged.csv", index=False)
    unsim_m.to_csv(OUTDIR / "FigS6B_unsim_scaled_merged.csv", index=False)

    r_train = pearson(train_m["epsilon"], train_m["B22_corr1_log10"])
    rho_train = spearman(train_m["epsilon"], train_m["B22_corr1_log10"])
    r_unsim = pearson(unsim_m["epsilon"], unsim_m["final_hybrid_prediction"])
    rho_unsim = spearman(unsim_m["epsilon"], unsim_m["final_hybrid_prediction"])
    r_train_scaled = pearson(train_m["eps_scaled"], train_m["y_scaled"])
    r_unsim_scaled = pearson(unsim_m["eps_scaled"], unsim_m["y_scaled"])

    save_panel_a(train_m, r_train_scaled)
    save_panel_b(unsim_m, r_unsim_scaled)

    test_audit = {"possible": False, "matched_n": 0, "unmatched_n": None, "Pearson": None, "Spearman": None}
    if TEST_CSV.exists():
        test = pd.read_csv(TEST_CSV, usecols=["pair_id", "simulated_target"])
        test = test.rename(columns={"pair_id": "name", "simulated_target": "B22_corr1_log10"})
        test_rec = reconcile(test["name"], eps_long["name"])
        matched = test.merge(eps_long[["name", "rgg", "tad", "epsilon"]], on="name", how="inner", validate="one_to_one")
        matched.to_csv(OUTDIR / "FINCHES_test_1794_audit.csv", index=False)
        if len(matched):
            test_audit = {
                "possible": True,
                "test_total_n": int(len(test)),
                "matched_n": int(len(matched)),
                "unmatched_n": int(len(test) - len(matched)),
                "Pearson": pearson(matched["epsilon"], matched["B22_corr1_log10"]),
                "Spearman": spearman(matched["epsilon"], matched["B22_corr1_log10"]),
                "test_only_examples": test_rec["left_only"][:50],
            }
        else:
            test_audit = {"possible": False, "test_total_n": int(len(test)), "matched_n": 0, "unmatched_n": int(len(test)), "reason": "FINCHES epsilon table does not contain the test pair IDs"}

    audit_lines = [
        "FINCHES final model audit",
        "",
        f"source filename: {eps_info['source_filename']}",
        f"pair-ID column: {eps_info['pair_id_column']}",
        f"epsilon column: {eps_info['epsilon_column']}",
        f"FINCHES total rows: {eps_info['total_rows']}",
        f"FINCHES unique n: {eps_info['unique_pair_ids']}",
        f"FINCHES duplicate pair IDs: {eps_info['duplicate_pair_ids']}",
        "",
        "Panel S6A — FINCHES vs definitive 917 simulated target",
        f"simulated-target source: {DEV_CSV}",
        f"matched n: {len(train_m)}",
        f"unmatched development IDs: {dev_rec['left_only'][:100]}",
        f"Pearson raw epsilon vs target: {r_train:.12g}",
        f"Spearman raw epsilon vs target: {rho_train:.12g}",
        f"Pearson scaled convention used in title: {r_train_scaled:.12g}",
        "",
        "Panel S6B — FINCHES vs definitive full-landscape final_hybrid_prediction",
        f"final prediction source: {FULL_CSV}",
        f"FINCHES n: {eps_info['unique_pair_ids']}",
        f"final prediction n: {len(full)}",
        f"matched n: {len(unsim_m)}",
        f"FINCHES-only IDs: {full_rec['left_only'][:100]}",
        f"prediction-only IDs: {full_rec['right_only'][:100]}",
        f"Pearson raw epsilon vs final prediction: {r_unsim:.12g}",
        f"Spearman raw epsilon vs final prediction: {rho_unsim:.12g}",
        f"Pearson scaled convention used in title: {r_unsim_scaled:.12g}",
        "",
        "Scaling",
        f"epsilon raw min: {eps_min:.12g}",
        f"epsilon raw max: {eps_max:.12g}",
        f"common y raw min: {y_min:.12g}",
        f"common y raw max: {y_max:.12g}",
        "",
        "Test 1,794 audit",
        json.dumps(test_audit, indent=2, sort_keys=True),
        "",
        "FINCHES epsilon values were taken from the pre-existing FINCHES output and were not recomputed.",
        "The 917 simulated targets were taken from the definitive development table.",
        "The full-landscape hybrid predictions were taken from the already-finalised full-landscape prediction table and were not regenerated.",
        "No hybrid model inference was performed during the FINCHES update.",
        "FINCHES panels use accepted epsilon and final prediction values.",
        "Pair reconciliation was performed using exact pair IDs.",
    ]
    (OUTDIR / "FINCHES_final_model_audit.txt").write_text("\n".join(audit_lines) + "\n")
    print("Wrote:", OUTDIR)


if __name__ == "__main__":
    main()
