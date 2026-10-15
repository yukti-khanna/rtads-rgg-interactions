#!/usr/bin/env python3
"""Shared utilities for revised Figure 5 plotting scripts."""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
MAIN_OUT = ROOT / "outputs" / "revised_fig5_main"
SUPP_OUT = ROOT / "outputs" / "revised_fig5_supplement"
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / "outputs" / "mplconfig"))


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path)


def metrics(y, p) -> dict[str, float]:
    y = pd.to_numeric(pd.Series(y), errors="coerce").to_numpy(float)
    p = pd.to_numeric(pd.Series(p), errors="coerce").to_numpy(float)
    ok = np.isfinite(y) & np.isfinite(p)
    y = y[ok]
    p = p[ok]
    ss_res = np.sum((y - p) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    return {
        "n": int(len(y)),
        "R2": float(1 - ss_res / ss_tot) if ss_tot else np.nan,
        "RMSE": float(np.sqrt(np.mean((y - p) ** 2))),
        "MAE": float(np.mean(np.abs(y - p))),
        "Pearson": float(pd.Series(y).corr(pd.Series(p), method="pearson")),
        "Spearman": float(pd.Series(y).corr(pd.Series(p), method="spearman")),
    }


def add_panel_label(ax, label: str) -> None:
    ax.text(-0.14, 1.08, label, transform=ax.transAxes, fontsize=13, fontweight="bold", va="top")


def save_all(fig, outdir: Path, stem: str) -> None:
    ensure_dir(outdir)
    try:
        from fig5_plotting_style import clean_figure_axes
        clean_figure_axes(fig)
    except Exception:
        pass
    fig.savefig(outdir / f"{stem}.pdf", bbox_inches="tight")
    fig.savefig(outdir / f"{stem}.svg", bbox_inches="tight")
    fig.savefig(outdir / f"{stem}.png", dpi=300, bbox_inches="tight")


def clean_model_label(model: str) -> str:
    mapping = {
        "F0_OCN_only": "-NCP only",
        "F1_OCN_plus_SCD": "-NCP + charge patterning",
        "C_only_hydropathy_stickiness": "non-electrostatic only",
        "F2_OCN_plus_hydropathy": "-NCP + hydropathy/stickiness",
        "F6_full_M4": "full non-redundant global model",
        "F2": "Global model",
        "F2-P1": "+ local charge heterogeneity",
        "F2-P7": "+ broader local sequence context",
        "F2-P8": "+ AH-energy heterogeneity",
    }
    return mapping.get(str(model), str(model).replace("OCN", "-NCP"))


def rename_charge_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "Opposite_Charge_Number" in out.columns and "-NCP" not in out.columns:
        out["-NCP"] = out["Opposite_Charge_Number"]
    if "OCN" in out.columns and "-NCP" not in out.columns:
        out["-NCP"] = out["OCN"]
    return out


def placeholder(ax, label: str, text: str) -> None:
    add_panel_label(ax, label)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_color("0.75")
    ax.text(0.5, 0.5, text, ha="center", va="center", transform=ax.transAxes, color="0.35", fontsize=10)
