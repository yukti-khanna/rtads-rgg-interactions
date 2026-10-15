#!/usr/bin/env python3
"""Create side-by-side original/new scatter style comparison."""
from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parents[1] / "outputs" / "mplconfig"))

import matplotlib.pyplot as plt
import matplotlib.image as mpimg

from fig5_revision_plot_utils import MAIN_OUT, ROOT, ensure_dir


ORIGINAL = ROOT / "Fig5_panels" / "Fig5D_heldout_pred_vs_sim.png"
REVISED = MAIN_OUT / "revised_Fig5.png"


def main():
    outdir = ensure_dir(MAIN_OUT)
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.4), constrained_layout=True)
    for ax, path, title in [
        (axes[0], ORIGINAL, "Original manuscript scatter style"),
        (axes[1], REVISED, "Revised Figure 5 using old scatter style"),
    ]:
        if path.exists():
            ax.imshow(mpimg.imread(path))
        else:
            ax.text(0.5, 0.5, f"Missing\n{path}", ha="center", va="center", transform=ax.transAxes)
        ax.set_title(title)
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)
    fig.savefig(outdir / "scatter_style_comparison.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
