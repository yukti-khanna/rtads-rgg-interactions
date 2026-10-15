#!/usr/bin/env python3
"""Build corrected S4D and proposed Supplementary Figure S4 v4.

This focused builder reads historical pair lists and completed audit outputs.
It does not run CALVADOS, recalculate B22, or invoke the full provenance audit.
"""
from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
import pandas as pd
from PIL import Image
from pypdf import PdfReader, PdfWriter, Transformation


ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
LISTS = Path("/Users/yuktikhanna/calvados/lists")
FINAL_FIG4 = ROOT / "reviewer_analysis" / "final_fig4_revision"

BASE_S4_PDF = OUT / "revised_Figure_S4_A-H_proposed_v2.pdf"
BIN_COVERAGE = FINAL_FIG4 / "S4_negative_NCP_bin_coverage.csv"
CURRENT_MAIN_SCRIPT = FINAL_FIG4 / "build_final_fig4_revision.py"

FINAL_LIST = LISTS / "sim_pairs_for_training_270525.txt"
HELD_AHD = LISTS / "tads_taken_out.txt"
HELD_RGG = LISTS / "rgs_taken_out.txt"

S4D_PDF = OUT / "S4D_training_set_construction.pdf"
S4D_PNG = OUT / "S4D_training_set_construction.png"
S4D_SOURCE = OUT / "S4D_training_set_construction_source_data.csv"
V4_PDF = OUT / "revised_Figure_S4_A-H_proposed_v4.pdf"
V4_PNG = OUT / "revised_Figure_S4_A-H_proposed_v4.png"
ADDENDUM = OUT / "TRAINING_SET_SELECTION_PROVENANCE_CORRECTED_STAGES_ADDENDUM.md"
TEXT_NOTES = OUT / "S4_revision_text_notes.md"
B22_CANDIDATE_PDF = OUT / "B22_distribution_stability_candidate.pdf"
B22_CANDIDATE_PNG = OUT / "B22_distribution_stability_candidate.png"

BLUE = "#0072B2"
ORANGE = "#D55E00"
GREY = "#C9C9C9"
BLACK = "#111111"


def require(path: Path) -> Path:
    if not path.exists():
        raise FileNotFoundError(path)
    return path


def read_names(path: Path) -> tuple[list[str], set[str]]:
    rows = [x.strip() for x in require(path).read_text().splitlines() if x.strip()]
    return rows, set(rows)


def pair_parts(pair: str) -> tuple[str, str]:
    if "__" not in pair:
        raise ValueError(f"Malformed pair ID: {pair}")
    return tuple(pair.split("__", 1))  # type: ignore[return-value]


def style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "font.size": 7,
            "axes.titlesize": 7,
            "axes.labelsize": 6.5,
            "xtick.labelsize": 5.5,
            "ytick.labelsize": 6,
            "legend.fontsize": 5.2,
            "axes.linewidth": 0.6,
            "xtick.major.width": 0.5,
            "ytick.major.width": 0.5,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def reconstruct_stages() -> tuple[pd.DataFrame, dict]:
    final_rows, final = read_names(FINAL_LIST)
    base_rows, base = read_names(LISTS / "final_test_names.txt")
    all_rows, all_test = read_names(LISTS / "all_test_names.txt")
    low_rows, low = read_names(LISTS / "200_pairs_outside_filter_less_than_lambda045_names.txt")
    ncp1_rows, ncp1 = read_names(LISTS / "ocn_names.txt")
    ncp2_rows, ncp2 = read_names(LISTS / "ocn_added_names.txt")
    _, held_ahd = read_names(HELD_AHD)
    _, held_rgg = read_names(HELD_RGG)

    expected_unique = {
        "final": (final_rows, final, 917),
        "base": (base_rows, base, 500),
        "all_test": (all_rows, all_test, 600),
        "low_lambda": (low_rows, low, 200),
        "first_ncp": (ncp1_rows, ncp1, 166),
        "further_ncp": (ncp2_rows, ncp2, 74),
    }
    for label, (rows, unique, expected) in expected_unique.items():
        if len(rows) != expected or len(unique) != expected:
            raise ValueError(f"{label}: expected {expected} unique rows, found {len(rows)} rows/{len(unique)} unique")
    if len(held_ahd) != 23 or len(held_rgg) != 78:
        raise ValueError("Reserved identity counts do not match 23 AHD/78 RG/RGG")
    if not base.issubset(all_test) or len(all_test - base) != 100:
        raise ValueError("final_test_names.txt is not a 500-pair subset plus exactly 100 additions in all_test_names.txt")

    def retained(pairs: set[str]) -> set[str]:
        return {
            pair
            for pair in pairs
            if pair_parts(pair)[0] not in held_rgg and pair_parts(pair)[1] not in held_ahd
        }

    raw_batches = [all_test, low, ncp1, ncp2]
    retained_batches = [retained(batch) for batch in raw_batches]
    sources = [
        LISTS / "all_test_names.txt",
        LISTS / "200_pairs_outside_filter_less_than_lambda045_names.txt",
        LISTS / "ocn_names.txt",
        LISTS / "ocn_added_names.txt",
    ]
    labels = [
        "Initial feature-guided + outside-filter sampling",
        "Low-lambda sampling",
        "First −NCP-directed sampling",
        "Further −NCP-directed sampling",
    ]

    cumulative: set[str] = set()
    stages: list[set[str]] = []
    rows = []
    for number, (label, source, raw, kept) in enumerate(
        zip(labels, sources, raw_batches, retained_batches), start=1
    ):
        before = set(cumulative)
        cumulative |= kept
        stages.append(set(cumulative))
        rows.append(
            {
                "stage_number": number,
                "stage_label": label,
                "historical_source_file": str(source),
                "raw_stage_size": len(raw),
                "heldout_excluded_from_stage": len(raw - kept),
                "retained_raw_pairs": len(kept),
                "overlap_with_previous_retained_set": len(kept & before),
                "net_new_retained_pairs": len(cumulative - before),
                "cumulative_retained_pairs": len(cumulative),
            }
        )

    stage_df = pd.DataFrame(rows)
    expected_cumulative = [498, 678, 843, 917]
    expected_new = [498, 180, 165, 74]
    if stage_df.cumulative_retained_pairs.tolist() != expected_cumulative:
        raise ValueError(f"Verified cumulative counts differ from expected: {stage_df.cumulative_retained_pairs.tolist()}")
    if stage_df.net_new_retained_pairs.tolist() != expected_new:
        raise ValueError(f"Verified net-new counts differ from expected: {stage_df.net_new_retained_pairs.tolist()}")
    if not all(stages[i].issubset(stages[i + 1]) for i in range(3)):
        raise ValueError("Post-exclusion scientific stages are not nested")

    missing = final - stages[-1]
    extra = stages[-1] - final
    if missing or extra:
        (OUT / "S4D_diagnostic_missing_from_reconstructed_final.txt").write_text("\n".join(sorted(missing)) + "\n")
        (OUT / "S4D_diagnostic_extra_in_reconstructed_final.txt").write_text("\n".join(sorted(extra)) + "\n")
        raise ValueError(f"Stage-4 reconstruction differs from authoritative final: missing={len(missing)}, extra={len(extra)}")

    final_reserved_ahd = sum(pair_parts(p)[1] in held_ahd for p in final)
    final_reserved_rgg = sum(pair_parts(p)[0] in held_rgg for p in final)
    if final_reserved_ahd or final_reserved_rgg:
        raise ValueError("Authoritative final contains reserved identities")

    held_features = pd.read_csv(require(ROOT / "task3_sampling_bias" / "heldout_23x78_features.csv"))
    exact_pair_overlap = len(final & set(held_features.pair_id.astype(str)))
    if exact_pair_overlap:
        raise ValueError("Final training set overlaps held-out pair set")

    # Historical bookkeeping checks, reported but not plotted.
    _, s911 = read_names(LISTS / "sim_pairs_for_training_090425.txt")
    _, s1039 = read_names(LISTS / "sim_pairs_for_training_220425.txt")
    _, temporarily_removed = read_names(LISTS / "names_taken_out.txt")
    raw_stage3_union = all_test | low | ncp1
    raw_stage4_union = raw_stage3_union | ncp2
    bookkeeping = {
        "state_911_is_raw_stage3_minus_names_taken_out": (raw_stage3_union - temporarily_removed) == s911,
        "state_1039_is_complete_raw_four_stage_union": raw_stage4_union == s1039,
        "state_1039_reconstructed_from_911_restoration_and_additions": (s911 | temporarily_removed | ncp2) == s1039,
        "names_taken_out_count": len(temporarily_removed),
        "pre_exclusion_count": len(s1039),
    }
    if not all(v for k, v in bookkeeping.items() if k.startswith("state_")):
        raise ValueError(f"Historical bookkeeping relationships were not reproduced: {bookkeeping}")

    meta = {
        "authoritative_final": str(FINAL_LIST),
        "final_rows": len(final_rows),
        "final_unique": len(final),
        "held_ahd_count": len(held_ahd),
        "held_rgg_count": len(held_rgg),
        "final_reserved_ahd": final_reserved_ahd,
        "final_reserved_rgg": final_reserved_rgg,
        "exact_pair_overlap": exact_pair_overlap,
        "reconstructed_missing": len(missing),
        "reconstructed_extra": len(extra),
        "bookkeeping": bookkeeping,
    }
    return stage_df, meta


def plot_s4d(ax: plt.Axes, panel_letter: bool = True) -> None:
    source = pd.read_csv(require(S4D_SOURCE))
    x = np.arange(1, 5)
    y = source.cumulative_retained_pairs.to_numpy(int)
    ax.plot(x, y, color=BLUE, marker="o", ms=4.0, lw=1.25, zorder=3)
    ax.fill_between(x, 0, y, color=BLUE, alpha=0.09, zorder=1)
    annotations = ["n = 498", "+180", "+165", "+74"]
    for xi, yi, annotation in zip(x, y, annotations):
        ax.annotate(
            annotation,
            (xi, yi),
            xytext=(0, 7),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=5.5,
        )
    ax.set_xticks(
        x,
        ["Initial +\noutside-filter", "+ low-$\\lambda$", "+ $-$NCP\nsampling", "+ further $-$NCP\nsampling"],
    )
    ax.set_xlabel("Sampling stage")
    ax.set_ylabel("Cumulative training pairs")
    ax.set_xlim(0.75, 4.25)
    ax.set_ylim(0, 1050)
    ax.grid(axis="y", color="#DDDDDD", lw=0.4, zorder=0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    if panel_letter:
        ax.text(-0.19, 1.07, "D", transform=ax.transAxes, fontsize=9, fontweight="bold", va="bottom")


def plot_s4h(ax: plt.Axes) -> plt.Axes:
    coverage = pd.read_csv(require(BIN_COVERAGE))
    centers = (coverage.bin_lower + coverage.bin_upper) / 2.0
    widths = coverage.bin_upper - coverage.bin_lower
    ax.bar(centers, coverage.candidate_count, width=widths * 0.85, color=GREY, label="candidate pairs")
    ax.set_yscale("log")
    ax.set_xlabel("−NCP")
    ax.set_ylabel("Candidate pairs, log scale")
    ax2 = ax.twinx()
    ax2.set_zorder(ax.get_zorder() + 1)
    ax2.patch.set_visible(False)
    ax2.plot(centers, coverage.selected_917_count, color=BLUE, marker="o", ms=2.2, lw=1.0, label="selected 917")
    ax2.set_ylabel("Selected pairs")
    ax.spines["top"].set_visible(False)
    ax2.spines["top"].set_visible(False)
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, frameon=False, loc="upper right", fontsize=4.6)
    return ax2


def save_s4d() -> None:
    fig, ax = plt.subplots(figsize=(3.05, 2.35))
    plot_s4d(ax, panel_letter=True)
    fig.subplots_adjust(left=0.18, right=0.98, bottom=0.27, top=0.91)
    fig.savefig(S4D_PDF, bbox_inches="tight", pad_inches=0.03)
    fig.savefig(S4D_PNG, dpi=600, bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)


def save_overlay(width_pt: float, height_pt: float, path: Path) -> None:
    fig = plt.figure(figsize=(width_pt / 72.0, height_pt / 72.0))
    fig.patch.set_alpha(0.0)
    # Replace only D and H. A-C and E-G remain inherited from the prior PDF.
    fig.add_artist(Rectangle((0.75, 0.50), 0.25, 0.50, transform=fig.transFigure, facecolor="white", edgecolor="none", zorder=1))
    fig.add_artist(Rectangle((0.75, 0.00), 0.25, 0.50, transform=fig.transFigure, facecolor="white", edgecolor="none", zorder=1))

    ax_d = fig.add_axes([0.82, 0.61, 0.17, 0.34], zorder=2)
    plot_s4d(ax_d, panel_letter=False)
    ax_d.tick_params(axis="x", labelsize=4.5)
    ax_d.set_xlabel("Sampling stage", fontsize=5.8)
    ax_d.set_ylabel("Cumulative training pairs", fontsize=5.8)
    for text in ax_d.texts:
        text.set_fontsize(min(text.get_fontsize(), 4.9))
    fig.text(0.755, 0.98, "D", fontsize=9, fontweight="bold", ha="left", va="top", zorder=3)

    # Leave room at the right edge for the selected-pair secondary-axis label.
    ax_h = fig.add_axes([0.81, 0.08, 0.15, 0.34], zorder=2)
    ax_h_right = plot_s4h(ax_h)
    ax_h.tick_params(labelsize=5.0)
    ax_h.set_xlabel("−NCP", fontsize=5.8)
    ax_h.set_ylabel("Candidate pairs, log scale", fontsize=5.4)
    ax_h_right.tick_params(labelsize=5.0)
    ax_h_right.set_ylabel("Selected pairs", fontsize=5.4)
    fig.text(0.755, 0.48, "H", fontsize=9, fontweight="bold", ha="left", va="top", zorder=3)

    fig.savefig(path, transparent=True, dpi=300)
    plt.close(fig)


def assemble_v4() -> None:
    reader = PdfReader(require(BASE_S4_PDF))
    if len(reader.pages) != 1:
        raise ValueError("Expected one-page proposed S4 base PDF")
    page = reader.pages[0]
    width_pt = float(page.mediabox.width)
    height_pt = float(page.mediabox.height)
    overlay = OUT / ".S4_v4_overlay.pdf"
    save_overlay(width_pt, height_pt, overlay)
    overlay_page = PdfReader(overlay).pages[0]
    overlay_page.add_transformation(
        Transformation().scale(
            width_pt / float(overlay_page.mediabox.width),
            height_pt / float(overlay_page.mediabox.height),
        )
    )
    page.merge_page(overlay_page)
    writer = PdfWriter()
    writer.add_page(page)
    with V4_PDF.open("wb") as handle:
        writer.write(handle)
    overlay.unlink()

    prefix = OUT / ".S4_v4_render"
    subprocess.run(
        ["pdftoppm", "-png", "-r", "600", "-singlefile", str(V4_PDF), str(prefix)],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    rendered = prefix.with_suffix(".png")
    with Image.open(rendered) as image:
        image.save(V4_PNG, dpi=(600, 600))
    rendered.unlink()


def write_text_outputs(stage: pd.DataFrame, meta: dict, main_content: str) -> None:
    stage_lines = "\n".join(
        f"- Stage {r.stage_number}: {r.stage_label}: raw batch {r.raw_stage_size}, "
        f"held-out-identity exclusions {r.heldout_excluded_from_stage}, "
        f"net-new retained {r.net_new_retained_pairs}, cumulative {r.cumulative_retained_pairs}."
        for r in stage.itertuples(index=False)
    )
    addendum = f"""# Training-set selection provenance: corrected-stage addendum

## Corrected publication-facing sampling stages

### A. Raw historical selected-pair counts

- `final_test_names.txt` contains 500 unique initial feature-guided pairs.
- `all_test_names.txt` contains those 500 plus exactly 100 outside-filter pairs, giving 600 raw Stage-1 pairs.
- The low-lambda batch contains 200 unique pairs.
- The first historical OCN-named batch contains 166 unique pairs; publication text refers to this coordinate as −NCP.
- The further OCN-named batch contains 74 unique pairs.

The temporary 911-pair bookkeeping file is exactly the 965-pair raw union through the first −NCP batch minus the 54 pairs in `names_taken_out.txt`. The 1,039-pair bookkeeping file exactly restores those 54 pairs and adds the 74-pair further −NCP batch. These file-management states are not plotted as scientific sampling stages.

### B. Post-held-out-exclusion counts contributing to the final training set

Reserved AHD and RG/RGG region identities were excluded consistently from every cumulative stage:

{stage_lines}

The reconstructed Stage-4 set contains 917 unique pairs and is exactly identical to the authoritative `sim_pairs_for_training_270525.txt` list (missing = {meta['reconstructed_missing']}, extra = {meta['reconstructed_extra']}). No final training pair contains a reserved AHD or RG/RGG identifier. Exact pair overlap with the held-out cross-product is zero.

The earlier audit found identical amino-acid strings under some different region identifiers. Therefore this addendum claims identifier-level exclusion, not literal absence of every duplicated sequence string.
"""
    ADDENDUM.write_text(addendum)

    notes = f"""# Supplementary Figure S4 revision notes

## 1. Verified final S4D stage counts

{stage_lines}

## 2. Manuscript-safe description

### Methods-level detail

The final 917-pair simulation training set was assembled by staged, feature-guided sampling. Reserved held-out AHD and RG/RGG region identities were excluded consistently from each historical batch. The combined initial feature-guided and outside-filter selections contributed 498 retained pairs. Low-lambda sampling increased the cumulative set to 678 pairs, followed by −NCP-directed additions to 843 and then 917 pairs. The reconstructed 917-pair set exactly matches `sim_pairs_for_training_270525.txt`.

### Results-level interpretation

The staged design broadened coverage beyond the initially favoured feature region and subsequently improved representation along −NCP. It should not be described as a single random or unbiased sample.

## 3. Proposed S4D legend

(D) Staged construction of the final simulation training set after exclusion of reserved held-out AHD and RG/RGG region identities. The initial feature-guided and outside-filter selections contributed 498 retained pairs. Addition of low-lambda controls increased the set to 678 pairs, followed by −NCP-directed sampling to 843 pairs and a final −NCP-directed addition to 917 pairs. Counts represent cumulative unique pairs contributing to the authoritative final training set.

## 4. Proposed complete S4 A-H legend outline

Supplementary Figure S4. Provenance and coverage of the staged Figure 4 simulation panel. (A-C) Historical pilot relationships between simulated interaction strength and charge complementarity, mean pair NCPR, and average lambda. (D) Cumulative construction of the final 917-pair training set after consistent exclusion of reserved held-out AHD and RG/RGG region identities. (E-G) Distributions of −NCP, total pair length, and average lambda across the complete candidate landscape, the feature-guided training set, and the held-out set. (H) Reconstructed −NCP-bin coverage of the final 917-pair simulation panel, showing candidate-pair and selected-pair counts.

## 5. Provenance caveats

- The 911- and 1,039-pair files are temporary bookkeeping states and are not plotted as scientific sampling stages.
- Historical filenames retain `ocn`, but publication-facing text uses −NCP.
- Reserved region identifiers are absent from training; literal sequence-string uniqueness is not claimed because distinct identifiers can share identical amino-acid strings.
- The B22 stability metrics remain retrospective support for the historical stopping decision, not a prospective numerical stopping rule.
- Current main Figure 4D content: {main_content}.
"""
    TEXT_NOTES.write_text(notes)


def current_main_content() -> str:
    text = require(CURRENT_MAIN_SCRIPT).read_text()
    if "fig4_917_sampling_membership.csv" in text:
        return "917-pair training relationship"
    if "heldout_23x78_features.csv" in text:
        return "1,794-pair held-out relationship"
    return "unresolved from current build script"


def main() -> None:
    style()
    stage, meta = reconstruct_stages()
    stage.to_csv(S4D_SOURCE, index=False)
    save_s4d()
    assemble_v4()

    shutil.copy2(require(OUT / "S4H_B22_distribution_stability.pdf"), B22_CANDIDATE_PDF)
    shutil.copy2(require(OUT / "S4H_B22_distribution_stability.png"), B22_CANDIDATE_PNG)
    main_content = current_main_content()
    write_text_outputs(stage, meta, main_content)

    print(f"AUTHORITATIVE FINAL TRAINING LIST: {FINAL_LIST}")
    print(f"FINAL UNIQUE PAIRS: {meta['final_unique']}")
    print(f"RAW STAGE 1: {stage.loc[0, 'raw_stage_size']}")
    print(f"POST-EXCLUSION STAGE 1: {stage.loc[0, 'cumulative_retained_pairs']}")
    print(f"LOW-LAMBDA RAW ADDITIONS: {stage.loc[1, 'raw_stage_size']}")
    print(f"LOW-LAMBDA RETAINED ADDITIONS: {stage.loc[1, 'net_new_retained_pairs']}")
    print(f"CUMULATIVE STAGE 2: {stage.loc[1, 'cumulative_retained_pairs']}")
    print(f"FIRST −NCP RAW PAIRS: {stage.loc[2, 'raw_stage_size']}")
    print(f"FIRST −NCP NET-NEW RETAINED: {stage.loc[2, 'net_new_retained_pairs']}")
    print(f"CUMULATIVE STAGE 3: {stage.loc[2, 'cumulative_retained_pairs']}")
    print(f"FURTHER −NCP RAW PAIRS: {stage.loc[3, 'raw_stage_size']}")
    print(f"FURTHER −NCP NET-NEW RETAINED: {stage.loc[3, 'net_new_retained_pairs']}")
    print(f"CUMULATIVE STAGE 4: {stage.loc[3, 'cumulative_retained_pairs']}")
    print("RECONSTRUCTED FINAL = AUTHORITATIVE FINAL: YES")
    print(f"MISSING FROM RECONSTRUCTED FINAL: {meta['reconstructed_missing']}")
    print(f"EXTRA IN RECONSTRUCTED FINAL: {meta['reconstructed_extra']}")
    print(f"RESERVED AHD IDENTITIES: {meta['held_ahd_count']}")
    print(f"RESERVED RG/RGG IDENTITIES: {meta['held_rgg_count']}")
    print(f"FINAL TRAINING PAIRS CONTAINING RESERVED AHD: {meta['final_reserved_ahd']}")
    print(f"FINAL TRAINING PAIRS CONTAINING RESERVED RG/RGG: {meta['final_reserved_rgg']}")
    print(f"S4D CREATED: {S4D_PDF}")
    print(f"ORIGINAL S4H PRESERVED: {BIN_COVERAGE}")
    print(f"FINAL S4 A-H CREATED: {V4_PDF}")
    print("S4I PRESENT IN FINAL SUPPLEMENT: NO")
    print(f"B22 STABILITY AUDIT PRESERVED: {B22_CANDIDATE_PDF}")
    print(f"MAIN FIG4D CURRENT CONTENT: {main_content}")
    print("OUTPUT PATHS:")
    for path in [S4D_SOURCE, S4D_PDF, S4D_PNG, V4_PDF, V4_PNG, ADDENDUM, TEXT_NOTES, B22_CANDIDATE_PDF, B22_CANDIDATE_PNG]:
        print(path)


if __name__ == "__main__":
    main()
