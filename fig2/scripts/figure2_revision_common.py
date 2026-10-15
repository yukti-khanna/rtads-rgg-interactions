#!/usr/bin/env python3
"""Plot-only finalisation using the supplied 168-model Figure 2 results.

The final catalogue criterion is 168/168 positive models (100% consensus),
yielding the stated 230 AHD regions in 190 TFs. The 85% threshold is sensitivity
only. If the summary lacks a 100% point, count exact unanimity in the existing
cached per-region maximum-support tables; never extrapolate or rescore models.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import platform
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrow
from matplotlib.ticker import PercentFormatter
import numpy as np
import pandas as pd
from PIL import Image

FEATURES = ["Aromatic fraction", "WF complexity", "SCD", "κ"]
UNION = "Union of >=1 external dataset"
MULTI = "External regions supported by >=2 datasets"
DATASETS = ["Soto 2022", "Kotha & Staller 2023 high-confidence",
            "Kotha & Staller 2023 GSL", "Staller 2022"]


def read_table(path):
    return pd.read_csv(path, sep="\t" if Path(path).suffix == ".tsv" else ",")


def wilson_interval(successes, total):
    """Wilson 95% CI for the added cached-data endpoint; no resampling."""
    if total <= 0:
        raise ValueError("Cannot create a recognition point without a denominator")
    z = 1.959963984540054
    rate = successes / total
    denominator = 1 + z*z/total
    centre = (rate + z*z/(2*total)) / denominator
    half = z * np.sqrt(rate*(1-rate)/total + z*z/(4*total*total)) / denominator
    return min(rate, max(0., centre-half)), max(rate, min(1., centre+half))


class Revision:
    def __init__(self, root, config, profile="all"):
        self.root = Path(root).resolve()
        self.config_path = Path(config).resolve()
        self.config = json.loads(self.config_path.read_text())
        self.paths = {key: self.root / value for key, value in self.config.items()
                      if isinstance(value, str) and key != "output_directory"}
        table_keys = {
            "plot_data", "plot_stats", "effect_sizes", "recurrence_curves",
            "external_recurrence", "matched_controls", "control_statistics",
            "nonfamily_curves", "family_enrichment", "identity_rates",
            "protein_membership",
        }
        profile_keys = {
            "manuscript": {
                "template_script", "recurrence_curves", "external_recurrence",
                "matched_controls", "control_statistics", "nonfamily_curves",
            },
            "main": {
                "template_script", "original_assembly_png", "plot_data",
                "plot_stats", "effect_sizes", "recurrence_curves",
                "external_recurrence", "matched_controls", "control_statistics",
            },
            "supplement": {
                "template_script", "original_assembly_png", "recurrence_curves",
                "external_recurrence", "matched_controls", "control_statistics",
                "nonfamily_curves", "family_enrichment", "identity_rates",
                "protein_membership",
            },
        }
        required_keys = set(self.paths) if profile == "all" else profile_keys[profile]
        missing = [str(self.paths[key]) for key in required_keys if not self.paths[key].is_file()]
        if missing:
            raise FileNotFoundError("Missing plotting input(s); no analysis will be rerun:\n" + "\n".join(missing))
        spec = importlib.util.spec_from_file_location("original_fig2_style", self.paths["template_script"])
        self.template = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.template)
        self.style = {key: mpl.rcParams[key] for key in [
            "font.family", "font.sans-serif", "font.size", "axes.labelsize",
            "axes.titlesize", "xtick.labelsize", "ytick.labelsize", "legend.fontsize",
            "axes.linewidth", "xtick.major.width", "ytick.major.width", "figure.dpi",
            "savefig.dpi", "savefig.bbox", "pdf.fonttype", "ps.fonttype", "svg.fonttype"]}
        self.tables = {key: read_table(self.paths[key]) for key in table_keys & required_keys}
        self.out = self.root / self.config["output_directory"]
        self.out.mkdir(parents=True, exist_ok=True)
        self.exports = []
        self.notes = []
        self.models = int(self.config["catalogue"]["models"])
        self.final_threshold = float(self.config["catalogue"]["consensus_fraction"])
        if self.models != 168 or self.final_threshold != 1.0:
            raise ValueError("Use the updated config: 168 models and consensus_fraction=1.0; 85% is sensitivity only")
        self.ensemble_label = f"{self.models}-model ensemble"
        if "plot_data" in self.tables:
            self.check_cached_features()
        self.prepare_curves()
        if "protein_membership" in self.tables:
            self.prepare_protein_counts()
        if self.paths["original_assembly_png"].is_file():
            im = Image.open(self.paths["original_assembly_png"])
            dpi = im.info.get("dpi")
            if dpi and min(dpi) > 0:
                self.assembly_size = (im.width / dpi[0], im.height / dpi[1])
                self.assembly_dimension_source = f"original PNG {im.width}x{im.height} pixels / DPI {dpi}"
            else:
                self.assembly_size = (8.214, 4.854)
                self.assembly_dimension_source = "fallback 8.214x4.854 inches; original PNG lacks physical DPI"
            im.close()
        else:
            self.assembly_size = (8.214, 4.854)
            self.assembly_dimension_source = "fallback 8.214x4.854 inches; original assembly not required by this profile"
        self.colors = {"AD": self.template.COLOR_BLUE, "control": "#7F7F7F",
                       "multi": self.template.COLOR_ORANGE,
                       DATASETS[0]: self.template.COLOR_GREEN,
                       DATASETS[1]: self.template.COLOR_PURPLE,
                       DATASETS[2]: self.template.COLOR_SKYBLUE,
                       DATASETS[3]: self.template.COLOR_VERMILLION}

    def check_cached_features(self):
        data, effects = self.tables["plot_data"], self.tables["effect_sizes"]
        expected = {"Seed R-TADs", "Random IDRs"}
        if set(data.group.unique()) != expected:
            raise ValueError(f"Unexpected cached group labels: {list(data.group.unique())}")
        if not set(FEATURES).issubset(data.feature):
            raise ValueError("The four required Fig. 2B features are not present")
        if effects.feature.duplicated().any():
            raise ValueError("Duplicate features in the effect-size summary")
        for name in FEATURES:
            sub = data[data.feature.eq(name)]
            row = effects.set_index("feature").loc[name]
            for group, ncol in [("Seed R-TADs", "n_seed"), ("Random IDRs", "n_random")]:
                n = int(sub.loc[sub.group.eq(group), "value"].notna().sum())
                if n != int(row[ncol]):
                    raise ValueError(f"Cached row count and effect analysis disagree for {name}: {group}")
            cached = self.tables["plot_stats"].set_index("feature").loc[name]
            if not np.isclose(float(cached.p_value), float(row.p_value), rtol=1e-10, atol=0):
                raise ValueError(f"Cached and effect-summary p values disagree for {name}")
        self.notes.append("B retains the exact cached 209 seed / 627 random windows per feature and signed effect sizes. WF complexity has a positive Hedges' g (seed higher); its direction is unchanged.")

    def group_records(self, name):
        data = self.tables["external_recurrence"]
        if name in (UNION, MULTI):
            group = data[data.record_type.eq("deduplicated_union_cluster") & data.eligible.eq(1)]
            return group[group.dataset_support_n.ge(2)] if name == MULTI else group
        return data[data.record_type.eq("source_external_region") & data.eligible.eq(1) & data.dataset.eq(name)]

    def prepare_curves(self):
        # Ignore the legacy named final (0.85) alias. The numeric 0.85 row
        # remains an intermediate sensitivity point, never the catalogue rule.
        # Add only missing 1.0 endpoints from cached region scores, not models.
        curves = self.tables["recurrence_curves"]
        self.curves = curves[~curves.threshold_label.astype(str).str.startswith("manuscript final")].copy()
        self.nonfamily = self.tables["nonfamily_curves"]
        self.nonfamily = self.nonfamily[~self.nonfamily.threshold_label.astype(str).str.startswith("manuscript final")].copy()
        if self.curves.duplicated(["population", "dataset", "threshold", "comparison"]).any():
            raise ValueError("Duplicate numeric recognition points in the supplied plotting table")
        self.curves = self.add_unanimous_points(self.curves)
        self.nonfamily = self.add_unanimous_points(self.nonfamily, exclude_seed_families=True)

    def add_unanimous_points(self, curves, exclude_seed_families=False):
        additions = []
        for (population, dataset), group in curves.groupby(["population", "dataset"], sort=False):
            if (group.threshold.eq(1.0) & group.comparison.eq("ge")).any():
                continue  # Reuse an existing exact 100% summary point if supplied.
            if population == "external_AD":
                raw = self.group_records(dataset)
            elif population == "matched_non_AD_control":
                raw = self.tables["matched_controls"]
                raw = raw[raw.analysis_group.eq(dataset)]
            else:
                raise ValueError(f"Unknown recurrence population: {population}")
            if exclude_seed_families:
                raw = raw[raw.family.eq("other")]
            values = pd.to_numeric(raw.max_vote_fraction, errors="raise")
            total = len(raw)
            if total != int(group.N_eligible.iloc[0]) or values.isna().any():
                raise ValueError(f"Missing cached region support or denominator mismatch for 100% point: {dataset}")
            # Exact equality: do not round 99.x% or use a tolerant isclose.
            recognised = int(values.eq(1.0).sum())
            low, high = wilson_interval(recognised, total)
            row = group.iloc[0].to_dict()
            row.update(threshold_label="1.00", threshold=1.0, comparison="ge",
                       N_eligible=total, N_recognized=recognised,
                       recovery_fraction=recognised/total, CI_95_low=low, CI_95_high=high)
            additions.append(row)
            family = "; exclude FOX/TCF/LEF" if exclude_seed_families else ""
            self.notes.append(f"100% endpoint from cached max_vote_fraction == 1.0: {population}, {dataset}{family}; {recognised}/{total} biological regions; Wilson 95% CI [{low:.6g}, {high:.6g}]. No scoring or extrapolation.")
        return pd.concat([curves, pd.DataFrame(additions)], ignore_index=True) if additions else curves

    def prepare_protein_counts(self):
        mem = self.tables["protein_membership"]
        cols = ["Soto_2022", "KothaHC_2023", "KothaGSL_2023", "Staller_2022"]
        mem = mem[mem.R_TADs.eq(1)]
        n = mem[cols].sum(axis=1)
        self.protein_counts = [int(n.eq(0).sum()), int(n.eq(1).sum()), int(n.ge(2).sum())]
        if len(mem) != 190 or self.protein_counts != [108, 19, 63]:
            raise ValueError(f"Original protein overlap does not reconcile: {len(mem)}, {self.protein_counts}")
        self.notes.append("S2G counts 108/19/63 TF proteins (190 total), not rAHD regions or windows.")

    def new_figure(self, size):
        return plt.figure(figsize=size)

    def export(self, fig, name, directory="main", fixed_canvas=False):
        out = self.out / directory
        out.mkdir(parents=True, exist_ok=True)
        fig.canvas.draw()
        for ext in ("pdf", "svg", "png"):
            path = out / f"{name}.{ext}"
            kwargs = {"bbox_inches": fig.bbox_inches if fixed_canvas else "tight"}
            if ext == "png":
                kwargs["dpi"] = 600
            fig.savefig(path, **kwargs)
            self.exports.append(str(path.relative_to(self.out)))
        plt.close(fig)

    @staticmethod
    def letter(fig, rect, text):
        x, y, w, h = rect
        fig.text(x + .006 * w, y + .975 * h, text, fontsize=9, fontweight="bold", ha="left", va="top")

    def feature_panel(self, fig, rect, letter="B", effects=True):
        x, y, w, h = rect
        self.letter(fig, rect, letter)
        data = self.tables["plot_data"]
        summary = self.tables["effect_sizes"].set_index("feature")
        for i, feature in enumerate(FEATURES):
            col, row = i % 2, i // 2
            ax = fig.add_axes([x + w * (.13 + col * .47), y + h * (.65 - row * .48), w * .35, h * .24])
            sub = data[data.feature.eq(feature)]
            a = sub.loc[sub.group.eq("Seed R-TADs"), "value"].to_numpy(float)
            b = sub.loc[sub.group.eq("Random IDRs"), "value"].to_numpy(float)
            # Original helper supplies exact box/jitter/whisker/spine aesthetics.
            # Its inexpensive MWU call is display-only; no bootstrap rerun.
            self.template._boxplot_one(ax, a, b, feature + "\n(z-score)",
                                       ["Seed\nwindows", "Random TF-IDR\nwindows"], .10)
            # Replace stars/brackets with the cached signed Hedges' g and CI.
            for artist in list(ax.texts):
                if set(artist.get_text()) <= {"*"}:
                    artist.remove()
            for line in list(ax.lines):
                if len(line.get_xdata()) == 4 and np.array_equal(line.get_xdata(), [1, 1, 2, 2]):
                    line.remove()
            if effects:
                r = summary.loc[feature]
                ax.text(.5, 1.08, f"g = {r.hedges_g:+.2f} [{r.hedges_g_CI_low:+.2f}, {r.hedges_g_CI_high:+.2f}]",
                        transform=ax.transAxes, ha="center", va="bottom", fontsize=6)
        fig.text(x + .54 * w, y + .012 * h,
                 "Signed Hedges' g [95% CI]; positive = seed windows higher" if effects else
                 "Exact cached Fig. 2B observations: 209 seed / 627 random windows per feature",
                 ha="center", va="bottom", fontsize=5.5)

    def workflow_panel(self, fig, rect, letter="C"):
        self.letter(fig, rect, letter)
        x, y, w, h = rect
        ax = fig.add_axes([x + .035 * w, y + .035 * h, .94 * w, .91 * h])
        ax.set(xlim=(-.01, 1.01), ylim=(0, 1)); ax.axis("off")
        steps = [
            ("Human TFs", "Human TF disordered regions"),
            ("Windowing", "Overlapping 15-aa windows"),
            ("Feature encoding", "143 sequence-derived features"),
            ("Classifier ensemble", f"{self.models} prediction models"),
            ("Consensus selection", "168/168 models positive (100% consensus)\nFinal catalogue criterion"),
            ("Region assembly", "Merge overlapping qualifying windows"),
            ("Candidate catalogue", "230 candidate rAHD regions in 190 TFs"),
        ]
        for i, (label, description) in enumerate(steps):
            bottom = .87 - i * .14
            for left, width, color in [(.0, .33, "#F4D99E"), (.345, .64, "#FAEACC")]:
                ax.add_patch(FancyBboxPatch((left, bottom), width, .108,
                    boxstyle="round,pad=0.004,rounding_size=0.014", linewidth=.6,
                    edgecolor="#B8B39F", facecolor=color))
            ax.text(.165, bottom + .054, label, fontsize=6.5, ha="center", va="center")
            ax.text(.665, bottom + .054, description, fontsize=6.5, ha="center", va="center",
                    fontweight="bold" if i == 4 else "normal")
            if i < len(steps) - 1:
                ax.add_patch(FancyArrow(.942, bottom-.003, 0, -.024,
                    width=.012, head_width=.035, head_length=.015,
                    length_includes_head=True, color="#666666", linewidth=0))

    def curve_axis(self, fig, rect, letter):
        self.letter(fig, rect, letter)
        x, y, w, h = rect
        ax = fig.add_axes([x + .16*w, y + .22*h, .79*w, .69*h])
        ax.spines[["top", "right"]].set_visible(False)
        ax.set(xlabel="Ensemble-support threshold (%)", ylabel="Fraction of regions recognised",
               xlim=(-2, 102), ylim=(0, 1.03))
        ax.set_xticks([0, 25, 50, 75, 85, 100], [">0", "25", "50", "75", "85", "100"])
        self.mark_final_threshold(ax, scale=100)
        fig.text(x + .56*w, y + .02*h, self.ensemble_label,
                 ha="center", va="bottom", fontsize=5.5)
        return ax

    def mark_final_threshold(self, ax, scale=1, position=None):
        threshold = self.final_threshold * scale if position is None else position
        ax.axvline(threshold, color="#777777", ls="--", lw=.6, zorder=0)
        ax.text(threshold, 1.035, "Window-selection threshold: 100% consensus",
                transform=ax.get_xaxis_transform(), ha="right", va="bottom",
                fontsize=5.4, color="#555555")

    def plot_curve(self, ax, group, label, color, linestyle="-", ribbon=True):
        group = group.sort_values("threshold")
        ax.plot(group.threshold*100, group.recovery_fraction, color=color, label=label,
                marker="o", markersize=2.7, linewidth=.9, linestyle=linestyle)
        if ribbon:
            ax.fill_between(group.threshold.to_numpy(float)*100,
                            group.CI_95_low.to_numpy(float), group.CI_95_high.to_numpy(float),
                            color=color, alpha=.12, linewidth=0)

    def main_benchmark_panel(self, fig, rect, letter="D"):
        ax = self.curve_axis(fig, rect, letter)
        selections = [("external_AD", UNION, "External ADs", "AD"),
                      ("matched_non_AD_control", UNION, "Matched TF-IDR controls", "control"),
                      ("external_AD", MULTI, "ADs supported by ≥2 datasets", "multi")]
        for population, dataset, label, color in selections:
            group = self.curves[self.curves.population.eq(population) & self.curves.dataset.eq(dataset)]
            n = int(group.N_eligible.iloc[0])
            self.plot_curve(ax, group, f"{label} (n={n})", self.colors[color])
        ax.legend(frameon=False, loc="lower left", fontsize=5.8, handlelength=1.6)

    def dataset_curve_panel(self, fig, rect, letter="A"):
        ax = self.curve_axis(fig, rect, letter)
        short = ["Soto 2022", "Kotha & Staller HC", "Kotha & Staller GSL", "Staller 2022"]
        for dataset, label in zip(DATASETS, short):
            group = self.curves[self.curves.population.eq("external_AD") & self.curves.dataset.eq(dataset)]
            self.plot_curve(ax, group, f"{label} (n={int(group.N_eligible.iloc[0])})", self.colors[dataset])
        ax.legend(frameon=False, loc="lower left", fontsize=5.7, handlelength=1.5)

    def distribution_panel(self, fig, rect, letter="B"):
        self.letter(fig, rect, letter)
        x, y, w, h = rect
        ax = fig.add_axes([x + .16*w, y + .23*h, .79*w, .68*h])
        external = self.group_records(UNION).max_vote_fraction.to_numpy(float)
        controls = self.tables["matched_controls"]
        controls = controls[controls.analysis_group.eq(UNION)].max_vote_fraction.to_numpy(float)
        for values, label, color in [(external, "External ADs", self.colors["AD"]),
                                     (controls, "Matched TF-IDR controls", self.colors["control"])]:
            values = np.sort(values)
            ax.step(np.r_[0, values], np.r_[0, np.arange(1, len(values)+1)/len(values)],
                    where="post", linewidth=.9, color=color, label=f"{label} (n={len(values)})")
        ax.set(xlabel="Maximum ensemble support in region", ylabel="Cumulative fraction",
               xlim=(0, 1.02), ylim=(0, 1.03))
        ax.spines[["top", "right"]].set_visible(False)
        self.mark_final_threshold(ax)
        ax.legend(frameon=False, loc="upper left", fontsize=5.8)
        row = self.tables["control_statistics"]
        row = row[row.dataset.eq(UNION)].iloc[0]
        ax.text(.98, .04, f"Cliff's δ = {row.Cliffs_delta_external_minus_control:.2f}\n"
                f"95% CI [{row.Cliffs_delta_CI_95_low:.2f}, {row.Cliffs_delta_CI_95_high:.2f}]",
                transform=ax.transAxes, ha="right", va="bottom", fontsize=6,
                bbox={"facecolor":"white","edgecolor":"none","pad":1})
        fig.text(x+.56*w, y+.02*h, self.ensemble_label, ha="center", fontsize=5.5)

    def selected_threshold_panel(self, fig, rect, letter="C"):
        """Same compact summary, retaining 85% and adding the 100% endpoint."""
        self.letter(fig, rect, letter)
        x, y, w, h = rect
        ax = fig.add_axes([x+.16*w, y+.26*h, .79*w, .65*h])
        thresholds = [.50, .75, .85, 1.0]
        selections = [("external_AD", UNION, "External ADs", "AD"),
                      ("matched_non_AD_control", UNION, "Matched TF-IDR controls", "control"),
                      ("external_AD", MULTI, "ADs supported by ≥2 datasets", "multi")]
        for i, (population, dataset, label, color) in enumerate(selections):
            group = self.curves[self.curves.population.eq(population) & self.curves.dataset.eq(dataset)]
            group = group.set_index("threshold").loc[thresholds]
            values = group.recovery_fraction.to_numpy(float)
            errors = np.vstack([np.maximum(0, values-group.CI_95_low.to_numpy(float)),
                                np.maximum(0, group.CI_95_high.to_numpy(float)-values)])
            ax.errorbar(np.arange(len(thresholds))+(i-1)*.18, values, yerr=errors, fmt="o",
                        color=self.colors[color], ms=3, lw=.8, capsize=2,
                        label=f"{label} (n={int(group.N_eligible.iloc[0])})")
        ax.axvspan(2.65, 3.35, color="#F4D99E", alpha=.2, linewidth=0, zorder=0)
        ax.set_xticks([0,1,2,3], ["50%", "75%", "85%", "100%\nWindow selection"])
        ax.set(xlim=(-.4,3.4), ylim=(0,1.03), ylabel="Fraction of regions recognised",
               xlabel="Selected ensemble-support thresholds")
        self.mark_final_threshold(ax, position=3)
        ax.spines[["top","right"]].set_visible(False)
        ax.legend(frameon=False, loc="lower left", fontsize=5.5, handlelength=1.2)
        fig.text(x+.56*w, y+.025*h, self.ensemble_label+"; existing Wilson 95% CIs",
                 ha="center", fontsize=5.3)

    def family_panel(self, fig, rect, letter="D"):
        self.letter(fig, rect, letter)
        x,y,w,h = rect
        ax=fig.add_axes([x+.17*w,y+.31*h,.78*w,.60*h])
        data=self.tables["family_enrichment"]
        if not data.statistical_unit.eq("biological_candidate_region").all():
            raise ValueError("S2D cannot use window-level enrichment")
        data=data[data.Family.eq("Combined") & ~data.Threshold.isin(["Top 10","Top 20"])].reset_index(drop=True)
        values=data.Observed_fraction.to_numpy(float)
        low=np.maximum(0,values-data.Observed_Wilson_CI_low.to_numpy(float))
        high=np.maximum(0,data.Observed_Wilson_CI_high.to_numpy(float)-values)
        ax.errorbar(np.arange(len(data)),values,yerr=np.vstack([low,high]),fmt="o-",
                    color=self.colors["AD"],lw=.9,ms=2.7,capsize=2)
        ax.axhline(data.Background_fraction.iloc[0],color="black",ls="--",lw=.7,label="Complete-universe background")
        labels=["≥85%-support\nregions" if str(label).lower().startswith("all ") else str(label)
                for label in data.Threshold]
        ax.set_xticks(np.arange(len(data)),labels,rotation=35,ha="right",fontsize=5.7)
        ax.set_ylabel("FOX/TCF/LEF region fraction")
        ax.yaxis.set_major_formatter(PercentFormatter(1))
        ax.spines[["top","right"]].set_visible(False)
        ax.legend(frameon=False,loc="upper right",fontsize=5.2)
        fig.text(x+.56*w,y+.035*h,"≥85% support sensitivity analysis",
                 ha="center",fontsize=5.3)

    def identity_panel(self, fig, rect, letter="E"):
        self.letter(fig,rect,letter)
        x,y,w,h=rect
        ax=fig.add_axes([x+.17*w,y+.35*h,.78*w,.56*h])
        data=self.tables["identity_rates"]
        if not data.statistical_unit.eq("biological_candidate_region").all():
            raise ValueError("S2E cannot use window-level prediction rates")
        # Empty bins are not measured zero rates. Retain genuine n=1 bins.
        data=data[data.denominator_regions.gt(0)].reset_index(drop=True)
        values=data.prediction_rate.to_numpy(float)
        yerr=np.vstack([np.maximum(0,values-data.Wilson_95_CI_low.to_numpy(float)),
                        np.maximum(0,data.Wilson_95_CI_high.to_numpy(float)-values)])
        ax.errorbar(np.arange(len(data)),values,yerr=yerr,fmt="o",lw=.9,ms=2.5,
                    capsize=2,color=self.colors["AD"])
        labels=[]
        for row in data.itertuples():
            label="No alignment" if row.identity_bin=="no reported alignment" else row.identity_bin
            labels.append(f"{label}\nn={row.denominator_regions:,}")
        ax.set_xticks(np.arange(len(data)),labels,rotation=45,ha="right",fontsize=5.2)
        ax.set_ylabel("Candidate-region prediction frequency")
        ax.set_ylim(0,1); ax.spines[["top","right"]].set_visible(False)
        fig.text(x+.56*w,y+.065*h,"Nearest-seed local identity (%) / biological-region denominators",ha="center",fontsize=5.3)
        fig.text(x+.56*w,y+.015*h,"≥85% support sensitivity analysis",ha="center",fontsize=5.3)

    def nonfamily_panel(self,fig,rect,letter="F"):
        ax=self.curve_axis(fig,rect,letter)
        all_ads=self.curves[self.curves.population.eq("external_AD") & self.curves.dataset.eq(UNION)]
        nonfamily=self.nonfamily[self.nonfamily.dataset.eq(UNION)]
        self.plot_curve(ax,all_ads,f"All ADs (n={int(all_ads.N_eligible.iloc[0])})",self.colors["AD"])
        self.plot_curve(ax,nonfamily,f"Exclude FOX/TCF/LEF (n={int(nonfamily.N_eligible.iloc[0])})",
                        self.colors["multi"],"--")
        ax.legend(frameon=False,loc="lower left",fontsize=5.7)

    def protein_panel(self,fig,rect,letter="G"):
        self.letter(fig,rect,letter)
        x,y,w,h=rect
        ax=fig.add_axes([x+.17*w,y+.25*h,.78*w,.66*h])
        labels=["No mapped\nexternal support","Mapped to\n1 dataset","Mapped to\n≥2 datasets"]
        bars=ax.bar(labels,self.protein_counts,color=self.colors["AD"],alpha=.9)
        for bar,n in zip(bars,self.protein_counts):
            ax.text(bar.get_x()+bar.get_width()/2,n+3,str(n),ha="center",va="bottom",fontsize=6.5)
        ax.set(ylabel="TF proteins carrying final candidate rAHDs",ylim=(0,128))
        ax.spines[["top","right"]].set_visible(False)
        ax.set_title("Protein-level support across external AD datasets",fontsize=6.5,pad=5)
        fig.text(x+.56*w,y+.04*h,"Protein-level overlap: 108 of 190 lack mapped external support",ha="center",fontsize=5.3)

    def manuscript(self):
        """Export analytical manuscript panels; omit graphical schematics."""
        manuscript_main = [
            ("Figure1E_external_AD_recurrence", (4.4, 2.8), self.main_benchmark_panel, "E"),
        ]
        for name, size, draw, letter in manuscript_main:
            fig = self.new_figure(size)
            draw(fig, (0, 0, 1, 1), letter)
            self.export(fig, name, "manuscript/Figure1")

        manuscript_s2 = [
            ("FigureS2D_ensemble_support_ECDF", self.distribution_panel, "D"),
            ("FigureS2E_selected_support_thresholds", self.selected_threshold_panel, "E"),
            ("FigureS2F_nonfamily_sensitivity", self.nonfamily_panel, "F"),
        ]
        for name, draw, letter in manuscript_s2:
            fig = self.new_figure((3.7, 2.9))
            draw(fig, (0, 0, 1, 1), letter)
            self.export(fig, name, "manuscript/FigureS2")

    def main(self):
        for name,size,draw in [
            ("Figure_2_panelB_revised",(4.8,2.3),self.feature_panel),
            ("Figure_2_panelC_revised",(4.4,2.8),self.workflow_panel),
            ("Figure_2_panelD_revised",(4.4,2.8),self.main_benchmark_panel),
        ]:
            fig=self.new_figure(size)
            draw(fig,(0,0,1,1))
            self.export(fig,name)
        fig=self.new_figure(self.assembly_size)
        # A contains only a placeholder, never experimental data or artwork.
        rect=(.01,.53,.47,.46)
        self.letter(fig,rect,"A")
        fig.text(.245,.76,"Experimental panel A\nPending",color="#888888",
                 fontsize=7,ha="center",va="center")
        self.feature_panel(fig,(.51,.53,.48,.46))
        self.workflow_panel(fig,(.01,.01,.47,.49))
        self.main_benchmark_panel(fig,(.51,.01,.48,.49))
        self.export(fig,"Figure_2_revised_BCD_assembled",fixed_canvas=True)

    def supplement(self):
        panels=[("A",self.dataset_curve_panel,(3.7,2.9)),("B",self.distribution_panel,(3.7,2.9)),
                ("C",self.selected_threshold_panel,(3.7,2.9)),("D",self.family_panel,(3.7,2.9)),
                ("E",self.identity_panel,(3.7,2.9)),("F",self.nonfamily_panel,(3.7,2.9)),
                ("G",self.protein_panel,(3.7,2.9))]
        for letter,draw,size in panels:
            fig=self.new_figure(size); draw(fig,(0,0,1,1),letter)
            self.export(fig,f"Figure_S2_panel{letter}_revised","supplement")
        # No legacy S2 assembly script was supplied: preserve original main
        # figure width, native font size, and two-column panel organization.
        fig=self.new_figure((self.assembly_size[0],10.6))
        for i,(letter,draw,size) in enumerate(panels):
            col,row=i%2,i//2
            draw(fig,(.01+col*.50,.755-row*.247,.48,.235),letter)
        self.export(fig,"Figure_S2_revised","supplement",fixed_canvas=True)

    def write_qa(self):
        catalogue=self.config["catalogue"]
        qa=["Figure 2 computational revision QA","==================================",
            "Formatting template: "+str(self.paths["template_script"]),
            "Reused original colors, rcParams and _boxplot_one; native B 4.8x2.3 in, C/D 4.4x2.8 in, S2 panels 3.7x2.9 in. PNG 600 dpi; vector PDF/SVG.",
            f"Main canvas: {self.assembly_size[0]:.6f} x {self.assembly_size[1]:.6f} in; A is a pending-experiment placeholder only. S2 canvas: {self.assembly_size[0]:.6f} x 10.6 in.",
            "",
            f"Ensemble: {self.models} prediction models/files. Final window-selection criterion: 168/168 models positive (100% consensus). Qualifying overlapping windows are merged to produce the final catalogue of {catalogue['regions']} candidate regulatory acidic-hydrophobic disordered regions (rAHDs) in {catalogue['TF_proteins']} TFs; {catalogue['sequence_features']} features.",
            "100% recurrence in benchmarking means an external AD contains at least one 15-aa window supported by all models. It is a window-level support criterion and is NOT numerically identical to final-catalogue overlap, which uses merged final regions and independent mapping criteria.",
            "The recurrence line is labelled 'Window-selection threshold: 100% consensus', never final-catalogue recovery. 85% is an intermediate relaxed recurrence/sensitivity threshold only; its legacy named-final alias is excluded.",
            "Supplied intermediate points are unchanged. Missing 100% endpoints are counted from existing per-region max_vote_fraction == 1.0 with Wilson 95% CIs; existing 100% summary points are reused. No extrapolation.",
            "No model/provenance audit, subset search, prediction-file recurrence rerun, model scoring, feature calculation, control regeneration or bootstrap rerun was performed.",
            "",
            "Supplementary changes:",
            "The existing A-G arrangement, values and highlight positions are unchanged. Main D and S2A-C/F use consistent window-selection terminology. ECDF x-axis extends to 1.02 to expose the final step; observations are unchanged.",
            "Top 10 and Top 20 omitted from the S2D plot only; all original table rows/statistics retained.",
            "S2E plots only bins with n>0, including removal of any empty <20% bin. No-alignment stays separate; every bin shows biological-region n. Wilson CIs are reused unchanged.",
            "S2D and S2E use the broader >=85%-support sensitivity set, not the final 230-region catalogue. The 4,048 positive biological regions in this sensitivity set are not final catalogue regions. The existing universe includes proteome and TF regions, so enrichment is not TF-restricted.",
            "S2G is protein-level: 108 TF proteins carrying final candidate rAHD regions have no mapped external AD support, 19 have support in exactly one dataset, and 63 in >=2 datasets. These are not unique rAHD regions or TADs.",
            *self.notes,"",
            "Exact source tables per panel:"]
        sources={
            "Main B": ["plot_data","plot_stats","effect_sizes"],
            "Main C": [],
            "Main D": ["recurrence_curves","external_recurrence","matched_controls"],
            "S2A (dataset curves)": ["recurrence_curves","external_recurrence"],
            "S2B (ECDF and existing effect size)": ["external_recurrence","matched_controls","control_statistics"],
            "S2C (50/75/85/100% summary)": ["recurrence_curves","external_recurrence","matched_controls"],
            "S2D (family enrichment)": ["family_enrichment"],
            "S2E (seed identity)": ["identity_rates"],
            "S2F (non-FOX/TCF/LEF sensitivity)": ["recurrence_curves","nonfamily_curves","external_recurrence"],
            "S2G (protein-level support)": ["protein_membership"],
        }
        for panel,keys in sources.items():
            qa.append(panel+": "+("; ".join(str(self.paths[k]) for k in keys) if keys else str(self.config_path)+" (user-supplied final catalogue definition)"))
        (self.out/"figure2_revision_QA.txt").write_text("\n".join(qa)+"\n")
        manifest={"config":self.config,"input_paths":{k:str(v) for k,v in self.paths.items()},
                  "style":self.style,"ensemble_model_count":self.models,"final_consensus_threshold":self.final_threshold,
                  "supplementary_panel_map":{panel:keys for panel,keys in sources.items() if panel.startswith("S2")},
                  "software_versions":{"Python":platform.python_version(),"NumPy":np.__version__,"pandas":pd.__version__,"Matplotlib":mpl.__version__},
                  "exports_this_run":self.exports,"main_assembly_inches":self.assembly_size}
        (self.out/"figure2_revision_plot_metadata.json").write_text(json.dumps(manifest,indent=2)+"\n")
        self.write_captions()

    def write_captions(self):
        row=self.tables["control_statistics"]
        row=row[row.dataset.eq(UNION)].iloc[0]
        caption=(
            "Figure 1, computational panels. C, Corrected raw sequence-feature distributions use parent-region cluster-bootstrap confidence intervals for signed Hedges' g. D, Human TF disordered regions are divided into overlapping 15-aa windows and encoded with 143 sequence-derived features. The ensemble comprises 168 prediction models; only windows positive in all 168 models (168/168; 100% consensus) are retained. Overlapping qualifying windows are merged into 230 candidate regulatory acidic-hydrophobic disordered regions (rAHDs) in 190 TFs. E, Eligible external functional AD regions, matched TF-IDR controls, and ADs supported by ≥2 datasets across ensemble-support thresholds. Maximum ensemble support within each biological region is used. The dashed line marks the window-selection threshold: 100% consensus; 85% is an intermediate relaxed sensitivity threshold, not the catalogue definition. 100% recurrence denotes the window-level consensus criterion used before merging; final-catalogue overlap is evaluated separately.\n\n"
            "Figure S2. A, Dataset-specific recognition curves (Soto 2022; Kotha & Staller HC and GSL; Staller 2022). B, ECDF of maximum support for external AD union regions and matched TF-IDR controls. "
            +f"The existing signed Cliff's δ is {row.Cliffs_delta_external_minus_control:.2f} (95% CI [{row.Cliffs_delta_CI_95_low:.2f}, {row.Cliffs_delta_CI_95_high:.2f}]), positive for external ADs with greater ensemble support. "
            +"C, Compact comparison at 50%, 75%, 85% and 100% support; 100% is highlighted as the window-selection consensus threshold, and 85% remains sensitivity only. D, Combined FOX/TCF/LEF region fractions across candidate-universe ranking thresholds in the >=85%-support sensitivity set. The 4,048 positive biological regions in this relaxed set are not the final 230-region catalogue. Top 10/20 are retained in the source table but omitted from the plot for readability. E, Candidate-region prediction frequency versus maximum nearest-seed local identity over constituent windows, also for the >=85%-support sensitivity analysis; bins with n=0 are omitted and no-alignment is separate. D/E include proteome and TF regions; they are not TF-only enrichment or final 100%-consensus catalogue summaries. F, External-union recognition before and after excluding FOX/TCF/LEF proteins. All recurrence panels use the supplied 168-model scores and label 100% as the window-selection consensus threshold. 100% recurrence denotes the window-level consensus criterion used before merging; final-catalogue overlap is evaluated separately using merged final regions and independent mapping criteria. Missing 100% points are counted from cached maxima, not extrapolated. G, Protein-level support across external AD datasets: 108 of 190 TF proteins carrying final candidate rAHD regions lack a corresponding mapped external AD annotation, 19 map to exactly one dataset, and 63 to ≥2 datasets. These are protein counts, not unique rAHD regions, TADs or windows.\n")
        (self.out/"figure2_revision_captions.txt").write_text(caption)


def cli(mode):
    parser=argparse.ArgumentParser(description="Plot cached Figure 2 reviewer outputs; no heavy analysis rerun")
    parser.add_argument("--project-root",type=Path,default=Path(__file__).resolve().parent.parent)
    parser.add_argument("--config",type=Path)
    args=parser.parse_args()
    config=args.config or args.project_root/"figure2_revision_inputs.json"
    revision=Revision(args.project_root,config,profile=mode)
    if mode in ("main","all"):
        revision.main()
    if mode in ("supplement","all"):
        revision.supplement()
    if mode in ("manuscript","all"):
        revision.manuscript()
    revision.write_qa()
    print(f"Plotting complete: {revision.out}")


if __name__=="__main__":
    cli("all")
