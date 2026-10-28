# Figure 6 workspace: NMR validation and Figure 5 revision package

This folder contains the curated experimental master, analysis scripts, frozen
prediction inputs, and the generated manuscript figure package.

## Build

From this folder, in the scientific Python environment:

```bash
python build_final_figure_package.py
```

The default uses cached trajectory summaries. Only use
`python build_final_figure_package.py --with-trajectories` when the local raw
trajectories and their verified topology/parameter files are available.

Edit `fig6_parent_master.csv` to update curated experimental measurements.
For an existing pair, enter `Kd_uM` and `Kd_uM_err` in micromolar and change
`Outcome` to `Kd_fit`. Leave unavailable errors blank; do not substitute zero.
The pipeline recalculates `pKd` and `pKd_error` from these input fields, so their
stored values do not need manual editing. Keep pair identifiers and constructs
unchanged. A completely new construct pair also needs verified sequence mapping
and frozen prediction/simulation inputs; adding an experimental-only row is not
sufficient. Preserve blank values as blank CSV cells.

The build rewrites the generated workbook. Keep manual experimental changes in
the master CSV rather than in Supplementary Table S11 alone.

Individual publication panels are exported to
`final_figure_package/figures/panels_9x5cm/` as PDF, editable SVG and PNG.
Each panel canvas is exactly 9 cm wide by 5 cm high with Arial text at 7 pt.
The larger assembled figures remain available as overview files.
The runner does not invoke the original Excel importer or retrain the model.
Outputs are written to `final_figure_package/`, including
`Supplementary_Table_S11.xlsx` (formerly Table S10).

The histidine sheets retain 390 matched pH 7.0/pH 6.5 pairs and distinguish the
audit's base predictions from the final hybrid predictions used in the plot.
Reference data in the sibling figure folders are read-only inputs. The build
also requires the bundled Node artifact-tool runtime for workbook generation;
see `build_final_figure_package.py` for its configured runtime discovery.

## Validation

```bash
python verify_revision_analysis.py --outputs
```

Raw trajectories, RDF pickle files, compressed trajectory caches, runtime state
and archived workflow copies are not distributed in Git. They must be supplied
locally for trajectory recalculation and the corresponding cache checks.
Generated source tables and SVG figures are included. PDF/PNG figures follow
the repository-wide ignore policy and can be regenerated locally.

## Current publication status

Main Figure 5A remains an author-supplied spectra placeholder. The code currently
uses the historical supplementary figure filenames S10/S11. The proposed
manuscript renumbering to S8/S9 and redistribution of threshold panels has not
yet been implemented. Supplementary **Table S11** is already renamed and includes
the histidine controls; table and figure numbers are separate.
