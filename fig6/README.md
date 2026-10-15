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
