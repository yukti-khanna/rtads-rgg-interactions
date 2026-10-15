# Figure 5 validation package 1: make CSVs

This package prepares the CSV inputs for model validation. It uses the packaged development/held-out tables plus packaged histidine/control metadata. RDF files are not bundled; pass their folders at runtime.

Setup:
```bash
conda env create -f environment.yml
conda activate fig5-make-csvs
```

Dry run:
```bash
./run_make_csvs.sh --dry-run --histidine-rdf-dir /path/to/histidines_rdfs --control-rdf-dir /path/to/controls_rdfs
```

Run:
```bash
./run_make_csvs.sh --histidine-rdf-dir /path/to/histidines_rdfs --control-rdf-dir /path/to/controls_rdfs
```

Outputs are written to `outputs/prepared_csvs/`. Copy that folder into package 2 with `--prepared-csv-dir`.
