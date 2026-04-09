# rtads-rgg-interactions

Figure-companion repository for the R-TAD–RGG interaction manuscript. This repository contains the figure-specific folders, plotting scripts, input tables, helper code, and exported outputs used to generate the main and supplementary figures.

## Overview

The repository is organized primarily by **figure**. Each main figure has its own folder, and the folder is intended to hold the scripts and files needed to reproduce or update that figure.

In practice, this is the repository to use when you want to:

- regenerate a manuscript figure from prepared inputs
- update a figure panel after data or styling changes
- track which scripts and files were used to build a figure
- keep figure-generation code and resources organized in a figure-specific way

## Repository structure

```text
rtads-rgg-interactions/
├── README.md
├── LICENSE
├── fig1/
├── fig2/
├── fig3/
├── fig4/
├── fig5/
├── fig6/
├── supplementary/
├── scripts/
```

### Folder roles

- `fig1/` to `fig6/` contain the files required to generate the corresponding main manuscript figures
- `supplementary/` contains files used for supplementary figures
- `scripts/` contains shared helper scripts or reusable utilities used across figures

## Figure-folder layout

The figure folders are intended to be as self-contained as possible. Based on the current organization of `fig1/`, `fig2/`, and `fig3/`, the preferred layout inside each figure folder is:

```text
figX/
├── README.md or README.txt
├── scripts/
├── inputs/ or input/
└── outputs/ or output/
```

Typical contents are:

- `scripts/`: plotting scripts and figure-specific helper scripts
- `inputs/` or `input/`: plotting-ready FASTA files, tables, metadata, and other required resources
- `outputs/` or `output/`: exported figure panels, supplementary panels, cached intermediate tables, and other generated outputs

This layout makes it easier to work on one figure at a time without mixing files across figures.

## Current organization status

- `fig1/`, `fig2/`, and `fig3/` are already organized and documented
- `fig4/`, `fig5/`, and `fig6/` are being cleaned so that each folder contains only the files required to regenerate that figure
- shared helper logic should remain in `scripts/` when it is reused across multiple figures

## Intended use

This repository is best used for:

1. reproducing the final figures used in the manuscript
2. updating figure panels while keeping the relevant resources grouped together
3. maintaining figure-generation code in a figure-specific way
4. preserving a clear mapping between manuscript figures and the files that produced them

## Reproducibility notes

To keep the repository easy to use and maintain:

- keep figure-specific files inside the corresponding figure folder
- keep shared helper logic in `scripts/` when it is reused across figures
- store plotting-ready inputs with clear file names
- avoid committing unnecessary temporary, duplicate, or machine-specific files when possible
- document figure entry points in script headers or figure-folder READMEs where useful

## Scope

This repository focuses on **manuscript figure generation and figure-supporting resources**. It complements the analysis and prediction repositories by organizing the files actually used to create the main and supplementary visual outputs.

## Status

This repository is under active cleanup and organization so that each manuscript figure can be reproduced more easily from its corresponding folder.