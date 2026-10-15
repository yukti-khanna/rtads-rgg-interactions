#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
DRY=""; RESUME=""; HIST_RDF=""; CTRL_RDF=""
while [ "$#" -gt 0 ]; do
  case "$1" in
    --dry-run) DRY="--dry-run"; shift ;;
    --resume) RESUME="--resume"; shift ;;
    --histidine-rdf-dir) HIST_RDF="--histidine-rdf-dir $2"; shift 2 ;;
    --control-rdf-dir) CTRL_RDF="--control-rdf-dir $2"; shift 2 ;;
    --help|-h) echo "Usage: ./run_all.sh [--dry-run] [--resume] [--histidine-rdf-dir PATH] [--control-rdf-dir PATH]"; exit 0 ;;
    *) echo "Unknown argument: $1"; exit 2 ;;
  esac
done
python scripts/validate_package.py $DRY $HIST_RDF $CTRL_RDF
python scripts/build_final_model.py $DRY $RESUME
python scripts/prepare_validation_features.py $DRY $RESUME $HIST_RDF $CTRL_RDF
python scripts/predict_validation_sets.py $DRY $RESUME
python scripts/analyze_validation.py $DRY $RESUME
python scripts/make_validation_figures.py $DRY $RESUME
if [ -z "$DRY" ]; then python scripts/validate_outputs.py; fi
