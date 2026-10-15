#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FIG5_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${FIG5_ROOT}"
python s8_extended_controls/scripts/prepare_new_s8_control_inputs.py "$@"
python s8_extended_controls/scripts/run_new_s8_control_predictions.py
python s8_extended_controls/scripts/make_extended_s8_outputs.py
