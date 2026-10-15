#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FIG5_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
python "${FIG5_ROOT}/controls/run_final_control_predictions.py"
