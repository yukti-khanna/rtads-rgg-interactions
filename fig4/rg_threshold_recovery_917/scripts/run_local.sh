#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/Users/yuktikhanna/miniconda3/bin/python3}"
MPLCONFIGDIR="${MPLCONFIGDIR:-/private/tmp/codex-fig4-threshold-mpl}"
export MPLCONFIGDIR
mkdir -p "${MPLCONFIGDIR}"

if [[ ! -x "${PYTHON_BIN}" ]]; then
  PYTHON_BIN="$(command -v python3)"
fi

"${PYTHON_BIN}" "${SCRIPT_DIR}/run_threshold_recovery.py" --output-dir "${PROJECT_DIR}" "$@"
"${PYTHON_BIN}" "${SCRIPT_DIR}/validate_outputs.py" --output-dir "${PROJECT_DIR}"
