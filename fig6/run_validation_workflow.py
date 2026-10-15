#!/usr/bin/env python3
"""Compatibility entrypoint for the reusable master-table validation workflow.

Existing simulation results and recovered historical predictions are reused.
For source overrides and replacement experiments, use run_validation.py.
"""
from run_validation import main


if __name__ == '__main__':
    main()
