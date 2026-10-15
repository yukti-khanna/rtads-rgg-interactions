#!/usr/bin/env python3
"""Build the current master CSV and generate its corrected figure panels.

Use run_validation.py for source overrides and replacement experimental tables.
This compatibility entrypoint uses exactly the same master-based workflow.
"""
from run_validation import main


if __name__ == '__main__':
    main()
