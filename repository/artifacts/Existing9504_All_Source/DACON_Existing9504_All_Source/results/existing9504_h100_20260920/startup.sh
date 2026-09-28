#!/usr/bin/env bash
set -euo pipefail
python3 -u /shared/existing-only-9504-h100-20260920/run_notebook.py 2>&1 | tee -a /shared/existing-only-9504-h100-20260920/native_launch.log
