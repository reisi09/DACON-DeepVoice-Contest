#!/usr/bin/env bash
set -euo pipefail
[[ "${PANNS_GUARD_PID:-}" == "$PPID" ]] || exit 2
trap 'wait; exit 130' TERM INT HUP
PANNS_RELEASE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
PANNS_ORIGINAL_WORK=/root/panns-cnn-v1-20260925/application-v11
PANNS_PINNED_PYTHON=/root/panns-cnn-v1-20260925/env/venv/bin/python
[[ -x "$PANNS_PINNED_PYTHON" && -f "$PANNS_ORIGINAL_WORK/run.json" ]] || exit 2
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1 PYTHONDONTWRITEBYTECODE=1 MPLBACKEND=Agg
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 CUBLAS_WORKSPACE_CONFIG=:4096:8
export PANNS_LOCAL_CACHE="$PANNS_ORIGINAL_WORK/cache_local"
export PYTHONPATH="$PANNS_ORIGINAL_WORK/bundle/code"
export HF_HOME="$PANNS_ORIGINAL_WORK/hf" MPLCONFIGDIR="$PANNS_ORIGINAL_WORK/mpl" NUMBA_CACHE_DIR="$PANNS_ORIGINAL_WORK/numba"
export PATH="$PANNS_ORIGINAL_WORK/package/model/runtime/bin:$PATH"
"$PANNS_PINNED_PYTHON" "$PANNS_RELEASE/panns_submission_workflow.py" --request "$PANNS_RELEASE/workflow_request.json"
