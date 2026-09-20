#!/usr/bin/env bash
set -Eeuo pipefail
ROOT=/root/redesign_full_20260917
SHARED=/shared/redesign-full-20260917
PYTHON=/root/fullft8/.venv/bin/python
mkdir -p "$ROOT"
if [[ "${1:-}" != "--worker" ]]; then
  nohup bash "$0" --worker </dev/null > "$ROOT/launcher.log" 2>&1 &
  echo "$!" > "$ROOT/launcher.pid"
  exit 0
fi
exec > >(tee -a "$ROOT/run.log") 2>&1
read -r SOURCE_SHA _ < "$SHARED/source.zip.sha256"
printf '%s  %s\n' "$SOURCE_SHA" "$SHARED/source.zip" | sha256sum --check --status
"$PYTHON" - "$SHARED/source.zip" "$ROOT/source" <<'PY'
import sys, zipfile
from pathlib import Path
with zipfile.ZipFile(sys.argv[1]) as archive:
    assert archive.testzip() is None
    assert all('/' not in x.filename and not x.is_dir() for x in archive.infolist())
    archive.extractall(sys.argv[2])
PY
cp "$SHARED/config.json" "$ROOT/config.json"
HARD_DEADLINE="$("$PYTHON" -c 'import json,sys; print(int(json.load(open(sys.argv[1]))["pause_deadline_epoch"]))' "$ROOT/config.json")"
REMAINING=$((HARD_DEADLINE - $(date +%s)))
(( REMAINING > 0 ))
nvidia-smi --query-gpu=timestamp,name,memory.total,memory.used --format=csv
nvidia-smi --query-gpu=timestamp,utilization.gpu,memory.used,power.draw --format=csv,noheader,nounits --loop=10 > "$ROOT/gpu_telemetry.live.csv" &
MONITOR_PID=$!
trap 'kill "$MONITOR_PID" 2>/dev/null || true; wait "$MONITOR_PID" 2>/dev/null || true' EXIT
export PYTHONPATH="$ROOT/source"
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
timeout --signal=TERM --kill-after=30s "${REMAINING}s" "$PYTHON" "$ROOT/source/redesign_full_refit_run.py" "$ROOT/config.json"
