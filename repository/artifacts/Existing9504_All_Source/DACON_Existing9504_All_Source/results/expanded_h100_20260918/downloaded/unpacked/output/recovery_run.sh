#!/usr/bin/env bash
set -euo pipefail

STARTED_EPOCH=${1:?original start epoch required}
DEADLINE_EPOCH=${2:?original absolute pause deadline required}
OBJECT=/shared/expanded-h100-20260918
JOB=/root/expanded_h100_20260918_retry1
INPUT=$JOB/input
SOURCE=$INPUT/source
OUTPUT=$JOB/output
PY=/root/fullft8/.venv/bin/python
SYS_PY=$PY
export STARTED_EPOCH DEADLINE_EPOCH

finalize() {
  code=$?
  trap - EXIT
  test -z "${GPU_MONITOR_PID:-}" || kill "$GPU_MONITOR_PID" 2>/dev/null || true
  test -z "${PROGRESS_PID:-}" || kill "$PROGRESS_PID" 2>/dev/null || true
  if [[ "${CREATED_JOB:-false}" == true && -f "$SOURCE/localization_export.py" ]]; then
    EXPORT_SECONDS=$("$PY" -c 'import os,time; print(max(1,int(float(os.environ["DEADLINE_EPOCH"])-time.time()-15)))')
    timeout --signal=TERM --kill-after=5 "$EXPORT_SECONDS" "$PY" "$SOURCE/localization_export.py" --output-dir "$OUTPUT" --object-dir "$OBJECT" \
      --workspace wsp-9683432r72m7 --exit-code "$code" --started-epoch "$STARTED_EPOCH" --deadline-epoch "$DEADLINE_EPOCH" && exit "$code"
  fi
  if [[ ! -x "$PY" ]]; then
    printf '%s\n' '{"status":"failed","archive_ready":false,"reason":"recorded_python_runtime_missing"}' > "$OBJECT/run_state.json"
    exit 1
  fi
  "$SYS_PY" - "$OBJECT" "$STARTED_EPOCH" "$DEADLINE_EPOCH" "$code" <<'PY'
import json,sys,time
from pathlib import Path
root=Path(sys.argv[1])
state={'status':'failed','archive_ready':False,'exit_code':int(sys.argv[4]) or 1,
       'started_epoch':float(sys.argv[2]),'pause_deadline_epoch':float(sys.argv[3]),
       'updated_at':time.time(),'reason':'bootstrap_or_export_failure'}
(root/'run_state.json').write_text(json.dumps(state,indent=2)+'\n')
PY
  exit 1
}
trap finalize EXIT

exec 9> /root/expanded_h100_20260918.lock
flock -n 9 || exit 1
test -d "$JOB"
CREATED_JOB=true
export PYTHONPATH="$JOB/runtime:$SOURCE" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 CUBLAS_WORKSPACE_CONFIG=:4096:8
export EXPANDED_EXECUTE_GPU=1 EXPANDED_JOB_ROOT="$JOB"
export JUPYTER_PATH="$JOB/kernel/share/jupyter:/root/fullft8/.venv/share/jupyter"
cp "$JOB/recover_expanded_cache.py" "$OUTPUT/source/recover_expanded_cache.py"
cp "$JOB/recovery_run.sh" "$OUTPUT/recovery_run.sh"
nvidia-smi --query-gpu=timestamp,utilization.gpu,memory.used,memory.total,power.draw --format=csv -l 5 >> "$OUTPUT/gpu_telemetry.csv" &
GPU_MONITOR_PID=$!
REMAINING=$("$PY" -c 'import os,time; print(max(1,int(float(os.environ["DEADLINE_EPOCH"])-600-time.time())))')
timeout --signal=TERM --kill-after=30 "$REMAINING" "$PY" "$JOB/recover_expanded_cache.py" > "$OUTPUT/recovery.log" 2>&1
