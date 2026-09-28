#!/usr/bin/env bash
set -Eeuo pipefail
ROOT=${RUN_ROOT:-/root/six_h100_20260917}
SHARED=${RUN_SHARED:-/shared/six-h100-20260917}
KERNEL=${RUN_KERNEL:-six-h100}
PYTHON=/root/fullft8/.venv/bin/python
mkdir -p "$ROOT"
if [[ "${1:-}" != "--worker" ]]; then
  nohup bash "$0" --worker </dev/null > "$ROOT/launcher.log" 2>&1 &
  echo "$!" > "$ROOT/launcher.pid"
  exit 0
fi
exec > >(tee -a "$ROOT/run.log") 2>&1
bootstrap_failure() {
  CODE=$?
  cp "$ROOT/run.log" "$SHARED/bootstrap_failure.log" || true
  "$PYTHON" - "$SHARED/run_state.json" <<'PY'
import json, sys, time
from pathlib import Path
p = Path(sys.argv[1])
state = json.loads(p.read_text()) if p.exists() else {}
if not state.get('archive_ready'):
    p.write_text(json.dumps(dict(status='failed', archive_ready=False, updated_unix=time.time(),
                                error='bootstrap failed; see bootstrap_failure.log')))
PY
  exit "$CODE"
}
trap bootstrap_failure ERR
read -r SOURCE_SHA _ < "$SHARED/source.zip.sha256"
printf '%s  %s\n' "$SOURCE_SHA" "$SHARED/source.zip" | sha256sum --check --status
"$PYTHON" - "$SHARED/source.zip" "$ROOT/source" <<'PY'
import hashlib, json, shutil, stat, sys, zipfile
from pathlib import Path
target = Path(sys.argv[2])
with zipfile.ZipFile(sys.argv[1]) as archive:
    assert archive.testzip() is None
    infos = archive.infolist()
    assert len({x.filename for x in infos}) == len(infos)
    assert shutil.disk_usage(target.parent).free > sum(x.file_size for x in infos) + 5_000_000_000
    for item in infos:
        path = Path(item.filename)
        assert not path.is_absolute() and '..' not in path.parts
        assert not stat.S_ISLNK(item.external_attr >> 16)
    manifest = json.loads(archive.read('INPUT_SHA256.json'))
    assert set(manifest) == {x.filename for x in infos} - {'INPUT_SHA256.json'}
    for name, expected in manifest.items():
        assert hashlib.sha256(archive.read(name)).hexdigest() == expected
    archive.extractall(target)
PY
cp "$SHARED/config.json" "$ROOT/config.json"
HARD_DEADLINE="$("$PYTHON" -c 'import json,sys; print(int(json.load(open(sys.argv[1]))["pause_deadline_epoch"]))' "$ROOT/config.json")"
REMAINING=$((HARD_DEADLINE - $(date +%s)))
(( REMAINING > 900 ))
if [[ -d "$ROOT/source/antideepfake" ]]; then
  cp "${RUN_MODEL_PATH:-$SHARED/model.safetensors}" "$ROOT/source/antideepfake/model.safetensors"
  (cd "$ROOT/source/antideepfake" && sha256sum --check model.sha256)
fi
export PYTHONPATH="$ROOT/source"
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export JUPYTER_PATH=/root/fullft8/.venv/share/jupyter
"$PYTHON" -m ipykernel install --prefix /root/fullft8/.venv --name "$KERNEL"
nvidia-smi --query-gpu=timestamp,name,memory.total,memory.used --format=csv
df -B1 "$ROOT"
nvidia-smi --query-gpu=timestamp,utilization.gpu,memory.used,power.draw --format=csv,noheader,nounits --loop=10 > "$ROOT/gpu_telemetry.live.csv" &
MONITOR_PID=$!
trap 'kill "$MONITOR_PID" 2>/dev/null || true' EXIT
timeout --signal=TERM --kill-after=30s "${REMAINING}s" "$PYTHON" "$ROOT/source/six_h100_run.py" "$ROOT/config.json"
