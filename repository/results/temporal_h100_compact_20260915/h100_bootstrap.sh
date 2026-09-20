#!/usr/bin/env bash
set -euo pipefail
export V7_RESUMED_EPOCH=$(date +%s)
ROOT=/root/Temporal_Compact
OBJECT=/root-1/temporal/h100-compact-20260915
if [ -e "$OBJECT/run_started.json" ]; then
    echo 'This experiment has already been dispatched; no automatic duplicate run.'
    exit 0
fi
mkdir -p "$ROOT" "$OBJECT"
cd "$ROOT"
python - <<'PY'
import hashlib,json,time,zipfile
from pathlib import Path
root=Path.cwd()
for path,expected in [('/root/Temporal_H100_Compact_Tuning_Input.zip','5250d6474985aa02579111c7c45825374796ebf318907726917e6ee05276f18d'),('/root-1/temporal/h100-transfer/features.zip','ddd3d481269e95172762f3ec988ccb15b541424d10ee05f9925deed463dcb13c')]:
    p=Path(path)
    with p.open('rb') as stream: assert hashlib.file_digest(stream,'sha256').hexdigest()==expected,p.name
    with zipfile.ZipFile(p) as z:
        for name in z.namelist(): assert (root/name).resolve().is_relative_to(root)
        assert z.testzip() is None
        z.extractall(root)
for name,digest in json.loads((root/'input_manifest.json').read_text())['sha256'].items():
    assert hashlib.sha256((root/name).read_bytes()).hexdigest()==digest,name
print('Input bundle and 1446 exported feature files unpacked',flush=True)
PY
if ! command -v uv >/dev/null 2>&1; then
    python -m pip install --disable-pip-version-check uv
fi
if [ ! -x /root/compact-venv/bin/python ]; then
    uv python install 3.13.9
    uv venv --python 3.13.9 /root/compact-venv
fi
uv pip install --python /root/compact-venv/bin/python 'torch==2.9.1' --index-url https://download.pytorch.org/whl/cu130
uv pip install --python /root/compact-venv/bin/python -r requirements-head.txt
/root/compact-venv/bin/python - <<'CHECK'
import sys, torch
assert sys.version_info[:3] == (3, 13, 9), sys.version
assert torch.__version__ == '2.9.1+cu130', torch.__version__
assert torch.cuda.is_available()
print('Pinned runtime verified:', sys.version.split()[0], torch.__version__, flush=True)
CHECK
/root/compact-venv/bin/python -m ipykernel install --user --name temporal-h100 --display-name 'Python (Temporal H100)'
export PATH=/root/compact-venv/bin:$PATH
export PYTHONPATH="$ROOT" MPLBACKEND=Agg OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 PYTHONUNBUFFERED=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export EXPERIMENT_NOTEBOOK=Temporal_H100_Compact_Tuning.ipynb EXPERIMENT_EXECUTED=Temporal_H100_Compact_Tuning_Executed.ipynb
export EXPERIMENT_KERNEL=temporal-h100 EXPERIMENT_OUTPUT_DIR=results
export EXPERIMENT_RESULTS_ZIP=Temporal_H100_Compact_Tuning_results.zip EXPERIMENT_COMPUTE_HOURS=0.5
python - <<'PY'
import json,os,time
from pathlib import Path
p=Path('/root-1/temporal/h100-compact-20260915/run_started.json')
p.write_text(json.dumps(dict(utc=time.time(),pid=os.getppid(),deadline=float(os.environ['V7_RESUMED_EPOCH'])+1800)))
PY
nvidia-smi --query-gpu=timestamp,utilization.gpu,memory.used,power.draw --format=csv -l 5 > "$ROOT/gpu_usage.csv" &
GPU_MONITOR_PID=$!
trap 'kill "$GPU_MONITOR_PID" 2>/dev/null || true' EXIT
/root/compact-venv/bin/python "$ROOT/v7_notebook_runner.py" > "$ROOT/notebook.log" 2>&1
cp "$ROOT/Temporal_H100_Compact_Tuning_results.zip" "$OBJECT/Temporal_H100_Compact_Tuning_results.zip"
cp "$ROOT/download_manifest.json" "$OBJECT/download_manifest.json"
echo 'RESULTS_PERSISTED_TO_OBJECT_STORAGE'
