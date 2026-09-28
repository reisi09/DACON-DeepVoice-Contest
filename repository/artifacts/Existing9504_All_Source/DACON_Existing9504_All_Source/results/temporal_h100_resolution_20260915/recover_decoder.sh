#!/usr/bin/env bash
set -euo pipefail
ROOT=/root/Temporal_Resolution
OBJECT=/root-1/temporal/h100-resolution-20260915
cd "$ROOT"
export V7_RESUMED_EPOCH=$(/root/resolution-venv/bin/python -c "import json; print(json.load(open('runtime.json'))['deadline']-7200)")
export PATH=/root/resolution-venv/bin:$PATH
export PYTHONPATH="$ROOT" MPLBACKEND=Agg OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 PYTHONUNBUFFERED=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
python -m ipykernel install --user --name temporal-h100 --display-name 'Python (Temporal H100)'
export EXPERIMENT_NOTEBOOK=Temporal_H100_Resolution_Softmax.ipynb EXPERIMENT_EXECUTED=Temporal_H100_Resolution_Softmax_Executed.ipynb
export EXPERIMENT_KERNEL=temporal-h100 EXPERIMENT_OUTPUT_DIR=results
export EXPERIMENT_RESULTS_ZIP=Temporal_Resolution_results.zip EXPERIMENT_COMPUTE_HOURS=2
python - <<'PY'
import json,os,time,sys,torch
from pathlib import Path
assert sys.version_info[:3]==(3,13,9)
assert torch.__version__=='2.9.1+cu130' and 'H100' in torch.cuda.get_device_name()
r=dict(utc=time.time(),pid=os.getppid(),deadline=float(os.environ['V7_RESUMED_EPOCH'])+7200,python=sys.version,torch=torch.__version__,gpu=torch.cuda.get_device_name())
Path('/root-1/temporal/h100-resolution-20260915/run_started.json').write_text(json.dumps(r))
Path('runtime.json').write_text(json.dumps(r))
PY
nvidia-smi --query-gpu=timestamp,utilization.gpu,memory.used,power.draw --format=csv -l 5 > "$ROOT/gpu_usage.csv" &
GPU_MONITOR_PID=$!
trap 'kill "$GPU_MONITOR_PID" 2>/dev/null || true' EXIT
python v7_notebook_runner.py > notebook.log 2>&1
cp Temporal_Resolution_results.zip "$OBJECT/Temporal_Resolution_results.zip"
cp download_manifest.json "$OBJECT/download_manifest.json"
python - <<'PY'
from pathlib import Path
import zipfile,hashlib,json
out=Path('/root-1/temporal/h100-resolution-20260915/Temporal_Resolution_features.zip')
with zipfile.ZipFile(out,'w',zipfile.ZIP_STORED) as z:
    for p in Path('feature_cache').glob('*.npz'):z.write(p,p.relative_to(Path.cwd()) if p.is_absolute() else p)
with out.open('rb') as f:digest=hashlib.file_digest(f,'sha256').hexdigest()
out.with_suffix('.json').write_text(json.dumps(dict(sha256=digest,bytes=out.stat().st_size)))
print('RESULTS_AND_FEATURES_PERSISTED',flush=True)
PY
