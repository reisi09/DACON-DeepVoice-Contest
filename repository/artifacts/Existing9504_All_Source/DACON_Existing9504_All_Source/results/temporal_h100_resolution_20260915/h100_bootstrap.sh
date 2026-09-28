#!/usr/bin/env bash
set -euo pipefail
export V7_RESUMED_EPOCH=$(date +%s)
ROOT=/root/Temporal_Resolution
OBJECT=/root-1/temporal/h100-resolution-20260915
mkdir -p "$ROOT"
cd "$ROOT"
if [ -e "$OBJECT/run_started.json" ]; then
    echo 'Already dispatched; inspect saved state before any explicit recovery.'
    exit 0
fi
python - <<'PY'
import hashlib,json,zipfile
from pathlib import Path
root=Path.cwd(); obj=Path('/root-1/temporal/h100-resolution-20260915')
for name,digest in [('Temporal_Resolution_Input.zip','b66032891e3d32383e6b2c76cf1d188cd2fa26d20ca36cb0af32aa12f8f7bf20')]:
    path=obj/name
    with path.open('rb') as f: assert hashlib.file_digest(f,'sha256').hexdigest()==digest,name
    with zipfile.ZipFile(path) as z:
        assert all((root/n).resolve().is_relative_to(root) for n in z.namelist())
        assert z.testzip() is None
        z.extractall(root)
for name,digest in json.loads((root/'input_manifest.json').read_text())['sha256'].items():
    with (root/name).open('rb') as f: assert hashlib.file_digest(f,'sha256').hexdigest()==digest,name
mapping=json.loads(Path('/root/resolution_existing_audio_map.json').read_text())
rows=[r for rows in json.loads((root/'locked_split.json').read_text()).values() for r in rows]
assert set(mapping)=={r['ID'] for r in rows}
(root/'audio').mkdir(exist_ok=True)
for key in sorted({v[0] for v in mapping.values()}):
    with zipfile.ZipFile(Path('/root-1')/key) as z:
        for row in rows:
            archive,member=mapping[row['ID']]
            if archive!=key:continue
            data=z.read(member)
            assert hashlib.sha256(data).hexdigest()==row['audio_sha256'],row['ID']
            (root/'audio'/(row['ID']+'.wav')).write_bytes(data)
print('All 1446 original audio hashes and source/asset hashes verified',flush=True)
PY
if ! command -v uv >/dev/null 2>&1; then python -m pip install uv; fi
apt-get update -qq
apt-get install -y ffmpeg
uv python install 3.13.9
uv venv --python 3.13.9 /root/resolution-venv
uv pip install --python /root/resolution-venv/bin/python 'torch==2.9.1' 'torchaudio==2.9.1' --index-url https://download.pytorch.org/whl/cu130
uv pip install --python /root/resolution-venv/bin/python -r requirements-resolution.txt
uv pip install --python /root/resolution-venv/bin/python 'torchcodec==0.8.1'
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
