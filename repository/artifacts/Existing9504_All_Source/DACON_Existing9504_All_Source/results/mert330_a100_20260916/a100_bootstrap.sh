#!/usr/bin/env bash
set -euo pipefail
export V7_RESUMED_EPOCH=$(date +%s)
ROOT=/root/C08_MERT330_A100
OBJECT=/root-1/temporal/mert330-a100-20260916
mkdir -p "$ROOT"
cd "$ROOT"
if [ -e "$OBJECT/run_started.json" ]; then
    echo 'Already dispatched; inspect saved state before explicit recovery.'
    exit 0
fi
python - <<'PY'
import hashlib,json,zipfile,urllib.request
from pathlib import Path
root=Path.cwd(); obj=Path('/root-1/temporal/mert330-a100-20260916')
def sha(path):
    with path.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
p=obj/'C08_MERT330_A100_Input.zip'
assert sha(p)=='445f3d237813c720949cad8afc02dd27f956a6f433607ec9c93c656488d44870'
with zipfile.ZipFile(p) as z:
    assert all((root/n).resolve().is_relative_to(root) for n in z.namelist())
    assert z.testzip() is None
    z.extractall(root)
for name,digest in json.loads((root/'input_manifest.json').read_text())['sha256'].items():
    assert sha(root/name)==digest,name
# Reuse the exact prior paired MERT95 cache and baseline separation assets.
p=Path('/root-1/temporal/h100-resolution-20260915/Temporal_Resolution_features.zip')
assert sha(p)=='3be777517142157d7928d4d605ce68ca8fcd0a92a019cbf30b8a1b5d34b231eb'
(root/'reference95_cache').mkdir(exist_ok=True)
with zipfile.ZipFile(p) as z:
    assert len(z.namelist())==1446
    for name in z.namelist():
        assert name.startswith('feature_cache/') and name.endswith('.npz')
        (root/'reference95_cache'/Path(name).name).write_bytes(z.read(name))
p=Path('/root-1/temporal/h100-resolution-20260915/Temporal_Resolution_Input.zip')
assert sha(p)=='b66032891e3d32383e6b2c76cf1d188cd2fa26d20ca36cb0af32aa12f8f7bf20'
with zipfile.ZipFile(p) as z:
    source=json.loads(z.read('input_manifest.json'))['sha256']
    for name in z.namelist():
        if name.startswith('assets/baseline/') and not name.endswith('/'):
            data=z.read(name)
            assert hashlib.sha256(data).hexdigest()==source[name],name
            path=root/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(data)
mapping=json.loads((root/'existing_audio_map.json').read_text())
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
meta=json.loads((root/'mert330_metadata.json').read_text())
weight=root/'assets/mert/pytorch_model.bin'
url=f"https://huggingface.co/{meta['model']}/resolve/{meta['revision']}/pytorch_model.bin"
urllib.request.urlretrieve(url,weight)
assert weight.stat().st_size==meta['weight_bytes'] and sha(weight)==meta['weight_sha256']
for name,digest in meta['source_sha256'].items():assert sha(root/'assets/mert'/name)==digest
assets=root/'assets'
(assets/'manifest.json').write_text(json.dumps({str(p.relative_to(assets)):sha(p) for p in assets.rglob('*') if p.is_file() and p.name!='manifest.json'},indent=2))
print('1446 audio, M95 reference cache and controls, Demucs, pinned MERT330 assets verified',flush=True)
PY
if ! command -v uv >/dev/null 2>&1; then python -m pip install uv; fi
apt-get update -qq
apt-get install -y ffmpeg
uv python install 3.13.9
uv venv --python 3.13.9 /root/mert330-a100-venv
uv pip install --python /root/mert330-a100-venv/bin/python 'torch==2.9.1' 'torchaudio==2.9.1' --index-url https://download.pytorch.org/whl/cu130
uv pip install --python /root/mert330-a100-venv/bin/python -r requirements-backbone.txt
export PATH=/root/mert330-a100-venv/bin:$PATH
export PYTHONPATH="$ROOT" MPLBACKEND=Agg OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 PYTHONUNBUFFERED=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
python -m ipykernel install --user --name temporal-a100 --display-name 'Python (Temporal A100)'
export EXPERIMENT_NOTEBOOK=C08_MERT330_A100.ipynb EXPERIMENT_EXECUTED=C08_MERT330_A100_Executed.ipynb
export EXPERIMENT_GPU=A100
export EXPERIMENT_KERNEL=temporal-a100 EXPERIMENT_OUTPUT_DIR=results
export EXPERIMENT_RESULTS_ZIP=C08_MERT330_A100_results.zip EXPERIMENT_COMPUTE_HOURS=2
python - <<'PY'
import json,os,time,sys,torch
from pathlib import Path
assert sys.version_info[:3]==(3,13,9)
assert torch.__version__=='2.9.1+cu130' and 'A100' in torch.cuda.get_device_name()
r=dict(utc=time.time(),pid=os.getppid(),deadline=float(os.environ['V7_RESUMED_EPOCH'])+7200,python=sys.version,torch=torch.__version__,gpu=torch.cuda.get_device_name())
Path('/root-1/temporal/mert330-a100-20260916/run_started.json').write_text(json.dumps(r))
Path('runtime.json').write_text(json.dumps(r))
PY
nvidia-smi --query-gpu=timestamp,utilization.gpu,memory.used,power.draw --format=csv -l 5 > "$ROOT/gpu_usage.csv" &
GPU_MONITOR_PID=$!
trap 'kill "$GPU_MONITOR_PID" 2>/dev/null || true' EXIT
python v7_notebook_runner.py > notebook.log 2>&1
cp C08_MERT330_A100_results.zip "$OBJECT/C08_MERT330_A100_results.zip"
cp download_manifest.json "$OBJECT/download_manifest.json"
python - <<'PY'
from pathlib import Path
import zipfile,hashlib,json
out=Path('/root-1/temporal/mert330-a100-20260916/C08_MERT330_A100_features.zip')
with zipfile.ZipFile(out,'w',zipfile.ZIP_STORED) as z:
    for p in Path('feature_cache').glob('*.npz'):z.write(p,p)
with zipfile.ZipFile(out) as z:assert z.testzip() is None
with out.open('rb') as f:digest=hashlib.file_digest(f,'sha256').hexdigest()
out.with_suffix('.json').write_text(json.dumps(dict(sha256=digest,bytes=out.stat().st_size)))
print('RESULTS_AND_FEATURES_PERSISTED',flush=True)
PY
