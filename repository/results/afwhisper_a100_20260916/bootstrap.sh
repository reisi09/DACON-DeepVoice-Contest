#!/usr/bin/env bash
set -euo pipefail
export V7_RESUMED_EPOCH=$(date +%s)
RUN_ROOT=/root/C08_AFWhisper_A100
RUN_OBJECT=/root-1/temporal/afwhisper-a100-20260916
if [ -e "$RUN_ROOT/run_started.json" ]; then
    echo 'Existing run found; inspect it before recovery.'
    exit 1
fi
mkdir -p "$RUN_ROOT" "$RUN_OBJECT"
cd "$RUN_ROOT"
export PATH=/root/mert330-a100-venv/bin:$PATH
python - <<'PY'
import hashlib,json,os,shutil,time,urllib.request,zipfile
from pathlib import Path
import torch
root=Path.cwd();old=Path('/root/C08_MERT330_A100')
def sha(p):
    with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
p=Path('/root/C08_AFWhisper_A100_Input.zip')
assert sha(p)=='3b4f0ad363a82724007a8cb5742c41dfb93321a26644b60a7924f80e2d72cecd'
with zipfile.ZipFile(p) as z:
    assert all((root/n).resolve().is_relative_to(root) for n in z.namelist())
    assert z.testzip() is None
    z.extractall(root)
for name,digest in json.loads((root/'input_manifest.json').read_text())['sha256'].items():
    assert sha(root/name)==digest,name
for name in ('audio','feature_cache','reference95_cache'):
    assert (old/name).is_dir();(root/name).symlink_to(old/name,target_is_directory=True)
(root/'assets/baseline').symlink_to(old/'assets/baseline',target_is_directory=True)
for name,digest in json.loads((old/'assets/manifest.json').read_text()).items():
    if name.startswith('baseline/'):assert sha(root/'assets'/name)==digest,name
meta=json.loads((root/'afwhisper_metadata.json').read_text())
target=root/'assets/afwhisper/model.safetensors'
url=f"https://huggingface.co/{meta['model']}/resolve/{meta['revision']}/sound_tower/model.safetensors?download=true"
with urllib.request.urlopen(url,timeout=120) as src,target.with_suffix('.partial').open('wb') as dst:
    shutil.copyfileobj(src,dst,1024*1024)
assert target.with_suffix('.partial').stat().st_size==meta['weight_bytes']
assert sha(target.with_suffix('.partial'))==meta['weight_sha256']
target.with_suffix('.partial').replace(target)
assert 'A100' in torch.cuda.get_device_name() and torch.__version__=='2.9.1+cu130'
receipt=dict(utc=time.time(),deadline=float(os.environ['V7_RESUMED_EPOCH'])+7200,
             gpu=torch.cuda.get_device_name(),torch=torch.__version__,pid=os.getppid(),
             audio_parameters=meta['audio_parameters'],weight_sha256=meta['weight_sha256'])
(root/'run_started.json').write_text(json.dumps(receipt,indent=2))
(root/'runtime.json').write_text(json.dumps(receipt,indent=2))
obj=Path('/root-1/temporal/afwhisper-a100-20260916')
shutil.copy2(root/'run_started.json',obj/'run_started.json');shutil.copy2(p,obj/p.name)
print(json.dumps(receipt),flush=True)
PY
export DEBIAN_FRONTEND=noninteractive NEEDRESTART_MODE=a
apt-get update -qq > ffmpeg_install.log 2>&1
apt-get install -y -qq ffmpeg >> ffmpeg_install.log 2>&1 < /dev/null
export PYTHONPATH="$RUN_ROOT" MPLBACKEND=Agg OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 PYTHONUNBUFFERED=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export EXPERIMENT_NOTEBOOK=C08_AFWhisper_A100.ipynb EXPERIMENT_EXECUTED=C08_AFWhisper_A100_Executed.ipynb
export EXPERIMENT_KERNEL=temporal-a100 EXPERIMENT_OUTPUT_DIR=results
export EXPERIMENT_RESULTS_ZIP=C08_AFWhisper_A100_results.zip EXPERIMENT_COMPUTE_HOURS=2
nvidia-smi --query-gpu=timestamp,utilization.gpu,memory.used,power.draw --format=csv -l 5 > gpu_usage.csv &
GPU_MONITOR_PID=$!
trap 'kill "$GPU_MONITOR_PID" 2>/dev/null || true' EXIT
python v7_notebook_runner.py > notebook.log 2>&1
cp C08_AFWhisper_A100_results.zip "$RUN_OBJECT/C08_AFWhisper_A100_results.zip"
cp download_manifest.json "$RUN_OBJECT/download_manifest.json"
cp run_state.json "$RUN_OBJECT/run_state.json"
python - <<'PY'
from pathlib import Path
import hashlib,json,zipfile
p=Path('/root-1/temporal/afwhisper-a100-20260916/AFWhisper_features.zip')
with zipfile.ZipFile(p.with_suffix('.partial'),'w',zipfile.ZIP_STORED) as z:
    for f in Path('afwhisper_cache').glob('*.npz'):z.write(f,f.name)
p.with_suffix('.partial').replace(p)
with p.open('rb') as f:digest=hashlib.file_digest(f,'sha256').hexdigest()
p.with_suffix('.json').write_text(json.dumps(dict(sha256=digest,bytes=p.stat().st_size)))
print('RESULTS_AND_FEATURES_PERSISTED',flush=True)
PY
