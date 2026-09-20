#!/usr/bin/env bash
set -euo pipefail
export V7_RESUMED_EPOCH=$(date +%s)
ROOT=/root/C08_Methods_A100
OBJECT=/root-1/temporal/methods-a100-20260916
export PATH=/root/mert330-a100-venv/bin:$PATH
mkdir -p "$OBJECT"
if [ -e "$ROOT/run_started.json" ]; then
    echo 'Already dispatched; inspect existing run before retrying.'
    exit 1
fi
mkdir -p "$ROOT"
cd "$ROOT"
python - <<'PY'
import hashlib,json,os,shutil,time,zipfile
from pathlib import Path
import torch
root=Path.cwd(); old=Path('/root/C08_MERT330_A100')
def sha(p):
    with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
p=Path('/root/C08_Methods_A100_Input.zip')
assert sha(p)=='5e29328628e1140892caac53052300cdf90e144b16f8a9468db7e09d48c084a2'
with zipfile.ZipFile(p) as z:
    assert all((root/n).resolve().is_relative_to(root) for n in z.namelist())
    assert z.testzip() is None
    z.extractall(root)
for name,digest in json.loads((root/'input_manifest.json').read_text())['sha256'].items():
    assert sha(root/name)==digest,name
for name in ('audio','feature_cache','reference95_cache'):
    assert (old/name).is_dir(),name
    (root/name).symlink_to(old/name,target_is_directory=True)
for path in (old/'assets').rglob('*'):
    if not path.is_file():continue
    target=root/path.relative_to(old)
    target.parent.mkdir(parents=True,exist_ok=True)
    if target.exists():assert sha(target)==sha(path),str(target)
    else:target.symlink_to(path)
rows=[r for split in json.loads((root/'locked_split.json').read_text()).values() for r in split]
for row in rows:assert sha(root/'audio'/(row['ID']+'.wav'))==row['audio_sha256']
for filename,folder in [('reference95_features_manifest.json','reference95_cache'),
                         ('reference330_features_manifest.json','feature_cache')]:
    manifest=json.loads((root/filename).read_text())
    assert set(manifest['records'])=={r['ID'] for r in rows}
    for row in rows:
        item=manifest['records'][row['ID']]
        assert item['identity']==row['audio_sha256']+':'+manifest['cache_id']
        assert item['audio_sha256']==row['audio_sha256']
        assert sha(root/folder/item['file'])==item['sha256'],row['ID']
assert 'A100' in torch.cuda.get_device_name()
assert torch.__version__=='2.9.1+cu130'
receipt=dict(utc=time.time(),deadline=float(os.environ['V7_RESUMED_EPOCH'])+3600,
             gpu=torch.cuda.get_device_name(),torch=torch.__version__,pid=os.getppid(),
             audio_verified=len(rows),cached_features_verified_per_backbone=len(rows),
             feature_cache_reused=True)
(root/'run_started.json').write_text(json.dumps(receipt,indent=2))
(root/'runtime.json').write_text(json.dumps(receipt,indent=2))
obj=Path('/root-1/temporal/methods-a100-20260916')
shutil.copy2(root/'run_started.json',obj/'run_started.json')
shutil.copy2(p,obj/p.name)
print(json.dumps(receipt),flush=True)
PY
export PYTHONPATH="$ROOT" MPLBACKEND=Agg OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 PYTHONUNBUFFERED=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export EXPERIMENT_NOTEBOOK=C08_Methods_A100.ipynb EXPERIMENT_EXECUTED=C08_Methods_A100_Executed.ipynb
export EXPERIMENT_GPU=A100 EXPERIMENT_KERNEL=temporal-a100 EXPERIMENT_OUTPUT_DIR=results
export EXPERIMENT_RESULTS_ZIP=C08_Methods_A100_results.zip EXPERIMENT_COMPUTE_HOURS=1
nvidia-smi --query-gpu=timestamp,utilization.gpu,memory.used,power.draw --format=csv -l 5 > gpu_usage.csv &
GPU_MONITOR_PID=$!
trap 'kill "$GPU_MONITOR_PID" 2>/dev/null || true' EXIT
python v7_notebook_runner.py > notebook.log 2>&1
cp C08_Methods_A100_results.zip "$OBJECT/C08_Methods_A100_results.zip"
cp download_manifest.json "$OBJECT/download_manifest.json"
cp run_state.json "$OBJECT/run_state.json"
echo RESULTS_PERSISTED
