#!/usr/bin/env bash
set -euo pipefail
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
export CUBLAS_WORKSPACE_CONFIG=:4096:8 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export FAST_DEADLINE="${FAST_DEADLINE:?}"
ROOT=/root/fast_h100_20260917
REMOTE=/shared/fast-h100-20260917
mkdir -p "$ROOT" "$REMOTE"
cd "$ROOT"
exec > >(tee -a "$ROOT/bootstrap.log") 2>&1
test "$(df -Pk /root | awk 'NR==2 {print $4}')" -gt 30000000
cp "$REMOTE/input.zip" "$ROOT/input.zip"
python - <<'PY'
import hashlib,json,zipfile
from pathlib import Path
root=Path.cwd()
manifest=json.loads(Path('/shared/fast-h100-20260917/upload_receipt.json').read_text())
with (root/'input.zip').open('rb') as stream:
    assert hashlib.file_digest(stream,'sha256').hexdigest()==manifest['sha256']
with zipfile.ZipFile(root/'input.zip') as archive:
    assert all((root/name).resolve().is_relative_to(root) for name in archive.namelist())
    assert archive.testzip() is None
    archive.extractall(root)
for name,digest in json.loads((root/'input_manifest.json').read_text()).items():
    with (root/name).open('rb') as stream:
        assert hashlib.file_digest(stream,'sha256').hexdigest()==digest,name
print('Uploaded code and additional audio hashes verified',flush=True)
PY
export PATH=/root/.local/bin:$PATH
if ! command -v uv >/dev/null 2>&1; then
    python -m pip install --disable-pip-version-check uv==0.8.22
fi
uv pip install --python /root/fullft8/.venv/bin/python librosa==0.11.0 scikit-learn==1.7.2 panns-inference==0.1.1 torchlibrosa==0.1.0
mkdir -p results
nvidia-smi --query-gpu=timestamp,utilization.gpu,memory.used,power.draw --format=csv -l 10 > results/gpu_usage.csv &
MONITOR_PID=$!
set +e
remaining=$((FAST_DEADLINE - $(date +%s)))
timeout -k 30s "${remaining}s" /root/fullft8/.venv/bin/python -u fast_h100_ablation.py 2>&1 | tee results/train.log
status=${PIPESTATUS[0]}
set -e
kill "$MONITOR_PID" 2>/dev/null || true
export FAST_EXIT_CODE="$status"
python - <<'PY'
import hashlib,json,os,shutil,time,zipfile
from pathlib import Path
root=Path.cwd();out=root/'results';remote=Path('/shared/fast-h100-20260917')
status='complete' if os.environ['FAST_EXIT_CODE']=='0' else 'failed'
(out/'run_state.json').write_text(json.dumps(dict(status=status,exit_code=int(os.environ['FAST_EXIT_CODE']),finished=time.time())))
for name in ['fast_h100_ablation.py','input_manifest.json','baseline_asset_hashes.json','mert330_metadata.json']:
    shutil.copy2(root/name,out/name)
manifest={str(p.relative_to(out)):hashlib.file_digest(p.open('rb'),'sha256').hexdigest() for p in out.rglob('*') if p.is_file()}
(out/'artifact_manifest.json').write_text(json.dumps(manifest,indent=2))
archive=root/'results.zip'
with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED) as z:
    for p in out.rglob('*'):
        if p.is_file():z.write(p,p.relative_to(out))
with archive.open('rb') as stream:digest=hashlib.file_digest(stream,'sha256').hexdigest()
shutil.copyfile(archive,remote/'results.zip')
state=dict(status=status,archive_ready=True,bytes=archive.stat().st_size,sha256=digest,finished=time.time())
(remote/'run_state.json').write_text(json.dumps(state))
print(json.dumps(state),flush=True)
PY
