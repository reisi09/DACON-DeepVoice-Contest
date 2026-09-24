#!/usr/bin/env bash
# Run only inside the existing A100 workspace after its explicit startup.
set -euo pipefail
# Interpreter override is restricted to Mac packaging QA, which cannot run CUDA.
if [[ "${PANNS_LOCAL_QA:-0}" == 1 ]]; then
  [[ "$(uname -s)" == Darwin && "${PANNS_PAUSE_ON_EXIT:-1}" == 0 && -n "${PANNS_PYTHON:-}" ]] || exit 2
elif [[ -n "${PANNS_PYTHON:-}" || "${PANNS_PAUSE_ON_EXIT:-1}" != 1 ]]; then
  echo 'Production requires the verified offline environment and automatic pause' >&2
  exit 2
fi
# Disable only for a local packaging check; the production default pauses compute.
if [[ "${PANNS_PAUSE_ON_EXIT:-1}" == 1 ]]; then
  PANNS_VESSLCTL=$(command -v vesslctl) || { echo 'vesslctl required for automatic workspace pause' >&2; exit 2; }
  trap '"$PANNS_VESSLCTL" --org qwdkw workspace pause wsp-nvxqw1jf9zdv' EXIT
fi
PANNS_SHARED=${PANNS_SHARED:-/shared}
PANNS_WORK=${PANNS_WORK:-/root/panns-cnn-v1-20260925}
PANNS_OVERRIDE_PYTHON=${PANNS_PYTHON:-}
PANNS_PYTHON=${PANNS_PYTHON:-$PANNS_WORK/env/venv/bin/python}
PANNS_INPUT="$PANNS_SHARED/panns-cnn-v1-20260925/inputs-v3"
mkdir -p "$PANNS_WORK"
python3 - "$PANNS_INPUT" "$PANNS_WORK" <<'PY'
import hashlib,json,sys,tarfile,shutil
from pathlib import Path,PurePosixPath
source,work=map(Path,sys.argv[1:])
lock=json.loads((source/'launch_lock.json').read_text())
for name,target_name in [('run_bundle.tar','bundle'),('noise-v2.tar','bundle/noise')]:
    archive=(source.parent/'inputs'/name) if name=='noise-v2.tar' else source/name
    with archive.open('rb') as f:actual=hashlib.file_digest(f,'sha256').hexdigest()
    if actual!=lock['sha256'][name]:raise ValueError('Uploaded archive hash mismatch: '+name)
    target=work/target_name
    target.mkdir(parents=True,exist_ok=True)
    with tarfile.open(archive,'r:') as tar:
        seen=set()
        for item in tar:
            path=PurePosixPath(item.name)
            if not item.isfile() or path.is_absolute() or '..' in path.parts or '\\' in item.name or item.name in seen or str(path)!=item.name:
                raise ValueError('Unsafe archive entry')
            seen.add(item.name)
            destination=(target/item.name).resolve()
            if not destination.is_relative_to(target.resolve()):raise ValueError('Archive escapes target')
            destination.parent.mkdir(parents=True,exist_ok=True)
            incoming=tar.extractfile(item)
            if destination.exists():
                with destination.open('rb') as f:old=hashlib.file_digest(f,'sha256').hexdigest()
                if old!=hashlib.file_digest(incoming,'sha256').hexdigest():raise ValueError('Existing bundle member differs')
            else:
                temporary=destination.with_suffix(destination.suffix+'.tmp')
                with temporary.open('wb') as output:shutil.copyfileobj(incoming,output)
                temporary.replace(destination)
print('Pre-uploaded bundle and noise verified')
PY
if [[ -z "$PANNS_OVERRIDE_PYTHON" ]]; then
  python3 "$PANNS_WORK/bundle/code/panns_restore_environment.py" --shared "$PANNS_SHARED" --work "$PANNS_WORK"
fi
[[ -x "$PANNS_PYTHON" ]] || { echo 'Verified Python environment unavailable' >&2; exit 2; }
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1 PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 CUBLAS_WORKSPACE_CONFIG=:4096:8
export PANNS_LOCAL_CACHE="$PANNS_WORK/cache_local"
export HF_HOME="$PANNS_WORK/hf" MPLCONFIGDIR="$PANNS_WORK/mpl" NUMBA_CACHE_DIR="$PANNS_WORK/numba"
"$PANNS_PYTHON" "$PANNS_WORK/bundle/code/panns_cloud_prepare.py" --bundle "$PANNS_WORK/bundle" --volume-root "$PANNS_SHARED" --work "$PANNS_WORK"
"$PANNS_PYTHON" - "$PANNS_WORK" <<'PY'
import hashlib,shutil,sys
from pathlib import Path
root=Path(sys.argv[1]);source=root/'package/model/assets/baseline/model/panns/class_labels_indices.csv'
target=Path.home()/'panns_data/class_labels_indices.csv'
if target.exists():
    if source.read_bytes()!=target.read_bytes():raise ValueError('Existing global label table differs')
else:
    raise FileNotFoundError('Existing offline PANNs label table unavailable; prepare it on CPU before startup')
import torch
if not torch.cuda.is_available() or 'A100' not in torch.cuda.get_device_name(0):raise RuntimeError('Expected existing A100')
print('A100:',torch.cuda.get_device_name(0),'torch:',torch.__version__)
PY
export PATH="$PANNS_WORK/package/model/runtime/bin:$PATH"
chmod u+x "$PANNS_WORK/package/model/runtime/bin/ffmpeg"
"$PANNS_PYTHON" "$PANNS_WORK/bundle/code/panns_cloud_pipeline.py" --run "$PANNS_WORK/run.json" --bundle "$PANNS_WORK/bundle" --hours 48
