#!/usr/bin/env bash
set -euo pipefail
[[ "${PANNS_GUARD_PID:-}" == "$PPID" ]] || exit 2
trap 'wait; exit 130' TERM INT HUP
export PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1
PANNS_RESUME_ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
PANNS_WORK=/root/panns-cnn-v1-20260925/application-v11
PANNS_PYTHON=/root/panns-cnn-v1-20260925/env/venv/bin/python
if [[ ! -x "$PANNS_PYTHON" || ! -f "$PANNS_WORK/run.json" ]]; then
  python3 - "$PANNS_WORK" <<'PY'
import hashlib,json,shutil,sys,tarfile
from pathlib import Path,PurePosixPath
work=Path(sys.argv[1]);source=Path('/shared/panns-cnn-v1-20260925/inputs-v12')
lock=json.loads((source/'launch_lock.json').read_text())
if lock['workspace']!='wsp-nvxqw1jf9zdv' or lock['sha256']['run_bundle.tar']!='2b6cd7790fd77d8aa04f5cbd8f561f397600ea33eefabdfc22bc7d5b47b9fb7c':
    raise ValueError('Frozen recovery bundle identity mismatch')
for name,target_name in [('run_bundle.tar','bundle'),('noise-v2.tar','bundle/noise')]:
    archive=source/name if name=='run_bundle.tar' else source.parent/'inputs'/name
    with archive.open('rb') as stream:
        if hashlib.file_digest(stream,'sha256').hexdigest()!=lock['sha256'][name]:
            raise ValueError('Frozen recovery archive changed: '+name)
    target=work/target_name;target.mkdir(parents=True,exist_ok=True)
    with tarfile.open(archive,'r:') as tar:
        seen=set()
        for item in tar:
            path=PurePosixPath(item.name)
            if not item.isfile() or path.is_absolute() or '..' in path.parts or '\\' in item.name or item.name in seen or str(path)!=item.name:
                raise ValueError('Unsafe recovery archive entry')
            seen.add(item.name);destination=(target/item.name).resolve()
            if not destination.is_relative_to(target.resolve()):raise ValueError('Recovery archive escapes target')
            destination.parent.mkdir(parents=True,exist_ok=True)
            incoming=tar.extractfile(item)
            if destination.exists():
                with destination.open('rb') as stream:old=hashlib.file_digest(stream,'sha256').hexdigest()
                if old!=hashlib.file_digest(incoming,'sha256').hexdigest():raise ValueError('Existing recovery member differs')
            else:
                temporary=destination.with_suffix(destination.suffix+'.tmp')
                with temporary.open('wb') as stream:shutil.copyfileobj(incoming,stream)
                temporary.replace(destination)
print('Frozen bundle and noise restored',flush=True)
PY
  python3 "$PANNS_WORK/bundle/code/panns_restore_environment.py" --shared /shared --work "$PANNS_WORK"
  "$PANNS_PYTHON" "$PANNS_WORK/bundle/code/panns_cloud_prepare.py" --bundle "$PANNS_WORK/bundle" --volume-root /shared --work "$PANNS_WORK"
fi
python3 - "$PANNS_WORK/run.json" <<'PY'
import hashlib,sys
from pathlib import Path
with Path(sys.argv[1]).open('rb') as stream:
    if hashlib.file_digest(stream,'sha256').hexdigest()!='dad5000a0b0b3d11cec4b94ac27aa4f2eba6ac4d1661c4ff4e740cdf110c5fe3':
        raise ValueError('Original training configuration changed during recovery')
print('Original training configuration restored',flush=True)
PY
"$PANNS_PYTHON" - "$PANNS_WORK" <<'PY'
import sys
from pathlib import Path
import torch
work=Path(sys.argv[1])
labels=work/'package/model/assets/baseline/model/panns/class_labels_indices.csv'
if labels.read_bytes()!=(Path.home()/'panns_data/class_labels_indices.csv').read_bytes():
    raise ValueError('Offline PANNs labels changed during recovery')
if not torch.cuda.is_available() or 'A100' not in torch.cuda.get_device_name(0):
    raise RuntimeError('Expected the authorized existing A100')
print('Offline labels and existing A100 verified',flush=True)
PY
chmod u+x "$PANNS_WORK/package/model/runtime/bin/ffmpeg"
PANNS_PREFETCH_DEADLINE=$(python3 - <<'PY'
import os,time
print(min(time.time()+900,float(os.environ['PANNS_COMPUTE_DEADLINE_EPOCH'])-60))
PY
)
"$PANNS_PYTHON" "$PANNS_RESUME_ROOT/panns_prefetch_cache.py" \
  --index /shared/panns-cnn-v1-20260925/run-v9/cache/index.json \
  --sha256 e9b651cbf1b55045a2f1a74bdbebc659655f3ee2691626755ef2b79f1db6b5db \
  --destination "$PANNS_WORK/cache_local" --deadline-epoch "$PANNS_PREFETCH_DEADLINE"
exec /bin/bash /shared/panns-cnn-v1-20260925/submission12h-inputs-v2/panns_submission_launch.sh
