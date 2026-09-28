#!/usr/bin/env bash
# CPU-only export from the verified Capella root to the existing project object volume.
# A separately authorized CPU job must bound this command with GNU timeout.
set -euo pipefail
export CUDA_VISIBLE_DEVICES='' HF_HUB_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1
PY=/root/envs/data/bin/python
WORK=/root/lb-improvement-20260924/a100-export-v1
SHARED=/shared/lb-improvement-20260924/a100-preparation-v1
test -x "$PY"
mkdir -p "$WORK"
"$PY" "$SHARED/export_bank_audio.py" \
  --manifest /root/lb-improvement-20260924/prepared-v1/inputs/feature_union.csv \
  --manifest-sha256 492bbe454abce816beab07d97baecdcf34e34ec1936d5230d955c2cda8f27906 \
  --input /root/data_v4/out --output "$WORK/audio.tar" | tee "$WORK/export.log"
"$PY" - "$WORK" "$SHARED" <<'PY'
import hashlib, json, shutil, sys
from pathlib import Path
work, shared = map(Path, sys.argv[1:])
receipt = json.loads((work / 'export.log').read_text().splitlines()[-1])
if receipt['files'] != 43516 or receipt['audio_bytes'] != 12329271166:
    raise ValueError('Export coverage differs from the verified bank')
print(json.dumps(dict(phase='upload_audio', bytes=receipt['archive_bytes'])), flush=True)
target = shared / 'audio/portable-audio.tar'
target.parent.mkdir(parents=True, exist_ok=True)
with (work / 'audio.tar').open('rb') as source, target.open('xb') as destination:
    shutil.copyfileobj(source, destination, 16 * 1024**2)
print(json.dumps(dict(phase='verify_remote_archive')), flush=True)
with target.open('rb') as stream:
    actual = hashlib.file_digest(stream, 'sha256').hexdigest()
if actual != receipt['archive_sha256'] or target.stat().st_size != receipt['archive_bytes']:
    raise ValueError('Object-volume readback differs from exported audio')
receipt.update(remote_archive='lb-improvement-20260924/a100-preparation-v1/audio/portable-audio.tar',
               full_remote_readback_verified=True, a100_workspace_started=False)
with (target.parent / 'export_receipt.json').open('x') as stream:
    json.dump(receipt, stream, indent=2)
print(json.dumps(receipt), flush=True)
PY
