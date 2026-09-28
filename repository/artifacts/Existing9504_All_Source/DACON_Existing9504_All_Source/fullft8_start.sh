#!/usr/bin/env bash
set -euo pipefail
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
export PATH="/root/.local/bin:$PATH"
export FULLFT_JOB_STARTED_AT="${FULLFT_JOB_STARTED_AT:-$(date +%s)}"
export FULLFT_RUN_ID="${FULLFT_RUN_ID:?}"
export FULLFT_REMOTE="/shared/fullft8/h100x1-20260916/runs/$FULLFT_RUN_ID"
mkdir -p /root/fullft8 "$FULLFT_REMOTE"
cd /root/fullft8
exec > >(tee -a "$FULLFT_REMOTE/bootstrap.log") 2>&1
echo "BOOTSTRAP $(date -u)"
python - <<'PY'
import json, os, time
from pathlib import Path
remote = Path(os.environ['FULLFT_REMOTE'])
marker = remote/'ARTIFACTS_SYNCED.json'
if marker.exists():
    marker.rename(remote/f'previous_attempt_{int(time.time())}.json')
(remote/'run_state.json').write_text(json.dumps({'status':'bootstrapping','utc':time.time(),'gpu_resource_stopped':False}))
PY
df -h /root /shared
test "$(df -Pk /root | awk 'NR==2 {print $4}')" -gt 100000000
python -m pip install --disable-pip-version-check uv==0.8.22
uv python install 3.11
test -x /root/fullft8/.venv/bin/python || uv venv --python 3.11 /root/fullft8/.venv
uv pip install --python .venv/bin/python torch==2.7.1 torchaudio==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cu128
uv pip install --python .venv/bin/python transformers==4.51.3 timm==1.0.15 numpy==1.26.4 soundfile==0.13.1 demucs==4.0.1 einops==0.8.1 nbformat==5.10.4 nnAudio==0.3.3
cp /shared/fullft8/h100x1-20260916/MERT_EAT_HTDemucs_H100x1_8Experiments.ipynb .
cp /shared/fullft8/h100x1-20260916/fullft8_runtime.py .
# ZIP seeks on the object mount are slow; stage existing archives once locally.
mkdir -p /root/fullft8/inputs/v6 /root/fullft8/inputs/c3/full-final-20260915
test -f /root/fullft8/inputs/v6/V6_H100_RunBundle.zip || cp /shared/v6/V6_H100_RunBundle.zip /root/fullft8/inputs/v6/
test -f /root/fullft8/inputs/c3/full-final-20260915/C3_Final_Input.zip || cp /shared/c3/full-final-20260915/C3_Final_Input.zip /root/fullft8/inputs/c3/full-final-20260915/
mkdir -p /root/fullft8/assets
test -d /root/fullft8/assets/models || cp -r /shared/fullft7/h100x1-20260916/models /root/fullft8/assets/
cp /shared/fullft7/h100x1-20260916/model_manifest.json /shared/fullft7/h100x1-20260916/existing_audio_map.json /root/fullft8/assets/
remaining=$((FULLFT_JOB_STARTED_AT + 55800 - $(date +%s)))
test "$remaining" -gt 0
exec timeout -k 60s "${remaining}s" .venv/bin/python -u fullft8_runtime.py
