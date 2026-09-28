#!/usr/bin/env bash
set -euo pipefail
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
export PATH="/root/.local/bin:$PATH"
export FULLFT_JOB_STARTED_AT="${FULLFT_JOB_STARTED_AT:-$(date +%s)}"
export FULLFT_REMOTE="/shared/fullft7/h100x1-20260916/runs/${FULLFT_RUN_ID:?}"
mkdir -p /root/fullft7 "$FULLFT_REMOTE"
cd /root/fullft7
exec > >(tee -a "$FULLFT_REMOTE/bootstrap.log") 2>&1
echo "BOOTSTRAP $(date -u)"
python -m pip install --disable-pip-version-check uv==0.8.22
uv python install 3.11
uv venv --python 3.11 /root/fullft7/.venv
uv pip install --python .venv/bin/python torch==2.7.1 torchaudio==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cu128
uv pip install --python .venv/bin/python transformers==4.51.3 timm==1.0.15 numpy==1.26.4 soundfile==0.13.1 demucs==4.0.1 einops==0.8.1 nbformat==5.10.4 nnAudio==0.3.3
cp /shared/fullft7/h100x1-20260916/MERT_EAT_FullFT_H100x1_7Experiments.ipynb .
cp /shared/fullft7/h100x1-20260916/launcher/run_job.py .
python - <<'PY'
import os, shutil
from pathlib import Path
previous = Path(os.environ['FULLFT_REMOTE']) / 'h100x1_fullft7_run'
if previous.exists():
    shutil.copytree(previous, Path.cwd() / previous.name,
                    ignore=shutil.ignore_patterns('*.uploading'), dirs_exist_ok=True)
PY
exec .venv/bin/python -u run_job.py
