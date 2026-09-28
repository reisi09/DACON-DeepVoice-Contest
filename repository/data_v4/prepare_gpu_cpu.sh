#!/usr/bin/env bash
# Explicit CPU preparation on the existing project volumes; never starts a GPU stage.
# Usage: bash prepare_gpu_cpu.sh /shared/.../gpu-preparation.tar.gz EXPECTED_SHA256
set -euo pipefail
export CUDA_VISIBLE_DEVICES='' HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=3 MKL_NUM_THREADS=3
PY=/root/envs/data/bin/python
BASE=/root/lb-improvement-20260924
SHARED=/shared/lb-improvement-20260924
test "$#" -eq 2
test -x "$PY"
test -d /root/data_v4/out/audio
mkdir -p "$BASE"
"$PY" - "$1" "$2" "$BASE" "$SHARED" <<'PY'
import hashlib, json, os, shutil, sys, tarfile, tempfile
from pathlib import Path
archive, expected, base, shared = sys.argv[1:]
base, shared = Path(base), Path(shared)
if base.is_symlink():
    raise ValueError('Preparation base must not be a symbolic link')
def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()
if digest(archive) != expected:
    raise ValueError('Preparation bundle hash mismatch')
if shutil.disk_usage(base).free < 12 * 1024**3:
    raise RuntimeError('Need 12 GiB free for assets, supplemental audio and initial cache space')
target = base / 'prepared-v1'
if target.is_symlink():
    raise ValueError('Preparation directory must not be a symbolic link')
with tempfile.TemporaryDirectory(prefix='.prepare-', dir=base) as temporary:
    unpacked = Path(temporary)
    with tarfile.open(archive, 'r:gz') as tar:
        members = tar.getmembers()
        names = [m.name for m in members]
        if len(names) != len(set(names)) or any(not m.isfile() or Path(m.name).is_absolute()
                or '..' in Path(m.name).parts for m in members):
            raise ValueError('Unsafe preparation archive')
        tar.extractall(unpacked, filter='data')
    hashes = json.loads((unpacked / 'SHA256SUMS.json').read_text())
    if set(names) != set(hashes) | {'SHA256SUMS.json'}:
        raise ValueError('Preparation member list mismatch')
    for name, value in hashes.items():
        if digest(unpacked / name) != value:
            raise ValueError('Preparation member hash mismatch: ' + name)
    if target.exists():
        existing = list(target.rglob('*'))
        if any(p.is_symlink() for p in existing) or {str(p.relative_to(target)) for p in existing if p.is_file()} != set(names):
            raise ValueError('Existing preparation has unexpected files or links')
        for name in names:
            if digest(target / name) != digest(unpacked / name):
                raise ValueError('Existing preparation differs; use a new version')
    else:
        unpacked.rename(target)
assets = base / 'assets'
if assets.is_symlink():
    raise ValueError('Asset directory must not be a symbolic link')
asset_hashes = json.loads((target / 'inputs/asset_hashes.json').read_text())
for name, value in asset_hashes.items():
    if Path(name).is_absolute() or '..' in Path(name).parts:
        raise ValueError('Unsafe asset path')
    destination = assets / name
    if any(p.is_symlink() for p in (destination, *destination.parents) if p.is_relative_to(base)):
        raise ValueError('Asset destination contains a symbolic link')
    if not destination.resolve().is_relative_to(assets.resolve()):
        raise ValueError('Escaping asset destination')
    destination.parent.mkdir(parents=True, exist_ok=True)
    if not destination.exists():
        with tempfile.NamedTemporaryFile(dir=destination.parent, delete=False) as stream:
            temporary = Path(stream.name)
        try:
            shutil.copyfile(shared / 'weights/baseline-assets' / name, temporary)
            if digest(temporary) != value:
                raise ValueError('Asset transfer hash mismatch: ' + name)
            os.link(temporary, destination)
        finally:
            temporary.unlink(missing_ok=True)
    if digest(destination) != value:
        raise ValueError('Existing asset differs: ' + name)
PY
CODE="$BASE/prepared-v1/code"
INPUTS="$BASE/prepared-v1/inputs"
RECEIPT_DIR=$(mktemp -d "$BASE/cpu-prepare-XXXXXXXX")
RECEIPT="$RECEIPT_DIR/staging.json"
"$PY" "$CODE/data_v4/stage_gpu_inputs.py" --input /root/data_v4/out \
  --archive "$SHARED/datasets/admitted-v2/admitted-v2.tar.gz" 9854dd539bf284a41f09020cf96928335de1adf40367afd59fd9ae6a6c5f8471 \
  --archive "$SHARED/datasets/mlaad-admitted-v1/mlaad-admitted-v1.tar.gz" 4f090b462ef76b422afe5d27608b41fb38e0b56f5ada83bae8a14aeae924ca1d \
  --receipt "$RECEIPT"
"$PY" "$CODE/data_v4/gpu_experiments.py" preflight \
  --feature-manifest "$INPUTS/feature_union.csv" --e00-manifest "$INPUTS/E00.csv" --e04-manifest "$INPUTS/E04.csv" \
  --config "$INPUTS/seed41.json" --input /root/data_v4/out --assets "$BASE/assets" \
  --asset-hashes "$INPUTS/asset_hashes.json" --cache "$BASE/cache" --output "$BASE/runs"
REPORTS="$SHARED/gpu-preparation-v1/cpu-reports"
mkdir -p "$REPORTS"
cp "$BASE/runs/preflight.json" "$REPORTS/preflight.json"
cp "$RECEIPT" "$REPORTS/$(basename "$RECEIPT_DIR")-staging.json"
printf '%s\n' 'CPU preparation finished. GPU feature extraction and training were not started.'
