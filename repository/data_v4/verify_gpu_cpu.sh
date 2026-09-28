#!/usr/bin/env bash
# Verify an already-staged bundle and audio bank; never stage data or start a GPU.
# Usage: bash verify_gpu_cpu.sh PREPARED INPUT ASSETS FRESH_ATTEMPT SHA256SUMS_SHA256
set -euo pipefail
test "$#" -eq 5
export CUDA_VISIBLE_DEVICES='' HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=3 MKL_NUM_THREADS=3
PY=/root/envs/data/bin/python
test -x "$PY"
"$PY" - "$@" <<'PY'
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time

started = time.monotonic()
def step(name, **values):
    print(json.dumps(dict(step=name, utc=datetime.now(timezone.utc).isoformat(),
                          elapsed_seconds=round(time.monotonic() - started, 3), **values)), flush=True)

def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()

def unique_keys(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError('Duplicate prepared manifest key')
        result[key] = value
    return result

prepared_arg, input_arg, assets_arg, attempt_arg, expected = sys.argv[1:]
step('integrity_started')
roots = []
for name in (prepared_arg, input_arg, assets_arg):
    path = Path(name)
    if path.is_symlink() or not path.is_dir():
        raise ValueError('Expected existing directory without root symlink: ' + name)
    roots.append(path.resolve(strict=True))
prepared, input_root, assets = roots
attempt = Path(attempt_arg)
if attempt.exists() or attempt.is_symlink():
    raise FileExistsError('A fresh attempt directory is required')
attempt = attempt.parent.resolve(strict=True) / attempt.name
if any(attempt.is_relative_to(root) or root.is_relative_to(attempt) for root in roots):
    raise ValueError('Attempt output must be separate from prepared code, input audio and assets')
manifest = prepared / 'SHA256SUMS.json'
if (not re.fullmatch('[0-9a-f]{64}', expected) or manifest.is_symlink()
        or manifest.stat().st_size > 10_000_000 or digest(manifest) != expected):
    raise ValueError('Prepared SHA256SUMS.json identity mismatch')
hashes = json.loads(manifest.read_text(), object_pairs_hook=unique_keys)
if not isinstance(hashes, dict) or not hashes:
    raise ValueError('Empty or invalid prepared hash manifest')
for name, value in hashes.items():
    path = Path(name)
    if (path.is_absolute() or '..' in path.parts or '\\' in name or str(path) != name
            or not isinstance(value, str) or not re.fullmatch('[0-9a-f]{64}', value)):
        raise ValueError('Unsafe prepared manifest entry')
existing = list(prepared.rglob('*'))
if (any(p.is_symlink() or not (p.is_file() or p.is_dir()) for p in existing)
        or {p.relative_to(prepared).as_posix() for p in existing if p.is_file()} != set(hashes) | {'SHA256SUMS.json'}):
    raise ValueError('Prepared bundle contains unexpected files or links')
for name, expected_file in hashes.items():
    if digest(prepared / name) != expected_file:
        raise ValueError('Prepared file hash mismatch: ' + name)
step('integrity_passed', verified_files=len(hashes), manifest_sha256=expected)
cache = prepared.parent / 'cache'
if cache.is_symlink():
    raise ValueError('Cache must not be a symbolic link')

def free_space():
    location = cache if cache.exists() else cache.parent
    free = shutil.disk_usage(location).free
    step('free_space', cache_filesystem=str(location), free_bytes=free, required_bytes=4 * 1024**3)
    if free < 4 * 1024**3:
        raise RuntimeError('Need at least 4 GiB free for GPU cache and run outputs after staging')
    return free

free_before = free_space()
attempt.mkdir(exist_ok=False)
environment = dict(python=sys.executable, python_version=sys.version, cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'),
                   hf_hub_offline=os.environ.get('HF_HUB_OFFLINE'), free_bytes_before=free_before,
                   prepared_manifest_sha256=expected)
(attempt / 'environment.json').write_text(json.dumps(environment, indent=2) + '\n')
inputs = prepared / 'inputs'
command = [sys.executable, str(prepared / 'code/data_v4/gpu_experiments.py'), 'preflight',
           '--feature-manifest', str(inputs / 'feature_union.csv'), '--e00-manifest', str(inputs / 'E00.csv'),
           '--e04-manifest', str(inputs / 'E04.csv'), '--config', str(inputs / 'seed41.json'),
           '--input', str(input_root), '--assets', str(assets), '--asset-hashes', str(inputs / 'asset_hashes.json'),
           '--cache', str(cache), '--output', str(attempt)]
step('cpu_preflight_started')
subprocess.run(command, check=True, cwd=prepared / 'code')
receipt = attempt / 'preflight.json'
result = json.loads(receipt.read_text())
if result.get('status') != 'cpu_preflight_passed' or result.get('gpu_started') is not False:
    raise ValueError('CPU preflight did not produce the required successful receipt')
free_after = free_space()
readiness = dict(status='cpu_readiness_passed', gpu_started=False, prepared_manifest_sha256=expected,
                 preflight_sha256=digest(receipt), free_bytes_after=free_after,
                 elapsed_seconds=time.monotonic() - started,
                 scope='Existing staged inputs, asset bytes and disk reserve; no model loading or GPU runtime validation')
(attempt / 'readiness.json').write_text(json.dumps(readiness, indent=2) + '\n')
step('cpu_readiness_passed', attempt=str(attempt), preflight_sha256=readiness['preflight_sha256'])
PY
