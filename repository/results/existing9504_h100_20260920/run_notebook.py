# /// script
# requires-python = ">=3.10"
# ///
# Run: python3 -u /shared/existing-only-9504-h100-20260920/run_notebook.py
"""Run the reviewed notebook pipeline through the workspace's native startup hook."""
import hashlib
import json
import os
import time
import uuid
from pathlib import Path

DEPLOYMENT = Path('/shared/existing-only-9504-h100-20260920')

def report(value):
    path = DEPLOYMENT / 'native_launch_status.json'
    with path.open('w') as stream:
        json.dump({**value, 'updated_epoch': time.time()}, stream, indent=2)
        stream.flush()
        os.fsync(stream.fileno())

report({'status': 'storage_probe', 'pid': os.getpid()})
probes = {}
for root in ['/root', '/shared']:
    target = Path(root) / ('.existing9504_probe_' + uuid.uuid4().hex)
    data = os.urandom(4 * 1024 * 1024)
    began = time.monotonic()
    with target.open('wb') as stream:
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())
    if hashlib.sha256(target.read_bytes()).digest() != hashlib.sha256(data).digest():
        raise RuntimeError('Storage probe readback mismatch: ' + root)
    target.unlink()
    probes[root] = {'passed': True, 'bytes': len(data), 'seconds': time.monotonic() - began}
report({'status': 'bootstrap', 'storage_probe': probes})
notebook = DEPLOYMENT / 'DACON_Existing9504_H100.ipynb'
exec(''.join(json.loads(notebook.read_text())['cells'][2]['source']))
if pipe.config['workspace'] != 'wsp-9683432r72m7' or pipe.config['expected_gpu'] != 'H100':
    raise RuntimeError('Notebook configuration does not target the authorized H100')
from dacon9504a100.common import atomic_json
atomic_json(pipe.job / 'status/storage_probe.json', probes)
for stage in ('preflight', 'stage_inputs', 'environment', 'pilot', 'features', 'train', 'test_eval', 'parity', 'handoff'):
    report({'status': 'running', 'stage': stage, 'storage_probe': probes})
    pipe.run(stage)
report({'status': 'complete', 'storage_probe': probes})
