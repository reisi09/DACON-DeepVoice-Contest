# /// script
# requires-python = ">=3.11"
# dependencies = []
# ///
# Run: .venv/bin/python expanded_h100_ops.py prepare
"""Reuse the existing H100, with its former startup command disabled before launch."""
from __future__ import annotations

import json
from pathlib import Path
import tempfile
import time

import localization_a100_ops as ops

ORIGINAL_PREPARE = ops.prepare
ORIGINAL_LAUNCH = ops.launch
INIT_PREFIX = 'redesign-full-20260917/'
INIT_NAME = 'redesign_full_refit_run.sh'
IDLE = b'#!/usr/bin/env bash\n# Previous experiment archived; launch is explicit from the verified run controller.\nexit 0\n'
STARTUP = b'''#!/usr/bin/env bash
set -euo pipefail
exec > /shared/expanded-h100-20260918/bootstrap.log 2>&1
/root/fullft8/.venv/bin/python - <<'PY'
import json,os,time
from pathlib import Path
r=json.loads(Path('/shared/expanded-h100-20260918/workspace_run.json').read_text())
assert r['workspace']=='wsp-9683432r72m7'
assert 0 < r['pause_deadline_epoch']-r['started_epoch'] <= 14400
assert time.time() < r['pause_deadline_epoch']-600
os.execv('/bin/bash',['bash','/shared/expanded-h100-20260918/bootstrap.sh',str(r['started_epoch']),str(r['pause_deadline_epoch'])])
PY
'''


def detail(*, stopped: bool = False) -> ops.Workspace:
    raw = json.loads(ops.shared._call(['workspace', 'show', ops.WORKSPACE, '-o', 'json']))
    item, gpu = raw['workspace'], raw['billingInfo']
    if (item['slug'] != ops.WORKSPACE or item['resourceSpecSlug'] != 'resourcespec-ch100x1'
            or gpu['resourceSpecAcceleratorName'] != 'H100 SXM' or gpu['resourceSpecAcceleratorCount'] != 1
            or item['initScript'] != 'bash /shared/redesign-full-20260917/redesign_full_refit_run.sh'
            or not any(v['mountPath'] == '/shared' and v['objectVolumeSlug'] == ops.VOLUME for v in raw['objectVolumes'])):
        raise RuntimeError('Unexpected H100 identity, startup command, or storage mount')
    if stopped and (item['state'], item['phase']) != ('stopped', 'standby'):
        raise RuntimeError('H100 must remain stopped until uploads are verified')
    endpoints = [v['httpUrl'] for v in raw.get('endpoints', []) if v.get('name') == 'jupyter']
    return ops.Workspace(item['state'], item['phase'], endpoints[0] if len(endpoints) == 1 else '')


def read_init(destination: Path) -> bytes:
    listing = json.loads(ops.shared._call(['volume', 'ls', ops.VOLUME, '--prefix', INIT_PREFIX, '-o', 'json']))
    command = ['volume', 'download', ops.VOLUME, str(destination), '--remote-prefix', INIT_PREFIX, '--overwrite']
    for item in listing:
        if item['name'] != INIT_NAME:
            command += ['--exclude', item['name']]
    ops.shared._call(command, timeout=300)
    files = [p for p in destination.rglob('*') if p.is_file()]
    if len(files) != 1 or files[0].name != INIT_NAME:
        raise RuntimeError('Unexpected startup-script readback')
    return files[0].read_bytes()


def prepare() -> None:
    ORIGINAL_PREPARE()
    backup = ops.ROOT / '_workspace/expanded_h100_20260918/previous_bootstrap.sh'
    with tempfile.TemporaryDirectory(dir=backup.parent) as temporary:
        content = read_init(Path(temporary))
    if not backup.exists():
        backup.write_bytes(content)
    elif content not in (backup.read_bytes(), IDLE):
        raise RuntimeError('Former startup script changed since its backup')
    with tempfile.TemporaryDirectory(dir=backup.parent) as temporary:
        idle = Path(temporary) / INIT_NAME
        idle.write_bytes(IDLE)
        ops.shared._call(['volume', 'upload', ops.VOLUME, str(idle), '--remote-prefix', INIT_PREFIX, '--overwrite'])
    with tempfile.TemporaryDirectory(dir=backup.parent) as temporary:
        if read_init(Path(temporary)) != IDLE:
            raise RuntimeError('Failed to disable former automatic experiment')
    detail(stopped=True)
    (ops.OUTPUT / 'startup_guard.json').write_text(json.dumps({'previous_bootstrap_sha256': ops.shared._sha256(backup),
          'disabled_old_auto_run': True, 'readback_verified': True, 'gpu_started': False, 'checked_at': time.time()}, indent=2))


def launch(*, release: bool = False, resume: bool = False) -> None:
    receipt = json.loads((ops.OUTPUT / 'startup_guard.json').read_text())
    if receipt.get('readback_verified') is not True:
        raise RuntimeError('Old experiment startup guard is unverified')
    with tempfile.TemporaryDirectory(dir=ops.OUTPUT) as temporary:
        if read_init(Path(temporary)) not in (IDLE, STARTUP):
            raise RuntimeError('Unexpected workspace startup content')
    detail(stopped=True)
    with tempfile.TemporaryDirectory(dir=ops.OUTPUT) as temporary:
        path = Path(temporary) / INIT_NAME
        path.write_bytes(STARTUP)
        ops.shared._call(['volume', 'upload', ops.VOLUME, str(path), '--remote-prefix', INIT_PREFIX, '--overwrite'])
    with tempfile.TemporaryDirectory(dir=ops.OUTPUT) as temporary:
        if read_init(Path(temporary)) != STARTUP:
            raise RuntimeError('New native startup script failed readback')
    ORIGINAL_LAUNCH(release=release, resume=resume)


def configure_profile() -> None:
    ops.WORKSPACE = 'wsp-9683432r72m7'
    ops.PREFIX = 'expanded-h100-20260918/'
    ops.UPLOAD = ops.ROOT / '_workspace/expanded_h100_20260918/upload'
    ops.OUTPUT = ops.ROOT / 'results/expanded_h100_20260918'
    ops.ENTRYPOINT = Path(__file__).resolve()
    ops.OBJECT_MOUNT = '/shared'
    ops.ENDPOINT_SCHEME = 'http'
    ops.ENDPOINT_HOST = 'capella.cloud.vessl.ai'
    ops.DISPATCH_VIA_STARTUP = True
    ops.workspace_detail = detail
    ops.prepare = prepare
    ops.launch = launch


if __name__ == '__main__':
    configure_profile()
    ops.main()
