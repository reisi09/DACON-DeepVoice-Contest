"""Wait for the verified upload, then execute and save the experiment notebook."""
import hashlib
import json
import os
import subprocess
import time
from datetime import datetime
from pathlib import Path

import nbformat
from nbclient import NotebookClient

ROOT = Path('/root')
BUNDLE = ROOT / 'V6_H100_RunBundle.zip'
EXPECTED_BYTES = 9247576070
EXPECTED_HASH = '55981049ccf5c15c7e85a4c6f32bf53e40bb4c0bef58033224962186e06c6ac2'
RESUMED_UTC = os.environ['V6_RESUMED_UTC']
AVAILABLE_CREDIT = float(os.environ['V6_AVAILABLE_CREDIT'])
os.environ['MPLBACKEND'] = 'Agg'
RESUMED = datetime.fromisoformat(RESUMED_UTC)
assert RESUMED.tzinfo is not None
BUDGET = min(15.0, AVAILABLE_CREDIT - 3.0)
assert BUDGET > 0
DEADLINE = RESUMED.timestamp() + min(4 * 3600, BUDGET / 1.8 * 3600)


def status(stage, **details):
    value = dict(stage=stage, utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), **details)
    tmp = ROOT / 'v6_run_state.tmp'
    tmp.write_text(json.dumps(value, indent=2))
    tmp.replace(ROOT / 'v6_run_state.json')


notebook = None
try:
    if not BUNDLE.is_file() or BUNDLE.stat().st_size != EXPECTED_BYTES:
        status('downloading_object_storage')
        download_root = ROOT / 'v6_object_download'
        subprocess.run(['vesslctl', 'volume', 'download', 'objvol-a5gxdg7er70i',
                        str(download_root), '--remote-prefix', 'v6/', '--overwrite'],
                       check=True, timeout=max(1, DEADLINE - time.time() - 900))
        candidates = list(download_root.rglob('V6_H100_RunBundle.zip'))
        assert len(candidates) == 1, 'Expected exactly one downloaded bundle'
        downloaded = candidates[0]
        assert downloaded.stat().st_size == EXPECTED_BYTES, 'Downloaded bundle size mismatch'
        with downloaded.open('rb') as stream:
            assert hashlib.file_digest(stream, 'sha256').hexdigest() == EXPECTED_HASH, 'Downloaded bundle SHA-256 mismatch'
        downloaded.replace(BUNDLE)
    while True:
        size = BUNDLE.stat().st_size if BUNDLE.exists() else 0
        if time.time() >= DEADLINE:
            raise TimeoutError('Upload/setup deadline reached; Pause the workspace')
        status('waiting_upload', uploaded_bytes=size, expected_bytes=EXPECTED_BYTES)
        if size == EXPECTED_BYTES:
            break
        if size > EXPECTED_BYTES:
            raise ValueError('Unexpected bundle size')
        time.sleep(60)
    status('checking_bundle')
    with BUNDLE.open('rb') as stream:
        digest = hashlib.file_digest(stream, 'sha256').hexdigest()
    if digest != EXPECTED_HASH:
        raise ValueError('Uploaded bundle SHA-256 mismatch')
    notebook = nbformat.read(ROOT / 'V6_L40S_Experiment.ipynb', as_version=4)
    for cell in notebook.cells:
        if cell.cell_type == 'code':
            cell.source = (cell.source.replace('min(3 * 3600,', 'min(4 * 3600,')
                           .replace('AVAILABLE_CREDIT_USD = 19.38', f'AVAILABLE_CREDIT_USD = {AVAILABLE_CREDIT}')
                           .replace('2026-09-14T16:27:43+00:00', RESUMED_UTC))
    nbformat.validate(notebook)
    status('executing_notebook', bundle_sha256=digest)
    client = NotebookClient(notebook, timeout=None, kernel_name='python3',
                            resources={'metadata': {'path': str(ROOT)}})
    client.execute()
    status('complete', results='/root/V6_L40S/v6_run', submission='/root/V6_L40S/V6_final_submission.zip')
except Exception as error:
    status('error', error=f'{type(error).__name__}: {error}')
    raise
finally:
    if notebook is not None:
        nbformat.write(notebook, ROOT / 'V6_L40S_Executed.ipynb')

