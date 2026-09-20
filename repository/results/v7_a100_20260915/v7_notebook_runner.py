"""Execute the A100 notebook with a wall-clock deadline and recoverable results."""

import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import zipfile


ROOT = Path(__file__).resolve().parent


def state(stage: str, details: str = '') -> None:
    path = ROOT / 'run_state.json'
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(dict(stage=stage, details=details, utc=time.time()), indent=2))
    temporary.replace(path)


def execute_notebook() -> None:
    import nbformat
    from nbclient import NotebookClient

    notebook = nbformat.read(ROOT / 'V7_A100_DataAblation.ipynb', as_version=4)
    target = ROOT / 'V7_A100_Executed.ipynb'
    def save_cell(**kwargs):
        nbformat.write(notebook, target)
    client = NotebookClient(notebook, timeout=None, kernel_name='v7-a100',
                            resources={'metadata': {'path': str(ROOT)}}, on_cell_executed=save_cell)
    try:
        client.execute()
    finally:
        nbformat.write(notebook, target)


def main() -> None:
    if '--worker' in sys.argv:
        execute_notebook()
        return
    os.environ['MPLBACKEND'] = 'Agg'
    os.environ['PYTHONUNBUFFERED'] = '1'
    deadline = float(os.environ['V7_RESUMED_EPOCH']) + 3.5 * 3600
    if time.time() >= deadline:
        raise TimeoutError('Workspace compute deadline already passed')
    state('running')
    child = subprocess.Popen([sys.executable, str(Path(__file__).resolve()), '--worker'],
                             cwd=ROOT, start_new_session=True)
    try:
        code = child.wait(timeout=max(1, deadline - time.time()))
        state('complete' if code == 0 else 'error', f'notebook exit code {code}')
    except subprocess.TimeoutExpired:
        os.killpg(child.pid, signal.SIGTERM)
        try:
            child.wait(timeout=15)
        except subprocess.TimeoutExpired:
            os.killpg(child.pid, signal.SIGKILL)
            child.wait()
        state('time_limit', 'Compute stopped; download partial artifacts and Pause workspace')
    finally:
        bundle = ROOT / 'V7_A100_results.zip'
        with zipfile.ZipFile(bundle.with_suffix('.partial'), 'w', zipfile.ZIP_DEFLATED, compresslevel=1) as archive:
            paths = list((ROOT / 'results').rglob('*'))
            paths += [ROOT / name for name in ('V7_A100_Executed.ipynb', 'V7_A100_DataAblation.ipynb',
                      'run_state.json', 'input_manifest.json', 'v7_ablation.py', 'v7_features.py',
                      'v7_notebook_runner.py', 'notebook.log')]
            for path in paths:
                if path.is_file():
                    archive.write(path, path.relative_to(ROOT))
        bundle.with_suffix('.partial').replace(bundle)
        with bundle.open('rb') as stream:
            digest = hashlib.file_digest(stream, 'sha256').hexdigest()
        (ROOT / 'download_manifest.json').write_text(json.dumps(
            dict(file=bundle.name, bytes=bundle.stat().st_size, sha256=digest), indent=2))
        print('RESULTS_READY', bundle.name, digest, flush=True)


if __name__ == '__main__':
    main()
