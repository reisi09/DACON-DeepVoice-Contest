"""Execute the H100 notebook and export even partial evidence before shutdown."""
from __future__ import annotations

import json
import shutil
import sys
import time
import traceback
from pathlib import Path

import nbformat
from nbclient import NotebookClient

from redesign_h100_run import atomic_json, sha256
from redesign_notebook_runtime import export_results


def main(config_path: str) -> int:
    config = json.loads(Path(config_path).read_text())
    root, shared = Path(config['root']), Path(config['shared'])
    output = root/'output'
    output.mkdir(parents=True, exist_ok=True)
    hard = config['pause_deadline_epoch']
    reserve = int(config.get('export_reserve_seconds', 900))
    maximum = int(config.get('max_gpu_seconds', 14400))
    if not 0 < hard-config['started_epoch'] <= maximum <= 14400 or time.time() >= hard-reserve:
        raise ValueError('Expired or invalid four-hour allocation')
    state = dict(status='running', archive_ready=False, started_unix=config['started_epoch'],
                 hard_deadline_unix=hard, updated_unix=time.time())
    atomic_json(shared/'run_state.json', state)
    notebook_name = config.get('notebook', 'DACON_Six_Experiments_H100.ipynb')
    notebook = nbformat.read(root/'source'/notebook_name, as_version=4)
    status = 'failed'
    try:
        NotebookClient(notebook, timeout=max(1,int(hard-time.time()-reserve)), kernel_name=config.get('kernel', 'six-h100'),
                       resources={'metadata': {'path': str(root/'source')}}).execute()
        status = 'exported'
    except Exception as error:  # Notebook boundary: preserve executed cells and traceback on failure.
        state['error'] = f'{type(error).__name__}: {error}'
        (output/'failure.txt').write_text(traceback.format_exc())
    finally:
        nbformat.write(notebook, output/(Path(notebook_name).stem+'_executed.ipynb'))
        for path in (root/'run.log', root/'gpu_telemetry.live.csv', root/'config.json'):
            if path.exists():
                shutil.copy2(path, output/path.name)
        # Keep per-member ZIP work off the object-storage mount; publish once afterward.
        exported = export_results((output,), root/'results.zip')
        shutil.copyfile(root/'results.zip', shared/'results.zip')
        if sha256(shared/'results.zip') != exported['archive_sha256']:
            raise ValueError('Published result archive differs from its verified local copy')
        receipt = dict(status='exported' if status=='exported' else 'partial_exported', archive_ready=True,
                       archive_verified=exported['verified'], archive_sha256=exported['archive_sha256'],
                       archive_bytes=exported['bytes'], export_member_count=exported['member_count'],
                       exported_unix=time.time())
        atomic_json(shared/'export_receipt.json', receipt)
        (shared/'results.zip.sha256').write_text(exported['archive_sha256']+'  results.zip\n')
        state.update(receipt, status=status, updated_unix=time.time())
        atomic_json(shared/'run_state.json', state)
    return 0 if status=='exported' else 1


if __name__ == '__main__':
    raise SystemExit(main(sys.argv[1]))
