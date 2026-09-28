# /// script
# requires-python = ">=3.11"
# dependencies = ["nbformat==5.10.4"]
# ///
# How to run: start_job.sh installs the notebook's pinned CUDA environment.
"""Execute the staged notebook in a bounded batch job, mirroring durable outputs."""
from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time

import nbformat

ROOT = Path('/root/fullft7')
REMOTE = Path(os.environ['FULLFT_REMOTE'])
STARTED = float(os.environ['FULLFT_JOB_STARTED_AT'])
NOTEBOOK = 'MERT_EAT_FullFT_H100x1_7Experiments.ipynb'


def execute() -> None:
    notebook = nbformat.read(ROOT / NOTEBOOK, as_version=4)
    namespace = {'__name__': 'fullft_notebook'}
    for index, cell in enumerate(notebook.cells):
        if cell.cell_type != 'code':
            continue
        print(f'CELL_START {index}', flush=True)
        exec(compile(cell.source, f'{NOTEBOOK}:cell{index}', 'exec', dont_inherit=True), namespace)
        if index == 2:
            namespace.update(PROJECT_ROOT=ROOT, NOTEBOOK_PATH=ROOT / NOTEBOOK,
                             OUTPUT=ROOT / 'h100x1_fullft7_run',
                             RUN_STAGE_INPUTS=True, RUN_CHECKS=True,
                             RUN_PREPARE=True, RUN_TRAIN=True,
                             SESSION_STARTED_AT=STARTED, MAX_SESSION_HOURS=15.5)
        print(f'CELL_DONE {index}', flush=True)


def mirror(seen: dict[str, tuple[int, int]]) -> None:
    """Copy completed file versions; notebook writes checkpoints by atomic replace."""
    paths = list((ROOT / 'h100x1_fullft7_run').rglob('*'))
    paths += [ROOT / NOTEBOOK, ROOT / 'notebook.log', ROOT / 'run_state.json']
    for source in paths:
        if not source.is_file() or '.partial' in source.name or source.name == 'session.lock':
            continue
        relative = source.relative_to(ROOT).as_posix()
        info = source.stat()
        signature = (info.st_mtime_ns, info.st_size)
        if seen.get(relative) == signature:
            continue
        target = REMOTE / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_name(target.name + '.uploading')
        shutil.copyfile(source, temporary)
        temporary.replace(target)
        seen[relative] = signature


def stop(child: subprocess.Popen) -> None:
    if child.poll() is not None:
        return
    os.killpg(child.pid, signal.SIGTERM)
    try:
        child.wait(timeout=20)
    except subprocess.TimeoutExpired:
        os.killpg(child.pid, signal.SIGKILL)
        child.wait()


def main() -> None:
    if '--worker' in sys.argv:
        execute()
        return
    os.chdir(ROOT)
    seen: dict[str, tuple[int, int]] = {}
    with (ROOT / 'notebook.log').open('a') as log:
        child = subprocess.Popen([sys.executable, __file__, '--worker'], stdout=log,
                                 stderr=subprocess.STDOUT, start_new_session=True)
        try:
            while child.poll() is None:
                if time.time() >= STARTED + 15.5 * 3600:
                    raise TimeoutError('Training deadline reached; preserving partial results')
                mirror(seen)
                tails = [ROOT / 'notebook.log'] + list((ROOT / 'h100x1_fullft7_run').glob('*.log'))
                for path in tails:
                    with path.open('rb') as stream:
                        stream.seek(max(0, path.stat().st_size - 2000))
                        print(path.name, stream.read().decode(errors='replace')[-1200:], flush=True)
                time.sleep(30)
            if child.returncode:
                raise subprocess.CalledProcessError(child.returncode, 'notebook')
            (ROOT / 'run_state.json').write_text(json.dumps({'status': 'complete', 'utc': time.time()}))
        finally:
            stop(child)
            mirror(seen)
            print('JOB_EXIT: artifacts persisted; batch allocation ends with this command.', flush=True)


if __name__ == '__main__':
    main()
