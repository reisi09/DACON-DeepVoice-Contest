#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = ["nbformat>=5.10,<6"]
# ///
"""Build the bounded, source-embedded V4 Colab experiment notebook."""

from __future__ import annotations

import argparse
import json
import textwrap
from pathlib import Path
from typing import Final

import nbformat as nb

ROOT: Final = Path(__file__).resolve().parent; NOTEBOOK: Final = ROOT / "DACON_236749_trainable_v4.ipynb"; INPUTS: Final = ROOT / "v4_validation" / "drive_inputs.json"
SOURCES: Final = ("train_v4.py", "v4_training.py", "v4_training_refit.py", "v4_training_trials.py", "v4_training_data.py", "v4_training_io.py", "v4_batches.py", "v4_model.py", "v4_model_components.py", "v4_data.py", "v4_evaluation.py", "v4_selection.py", "graph_features.py", "graph_fusion.py", "graph_scoring.py", "v2_scoring.py", "v2_data.py", "recovery_sonics.py", "ast_multitask.py")
RECHECK_NOTEBOOK: Final = ROOT / "DACON_236749_v4_matched_recheck.ipynb"; RECHECK_SOURCES: Final = ("recheck_v4_matched.py", "v4_training_trials.py", "v4_training.py", "v4_training_data.py", "v4_batches.py", "v4_data.py", "v4_model.py", "v4_model_components.py", "v4_evaluation.py", "graph_scoring.py", "v2_scoring.py", "recovery_sonics.py", "ast_multitask.py", "v2_data.py")


def code(source: str) -> nb.NotebookNode:
    source = textwrap.dedent(source).strip(); compile(source, "v4_notebook_cell", "exec")
    return nb.v4.new_code_cell(source)


def embedded_source(name: str) -> nb.NotebookNode:
    source = (ROOT / name).read_text(encoding="utf-8"); compile(source, name, "exec")
    return nb.v4.new_code_cell(f"%%writefile /content/dacon_trainable_v4/{name}\n{source}")


def matched_recheck(archives: list[dict], allocation_start: float) -> None:
    inputs = [archive for archive in archives if archive['name'] in {'v4_train.zip', 'v4_dev.zip', 'v4_metadata.zip', 'recovery_sonics_assets.zip'}]
    if len(inputs) != 4:
        raise ValueError('Matched recheck requires only train/dev/metadata/SONICS archive receipts')
    cells = [nb.v4.new_markdown_cell("# DACON 236749 V4 matched recheck\n\n## tl;dr\n\n**Initial state: not run.** This is a distinct, receipt-bound P1/P3 causal recheck. It never opens sealed data, selects a model, refits, or submits."), nb.v4.new_markdown_cell("## Context & Methods\n\nThe recheck uses the original allocation clock, fresh matched P1/P3 initialization and RNG receipts, ten epochs per arm, and the audited G0 development predictions. It is not a continuation of the earlier P3 result."), nb.v4.new_markdown_cell("## Data\n\nOnly train, development, metadata, and SONICS input archives are staged after part and archive receipt verification. AST assets, sealed data, the official baseline, and `open.zip` are not accessed."), code(f'''
        import hashlib, importlib.metadata, json, os, shutil, subprocess, sys, threading, time, traceback, zipfile
        from pathlib import Path
        from google.colab import drive, runtime

        ALLOCATION_START = {allocation_start!r}; SAFE_DEADLINE = ALLOCATION_START + 7080; SESSION_START = time.time()
        if SAFE_DEADLINE <= SESSION_START: raise TimeoutError('Matched recheck original safe deadline already elapsed')
        def watchdog_stop():
            try:
                if 'RUN' in globals() and RUN.exists(): budget_event('watchdog_release_requested')
            finally: runtime.unassign()
        watchdog = threading.Timer(max(1, SAFE_DEADLINE - time.time()), watchdog_stop); watchdog.daemon = True; watchdog.start()
        drive.mount('/content/drive'); DRIVE = Path('/content/drive/MyDrive/DACON_236749'); WORK = Path('/content/dacon_trainable_v4'); WORK.mkdir(exist_ok=True)
        RUN = DRIVE / 'runs_trainable_v4_matched_recheck' / time.strftime('%Y%m%d_%H%M%S', time.gmtime(SESSION_START)); RUN.mkdir(parents=True, exist_ok=False)
        def write_json(path, value):
            temporary = path.with_suffix(path.suffix + '.tmp'); temporary.write_text(json.dumps(value, indent=2, sort_keys=True), encoding='utf-8'); temporary.replace(path)
        def sha256(path):
            with path.open('rb') as stream: return hashlib.file_digest(stream, 'sha256').hexdigest()
        def budget_event(event, **extra):
            with (RUN / 'budget_events.jsonl').open('a', encoding='utf-8') as stream: stream.write(json.dumps({{'event': event, 'unix': time.time(), 'elapsed_seconds': time.time() - SESSION_START, **extra}}) + '\\n')
        write_json(RUN / 'budget.json', {{'start_unix': SESSION_START, 'allocation_start_unix': ALLOCATION_START, 'safe_deadline_unix': SAFE_DEADLINE, 'purpose': 'fresh_matched_P1_P3_only'}}); budget_event('notebook_started'); print('RESULTS_DIRECTORY', RUN, flush=True)
    ''')]
    cells.extend(embedded_source(name) for name in RECHECK_SOURCES)
    cells.extend([code('INPUT_ARCHIVES = ' + repr(inputs)), nb.v4.new_markdown_cell("## Results\n\nNo result is embedded initially. The runner writes `remedy_contract.json`, both trial receipts, and `matched_comparison.json` only after actual execution."), code('''
        def safe_extract(archive, destination):
            with zipfile.ZipFile(archive) as packed:
                if any(Path(member.filename).is_absolute() or '..' in Path(member.filename).parts for member in packed.infolist()): raise ValueError(f'Unsafe archive member in {archive.name}')
                packed.extractall(destination)
        def capture_notebook():
            try:
                from google.colab import _message
                payload = _message.blocking_request('get_ipynb', timeout_sec=30)['ipynb']; payload = json.loads(payload) if isinstance(payload, str) else payload
                destination = RUN / 'executed.ipynb'; destination.write_text(json.dumps(payload), encoding='utf-8'); write_json(RUN / 'executed_notebook_capture.json', {'status': 'captured', 'sha256': sha256(destination), 'current_cell_may_be_incomplete': True})
            except Exception:
                (RUN / 'executed_notebook_capture_error.txt').write_text(traceback.format_exc(), encoding='utf-8')
        process = None
        try:
            (RUN / 'code').mkdir()
            for source in WORK.glob('*.py'): shutil.copy2(source, RUN / 'code' / source.name)
            write_json(RUN / 'source_hashes.json', {path.name: sha256(path) for path in sorted((RUN / 'code').glob('*.py'))}); write_json(RUN / 'input_archives.json', INPUT_ARCHIVES)
            packages = ['transformers==4.57.6', 'librosa==0.10.2.post1', 'diffq==0.2.4', 'einops==0.8.1', 'timm==1.0.15', 'scikit-learn==1.8.0', 'soundfile==0.12.1', 'imageio-ffmpeg==0.6.0', 'safetensors==0.8.0', 'fvcore==0.1.5.post20221221']
            with (RUN / 'install.log').open('w', encoding='utf-8') as log: subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', *packages], stdout=log, stderr=subprocess.STDOUT, check=True, timeout=600)
            import torch
            if not torch.cuda.is_available() or 'A100' not in torch.cuda.get_device_name(): raise RuntimeError('Requested A100 runtime is unavailable')
            requested = {package.split('==', 1)[0]: package.split('==', 1)[1] for package in packages}; write_json(RUN / 'environment.json', {'python': sys.version, 'gpu': torch.cuda.get_device_name(), 'requested_packages': requested, 'packages': {name: importlib.metadata.version(name) for name in (*requested, 'torch', 'torchaudio', 'torchvision')}})
            for archive in INPUT_ARCHIVES:
                destination = WORK / archive['name']; reusable = destination.is_file() and destination.stat().st_size == archive['bytes'] and sha256(destination) == archive['sha256']
                if not reusable:
                    with destination.open('wb') as output:
                        for part in archive['parts']:
                            source = DRIVE / part['name']
                            if source.stat().st_size != part['bytes'] or sha256(source) != part['sha256']: raise ValueError(f'Drive part receipt mismatch: {source.name}')
                            with source.open('rb') as incoming: shutil.copyfileobj(incoming, output, 8 * 1024**2)
                if destination.stat().st_size != archive['bytes'] or sha256(destination) != archive['sha256']: raise ValueError(f'Archive receipt mismatch: {destination.name}')
                safe_extract(destination, WORK); budget_event('archive_verified', archive=destination.name, reused_local_archive=reusable)
            data = WORK / 'v4_validation'
            if sha256(data / 'train.csv') != 'df5c083d7deee6f5efa195c4fce2c8b4298f2e8ca743f2c7bce7f7129877c07f' or sha256(data / 'dev.csv') != 'f78185faee0578a311eea420a1776256ccea0346d2152e77b2288db477c605d9': raise ValueError('Train/dev manifest receipt mismatch')
            if not all(path.is_file() for path in (data / 'train.csv', data / 'dev.csv', WORK / 'sonics_assets/config.json', WORK / 'sonics_assets/pytorch_model.bin')): raise FileNotFoundError('Recheck inputs are incomplete')
            g0_root = DRIVE / 'runs_trainable_v4' / '20260913_074847' / 'dev'; g0_matches = [path for path in g0_root.rglob('*.csv') if sha256(path) == 'bd1c27227829f36e9ae6191db7600920bf6649810f0896a791730b10555f0854']
            if len(g0_matches) != 1: raise ValueError(f'Expected one pinned raw G0 CSV, found {{len(g0_matches)}}')
            g0 = RUN / 'G0_predictions.csv'; shutil.copy2(g0_matches[0], g0)
            if sha256(g0) != 'bd1c27227829f36e9ae6191db7600920bf6649810f0896a791730b10555f0854': raise ValueError('Copied G0 receipt mismatch')
            command = [sys.executable, '-u', 'recheck_v4_matched.py', '--data', str(data), '--sonics', str(WORK / 'sonics_assets'), '--g0', str(g0), '--g0-sha256', 'bd1c27227829f36e9ae6191db7600920bf6649810f0896a791730b10555f0854', '--output', str(RUN / 'matched'), '--allocation-start', '1789283305', '--deadline', '1789290385']; budget_event('runner_started', command=command)
            with (RUN / 'experiment.log').open('w', buffering=1, encoding='utf-8') as log:
                process = subprocess.Popen(command, cwd=WORK, stdout=log, stderr=subprocess.STDOUT, env=dict(os.environ, HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1'))
                while process.poll() is None:
                    if time.time() >= SAFE_DEADLINE - 60: budget_event('runner_terminated_for_safe_shutdown'); process.terminate(); process.wait(timeout=30); raise TimeoutError('Matched recheck reached the original safe allocation boundary')
                    time.sleep(15)
            if process.returncode: raise RuntimeError(f'Matched recheck exited with {{process.returncode}}')
            shutil.copy2(RUN / 'matched' / 'matched_comparison.json', RUN / 'matched_comparison.json'); budget_event('runner_finished'); print('MATCHED_RECHECK_SAVED', RUN, flush=True)
        except Exception:
            (RUN / 'failure.txt').write_text(traceback.format_exc(), encoding='utf-8'); budget_event('notebook_failure'); print(traceback.format_exc(), flush=True); raise
        finally:
            capture_notebook(); receipt = {name: {'bytes': path.stat().st_size, 'sha256': sha256(path)} for name in ('experiment.log', 'failure.txt', 'matched_comparison.json') if (path := RUN / name).is_file()}; write_json(RUN / 'drive_receipt.json', receipt); write_json(RUN / 'release_requested.json', {'unix': time.time()}); watchdog.cancel(); runtime.unassign()
    '''), nb.v4.new_markdown_cell("## Takeaways\n\nUse `matched_comparison.json` and both trial receipts to decide whether the fresh causal comparison completed. This notebook does not make a selection or open sealed data.")])
    notebook = nb.v4.new_notebook(cells=cells, metadata={'kernelspec': {'name': 'python3', 'display_name': 'Python 3'}, 'language_info': {'name': 'python'}, 'accelerator': 'GPU', 'colab': {'name': RECHECK_NOTEBOOK.name, 'gpuType': 'A100', 'provenance': []}}); nb.validate(notebook); nb.write(notebook, RECHECK_NOTEBOOK)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--allocation-start', type=float, help='Pinned original A100 allocation UNIX timestamp')
    parser.add_argument('--resume-run', help='Existing runs_trainable_v4 child directory to resume')
    parser.add_argument('--matched-recheck', action='store_true', help='Build the distinct sealed-free P1/P3 recheck notebook')
    args = parser.parse_args()
    if args.allocation_start is not None and args.allocation_start <= 0:
        parser.error('--allocation-start must be a positive UNIX timestamp')
    if args.resume_run and Path(args.resume_run).name != args.resume_run:
        parser.error('--resume-run must be a runs_trainable_v4 child directory name')
    archives = json.loads(INPUTS.read_text(encoding="utf-8"))
    if args.matched_recheck:
        if args.allocation_start != 1789283305:
            parser.error('--matched-recheck requires --allocation-start 1789283305')
        if args.resume_run:
            parser.error('--matched-recheck creates a new run and cannot use --resume-run')
        matched_recheck(archives, args.allocation_start)
        return
    cells = [
        nb.v4.new_markdown_cell(
            "# DACON 236749 V4 trainable recovery\n\n"
            "## tl;dr\n\n"
            "**Initial state: not run.** This notebook performs the bounded V4 training experiment only "
            "after its Drive inputs pass SHA-256 verification. It records an actual result or a stated "
            "censoring/failure reason for six LR probes and the five G0/P1/P2/P3/P4 pipeline rows. "
            "It never submits to DACON."
        ),
        nb.v4.new_markdown_cell(
            "## Context & Methods\n\n"
            "The music branch is the only newly trainable component. The runner owns role separation, "
            "full-encoder gradient/update checks, matched controls, recipe locking, and the rule that a "
            "failed or incomplete candidate retains G0. The 7,200-second A100 allocation includes setup; "
            "the runner receives a 7,080-second safe deadline and the notebook watchdog releases the runtime."
        ),
        nb.v4.new_markdown_cell(
            "## Data\n\n"
            "Every archive is reconstructed from Drive parts only after each part and its assembled archive "
            "match the embedded receipt. Archives are rejected if a member could escape the workspace. "
            "`v4_validation` supplies train, dev, and sealed manifests; only the runner may decide whether "
            "the sealed panel is opened."
        ),
        code('''
            import csv, hashlib, importlib.metadata, json, os, shutil, subprocess, sys, threading, time, traceback, zipfile
            from pathlib import Path
            from google.colab import drive, runtime

            PINNED_ALLOCATION_START = __PINNED_ALLOCATION_START__
            PINNED_RESUME_RUN = __PINNED_RESUME_RUN__
            SESSION_START = time.time()
            ALLOCATION_START = float(PINNED_ALLOCATION_START if PINNED_ALLOCATION_START is not None else os.environ.get('V4_ALLOCATION_STARTED_UNIX', SESSION_START))
            SAFE_DEADLINE = ALLOCATION_START + 7080
            if SAFE_DEADLINE <= SESSION_START:
                raise TimeoutError('V4 original allocation safe deadline already elapsed')
            def watchdog_stop():
                try:
                    if 'RUN' in globals() and RUN.exists():
                        budget_event('watchdog_release_requested')
                finally:
                    runtime.unassign()

            watchdog = threading.Timer(max(1, SAFE_DEADLINE - time.time()), watchdog_stop)
            watchdog.daemon = True
            watchdog.start()
            drive.mount('/content/drive')
            DRIVE = Path('/content/drive/MyDrive/DACON_236749')
            WORK = Path('/content/dacon_trainable_v4')
            WORK.mkdir(exist_ok=True)
            resume_run = PINNED_RESUME_RUN or os.environ.get('V4_RESUME_RUN')
            if resume_run and Path(resume_run).name != resume_run:
                raise ValueError('V4_RESUME_RUN must be a runs_trainable_v4 child directory name')
            RUN = DRIVE / 'runs_trainable_v4' / (resume_run or time.strftime('%Y%m%d_%H%M%S', time.gmtime(SESSION_START)))
            if resume_run and not RUN.is_dir():
                raise FileNotFoundError(f'Resume run does not exist: {RUN}')
            RUN.mkdir(parents=True, exist_ok=bool(resume_run))

            def write_json(path, value):
                temporary = path.with_suffix(path.suffix + '.tmp')
                temporary.write_text(json.dumps(value, indent=2, sort_keys=True), encoding='utf-8')
                temporary.replace(path)

            def sha256(path):
                with path.open('rb') as stream:
                    return hashlib.file_digest(stream, 'sha256').hexdigest()

            def budget_event(event, **extra):
                with (RUN / 'budget_events.jsonl').open('a', encoding='utf-8') as stream:
                    stream.write(json.dumps({'event': event, 'unix': time.time(), 'elapsed_seconds': time.time() - SESSION_START, **extra}) + '\\n')

            write_json(RUN / ('retry_budget.json' if resume_run else 'budget.json'), {
                'start_unix': SESSION_START, 'allocation_start_unix': ALLOCATION_START, 'safe_deadline_unix': SAFE_DEADLINE, 'resumed_run': bool(resume_run),
                'allocation_limit_seconds': 7200, 'safe_shutdown_seconds': 7080,
            })

            budget_event('notebook_started', resumed_run=bool(resume_run))
            print('RESULTS_DIRECTORY', RUN, flush=True)
        '''.replace('__PINNED_ALLOCATION_START__', repr(args.allocation_start)).replace('__PINNED_RESUME_RUN__', repr(args.resume_run))),
        nb.v4.new_markdown_cell(
            "### Embedded execution source\n\n"
            "These cells are the exact local sources compiled by the builder. The run receipt stores their "
            "SHA-256 values before any experiment command starts."
        ),
    ]
    cells.extend(embedded_source(name) for name in SOURCES)
    cells.extend([
        code("INPUT_ARCHIVES = " + repr(archives)),
        nb.v4.new_markdown_cell(
            "## Results\n\n"
            "No experimental output is embedded initially. During a successful run, this notebook renders a "
            "plot only from the runner-written `epoch_metrics.csv`; its Drive receipt records every displayed input."
        ),
        code('''
            def safe_extract(archive, destination):
                with zipfile.ZipFile(archive) as packed:
                    members = packed.infolist()
                    if any(Path(member.filename).is_absolute() or '..' in Path(member.filename).parts for member in members):
                        raise ValueError(f'Unsafe archive member in {archive.name}')
                    packed.extractall(destination)

            def extract_baseline_member(outer, destination):
                with zipfile.ZipFile(outer) as packed:
                    members = [member for member in packed.infolist() if not member.is_dir() and not Path(member.filename).is_absolute() and '..' not in Path(member.filename).parts and Path(member.filename).name == 'baseline_submit.zip']
                    if len(members) != 1 or members[0].file_size != 4562809110 or members[0].compress_size > 4562809110:
                        raise ValueError('Expected one bounded baseline_submit.zip member')
                    with packed.open(members[0]) as incoming, destination.open('wb') as output:
                        shutil.copyfileobj(incoming, output, 8 * 1024**2)
                if destination.stat().st_size != 4562809110 or sha256(destination) != 'cc86f143915c95fd958a0066cbf14b0db9f4c41fd213310c3650b9fe870d0c70':
                    raise ValueError('Official baseline archive receipt mismatch')

            def capture_executed_notebook():
                try:
                    from google.colab import _message
                    reply = _message.blocking_request('get_ipynb', timeout_sec=30)
                    payload = reply['ipynb']
                    if isinstance(payload, str):
                        payload = json.loads(payload)
                    destination = RUN / 'executed.ipynb'
                    destination.write_text(json.dumps(payload), encoding='utf-8')
                    write_json(RUN / 'executed_notebook_capture.json', {'status': 'captured', 'capture_stage': 'before_runtime_unassign', 'current_cell_may_be_incomplete': True, 'sha256': sha256(destination)})
                except Exception:
                    (RUN / 'executed_notebook_capture_error.txt').write_text(traceback.format_exc(), encoding='utf-8')
                    write_json(RUN / 'executed_notebook_capture.json', {'status': 'not_captured'})

            def record_artifacts():
                names = ('trials.jsonl', 'epoch_metrics.csv', 'exposures.json', 'selection.json', 'recipe_lock.json', 'refit_manifest.csv', 'final_weight_audit.json', 'weight_lock.json', 'sealed_opened.json', 'failure.txt', 'experiment.log')
                receipt = {}
                for name in names:
                    path = RUN / name
                    if path.is_file():
                        receipt[name] = {'bytes': path.stat().st_size, 'sha256': sha256(path)}
                write_json(RUN / 'drive_receipt.json', receipt)

            def plot_results():
                metrics_path = RUN / 'epoch_metrics.csv'
                if not metrics_path.is_file():
                    write_json(RUN / 'plot_status.json', {'status': 'not_created', 'reason': 'no executed epoch_metrics.csv'})
                    return
                rows = list(csv.DictReader(metrics_path.open(encoding='utf-8')))
                if not rows:
                    write_json(RUN / 'plot_status.json', {'status': 'not_created', 'reason': 'empty epoch_metrics.csv'})
                    return
                x_name = next((name for name in ('epoch', 'step') if name in rows[0]), None)
                group_name = next((name for name in ('pipeline', 'trial', 'name') if name in rows[0]), None)
                numeric = []
                for name in rows[0]:
                    try:
                        [float(row[name]) for row in rows if row.get(name, '') != '']
                    except ValueError:
                        continue
                    numeric.append(name)
                metric_name = next((name for name in ('rank_score', 'dev_score', 'proxy_score', 'loss', 'dev_loss') if name in numeric), next((name for name in numeric if name != x_name), None))
                if x_name is None or group_name is None or metric_name is None:
                    write_json(RUN / 'plot_status.json', {'status': 'not_created', 'reason': 'missing plotted metric columns', 'columns': list(rows[0])})
                    return
                import matplotlib.pyplot as plt
                grouped = {}
                for row in rows:
                    if row.get(x_name, '') and row.get(metric_name, ''):
                        grouped.setdefault(row[group_name], []).append((float(row[x_name]), float(row[metric_name])))
                if not grouped:
                    write_json(RUN / 'plot_status.json', {'status': 'not_created', 'reason': 'no numeric plotted rows'})
                    return
                figure, axis = plt.subplots(figsize=(9, 5))
                for label, values in sorted(grouped.items()):
                    values.sort()
                    axis.plot(*zip(*values), marker='o', label=label)
                axis.set(title=f'Executed V4 {metric_name}', xlabel=x_name, ylabel=metric_name)
                axis.grid(alpha=.25)
                axis.legend()
                figure.tight_layout()
                output = RUN / 'epoch_metric.png'
                figure.savefig(output, dpi=160)
                plt.show()
                write_json(RUN / 'plot_status.json', {'status': 'created', 'path': output.name, 'sha256': sha256(output)})

            process = None
            try:
                (RUN / 'code').mkdir(exist_ok=True)
                for source in WORK.glob('*.py'):
                    shutil.copy2(source, RUN / 'code' / source.name)
                write_json(RUN / 'source_hashes.json', {path.name: sha256(path) for path in sorted((RUN / 'code').glob('*.py'))})
                write_json(RUN / 'input_archives.json', INPUT_ARCHIVES)
                packages = ['demucs==4.0.1', 'panns-inference==0.1.1', 'transformers==4.57.6', 'librosa==0.10.2.post1', 'diffq==0.2.4', 'einops==0.8.1', 'timm==1.0.15', 'scikit-learn==1.8.0', 'soundfile==0.12.1', 'imageio-ffmpeg==0.6.0', 'safetensors==0.8.0', 'fvcore==0.1.5.post20221221']
                with (RUN / 'install.log').open('w', encoding='utf-8') as log:
                    subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', *packages], stdout=log,
                                   stderr=subprocess.STDOUT, check=True, timeout=600)
                import torch
                if not torch.cuda.is_available() or 'A100' not in torch.cuda.get_device_name():
                    raise RuntimeError('Requested A100 runtime is unavailable')
                requested_packages = {package.split('==', 1)[0]: package.split('==', 1)[1] for package in packages}
                version_names = (*requested_packages, 'torch', 'torchaudio', 'torchvision')
                write_json(RUN / 'environment.json', {'python': sys.version, 'gpu': torch.cuda.get_device_name(), 'requested_packages': requested_packages, 'packages': {name: importlib.metadata.version(name) for name in version_names}})
                for archive in INPUT_ARCHIVES:
                    destination = WORK / archive['name']
                    reusable = destination.is_file() and destination.stat().st_size == archive['bytes'] and sha256(destination) == archive['sha256']
                    if not reusable:
                        with destination.open('wb') as output:
                            for part in archive['parts']:
                                source = DRIVE / part['name']
                                if source.stat().st_size != part['bytes'] or sha256(source) != part['sha256']:
                                    raise ValueError(f'Drive part receipt mismatch: {source.name}')
                                with source.open('rb') as incoming:
                                    shutil.copyfileobj(incoming, output, 8 * 1024**2)
                    if destination.stat().st_size != archive['bytes'] or sha256(destination) != archive['sha256']:
                        raise ValueError(f'Archive receipt mismatch: {destination.name}')
                    safe_extract(destination, WORK)
                    budget_event('archive_verified', archive=destination.name, reused_local_archive=reusable)
                if not all(path.is_file() for path in (WORK / 'v4_validation/train.csv', WORK / 'v4_validation/dev.csv', WORK / 'v4_validation/sealed.csv', WORK / 'sonics_assets/config.json', WORK / 'sonics_assets/pytorch_model.bin', WORK / 'ast/config.json', WORK / 'ast/preprocessor_config.json', WORK / 'ast/model.safetensors')): raise FileNotFoundError('Verified archives did not produce the required V4 roots')
                original = extract_baseline_member(DRIVE / 'open.zip', WORK / 'baseline_submit.zip')
                baseline = WORK / 'baseline'; safe_extract(original, baseline)
                import demucs, diffq, panns_inference
                if not (baseline / 'script.py').is_file() or not (baseline / 'model' / 'SHA256SUMS.txt').is_file():
                    raise FileNotFoundError('Baseline script or signed model inventory is missing')
                print('CPU_PREFLIGHT_OK', importlib.metadata.version('demucs'), flush=True)
                command = [sys.executable, '-u', 'train_v4.py', '--data', str(WORK / 'v4_validation'), '--baseline', str(baseline), '--sonics', str(WORK / 'sonics_assets'), '--ast', str(WORK / 'ast'), '--output', str(RUN), '--deadline', str(SAFE_DEADLINE)]
                environment = dict(os.environ, PYTHONPATH=str(WORK / 'sonics_vendor'), HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1')
                budget_event('runner_started', command=command)
                with (RUN / 'experiment.log').open('w', buffering=1, encoding='utf-8') as log:
                    process = subprocess.Popen(command, cwd=WORK, stdout=log, stderr=subprocess.STDOUT, env=environment)
                    while process.poll() is None:
                        if time.time() >= SAFE_DEADLINE - 60:
                            budget_event('runner_terminated_for_safe_shutdown')
                            process.terminate()
                            process.wait(timeout=30)
                            raise TimeoutError('V4 runner reached the safe allocation boundary')
                        time.sleep(15)
                        lines = (RUN / 'experiment.log').read_text(encoding='utf-8', errors='replace').splitlines()
                        print(lines[-1][:1000] if lines else 'Preparing V4 experiment...', flush=True)
                if process.returncode:
                    raise RuntimeError(f'V4 runner exited with {process.returncode}')
                budget_event('runner_finished')
                plot_results()
                print('V4_SAVED', RUN, flush=True)
            except Exception:
                (RUN / 'failure.txt').write_text(traceback.format_exc(), encoding='utf-8')
                budget_event('notebook_failure')
                print(traceback.format_exc(), flush=True)
                raise
            finally:
                capture_executed_notebook()
                record_artifacts()
                write_json(RUN / 'release_requested.json', {'unix': time.time(), 'elapsed_seconds': time.time() - SESSION_START})
                watchdog.cancel()
                runtime.unassign()
        '''),
        nb.v4.new_markdown_cell(
            "## Takeaways\n\n"
            "Initial state: no V4 result, no selected candidate, and no DACON submission. Read `selection.json`, "
            "the CSVs, budget events, and the executed-notebook capture after a run before making any performance claim."
        ),
    ])
    notebook = nb.v4.new_notebook(cells=cells, metadata={
        'kernelspec': {'name': 'python3', 'display_name': 'Python 3'},
        'language_info': {'name': 'python'}, 'accelerator': 'GPU',
        'colab': {'name': NOTEBOOK.name, 'gpuType': 'A100', 'provenance': []},
    })
    nb.validate(notebook)
    nb.write(notebook, NOTEBOOK)


if __name__ == '__main__':
    main()
