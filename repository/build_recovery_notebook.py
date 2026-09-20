#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = ["nbformat>=5.10,<6"]
# ///
# How to run: uv run build_recovery_notebook.py
"""Embed the reviewed recovery experiment and verified Drive inputs in Colab."""

from __future__ import annotations

import json
import textwrap
from pathlib import Path
from typing import Final

import nbformat as nb

ROOT: Final = Path(__file__).resolve().parent
SOURCES: Final = (
    "graph_features.py", "graph_fusion.py", "graph_scoring.py", "v2_scoring.py",
    "run_experiment.py", "recovery_sonics.py", "recovery_fusion.py",
    "recovery_selection.py", "run_recovery.py",
)


def code(source: str) -> nb.NotebookNode:
    """Validate Python cells before they can reach the remote runtime."""
    source = textwrap.dedent(source).strip()
    compile(source, "recovery_cell", "exec")
    return nb.v4.new_code_cell(source)


def main() -> None:
    inputs = json.loads((ROOT / "recovery_validation/drive_inputs.json").read_text())
    cells = [nb.v4.new_markdown_cell(
        "# DACON recovery V3 — music specialist comparison\n\n"
        "The previous AST candidate failed its promotion gates and underperformed the official baseline. "
        "This run preserves PANNs presence and DF-Arena voice detection, and compares five frozen "
        "music branches. It performs no training or private-test adaptation.\n\n"
        "Development and old challenge results are reused diagnostics. A new source-disjoint "
        "challenge accepts or rejects the locked candidate; it does not choose another candidate. "
        "Unknown labels remain unknown, and partial metrics are not a DACON leaderboard score. "
        "Every input file is processed through its retained duration. A100 allocation is capped "
        "at two hours and released on completion or failure."
    )]
    cells.append(code('''
        import time, json, shutil, subprocess, sys, zipfile, hashlib, threading, traceback, os
        from pathlib import Path
        from google.colab import drive, runtime
        SESSION_START = time.time()
        DEADLINE = SESSION_START + 7080
        drive.mount('/content/drive')
        DRIVE = Path('/content/drive/MyDrive/DACON_236749')
        WORK = Path('/content/dacon_recovery_v3')
        WORK.mkdir(exist_ok=True)
        RUN = DRIVE / 'runs_recovery' / time.strftime('%Y%m%d_%H%M%S',time.gmtime(SESSION_START))
        RUN.mkdir(parents=True)
        watchdog = threading.Timer(max(1,DEADLINE-time.time()),runtime.unassign)
        watchdog.daemon = True
        watchdog.start()
        (RUN/'budget.json').write_text(json.dumps({'start_unix':SESSION_START,'deadline':DEADLINE,'allocation_limit_seconds':7200}))
        print('RESULTS_DIRECTORY',RUN,flush=True)
    '''))
    for name in SOURCES:
        source = (ROOT / name).read_text()
        compile(source, name, "exec")
        cells.append(nb.v4.new_code_cell(f"%%writefile /content/dacon_recovery_v3/{name}\n" + source))
    cells.append(code("INPUT_ARCHIVES = " + repr(inputs)))
    cells.append(code('''
        try:
            (RUN/'code').mkdir()
            for source in WORK.glob('*.py'):
                shutil.copy2(source,RUN/'code'/source.name)
            def sha(path):
                with path.open('rb') as stream:
                    return hashlib.file_digest(stream,'sha256').hexdigest()
            (RUN/'source_hashes.json').write_text(json.dumps({p.name:sha(p) for p in (RUN/'code').glob('*.py')},indent=2))
            (RUN/'input_archives.json').write_text(json.dumps(INPUT_ARCHIVES,indent=2))
            packages = ['demucs==4.0.1','panns-inference==0.1.1','transformers==4.57.6',
                'librosa==0.10.2.post1','diffq==0.2.4','einops==0.8.1','timm==1.0.15',
                'scikit-learn==1.8.0','soundfile==0.12.1','safetensors','fvcore']
            with (RUN/'install.log').open('w') as log:
                subprocess.run([sys.executable,'-m','pip','install','-q',*packages],
                    stdout=log,stderr=subprocess.STDOUT,check=True,timeout=600)
            import torch, importlib.metadata
            if not torch.cuda.is_available() or 'A100' not in torch.cuda.get_device_name():
                raise RuntimeError('This bounded comparison requires the requested A100 runtime')
            (RUN/'environment.json').write_text(json.dumps({'python':sys.version,'gpu':torch.cuda.get_device_name(),
                'packages':{k:importlib.metadata.version(k) for k in ['torch','torchaudio','torchvision','transformers','timm','numpy','scipy','librosa','scikit-learn']}}))
            for archive in INPUT_ARCHIVES:
                destination = WORK/archive['name']
                with destination.open('wb') as output:
                    for part in archive['parts']:
                        source = DRIVE/part['name']
                        if sha(source) != part['sha256']:
                            raise ValueError(f'Drive part checksum mismatch: {source.name}')
                        with source.open('rb') as incoming:
                            shutil.copyfileobj(incoming,output,8*1024**2)
                if sha(destination) != archive['sha256']:
                    raise ValueError(f'Archive checksum mismatch: {destination.name}')
                with zipfile.ZipFile(destination) as packed:
                    if any(Path(m.filename).is_absolute() or '..' in Path(m.filename).parts for m in packed.infolist()):
                        raise ValueError('Unsafe archive path')
                    packed.extractall(WORK)
                print('VERIFIED_INPUT',destination.name,flush=True)
            shutil.copy2(DRIVE/'open.zip',WORK/'open.zip')
            with zipfile.ZipFile(WORK/'open.zip') as packed:
                packed.extractall(WORK/'official')
            original = next((WORK/'official').rglob('baseline_submit.zip'))
            if sha(original) != 'cc86f143915c95fd958a0066cbf14b0db9f4c41fd213310c3650b9fe870d0c70':
                raise ValueError('Original baseline archive changed')
            BASELINE = WORK/'baseline'
            with zipfile.ZipFile(original) as packed:
                packed.extractall(BASELINE)
            public = next((WORK/'official').rglob('sample_submission.csv')).parent
            shutil.copytree(public,BASELINE/'data',dirs_exist_ok=True)
            os.chdir(WORK)
            env = dict(os.environ, PYTHONPATH=str(WORK/'sonics_vendor'), HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1')
            command = [sys.executable,'-u','run_recovery.py','--data',str(WORK/'v2_validation'),
                '--fresh',str(WORK/'recovery_validation'),'--baseline',str(BASELINE),
                '--assets',str(WORK/'sonics_assets'),'--output',str(RUN),'--deadline',str(DEADLINE)]
            with (RUN/'experiment.log').open('w',buffering=1) as log:
                process = subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT,env=env)
                while process.poll() is None:
                    if time.time() >= DEADLINE-60:
                        process.terminate()
                        process.wait(timeout=30)
                        raise TimeoutError('Recovery experiment reached allocation limit')
                    time.sleep(15)
                    lines = (RUN/'experiment.log').read_text().splitlines()
                    print(lines[-1][:800] if lines else 'Loading frozen models...',flush=True)
                if process.returncode:
                    raise RuntimeError((RUN/'experiment.log').read_text()[-12000:])
            print('RESULT',json.loads((RUN/'finished.json').read_text()))
            print('DRIVE_RESULTS',RUN)
        except Exception:
            (RUN/'failure.txt').write_text(traceback.format_exc())
            raise
        finally:
            (RUN/'release_requested.json').write_text(json.dumps({'unix':time.time(),'elapsed_seconds':time.time()-SESSION_START}))
            watchdog.cancel()
            runtime.unassign()
    '''))
    notebook = nb.v4.new_notebook(cells=cells, metadata={
        "kernelspec": {"name": "python3", "display_name": "Python 3"},
        "language_info": {"name": "python"}, "accelerator": "GPU",
        "colab": {"name": "DACON_236749_recovery_v3.ipynb", "gpuType": "A100", "provenance": []},
    })
    nb.validate(notebook)
    nb.write(notebook, ROOT / "DACON_236749_recovery_v3.ipynb")


if __name__ == "__main__":
    main()
