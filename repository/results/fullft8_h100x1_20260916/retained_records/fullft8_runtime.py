# /// script
# requires-python = ">=3.11"
# dependencies = ["nbformat==5.10.4"]
# ///
# Run: bash fullft8_start.sh (inside the authorized H100 workspace)
"""Execute the notebook and preserve artifacts before the external monitor pauses GPU."""
from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time
import gc
import hashlib

import nbformat

ROOT = Path('/root/fullft8')
REMOTE = Path(os.environ['FULLFT_REMOTE'])
STARTED = float(os.environ['FULLFT_JOB_STARTED_AT'])
NOTEBOOK = 'MERT_EAT_HTDemucs_H100x1_8Experiments.ipynb'


def preflight(namespace) -> None:
    """Real pretrained backward/update checks before expensive full-data runs."""
    import torch
    import torchaudio
    from torch.nn import functional as F
    manifest = json.loads((ROOT/'assets/model_manifest.json').read_text())
    for relative, expected in manifest['sha256'].items():
        with (ROOT/'assets/models'/relative).open('rb') as stream:
            if hashlib.file_digest(stream,'sha256').hexdigest() != expected:
                raise RuntimeError(f'Preflight model hash mismatch: {relative}')
    if not torch.cuda.is_available() or 'H100' not in torch.cuda.get_device_name(0):
        raise RuntimeError('Preflight requires the requested H100')
    chosen, parents = [], set()
    for row in namespace['SPLITS']['S0']['train']:
        lineage = set(row['group_id'].split('|'))
        if row['MUSIC_PRESENT']=='1' and row['MUSIC_FAKE']=='0' and not lineage & parents:
            chosen.append(row)
            parents.update(lineage)
        if len(chosen) == 16:
            break
    output = ROOT/'h100x1_fullft8_run'/'preflight'
    config = dict(namespace['TRAIN_CONFIG'],project_root=str(ROOT),output=str(output),
        model_root=str(ROOT/'assets/models'),deadline=min(STARTED+15*3600,time.time()+1800),
        separator_steps=1,separator_validation_every=1,separator_lr=1e-5,separator_seconds=5.,
        separator_code_sha256=namespace['SEPARATOR_CODE_SHA256'])
    namespace['separator_module'].adapt_separator(chosen,config)
    crop_path = sorted((output/'separator_adaptation/teacher_cache').glob('*.pt'))[0]
    stems = torch.load(crop_path,weights_only=True)
    music = torchaudio.functional.resample(stems[:3].sum(0).mean(0),44100,16000)
    bags = music[None,None].repeat(1,namespace['TRAIN_CONFIG']['train_windows'],1).cuda()
    reports = {}
    for kind in ['mert','eat']:
        model = namespace['MusicClassifier'](kind,False,ROOT/'assets/models').cuda().train()
        optimizer = torch.optim.AdamW(namespace['optimization_groups'](model,namespace['TRAIN_CONFIG']))
        first_name, first = next((name,p) for name,p in model.encoder.named_parameters() if p.requires_grad)
        before = first.detach().clone()
        with torch.autocast('cuda',dtype=torch.bfloat16):
            loss = F.cross_entropy(model(bags).float(),torch.zeros(1,dtype=torch.long,device='cuda'))
        loss.backward()
        evidence = namespace['gradient_evidence'](model)
        torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True)
        optimizer.step()
        update = float((first.detach()-before).abs().sum())
        if not update > 0:
            raise RuntimeError(f'Preflight optimizer did not update {kind}/{first_name}')
        reports[kind] = dict(loss=float(loss.detach()),first_parameter=first_name,update_l1=update,gradients=evidence)
        del model,optimizer,loss,before,first,bags
        gc.collect(); torch.cuda.empty_cache()
        bags = music[None,None].repeat(1,namespace['TRAIN_CONFIG']['train_windows'],1).cuda()
    namespace['save_json'](output/'encoder_gradients.json',reports)
    print('PREFLIGHT_PASSED: real HTDemucs, MERT, EAT backward and optimizer updates',flush=True)


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
                             OUTPUT=ROOT / 'h100x1_fullft8_run', STORAGE_ROOT=ROOT / 'inputs',
                             ASSET_ROOT=ROOT / 'assets', RUN_STAGE_INPUTS=True, RUN_CHECKS=True,
                             RUN_PREPARE=True, RUN_TRAIN=True, RUN_SEPARATOR=True,
                             SESSION_STARTED_AT=STARTED, MAX_SESSION_HOURS=15.0)
        if index == 19:
            preflight(namespace)
        print(f'CELL_DONE {index}', flush=True)
    output = ROOT / 'h100x1_fullft8_run'
    completed = sorted(str(p.relative_to(output)) for p in output.rglob('complete.json')
                       if p.parent.name[:2] in {'01','02','03','04','05','06','07','08'})
    if len(completed) != 8:
        raise RuntimeError(f'Expected eight completed classifier trials, observed {completed}')


def mirror(seen: dict[str, tuple[int, int]]) -> None:
    paths = list((ROOT / 'h100x1_fullft8_run').rglob('*'))
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
    status = 'failed'
    (ROOT / 'run_state.json').write_text(json.dumps({'status':'running','utc':time.time(),'gpu_resource_stopped':False}))
    mirror(seen)
    with (ROOT / 'notebook.log').open('a') as log:
        child = subprocess.Popen([sys.executable, __file__, '--worker'], stdout=log,
                                 stderr=subprocess.STDOUT, start_new_session=True)
        try:
            while child.poll() is None:
                if time.time() >= STARTED + 15.0 * 3600:
                    status = 'budget_stopped'
                    raise TimeoutError('Training deadline reached; preserving partial results')
                mirror(seen)
                tails = [ROOT / 'notebook.log'] + list((ROOT / 'h100x1_fullft8_run').rglob('*.log'))
                for path in tails:
                    with path.open('rb') as stream:
                        stream.seek(max(0, path.stat().st_size - 2000))
                        print(path.name, stream.read().decode(errors='replace')[-1200:], flush=True)
                time.sleep(30)
            if child.returncode:
                raise subprocess.CalledProcessError(child.returncode, 'notebook')
            status = 'complete'
        finally:
            stop(child)
            (ROOT / 'run_state.json').write_text(json.dumps({
                'status':status,'utc':time.time(),'returncode':child.returncode,
                'gpu_resource_stopped':False}))
            mirror(seen)
            (REMOTE / 'ARTIFACTS_SYNCED.json').write_text(json.dumps({
                'status':status,'utc':time.time(),'files':len(seen)}))
            print('ARTIFACTS_SYNCED: external monitor must pause workspace.', flush=True)


if __name__ == '__main__':
    main()
