"""Bounded A100 pipeline; invoke inside an explicitly started workspace."""
from __future__ import annotations
import argparse
import contextlib
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from panns_adaptation_data import identity, sha256, write_json

CHILD_GRACE_SECONDS = 45.
CHILD_REAP_SECONDS = 1.


def fixed_preflight_config(config):
    panel = set(config['panel_ids'])
    rows = [row for row in map(json.loads, Path(config['rows']).read_text().splitlines())
            if row['ID'] in panel and row['VOICE_PRESENT'] == row['MUSIC_PRESENT'] == 1]
    selected = []
    for role in ('development', 'selection'):
        row = min((row for row in rows if row['role'] == role),
                  key=lambda row: (row['verified_audio']['seconds'], row['ID']), default=None)
        if row is None:
            raise ValueError('Fixed Engine preflight requires a mixed-presence '+role+' row in the locked panel')
        selected.append(row['ID'])
    output = Path(config['output'])/'fixed_preflight'
    probe = dict(config, panel_ids=selected, output=str(output), fixed_predictions=str(output/'fixed_predictions.json'))
    probe.pop('identity')
    probe['identity'] = identity(probe)
    return probe


def training_phase_binding(config):
    from panns_cloud_run import training_identity
    from panns_training_budget import load_budget
    from panns_training_state import validate_completion
    output = Path(config['output'])
    budget, budget_sha = load_budget(config, 'cuda')
    conditions = {}
    for arm in config['arms']:
        for seed in config['seeds']:
            path = output/f'train_{arm}_{seed}'/'complete.json'
            done = json.loads(path.read_text())
            validate_completion(done, training_identity(config, arm, seed, 'cuda'),
                                sha256(path.with_name('checkpoint.pt')), budget)
            conditions[str(path.relative_to(output))] = sha256(path)
    return dict(config_identity=config['identity'], training_budget_sha256=budget_sha, conditions=conditions)


def parity(config, bundle):
    import torch
    import numpy as np
    from panns_adaptation import presence_logits
    from panns_adaptation_cache import readout_assets
    from panns_adaptation_data import load_cache
    from panns_adaptation_run import model_for_arm, optimization_step
    from panns_cloud_run import setup
    setup('cuda')
    reference=json.loads(Path(config['reference']).read_text())
    package=Path(reference['package'])
    heads,labels,groups=readout_assets(package)
    gold=json.loads((bundle/'gold/index.json').read_text())
    actual=json.loads((Path(config['output'])/'cache/index.json').read_text())
    by_id={e['ID']:e for e in actual['entries']}
    tag_error=presence_error=0.
    for entry in gold['entries']:
        entry['path']=str(bundle/'gold'/Path(entry['path']).name)
        cpu=load_cache(entry); gpu=load_cache(by_id[entry['ID']])
        if not np.isfinite(cpu['tags']).all() or not np.isfinite(gpu['tags']).all():
            raise ValueError('Nonfinite CPU/CUDA tags')
        tag_error=max(tag_error,float(np.abs(cpu['tags']-gpu['tags']).max()))
        old=presence_logits(torch.from_numpy(cpu['tags']),entry['stems'],heads,labels,groups).sigmoid()
        new=presence_logits(torch.from_numpy(gpu['tags']),by_id[entry['ID']]['stems'],heads,labels,groups).sigmoid()
        if not torch.isfinite(old).all() or not torch.isfinite(new).all():
            raise ValueError('Nonfinite CPU/CUDA presence probabilities')
        presence_error=max(presence_error,float((old-new).abs().max()))
    if tag_error>1e-5 or presence_error>1e-5:
        raise ValueError(f'CPU/CUDA preprocessing parity failed: tags={tag_error}, presence={presence_error}')
    entries=sorted(actual['entries'],key=lambda e:e['seconds'],reverse=True)[:4]
    gradients=[]
    for device in ('cpu','cuda'):
        model=model_for_arm(package,'R3',41,device)
        optimizer=torch.optim.AdamW(model.optimizer_groups(),weight_decay=1e-4,foreach=False)
        _,grad=optimization_step(model,optimizer,entries,heads,labels,groups,[1.,1.],device,4,True)
        gradients.append(grad)
        del model,optimizer
    relative=max(float((gradients[0][k]-gradients[1][k]).double().norm()/gradients[0][k].double().norm().clamp_min(1e-12)) for k in gradients[0])
    if not math.isfinite(relative) or relative>1e-3:
        raise ValueError('CPU/CUDA gradient parity failed')
    write_json(Path(config['output'])/'cuda_parity.json',dict(tag_max_abs=tag_error,presence_max_abs=presence_error,
        gradient_relative_l2=relative,config_identity=config['identity']))


def run_command(command, deadline, *, deadline_epoch=None):
    stopped = []
    previous = {sig: signal.signal(sig, lambda *_: stopped.append(True))
                for sig in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP)}
    process = None
    def remaining():
        return min(deadline-time.monotonic(), deadline_epoch-time.time() if deadline_epoch is not None else math.inf)
    try:
        if remaining() <= CHILD_GRACE_SECONDS+CHILD_REAP_SECONDS:
            raise TimeoutError('Session budget exhausted')
        process = subprocess.Popen(command, start_new_session=True)
        while True:
            if stopped:
                raise InterruptedError('Pipeline terminated; stopping active process group')
            available = remaining()-CHILD_GRACE_SECONDS-CHILD_REAP_SECONDS
            if available <= 0:
                raise TimeoutError('Session budget exhausted')
            try:
                code = process.wait(timeout=min(available, .5))
            except subprocess.TimeoutExpired:
                continue
            if stopped:
                raise InterruptedError('Pipeline terminated at stage boundary')
            if code:
                raise RuntimeError(f'{Path(command[1]).name} exited {code}; durable partial outputs retained')
            with contextlib.suppress(ProcessLookupError):
                os.killpg(process.pid, signal.SIGKILL)
            break
    except (TimeoutError, KeyboardInterrupt, InterruptedError, RuntimeError):
        if process is not None:
            with contextlib.suppress(ProcessLookupError):
                os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=min(CHILD_GRACE_SECONDS, max(0., remaining())))
            except subprocess.TimeoutExpired:
                print('Checkpoint grace exhausted; killing owned process group', file=sys.stderr)
            with contextlib.suppress(ProcessLookupError):
                os.killpg(process.pid, signal.SIGKILL)
            try:
                process.wait(timeout=min(CHILD_REAP_SECONDS, max(0., remaining())))
            except subprocess.TimeoutExpired as exc:
                raise RuntimeError('Owned child did not reap after SIGKILL; workspace pause required') from exc
        raise
    finally:
        for sig, handler in previous.items():
            signal.signal(sig, handler)
    if stopped:
        raise InterruptedError('Pipeline termination requested; no further stage may start')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run',type=Path,required=True)
    p.add_argument('--bundle',type=Path,required=True)
    p.add_argument('--hours',type=float,required=True)
    p.add_argument('--deadline-epoch',type=float,required=True)
    p.add_argument('--parity-only',action='store_true')
    a=p.parse_args()
    if not 0<a.hours<=72: p.error('Finite session bound must be between 0 and 72 hours')
    if not math.isfinite(a.deadline_epoch) or not 0<a.deadline_epoch-time.time()<=72*3600:
        p.error('Original absolute deadline must be future and within 72 hours')
    os.environ.update(HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',HF_HUB_DISABLE_TELEMETRY='1',CUBLAS_WORKSPACE_CONFIG=':4096:8')
    config=json.loads(a.run.read_text())
    if a.parity_only:
        parity(config,a.bundle);return
    if (Path(config['output'])/'training_deadline.json').exists():
        raise ValueError('Fixed-duration training contract requires a fresh experiment namespace')
    root=Path(__file__).resolve().parent
    start=time.monotonic()
    deadline=min(start+a.hours*3600,start+a.deadline_epoch-time.time())
    def run(script,*args):
        command=[sys.executable,str(root/script),*map(str,args)]
        run_command(command,deadline,deadline_epoch=a.deadline_epoch)
    gold=json.loads((a.bundle/'gold/index.json').read_text())
    probe={**config,'panel_ids':[e['ID'] for e in gold['entries']], 'output':str(Path(config['output'])/'cuda_preflight')}
    probe.pop('identity');probe['identity']=identity(probe)
    probe_path=a.run.with_name('cuda_preflight.json');write_json(probe_path,probe)
    run('panns_cloud_run.py','cache','--run',probe_path,'--device','cuda','--resume')
    run('panns_cloud_pipeline.py','--run',probe_path,'--bundle',a.bundle,'--parity-only',
        '--hours',a.hours,'--deadline-epoch',a.deadline_epoch)
    fixed_probe_path=a.run.with_name('fixed_preflight.json')
    write_json(fixed_probe_path,fixed_preflight_config(config))
    run('panns_fixed_predictions.py','--run',fixed_probe_path,'--device','cuda')
    run('panns_cloud_run.py','cache','--run',a.run,'--device','cuda','--resume')
    output=Path(config['output'])
    completed=output/'training_phase_complete.json'
    if completed.exists():
        record=json.loads(completed.read_text())
        expected=training_phase_binding(config)
        if ({key:value for key,value in record.items() if key!='completed_epoch'}!=expected
                or not math.isfinite(record['completed_epoch']) or record['completed_epoch']<=0):
            raise ValueError('Completed training phase differs from the locked condition receipts')
    else:
        run('panns_training_budget.py','--run',a.run,'--device','cuda')
        for arm in config['arms']:
            for seed in config['seeds']:
                run('panns_cloud_run.py','train','--run',a.run,'--arm',arm,'--seed',seed,'--resume')
        write_json(completed,dict(training_phase_binding(config),completed_epoch=time.time()))
    run('panns_fixed_predictions.py','--run',a.run)
    run('panns_cloud_run.py','evaluate','--run',a.run,'--arm','R0')
    for arm in config['arms']:
        for seed in config['seeds']:
            run('panns_cloud_run.py','evaluate','--run',a.run,'--arm',arm,'--seed',seed)
    run('panns_cloud_report.py','--run',a.run)
    write_json(Path(config['output'])/'pipeline_complete.json',dict(run_identity=config['identity'],
        seconds=time.monotonic()-start,deadline_epoch=a.deadline_epoch,
        independent_evaluation=False,adoption_authorized=False))

if __name__=='__main__':main()
