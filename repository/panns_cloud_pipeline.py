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


def training_phase_deadline(config, session_deadline_epoch, training_hours):
    if not math.isfinite(training_hours) or not 0 < training_hours <= 72 or not math.isfinite(session_deadline_epoch):
        raise ValueError('Invalid training deadline contract')
    path = Path(config['output'])/'training_deadline.json'
    binding = dict(config_identity=config['identity'], session_deadline_epoch=session_deadline_epoch,
                   training_hours=training_hours)
    if path.exists():
        record = json.loads(path.read_text())
    else:
        output = Path(config['output'])
        if (any((output/name).exists() for name in ('throughput_profile.json', 'training_budget.json', 'training_phase_complete.json'))
                or any(output.glob('train_*')) or any(output.glob('.checkpoint_profile_*'))):
            raise FileNotFoundError('Training deadline missing for existing phase; a fresh allowance is refused')
        started = time.time()
        record = dict(binding, started_epoch=started,
                      deadline_epoch=min(session_deadline_epoch, started+training_hours*3600))
        record['planning_hours'] = (record['deadline_epoch']-started-CHILD_GRACE_SECONDS-CHILD_REAP_SECONDS)/3600
        record['identity'] = identity(record)
        write_json(path, record)
    if (any(record.get(key) != value for key, value in binding.items())
            or record['identity'] != identity({key: value for key, value in record.items() if key != 'identity'})
            or not math.isfinite(record['started_epoch'])
            or record['planning_hours'] != (record['deadline_epoch']-record['started_epoch']-CHILD_GRACE_SECONDS-CHILD_REAP_SECONDS)/3600
            or record['deadline_epoch'] != min(session_deadline_epoch, record['started_epoch']+training_hours*3600)):
        raise ValueError('Training deadline contract changed; retain the original allowance and namespace')
    return record['deadline_epoch']


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
    p.add_argument('--training-hours',type=float,required=True)
    p.add_argument('--deadline-epoch',type=float,required=True)
    p.add_argument('--parity-only',action='store_true')
    a=p.parse_args()
    if not 0<a.hours<=72: p.error('Finite session bound must be between 0 and 72 hours')
    if not 0<a.training_hours<=72: p.error('Finite aggregate training bound must be between 0 and 72 hours')
    if not math.isfinite(a.deadline_epoch) or not 0<a.deadline_epoch-time.time()<=72*3600:
        p.error('Original absolute deadline must be future and within 72 hours')
    os.environ.update(HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',HF_HUB_DISABLE_TELEMETRY='1',CUBLAS_WORKSPACE_CONFIG=':4096:8')
    config=json.loads(a.run.read_text())
    if a.parity_only:
        parity(config,a.bundle);return
    root=Path(__file__).resolve().parent
    start=time.monotonic()
    deadline=min(start+a.hours*3600,start+a.deadline_epoch-time.time())
    def run(script,*args,cutoff=deadline,cutoff_epoch=a.deadline_epoch):
        command=[sys.executable,str(root/script),*map(str,args)]
        run_command(command,min(deadline,cutoff),deadline_epoch=min(a.deadline_epoch,cutoff_epoch))
    gold=json.loads((a.bundle/'gold/index.json').read_text())
    probe={**config,'panel_ids':[e['ID'] for e in gold['entries']], 'output':str(Path(config['output'])/'cuda_preflight')}
    probe.pop('identity');probe['identity']=identity(probe)
    probe_path=a.run.with_name('cuda_preflight.json');write_json(probe_path,probe)
    run('panns_cloud_run.py','cache','--run',probe_path,'--device','cuda','--resume')
    run('panns_cloud_pipeline.py','--run',probe_path,'--bundle',a.bundle,'--parity-only',
        '--hours',a.hours,'--training-hours',a.training_hours,'--deadline-epoch',a.deadline_epoch)
    run('panns_cloud_run.py','cache','--run',a.run,'--device','cuda','--resume')
    training_epoch=training_phase_deadline(config,a.deadline_epoch,a.training_hours)
    training_cutoff=min(deadline,time.monotonic()+training_epoch-time.time())
    output=Path(config['output'])
    planning_hours=json.loads((output/'training_deadline.json').read_text())['planning_hours']
    completed=output/'training_phase_complete.json'
    condition_paths=[output/f'train_{arm}_{seed}'/'complete.json' for arm in config['arms'] for seed in config['seeds']]
    phase_binding=dict(config_identity=config['identity'],training_deadline_sha256=sha256(output/'training_deadline.json'))
    if completed.exists():
        record=json.loads(completed.read_text())
        expected=dict(phase_binding,conditions={str(path.relative_to(output)):sha256(path) for path in condition_paths})
        if ({key:value for key,value in record.items() if key!='completed_epoch'}!=expected
                or not 0<record['completed_epoch']<=training_epoch):
            raise ValueError('Completed training phase differs from the locked condition receipts')
    else:
        run('panns_training_budget.py','--run',a.run,'--device','cuda','--training-hours',planning_hours,
            cutoff=training_cutoff,cutoff_epoch=training_epoch)
        for arm in config['arms']:
            for seed in config['seeds']:
                run('panns_cloud_run.py','train','--run',a.run,'--arm',arm,'--seed',seed,'--resume',
                    cutoff=training_cutoff,cutoff_epoch=training_epoch)
        write_json(completed,dict(phase_binding,completed_epoch=time.time(),
            conditions={str(path.relative_to(output)):sha256(path) for path in condition_paths}))
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
