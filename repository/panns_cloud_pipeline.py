"""Bounded A100 pipeline; invoke inside an explicitly started workspace."""
from __future__ import annotations
import argparse
import contextlib
import json
import math
import os
from pathlib import Path
import platform
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


def parity_runtime():
    return dict(system=platform.system(), architecture=platform.machine(), python=platform.python_version())


def cpu_reference_config(config):
    if config.get('cache_device') != 'cuda' or config.get('parity_runtime') != parity_runtime():
        raise ValueError('Parity probe device or producer platform mismatch')
    cpu = dict(config, output=str(Path(config['output'])/'cpu_reference'), cache_device='cpu')
    cpu.pop('identity')
    cpu['identity'] = identity(cpu)
    return cpu


def gold_index(config, bundle):
    manifest_path = bundle/'bundle_manifest.json'
    if config['inputs'].get(str(manifest_path)) != sha256(manifest_path):
        raise ValueError('Bundled Mac gold manifest is not pinned')
    manifest = json.loads(manifest_path.read_text())['sha256']
    path = bundle/'gold/index.json'
    if sha256(path) != manifest['gold/index.json']:
        raise ValueError('Bundled Mac gold index hash mismatch')
    return json.loads(path.read_text()), manifest


def parity_indexes(config, bundle):
    from panns_adaptation_cache import validate_cache_entry
    from panns_adaptation_data import read_cache_index, validate_config
    cpu_config = cpu_reference_config(config)
    indexes = []
    rows = {row['ID']: row for row in map(json.loads, Path(config['rows']).read_text().splitlines())}
    for current in (cpu_config, config):
        validate_config(current)
        index = read_cache_index(current)
        for entry in index['entries']:
            validate_cache_entry(current, rows[entry['ID']], entry)
            threshold = 1e-6 if current['cache_device'] == 'cpu' else 1e-5
            if (entry['tag_error'] > threshold or entry['presence_error'] > threshold
                    or entry['stats_error'] > 2*2**-23):
                raise ValueError('Cached original-forward/readout parity exceeds device threshold')
        indexes.append(index)
    gold, manifest = gold_index(config, bundle)
    ids = [entry['ID'] for entry in gold['entries']]
    if not ids or len(ids) != len(set(ids)) or ids != config['panel_ids']:
        raise ValueError('Parity requires the exact ordered bundled gold panel')
    for entry in gold['entries']:
        path = bundle/'gold'/Path(entry['path']).name
        if manifest[str(path.relative_to(bundle))] != entry['sha256'] or sha256(path) != entry['sha256']:
            raise ValueError('Bundled Mac gold activation hash mismatch')
        entry['path'] = str(path)
    return cpu_config, indexes[0], indexes[1], gold


class ParityLayoutMismatch(ValueError):
    """Authenticated caches cannot be paired segment for segment."""


def prediction_errors(reference, actual, assets, *, shared_actual_stems=False):
    import numpy as np
    import torch
    from panns_adaptation import presence_logits
    from panns_adaptation_data import load_cache
    if [entry['ID'] for entry in reference['entries']] != [entry['ID'] for entry in actual['entries']]:
        raise ValueError('Parity cache panel IDs or order mismatch')
    tag_error = presence_error = 0.
    for left, right in zip(reference['entries'], actual['entries'], strict=True):
        cpu, gpu = load_cache(left), load_cache(right)
        if (not np.isfinite(cpu['tags']).all() or not np.isfinite(gpu['tags']).all()
                or not np.isfinite(list(left['stems'].values())).all()
                or not np.isfinite(list(right['stems'].values())).all()):
            raise ValueError('Nonfinite parity tags or stems')
        if (not np.array_equal(cpu['starts'], gpu['starts'])
                or cpu['tags'].shape != gpu['tags'].shape or cpu['block5'].shape != gpu['block5'].shape
                or cpu['tags'].ndim != 2 or len(cpu['tags']) != len(cpu['starts'])
                or len(cpu['tags']) != left['segments'] or left['segments'] != right['segments']):
            raise ParityLayoutMismatch(f"Parity segment starts, counts or tensor shapes mismatch: {left['ID']}")
        tag_error = max(tag_error, float(np.abs(cpu['tags']-gpu['tags']).max()))
        stems = right['stems'] if shared_actual_stems else left['stems']
        old = presence_logits(torch.from_numpy(cpu['tags']), stems, *assets).sigmoid()
        new = presence_logits(torch.from_numpy(gpu['tags']), right['stems'], *assets).sigmoid()
        if not torch.isfinite(old).all() or not torch.isfinite(new).all():
            raise ValueError('Nonfinite parity presence probabilities')
        presence_error = max(presence_error, float((old-new).abs().max()))
    return dict(tag_max_abs=tag_error, presence_max_abs=presence_error)


def cross_platform_errors(gold, cpu, assets):
    errors = prediction_errors(gold, cpu, assets)
    return dict(errors, comparable=True, within_legacy_tolerance=all(v <= 1e-5 for v in errors.values()))


def parity(config, bundle):
    import torch
    from panns_adaptation_cache import readout_assets
    from panns_adaptation_run import model_for_arm, optimization_step
    from panns_cloud_run import setup
    setup('cuda')
    cpu_config, cpu, actual, gold = parity_indexes(config, bundle)
    package = Path(json.loads(Path(config['reference']).read_text())['package'])
    assets = readout_assets(package)
    output = Path(config['output'])
    binding = dict(reference_kind='same_runtime_cpu_original_forward', config_identity=config['identity'],
        cpu_config_identity=cpu_config['identity'], cpu_config_canonical_sha256=identity(cpu_config),
        cuda_config_canonical_sha256=identity(config), runtime=config['parity_runtime'],
        provenance=actual['provenance'], cpu_cache_sha256=sha256(Path(cpu_config['output'])/'cache/index.json'),
        cuda_cache_sha256=sha256(output/'cache/index.json'), panel_ids=config['panel_ids'])
    cross_platform = cross_platform_errors(gold, cpu, assets)
    write_json(output/'cross_platform_diagnostic.json', dict(cross_platform,
        reference_kind='bundled_mac_gold_vs_same_runtime_cpu', gold_index_sha256=sha256(bundle/'gold/index.json'),
        cpu_reference=binding, legacy_tolerance=1e-5,
        gates_training=False, quality_evaluation=False))
    frontend_errors = prediction_errors(cpu, actual, assets)
    write_json(output/'frozen_frontend_diagnostic.json', dict(frontend_errors,
        reference_kind='same_runtime_cpu_vs_cuda_independent_frozen_stems', cpu_reference=binding,
        within_legacy_tolerance=all(error <= 1e-5 for error in frontend_errors.values()),
        legacy_tolerance=1e-5, gates_training=False, quality_evaluation=False))
    errors = prediction_errors(cpu, actual, assets, shared_actual_stems=True)
    if any(error > 1e-5 for error in errors.values()):
        raise ValueError(f'Same-runtime PANNs CPU/CUDA parity with shared frozen stems failed: {errors}')
    entries = sorted(actual['entries'], key=lambda entry: entry['seconds'], reverse=True)[:4]
    gradients = []
    for device in ('cpu', 'cuda'):
        model = model_for_arm(package, 'R3', 41, device)
        optimizer = torch.optim.AdamW(model.optimizer_groups(), weight_decay=1e-4, foreach=False)
        _, grad = optimization_step(model, optimizer, entries, *assets, [1.,1.], device, 4, True)
        gradients.append(grad)
        del model, optimizer
    relative = max(float((gradients[0][key]-gradients[1][key]).double().norm()/gradients[0][key].double().norm().clamp_min(1e-12)) for key in gradients[0])
    if not math.isfinite(relative) or relative > 1e-3:
        raise ValueError('CPU/CUDA gradient parity failed')
    write_json(output/'cuda_parity.json', dict(binding, **errors, gradient_relative_l2=relative,
        tag_presence_tolerance=1e-5, gradient_tolerance=1e-3,
        presence_scope='CPU/CUDA PANNs tags with identical frozen stems from the target CUDA cache',
        independent_demucs_parity=False,
        gradient_scope='R3 seed41 same CUDA cache, four longest files; suffix/backward only',
        cpu_stem_reused_files=cpu['stem_reused_files'], cuda_stem_reused_files=actual['stem_reused_files']))


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
    gold,_=gold_index(config,a.bundle)
    probe={**config,'panel_ids':[e['ID'] for e in gold['entries']], 'output':str(Path(config['output'])/'cuda_preflight'),
           'parity_runtime':parity_runtime()}
    probe.pop('identity');probe['identity']=identity(probe)
    probe_path=a.run.with_name('cuda_preflight.json')
    cpu_probe_path=a.run.with_name('cpu_reference.json')
    for path, value in ((probe_path,probe),(cpu_probe_path,cpu_reference_config(probe))):
        if path.exists() and json.loads(path.read_text()) != value:
            raise ValueError('Existing parity probe config differs; use a fresh experiment namespace')
        if not path.exists(): write_json(path,value)
    run('panns_cloud_run.py','cache','--run',cpu_probe_path,'--device','cpu','--resume')
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
