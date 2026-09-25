"""Freeze a common full-pass budget from real suffix-update throughput before evaluation."""
from __future__ import annotations
import argparse
import gc
import json
import math
from pathlib import Path
import platform
import time
from collections.abc import Mapping

from panns_adaptation_data import identity, read_cache_index, sha256, validate_config, write_json

ARMS = ('R1', 'R2', 'R3', 'R5')


def budget_steps(seconds: Mapping[str, float], train_files: int, hours: float) -> int:
    if not math.isfinite(hours) or not 0 < hours <= 72:
        raise ValueError('Training allowance must be finite and between 0 and 72 hours')
    if set(seconds) != set(ARMS) or any(not math.isfinite(v) or v <= 0 for v in seconds.values()):
        raise ValueError('All four arms require measured finite positive throughput')
    if train_files < 4:
        raise ValueError('At least four eligible train files required')
    steps_per_pass = math.ceil(train_files / 4)
    epochs = min(int(hours * 3600 * .8 / (3 * sum(seconds.values()) * steps_per_pass)),
                 100000 // steps_per_pass)
    if epochs < 1:
        raise ValueError('Measured training allowance cannot cover one whole pass for all 12 conditions')
    return epochs * steps_per_pass


def profile_binding(config, device):
    import torch
    if device not in ('cpu', 'cuda'):
        raise ValueError('Only CPU integration or CUDA training profiles are supported')
    hardware = dict(machine=platform.machine(), processor=platform.processor())
    if device == 'cuda':
        prop = torch.cuda.get_device_properties(torch.cuda.current_device())
        hardware = dict(name=prop.name, capability=[prop.major, prop.minor], memory=prop.total_memory)
    root = Path(__file__).resolve().parent
    names = {'panns_cloud_run.py', 'panns_adaptation_run.py', 'panns_adaptation.py'}
    names.update(p.name for p in root.glob('panns_training*.py'))
    return dict(config_identity=config['identity'], cache_sha256=sha256(Path(config['output'])/'cache/index.json'),
                device=device, hardware=hardware, torch=torch.__version__, cuda=torch.version.cuda,
                code={name: sha256(root/name) for name in sorted(names)}, formula='max-step-20pct-reserve-full-pass-v1')


def load_budget(config, device):
    out = Path(config['output'])
    budget_path, profile_path = out/'training_budget.json', out/'throughput_profile.json'
    budget, profile = json.loads(budget_path.read_text()), json.loads(profile_path.read_text())
    if budget['binding']['device'] != device or profile['binding']['device'] != device:
        raise ValueError('Measured profile device does not authorize requested training device')
    binding = profile_binding(config, device)
    if budget['binding'] != binding or profile['binding'] != binding:
        raise ValueError('Measured budget binding changed; use a fresh experiment namespace')
    if budget['profile_sha256'] != sha256(profile_path):
        raise ValueError('Measured throughput profile changed after budget lock')
    if budget['identity'] != identity({k: v for k, v in budget.items() if k != 'identity'}):
        raise ValueError('Training budget identity mismatch')
    steps = budget_steps(profile['seconds_per_step'], profile['train_files'], budget['training_hours'])
    if budget['training_steps'] != steps or budget['steps_per_pass'] != math.ceil(profile['train_files']/4):
        raise ValueError('Fixed training steps do not match the measured budget')
    return budget, sha256(budget_path)


def profile_and_freeze(config, device, training_hours):
    import torch
    from panns_adaptation_cache import readout_assets
    from panns_adaptation_run import frozen_digest, model_for_arm, optimization_step, resource_usage
    from panns_cloud_run import setup
    if not math.isfinite(training_hours) or not 0 < training_hours <= 72:
        raise ValueError('Finite training-hours limit between 0 and 72 is required')
    setup(device)
    package = validate_config(config)
    entries = [e for e in read_cache_index(config)['entries'] if e['role'] == 'train' and any(y is not None for y in e['targets'])]
    if len(entries) < 4 or config['arms'] != list(ARMS) or config['seeds'] != [41, 42, 43]:
        raise ValueError('Measured v1 budget requires all four arms, three seeds and at least four train files')
    out = Path(config['output'])
    budget_path, profile_path = out/'training_budget.json', out/'throughput_profile.json'
    if budget_path.exists():
        budget, _ = load_budget(config, device)
        if budget['training_hours'] != training_hours:
            raise ValueError('Training-hours differs from immutable measured budget')
        return budget
    if list(out.glob('train_*')) or list(out.glob('evaluation_*.json')):
        raise ValueError('Measured budget must be frozen before training or quality evaluation')
    binding = profile_binding(config, device)
    if profile_path.exists():
        profile = json.loads(profile_path.read_text())
        if profile['binding'] != binding or profile['train_files'] != len(entries):
            raise ValueError('Existing measured throughput has stale binding')
    else:
        heads, labels, groups = readout_assets(package)
        fraction = [sum(e['targets'][i] is not None for e in entries)/len(entries) for i in range(2)]
        ordered = sorted(entries, key=lambda e: (e['seconds'], e['ID']))
        sampled = [ordered[round(i*(len(ordered)-1)/27)] for i in range(28)]
        batches = [sampled[i:i+4] for i in range(0, 28, 4)] + [ordered[-4:]]
        timings, started = {}, time.monotonic()
        for arm in ARMS:
            model = model_for_arm(package, arm, 41, device)
            frozen = frozen_digest(model)
            optimizer = torch.optim.AdamW(model.optimizer_groups(), weight_decay=1e-4, foreach=False)
            elapsed = []
            for step in range(10):
                if time.monotonic()-started >= 3600:
                    raise TimeoutError('Throughput pilot exceeded 60 minutes; no budget frozen')
                if device == 'cpu' and sum(resource_usage(device).values()) > 18*1024**3:
                    raise RuntimeError('CPU throughput pilot exceeded 18 GiB memory ceiling')
                if device == 'cuda':
                    torch.cuda.synchronize()
                begin = time.monotonic()
                optimization_step(model, optimizer, batches[max(0, step-2)], heads, labels,
                                  groups, fraction, device, config['microbatch'])
                if device == 'cuda':
                    torch.cuda.synchronize()
                duration = time.monotonic()-begin
                if time.monotonic()-started >= 3600:
                    raise TimeoutError('Throughput pilot exceeded 60 minutes; no budget frozen')
                if device == 'cpu' and sum(resource_usage(device).values()) > 18*1024**3:
                    raise RuntimeError('CPU throughput pilot exceeded 18 GiB memory ceiling')
                if not math.isfinite(duration) or duration <= 0:
                    raise ValueError('Invalid measured optimizer-step duration')
                if step >= 2:
                    elapsed.append(duration)
            if frozen_digest(model) != frozen:
                raise ValueError('Throughput pilot changed frozen model state')
            timings[arm] = elapsed
            del model, optimizer
            gc.collect()
            if device == 'cuda':
                torch.cuda.empty_cache()
        profile = dict(binding=binding, train_files=len(entries), microbatch=config['microbatch'],
                       known_fraction=fraction, timed_batch_ids=[[e['ID'] for e in b] for b in batches],
                       warmup_updates=2, raw_seconds=timings, seconds_per_step={a: max(t) for a, t in timings.items()},
                       elapsed_seconds=time.monotonic()-started, quality_evaluation=False)
        write_json(profile_path, profile)
    steps = budget_steps(profile['seconds_per_step'], len(entries), training_hours)
    budget = dict(binding=binding, profile_sha256=sha256(profile_path), training_hours=training_hours,
                  training_steps=steps, steps_per_pass=math.ceil(len(entries)/4))
    budget['identity'] = identity(budget)
    write_json(budget_path, budget)
    return budget


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--device', choices=['cpu', 'cuda'], default='cuda')
    parser.add_argument('--training-hours', type=float, required=True)
    args = parser.parse_args()
    print(json.dumps(profile_and_freeze(json.loads(args.run.read_text()), args.device, args.training_hours)), flush=True)
