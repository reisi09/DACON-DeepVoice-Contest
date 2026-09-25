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


def validate_policy(policy):
    if (not isinstance(policy, dict) or set(policy) != {'max_epochs', 'patience', 'min_delta', 'monitor'}
            or type(policy['max_epochs']) is not int or policy['max_epochs'] < 1
            or type(policy['patience']) is not int or policy['patience'] < 1
            or type(policy['min_delta']) not in (int, float) or not math.isfinite(policy['min_delta'])
            or policy['min_delta'] < 0 or policy['monitor'] != 'selection_presence_bce'):
        raise ValueError('Invalid immutable early-stopping policy')
    return dict(policy)


def budget_steps(seconds: Mapping[str, float], train_files: int, max_epochs: int, *,
                 checkpoint_seconds: Mapping[str, float] | None = None,
                 checkpoint_hash_seconds: Mapping[str, float] | None = None,
                 checkpoint_every: int = 100) -> int:
    if type(max_epochs) is not int or max_epochs < 1:
        raise ValueError('Maximum epochs must be a positive integer')
    if set(seconds) != set(ARMS) or any(not math.isfinite(v) or v <= 0 for v in seconds.values()):
        raise ValueError('All four arms require measured finite positive throughput')
    if train_files < 4:
        raise ValueError('At least four eligible train files required')
    if type(checkpoint_every) is not int or checkpoint_every < 1:
        raise ValueError('Checkpoint interval must be a positive integer')
    writes = checkpoint_seconds if checkpoint_seconds is not None else dict.fromkeys(ARMS, 0.)
    hashes = checkpoint_hash_seconds if checkpoint_hash_seconds is not None else dict.fromkeys(ARMS, 0.)
    if any(set(values) != set(ARMS) or any(not math.isfinite(v) or v < 0 for v in values.values())
           for values in (writes, hashes)):
        raise ValueError('Every arm requires finite nonnegative persistence timing')
    return max_epochs * math.ceil(train_files / 4)


def training_estimate(profile, steps, max_epochs, checkpoint_every):
    writes = math.ceil(steps/checkpoint_every) + max_epochs + 1
    return 3 * (steps*sum(profile['seconds_per_step'].values())
                + writes*sum(profile['checkpoint_seconds'].values())
                + sum(profile['checkpoint_hash_seconds'].values()))


def profile_binding(config, device, *, validate_runtime=True):
    if device not in ('cpu', 'cuda'):
        raise ValueError('Only CPU integration or CUDA training profiles are supported')
    root = Path(__file__).resolve().parent
    names = {'panns_cloud_run.py', 'panns_adaptation_run.py', 'panns_adaptation.py'}
    names.update(p.name for p in root.glob('panns_training*.py'))
    binding = dict(config_identity=config['identity'], cache_sha256=sha256(Path(config['output'])/'cache/index.json'),
                   device=device, code={name: sha256(root/name) for name in sorted(names)},
                   formula='max-epochs-selection-early-stopping-v3',
                   early_stopping=validate_policy(config['early_stopping']))
    if validate_runtime:
        import torch
        hardware = dict(machine=platform.machine(), processor=platform.processor())
        if device == 'cuda':
            prop = torch.cuda.get_device_properties(torch.cuda.current_device())
            hardware = dict(name=prop.name, capability=[prop.major, prop.minor], memory=prop.total_memory)
        binding.update(hardware=hardware, torch=torch.__version__, cuda=torch.version.cuda)
    return binding


def load_budget(config, device, *, validate_runtime=True):
    out = Path(config['output'])
    budget_path, profile_path = out/'training_budget.json', out/'throughput_profile.json'
    budget, profile = json.loads(budget_path.read_text()), json.loads(profile_path.read_text())
    if budget['binding']['device'] != device or profile['binding']['device'] != device:
        raise ValueError('Measured profile device does not authorize requested training device')
    binding = profile_binding(config, device, validate_runtime=validate_runtime)
    if (budget['binding'] != profile['binding']
            or set(budget['binding']) != set(binding) | {'hardware', 'torch', 'cuda'}
            or any(budget['binding'].get(key) != value for key, value in binding.items())):
        raise ValueError('Measured budget binding changed; use a fresh experiment namespace')
    if budget['profile_sha256'] != sha256(profile_path):
        raise ValueError('Measured throughput profile changed after budget lock')
    if budget['identity'] != identity({k: v for k, v in budget.items() if k != 'identity'}):
        raise ValueError('Training budget identity mismatch')
    policy = validate_policy(config['early_stopping'])
    if budget.get('early_stopping') != policy or 'training_hours' in budget:
        raise ValueError('Early-stopping policy differs from the locked run')
    steps = budget_steps(profile['seconds_per_step'], profile['train_files'], policy['max_epochs'],
                         checkpoint_seconds=profile['checkpoint_seconds'],
                         checkpoint_hash_seconds=profile['checkpoint_hash_seconds'],
                         checkpoint_every=config['checkpoint_every'])
    if budget['training_steps'] != steps or budget['steps_per_pass'] != math.ceil(profile['train_files']/4):
        raise ValueError('Maximum training steps do not match the locked epoch policy')
    estimate = training_estimate(profile, steps, policy['max_epochs'], config['checkpoint_every'])
    if (budget.get('estimated_training_seconds') != estimate or budget.get('validation_time_included') is not False
            or not math.isfinite(estimate)):
        raise ValueError('Training estimate or validation-time disclosure differs from the profile')
    return budget, sha256(budget_path)


def profile_and_freeze(config, device):
    import torch
    from panns_adaptation_cache import readout_assets
    from panns_adaptation_run import frozen_digest, model_for_arm, optimization_step, resource_usage, trainable_state
    from panns_cloud_run import save_checkpoint, setup
    from panns_training_state import EpochSampler, accumulate_diagnostics
    policy = validate_policy(config['early_stopping'])
    setup(device)
    package = validate_config(config)
    entries = [e for e in read_cache_index(config)['entries'] if e['role'] == 'train' and any(y is not None for y in e['targets'])]
    if len(entries) < 4 or config['arms'] != list(ARMS) or config['seeds'] != [41, 42, 43]:
        raise ValueError('Training profile requires all four arms, three seeds and at least four train files')
    out = Path(config['output'])
    budget_path, profile_path = out/'training_budget.json', out/'throughput_profile.json'
    if budget_path.exists():
        budget, _ = load_budget(config, device)
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
        timings, checkpoint_seconds, checkpoint_hash_seconds, checkpoint_bytes = {}, {}, {}, {}
        started = time.monotonic()
        for arm in ARMS:
            model = model_for_arm(package, arm, 41, device)
            frozen = frozen_digest(model)
            optimizer = torch.optim.AdamW(model.optimizer_groups(), weight_decay=1e-4, foreach=False)
            elapsed, diagnostics = [], {}
            for step in range(10):
                if time.monotonic()-started >= 3600:
                    raise TimeoutError('Throughput pilot exceeded 60 minutes; no budget frozen')
                if device == 'cpu' and sum(resource_usage(device).values()) > 18*1024**3:
                    raise RuntimeError('CPU throughput pilot exceeded 18 GiB memory ceiling')
                if device == 'cuda':
                    torch.cuda.synchronize()
                begin = time.monotonic()
                record, _ = optimization_step(model, optimizer, batches[max(0, step-2)], heads, labels,
                                              groups, fraction, device, config['microbatch'])
                if device == 'cuda':
                    torch.cuda.synchronize()
                duration = time.monotonic()-begin
                accumulate_diagnostics(diagnostics, record)
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
            checkpoint = out/f'.checkpoint_profile_{arm}.pt'
            begin = time.monotonic()
            save_checkpoint(checkpoint, model, optimizer, device,
                dict(run_identity='throughput-pilot', arm=arm, seed=41, step=10,
                     sampler=EpochSampler(len(entries), 41).state_dict(), diagnostics=diagnostics,
                     frozen_sha256=frozen, elapsed_seconds=time.monotonic()-started,
                     best_model=trainable_state(model)))
            checkpoint_seconds[arm] = time.monotonic()-begin
            begin = time.monotonic()
            sha256(checkpoint)
            checkpoint_hash_seconds[arm] = time.monotonic()-begin
            checkpoint_bytes[arm] = checkpoint.stat().st_size
            checkpoint.unlink()
            del model, optimizer
            gc.collect()
            if device == 'cuda':
                torch.cuda.empty_cache()
        profile = dict(binding=binding, train_files=len(entries), microbatch=config['microbatch'],
                       known_fraction=fraction, timed_batch_ids=[[e['ID'] for e in b] for b in batches],
                       warmup_updates=2, raw_seconds=timings, seconds_per_step={a: max(t) for a, t in timings.items()},
                       checkpoint_seconds=checkpoint_seconds, checkpoint_hash_seconds=checkpoint_hash_seconds,
                       checkpoint_bytes=checkpoint_bytes,
                       elapsed_seconds=time.monotonic()-started, quality_evaluation=False)
        write_json(profile_path, profile)
    steps = budget_steps(profile['seconds_per_step'], len(entries), policy['max_epochs'],
                         checkpoint_seconds=profile['checkpoint_seconds'],
                         checkpoint_hash_seconds=profile['checkpoint_hash_seconds'],
                         checkpoint_every=config['checkpoint_every'])
    budget = dict(binding=binding, profile_sha256=sha256(profile_path), early_stopping=policy,
                  training_steps=steps, steps_per_pass=math.ceil(len(entries)/4),
                  estimated_training_seconds=training_estimate(profile, steps, policy['max_epochs'], config['checkpoint_every']),
                  validation_time_included=False)
    budget['identity'] = identity(budget)
    write_json(budget_path, budget)
    return budget


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--device', choices=['cpu', 'cuda'], default='cuda')
    args = parser.parse_args()
    print(json.dumps(profile_and_freeze(json.loads(args.run.read_text()), args.device)), flush=True)
