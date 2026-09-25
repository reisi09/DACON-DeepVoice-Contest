"""Audited, file-level PANNs v1 experiments. No provisioning or network operations."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import time
import numpy as np
from panns_adaptation_data import (TARGETS, audit, audit_archive, cache_provenance,
    identity, load_cache, quarantine_roles, read_cache_index, sha256, validate_config, write_json)

from panns_adaptation_cache import cache, load_base, readout_assets


def model_for_arm(package, arm, seed, device):
    import torch
    from panns_adaptation import AdaptedCnn14
    if arm not in ('R0', 'R1', 'R2', 'R3', 'R5'):
        raise ValueError('Unapproved v1 arm')
    torch.manual_seed(seed)
    model = AdaptedCnn14(load_base(package), channel_se=arm in ('R2', 'R3'),
                        temporal_pool=arm=='R5', train_blocks=() if arm in ('R0', 'R2') else (6,))
    return model.to(device).train()


def frozen_digest(model):
    trainable = {name for name, param in model.named_parameters() if param.requires_grad}
    digest = hashlib.sha256()
    for name, value in model.state_dict().items():
        if name not in trainable:
            digest.update(name.encode())
            digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def trainable_state(model):
    return {name: value.detach().cpu().clone() for name, value in model.named_parameters() if value.requires_grad}


def restore_checkpoint(path, expected_identity, model, optimizer):
    import torch
    record = torch.load(path, map_location='cpu', weights_only=True)
    if record['run_identity'] != expected_identity:
        raise ValueError('Checkpoint identity mismatch')
    expected = {name for name, p in model.named_parameters() if p.requires_grad}
    if set(record['model']) != expected:
        raise ValueError('Checkpoint trainable parameter set mismatch')
    with torch.no_grad():
        for name, param in model.named_parameters():
            if name in expected:
                value = record['model'][name]
                if not torch.isfinite(value).all():
                    raise ValueError('Nonfinite checkpoint')
                param.copy_(value)
    optimizer.load_state_dict(record['optimizer'])
    return record


def resource_usage(device):
    import resource
    import sys
    import torch
    driver = torch.mps.driver_allocated_memory() if device == 'mps' else 0
    return dict(peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * (1 if sys.platform == 'darwin' else 1024),
                mps_driver_bytes=driver)


def optimization_step(model, optimizer, entries, heads, labels, groups, fraction, device, microbatch, capture_gradients=False):
    import torch
    from panns_adaptation import masked_file_loss, presence_logits
    optimizer.zero_grad(set_to_none=True)
    loss_sum, predictions = 0., []
    attention = dict(segments=0, entropy_sum=0., peak_sum=0.)
    for entry in entries:
        features = torch.from_numpy(load_cache(entry)['block5'])
        parts = []
        for part in features.split(microbatch):
            parts.append(model.tags_from_block5(part.to(device)))
            if model.pool is not None:
                weights = model.pool.last_attention.to(device='cpu', dtype=torch.float64)
                attention['segments'] += weights.size(0)
                attention['entropy_sum'] += float(-(weights * weights.clamp_min(1e-300).log()).sum())
                attention['peak_sum'] += float(weights.max(1).values.sum())
        tags = torch.cat(parts)
        logits = presence_logits(tags, entry['stems'], heads, labels, groups)
        known = torch.tensor([[y is not None for y in entry['targets']]])
        targets = torch.tensor([[float('nan') if y is None else y for y in entry['targets']]], dtype=torch.float64)
        loss = masked_file_loss(logits[None], targets, known, fraction) / len(entries)
        if not torch.isfinite(loss):
            raise ValueError('Nonfinite file loss')
        loss.backward()
        loss_sum += float(loss.detach())
        predictions.append(logits.detach().sigmoid().tolist())
    gradients = {}
    for name, param in model.named_parameters():
        if param.requires_grad:
            if param.grad is None or not torch.isfinite(param.grad).all():
                raise ValueError(f'Missing or nonfinite gradient: {name}')
            if capture_gradients:
                gradients[name] = param.grad.detach().cpu().clone()
    norms = {group['name']: float(torch.nn.utils.clip_grad_norm_(group['params'], 1., error_if_nonfinite=True))
             for group in optimizer.param_groups}
    optimizer.step()
    return dict(loss=loss_sum, probabilities=predictions, IDs=[r['ID'] for r in entries],
                gradient_norms=norms, attention=attention if attention['segments'] else None), gradients


def prepare_pilot_output(out, resume):
    out.mkdir(parents=True, exist_ok=True)
    if (out/'pilot.json').exists():
        raise FileExistsError('Completed pilot must not be overwritten')
    if any(out.iterdir()) and not resume:
        raise FileExistsError('Existing incomplete pilot; use --resume or a fresh namespace')
    if resume and not (out/'resume_step2.pt').exists():
        raise FileNotFoundError('No validated pilot checkpoint to resume')


def validate_pilot_steps(steps):
    # Continuous updates plus step-two replay share the condition's 200-update cap.
    if not 3 <= steps or 2*steps-2 > 200:
        raise ValueError('Pilot must use 3-101 steps, at most 200 including replay')


def check_budget(config, elapsed, usage):
    if sum(usage.values()) > config['memory_limit_bytes'] or elapsed > config['pilot_timeout_seconds']:
        raise RuntimeError('Pilot resource budget exceeded; no quality claim')


def pilot(config, arm, device, microbatch, resume=False):
    import gc
    import random
    import torch
    validate_pilot_steps(config['pilot_steps'])
    start = time.monotonic()
    package = validate_config(config)
    if arm not in config['arms'] or microbatch not in (1, 2, 4) or device not in ('cpu', 'mps'):
        raise ValueError('Pilot condition outside fixed plan')
    if device == 'mps' and not torch.backends.mps.is_available():
        raise RuntimeError('MPS unavailable')
    torch.set_num_threads(4)
    if device == 'mps':
        torch.mps.set_per_process_memory_fraction(.5)
    cache_index = read_cache_index(config)
    entries = cache_index['entries']
    if len(entries) != len(config['panel_ids']) or any(e['role'] != 'train' for e in entries):
        raise ValueError('Engineering pilot requires exact train-only panel')
    training_rows = [r for r in (json.loads(line) for line in Path(config['rows']).open()) if r['role']=='train']
    fraction = [sum(r[k] is not None for r in training_rows)/len(training_rows) for k in TARGETS]
    out = Path(config['output'])/f'pilot_{arm}_{device}_mb{microbatch}'
    prepare_pilot_output(out, resume)
    run_identity = identity(dict(config=config['identity'], arm=arm, device=device,
                                 microbatch=microbatch, code=sha256(__file__), seed=41,
                                 cache_index=sha256(Path(config['output'])/'cache/index.json')))
    model = model_for_arm(package, arm, 41, device)
    frozen_before = frozen_digest(model)
    initial = trainable_state(model)
    optimizer = torch.optim.AdamW(model.optimizer_groups(), weight_decay=1e-4, foreach=False)
    sampler = random.Random(41)
    heads, labels, groups = readout_assets(package)
    initial_tag_error = 0.
    with torch.no_grad():
        for entry in entries:
            cached = load_cache(entry)
            tags = torch.cat([model.tags_from_block5(part.to(device)).cpu()
                              for part in torch.from_numpy(cached['block5']).split(microbatch)])
            initial_tag_error = max(initial_tag_error, float(np.abs(tags.numpy()-cached['tags']).max()))
    if initial_tag_error > (1e-6 if device == 'cpu' else 1e-5):
        raise ValueError(f'Adapted initial tags fail parity: {initial_tag_error}')
    records = []
    prior_seconds = 0.
    if resume:
        saved = restore_checkpoint(out/'resume_step2.pt', run_identity, model, optimizer)
        sampler.setstate(saved['sampler'])
        records = saved['records']
        prior_seconds = saved['elapsed_seconds']
    steps = config['pilot_steps']
    longest = max(entries, key=lambda e: e['seconds'])
    for step in range(len(records), steps):
        check_budget(config, prior_seconds+time.monotonic()-start, resource_usage(device))
        selected = sampler.sample(entries, 4)
        if step == 0 and longest['ID'] not in [e['ID'] for e in selected]:
            selected[0] = longest
        t = time.monotonic()
        record, gradients = optimization_step(model, optimizer, selected, heads, labels, groups, fraction,
                    device, microbatch, capture_gradients=step==0)
        if device == 'mps':
            torch.mps.synchronize()
        record.update(step=step+1, seconds=time.monotonic()-t, **resource_usage(device))
        check_budget(config, prior_seconds+time.monotonic()-start, resource_usage(device))
        records.append(record)
        if step == 0:
            torch.save(dict(gradients=gradients, record=record), out/'first_gradient.pt')
        del gradients
        if step == 1:
            torch.save(dict(run_identity=run_identity, step=2, model=trainable_state(model),
                            optimizer=optimizer.state_dict(), sampler=sampler.getstate(), records=records.copy(),
                            elapsed_seconds=prior_seconds+time.monotonic()-start), out/'resume_step2.pt.tmp')
            (out/'resume_step2.pt.tmp').replace(out/'resume_step2.pt')
        print(json.dumps(record), flush=True)
        write_json(out/'progress.json', records)
    final = trainable_state(model)
    if frozen_digest(model) != frozen_before:
        raise ValueError('Frozen parameter or BN buffer changed')
    changes = {name: float((final[name]-value).double().norm()) for name, value in initial.items()}
    if any(value <= 0 for value in changes.values()):
        raise ValueError(f'Intended parameters did not update: {changes}')
    del model, optimizer, initial
    gc.collect()
    if device == 'mps':
        torch.mps.empty_cache()
    model = model_for_arm(package, arm, 41, device)
    optimizer = torch.optim.AdamW(model.optimizer_groups(), weight_decay=1e-4, foreach=False)
    resumed = restore_checkpoint(out/'resume_step2.pt', run_identity, model, optimizer)
    sampler.setstate(resumed['sampler'])
    heads, labels, groups = readout_assets(package)
    resume_prob_error = 0.
    replay_usage = []
    for step in range(resumed['step'], steps):
        check_budget(config, prior_seconds+time.monotonic()-start, resource_usage(device))
        record, gradients = optimization_step(model, optimizer, sampler.sample(entries, 4),
                         heads, labels, groups, fraction, device, microbatch)
        del gradients
        if device == 'mps':
            torch.mps.synchronize()
        usage = resource_usage(device)
        check_budget(config, prior_seconds+time.monotonic()-start, usage)
        replay_usage.append(usage)
        if record['IDs'] != records[step]['IDs']:
            raise ValueError('Resume sampler order mismatch')
        resume_prob_error = max(resume_prob_error, float(np.abs(np.array(record['probabilities'])-
                                                 np.array(records[step]['probabilities'])).max()))
    resumed_final = trainable_state(model)
    resume_weight_error = max(float((resumed_final[k]-v).abs().max()) for k, v in final.items())
    if resume_weight_error > (0 if device == 'cpu' else 1e-6) or resume_prob_error > 1e-6:
        raise ValueError('Resume changed continuation')
    if frozen_digest(model) != frozen_before:
        raise ValueError('Resume changed frozen state')
    report = dict(arm=arm, seed=41, device=device, microbatch=microbatch, steps=steps,
        config_identity=config['identity'], run_identity=run_identity, records=records,
        trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad),
        frozen_sha256=frozen_before, frozen_unchanged=True, initial_tag_max_abs=initial_tag_error, parameter_update_norms=changes,
        resume_weight_max_abs=resume_weight_error, resume_probability_max_abs=resume_prob_error,
        total_seconds=prior_seconds+time.monotonic()-start, engineering_pilot_only=True,
        resumed_cli=resume, optimizer_steps_including_replay=steps+steps-2, replay_resources=replay_usage,
        longest_backward_ID=longest['ID'], longest_backward_seconds=longest['seconds'],
        memory_limit_respected=all(r['peak_rss_bytes']+r['mps_driver_bytes'] <= config['memory_limit_bytes'] for r in [*records, *replay_usage]))
    if not report['memory_limit_respected']:
        raise RuntimeError('Pilot memory limit exceeded')
    torch.save(dict(run_identity=run_identity, arm=arm, seed=41, model=final,
                    config_identity=config['identity']), out/'pilot_checkpoint.pt')
    write_json(out/'pilot.json', report)
    print(json.dumps({k:v for k,v in report.items() if k not in ('records','parameter_update_norms')}), flush=True)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['audit', 'cache', 'pilot'])
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--resume', action='store_true', help='Continue a validated incomplete pilot')
    parser.add_argument('--arm', choices=['R1', 'R2', 'R3', 'R5'], default='R3')
    parser.add_argument('--device', choices=['cpu', 'mps'], default='cpu')
    parser.add_argument('--microbatch', type=int, choices=[1, 2, 4], default=1)
    args = parser.parse_args()
    config = json.loads(args.run.read_text())
    if args.command == 'pilot':
        pilot(config, args.arm, args.device, args.microbatch, resume=args.resume)
    else:
        if args.command == 'cache':
            cache(config, resume=args.resume)
        else:
            audit(config)


if __name__ == '__main__':
    main()
