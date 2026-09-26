from __future__ import annotations
import argparse
from importlib import import_module
import json
import math
from pathlib import Path
import signal
import time

from panns_adaptation_data import identity, read_cache_index, sha256, validate_config, write_json
from panns_adaptation_cache import readout_assets
from panns_adaptation_run import (frozen_digest, model_for_arm, optimization_step,
                                  restore_checkpoint, trainable_state)
from panns_cloud_run import save_checkpoint, selection_metrics, setup
from panns_training_state import (EpochSampler, accumulate_diagnostics, record_validation,
                                  validate_stopping_state)


def condition(config, request, arm):
    if arm not in ('R1', 'R3') or request['schema_version'] != 1 or request['device'] not in ('cpu', 'cuda'):
        raise ValueError('Unsupported submission training condition')
    if (type(request['microbatch']) is not int or not 1 <= request['microbatch'] <= 64
            or request['checkpoint_seconds'] != 300 or request['progress_steps'] != 100):
        raise ValueError('Invalid submission execution settings')
    deadlines = request['deadlines']
    if (set(deadlines) != {'R1', 'R3'} or any(type(v) not in (int, float) or not math.isfinite(v)
            for v in [*deadlines.values(), request['hard_deadline_epoch']])
            or not deadlines['R1'] < deadlines['R3'] < request['hard_deadline_epoch']):
        raise ValueError('Invalid absolute submission deadlines')
    output, old = Path(request['output']).resolve(), Path(config['output']).resolve()
    if not Path(request['output']).is_absolute() or output.is_relative_to(old) or old.is_relative_to(output):
        raise ValueError('Submission output must be separate from the original run')
    index_sha = sha256(old/'cache/index.json')
    if request['cache_index_sha256'] != index_sha:
        raise ValueError('Submission cache index hash mismatch')
    sources = {name: sha256(Path(__file__).with_name(name)) for name in
               ('panns_submission_train.py', 'panns_fast_step.py')}
    if request['sources'] != sources:
        raise ValueError('Submission trainer source hash mismatch')
    parent = request['parent']
    for digest in (parent['checkpoint_sha256'], parent['run_identity'], config['identity']):
        if not isinstance(digest, str) or len(digest) != 64 or any(c not in '0123456789abcdef' for c in digest):
            raise ValueError('Invalid submission parent or configuration identity')
    if sha256(parent['checkpoint']) != parent['checkpoint_sha256']:
        raise ValueError('Submission parent checkpoint hash mismatch')
    fast, step_function = request['fast_step'], optimization_step
    if type(fast['enabled']) is not bool:
        raise ValueError('Fast-step activation must be explicit')
    if fast['enabled']:
        if sha256(fast['receipt']) != fast['receipt_sha256']:
            raise ValueError('Fast-step benchmark receipt hash mismatch')
        report = json.loads(Path(fast['receipt']).read_text())
        expected = dict(passed=True, numerical_parity=True, optimizer_parity=True, frozen_unchanged=True,
            config_identity=config['identity'], cache_index_sha256=index_sha, device=request['device'],
            fast_source_sha256=sources['panns_fast_step.py'], microbatch=request['microbatch'], file_batch=4)
        if (any(report.get(key) != value for key, value in expected.items())
                or any(report.get(key) is not True for key in
                       ('passed', 'numerical_parity', 'optimizer_parity', 'frozen_unchanged'))
                or set(report.get('arms', [])) != {'R1', 'R3'}):
            raise ValueError('Fast-step benchmark did not pass the exact execution condition')
        from panns_fast_step import optimization_step as step_function
    elif fast['receipt'] is not None or fast['receipt_sha256'] is not None:
        raise ValueError('Disabled fast step must not carry a benchmark receipt')
    if not fast['enabled'] and request['microbatch'] != config['microbatch']:
        raise ValueError('Changing segment microbatch requires a passed benchmark')
    binding = dict(config_identity=config['identity'], arm=arm, seed=41, request=request,
        cache_index_sha256=index_sha, sources=sources,
        helpers={name: sha256(import_module(name).__file__) for name in
                 ('panns_cloud_run', 'panns_adaptation_run', 'panns_training_state', 'panns_fixed_predictions')})
    return output/f'train_{arm}_41', identity(binding), step_function


def train(config, request, arm):
    import torch
    out, run_id, step_function = condition(config, request, arm)
    device, microbatch, deadline = request['device'], request['microbatch'], request['deadlines'][arm]
    setup(device)
    package = validate_config(config)
    policy = dict(max_epochs=200, patience=15, min_delta=1e-4, monitor='selection_presence_bce')
    if config['early_stopping'] != policy:
        raise ValueError('Submission must retain the original early-stopping policy')
    cached = read_cache_index(config)['entries']
    entries = [e for e in cached if e['role'] == 'train' and any(y is not None for y in e['targets'])]
    selection = [e for e in cached if e['role'] == 'selection']
    if (len(entries) < 4 or not selection
            or any({e['targets'][i] for e in entries if e['targets'][i] is not None} != {0, 1} for i in range(2))
            or any(not any(e['targets'][i] is not None for e in selection) for i in range(2))):
        raise ValueError('Invalid locked training or selection panel')
    fraction = [sum(e['targets'][i] is not None for e in entries)/len(entries) for i in range(2)]
    steps_per_pass = math.ceil(len(entries)/4)
    out.mkdir(parents=True, exist_ok=True)
    receipt, checkpoint = out/'run_identity.json', out/'checkpoint.pt'
    if receipt.exists():
        if receipt.is_symlink() or json.loads(receipt.read_text()) != dict(run_identity=run_id):
            raise ValueError('Submission continuation identity mismatch')
    elif any(out.iterdir()):
        raise FileExistsError('Unowned submission training output')
    else:
        write_json(receipt, dict(run_identity=run_id))
    if checkpoint.is_symlink():
        raise ValueError('Submission checkpoint must not be a symlink')
    model = model_for_arm(package, arm, 41, device)
    frozen = frozen_digest(model)
    optimizer = torch.optim.AdamW(model.optimizer_groups(), weight_decay=1e-4, foreach=False)
    sampler = EpochSampler(len(entries), 41)
    heads, labels, groups = readout_assets(package)
    step, elapsed, diagnostics, best_model = 0, 0., {}, {}
    stopping = dict(policy=policy, best_loss=None, best_epoch=0, patience_anchor=None,
                    bad_epochs=0, last_validated_epoch=-1, history=[])
    source = checkpoint if checkpoint.exists() else Path(request['parent']['checkpoint']) if arm == 'R1' else None
    if source is not None:
        saved = restore_checkpoint(source, run_id if source == checkpoint else request['parent']['run_identity'], model, optimizer)
        if saved['arm'] != arm or saved['seed'] != 41 or saved['frozen_sha256'] != frozen:
            raise ValueError('Parent or continuation model condition mismatch')
        sampler.load_state_dict(saved['sampler'])
        torch.set_rng_state(saved['torch_rng'])
        if device == 'cuda':
            torch.cuda.set_rng_state_all(saved['cuda_rng'])
        step, elapsed, diagnostics = saved['step'], saved['elapsed_seconds'], saved['diagnostics']
        stopping, best_model = saved['early_stopping'], saved['best_model']
        if not 0 <= step <= policy['max_epochs']*steps_per_pass:
            raise ValueError('Checkpoint step exceeds the original epoch limit')
        if step or stopping['last_validated_epoch'] >= 0:
            validate_stopping_state(stopping, policy, sampler, step, steps_per_pass)
        elif stopping != dict(policy=policy, best_loss=None, best_epoch=0, patience_anchor=None,
                              bad_epochs=0, last_validated_epoch=-1, history=[]) or sampler.epoch != 0 or best_model or sampler.cursor != sampler.size:
            raise ValueError('Invalid interrupted initial validation')
        expected = {name: p for name, p in model.named_parameters() if p.requires_grad}
        if stopping['last_validated_epoch'] >= 0 and (set(best_model) != set(expected) or any(
                best_model[name].shape != p.shape or best_model[name].dtype != p.dtype
                or not torch.isfinite(best_model[name]).all() for name, p in expected.items())):
            raise ValueError('Invalid fully validated best weights')
    stop = []
    previous = signal.signal(signal.SIGTERM, lambda *_: stop.append(True))
    started, last_save = time.monotonic(), time.monotonic()
    reason = None

    def requested():
        return bool(stop) or time.time() >= deadline

    def save():
        nonlocal last_save
        save_checkpoint(checkpoint, model, optimizer, device, dict(run_identity=run_id, arm=arm, seed=41,
            step=step, sampler=sampler.state_dict(), diagnostics=diagnostics, frozen_sha256=frozen,
            elapsed_seconds=elapsed+time.monotonic()-started, early_stopping=stopping, best_model=best_model))
        last_save = time.monotonic()

    def progress(record=None):
        data = dict(run_identity=run_id, arm=arm, seed=41, step=step, epoch=sampler.epoch,
                    deadline_epoch=deadline, early_stopping=stopping, diagnostics=diagnostics,
                    updated_epoch=time.time(), last=record)
        write_json(out/'progress.json', data)
        print(json.dumps(dict(arm=arm, step=step, epoch=sampler.epoch, best_loss=stopping['best_loss'])), flush=True)

    try:
        save()
        while not requested():
            if sampler.cursor == sampler.size and stopping['last_validated_epoch'] < sampler.epoch:
                metrics = selection_metrics(model, selection, heads, labels, groups, device, microbatch, requested)
                if metrics is None or requested():
                    break
                if record_validation(stopping, metrics, sampler.epoch):
                    best_model = trainable_state(model)
                save()
                progress()
            if stopping['last_validated_epoch'] == sampler.epoch:
                reason = ('early_stopping' if stopping['bad_epochs'] >= policy['patience'] else
                          'max_epochs' if sampler.epoch == policy['max_epochs'] else None)
                if reason:
                    break
            if requested():
                break
            selected = [entries[i] for i in sampler.next_batch()]
            record, _ = step_function(model, optimizer, selected, heads, labels, groups, fraction, device, microbatch)
            step += 1
            accumulate_diagnostics(diagnostics, record)
            if time.monotonic()-last_save >= request['checkpoint_seconds']:
                save()
            if step % request['progress_steps'] == 0:
                progress(record)
        reason = 'interrupted' if stop else reason or 'time_budget'
        save()
        progress()
        if frozen_digest(model) != frozen:
            raise ValueError('Frozen parameter or BN state changed')
        checkpoint_sha = sha256(checkpoint)
        ready = bool(best_model) and reason != 'interrupted'
        if ready:
            export = out/'export'
            export.mkdir(exist_ok=True)
            temporary = export/'best_model.pt.tmp'
            torch.save(best_model, temporary)
            temporary.replace(export/'best_model.pt')
            write_json(export/'selection.json', dict(arm=arm, seed=41, checkpoint_sha256=checkpoint_sha,
                best_epoch=stopping['best_epoch'], stop_reason=reason, run_identity=run_id,
                config_identity=config['identity'], weights_sha256=sha256(export/'best_model.pt'),
                reference_sha256=sha256(config['reference'])))
        terminal = dict(run_identity=run_id, arm=arm, seed=41, stop_reason=reason,
            training_converged=reason == 'early_stopping', export_ready=ready, step=step,
            epoch=sampler.epoch, best_epoch=stopping['best_epoch'], best_loss=stopping['best_loss'],
            checkpoint_sha256=checkpoint_sha, deadline_epoch=deadline, finished_epoch=time.time())
        write_json(out/'terminal.json', terminal)
        return terminal
    finally:
        signal.signal(signal.SIGTERM, previous)


def main():
    parser = argparse.ArgumentParser(description='Continue R1 or start R3 under an absolute submission deadline')
    parser.add_argument('--run', required=True, type=Path)
    parser.add_argument('--request', required=True, type=Path)
    parser.add_argument('--arm', required=True, choices=('R1', 'R3'))
    args = parser.parse_args()
    result = train(json.loads(args.run.read_text()), json.loads(args.request.read_text()), args.arm)
    print(json.dumps(result), flush=True)
    raise SystemExit(0 if result['export_ready'] else 2)


if __name__ == '__main__':
    main()
