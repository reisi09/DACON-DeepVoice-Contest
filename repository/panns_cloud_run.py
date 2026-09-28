"""Resumable PANNs training and development evaluation; never provisions compute."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import signal
import time
import numpy as np
from panns_adaptation_data import TARGETS, identity, load_cache, read_cache_index, sha256, validate_config, write_json
from panns_adaptation_cache import cache, readout_assets
from panns_adaptation_run import frozen_digest, model_for_arm, optimization_step, restore_checkpoint, trainable_state
from panns_training_state import (EpochSampler, accumulate_diagnostics, record_validation,
                                  validate_completion, validate_stopping_state)
from panns_training_budget import load_budget
from panns_fixed_predictions import configure_numerics, load_fixed_predictions


def setup(device):
    configure_numerics(device)


def training_identity(config, arm, seed, device, *, validate_runtime=True):
    _, budget_sha = load_budget(config, device, validate_runtime=validate_runtime)
    return identity(dict(config=config['identity'], arm=arm, seed=seed, device=device,
                         code=sha256(__file__), budget=budget_sha,
                         sampler=sha256(Path(__file__).with_name('panns_training_state.py')),
                         cache=sha256(Path(config['output'])/'cache/index.json')))


def save_checkpoint(path, model, optimizer, device, record):
    import torch
    temporary = path.with_suffix('.pt.tmp')
    torch.save(dict(record, model=trainable_state(model), optimizer=optimizer.state_dict(),
                    torch_rng=torch.get_rng_state(),
                    cuda_rng=torch.cuda.get_rng_state_all() if device == 'cuda' else []), temporary)
    temporary.replace(path)


def selection_metrics(model, entries, heads, labels, groups, device, microbatch, stop_requested):
    import torch
    from panns_adaptation import presence_logits
    if not entries or any(entry['role'] != 'selection' for entry in entries):
        raise ValueError('Early stopping requires the fixed selection panel only')
    counts = [sum(entry['targets'][i] is not None for entry in entries) for i in range(2)]
    if not all(counts):
        raise ValueError('Selection needs known labels for both presence targets')
    sums = torch.zeros(2, dtype=torch.float64)
    with torch.inference_mode():
        for entry in entries:
            if stop_requested():
                return None
            features = torch.from_numpy(load_cache(entry)['block5'])
            parts = []
            for part in features.split(microbatch):
                if stop_requested():
                    return None
                parts.append(model.tags_from_block5(part.to(device)).cpu())
            logits = presence_logits(torch.cat(parts), entry['stems'], heads, labels, groups)
            if not torch.isfinite(logits).all():
                raise ValueError('Nonfinite selection logits')
            known = torch.tensor([value is not None for value in entry['targets']])
            targets = torch.tensor([value if value is not None else 0. for value in entry['targets']], dtype=torch.float64)
            if not torch.isfinite(targets).all() or ((targets < 0) | (targets > 1)).any():
                raise ValueError('Invalid known selection labels')
            sums += torch.nn.functional.binary_cross_entropy_with_logits(logits, targets, reduction='none') * known
    if stop_requested():
        return None
    losses = sums / torch.tensor(counts)
    if not torch.isfinite(losses).all():
        raise ValueError('Nonfinite selection loss')
    return dict(selection_presence_bce=float(losses.mean()), voice_bce=float(losses[0]),
                music_bce=float(losses[1]), known_counts=counts)


def train(config, arm, seed, device='cuda', resume=False, stop_after=None):
    import torch
    setup(device)
    package = validate_config(config)
    if arm not in config['arms'] or seed not in config['seeds']:
        raise ValueError('Training condition outside locked experiment')
    budget, _ = load_budget(config, device)
    steps = budget['training_steps']
    policy = budget['early_stopping']
    cached = read_cache_index(config)['entries']
    entries = [r for r in cached if r['role'] == 'train' and any(y is not None for y in r['targets'])]
    selection = [r for r in cached if r['role'] == 'selection']
    if not selection or any(not any(row['targets'][i] is not None for row in selection) for i in range(2)):
        raise ValueError('Selection needs known labels for both presence targets')
    if len(entries) < 4:
        raise ValueError('Need at least four train files')
    if any({r['targets'][i] for r in entries if r['targets'][i] is not None} != {0, 1} for i in range(2)):
        raise ValueError('Training needs known positives and negatives for both targets')
    fraction = [sum(r['targets'][i] is not None for r in entries)/len(entries) for i in range(2)]
    out = Path(config['output'])/f'train_{arm}_{seed}'
    out.mkdir(parents=True, exist_ok=True)
    run_id = training_identity(config, arm, seed, device)
    checkpoint = out/'checkpoint.pt'
    if (out/'complete.json').exists():
        done = json.loads((out/'complete.json').read_text())
        validate_completion(done, run_id, sha256(checkpoint), budget)
        return done
    existing = set(out.iterdir())
    if existing and not resume:
        raise FileExistsError('Incomplete training exists; explicit --resume required')
    receipt, temporary = out/'run_identity.json', out/'checkpoint.pt.tmp'
    if receipt in existing and (receipt.is_symlink() or not receipt.is_file()
                               or json.loads(receipt.read_text()) != dict(run_identity=run_id)):
        raise ValueError('Training directory run identity mismatch')
    if resume and not checkpoint.exists() and existing:
        if (receipt not in existing or not existing <= {receipt, temporary}
                or any(path.is_symlink() or not path.is_file() for path in existing)):
            raise FileNotFoundError('No checkpoint for incomplete training; partial ownership is unverified')
        temporary.unlink(missing_ok=True)
    if not existing:
        write_json(receipt, dict(run_identity=run_id))
    model = model_for_arm(package, arm, seed, device)
    frozen = frozen_digest(model)
    optimizer = torch.optim.AdamW(model.optimizer_groups(), weight_decay=1e-4, foreach=False)
    sampler = EpochSampler(len(entries), seed)
    heads, labels, groups = readout_assets(package)
    step, elapsed, diagnostics = 0, 0., {}
    stopping = dict(policy=policy, best_loss=None, best_epoch=0, patience_anchor=None,
                    bad_epochs=0, last_validated_epoch=-1, history=[])
    best_model = {}
    if checkpoint.exists():
        saved = restore_checkpoint(checkpoint, run_id, model, optimizer)
        sampler.load_state_dict(saved['sampler'])
        torch.set_rng_state(saved['torch_rng'])
        if device == 'cuda':
            torch.cuda.set_rng_state_all(saved['cuda_rng'])
        step, elapsed = saved['step'], saved['elapsed_seconds']
        diagnostics = saved['diagnostics']
        if not 0 <= step <= steps:
            raise ValueError('Checkpoint step exceeds locked budget')
        if saved['frozen_sha256'] != frozen:
            raise ValueError('Frozen state differs from checkpoint')
        stopping, best_model = saved['early_stopping'], saved['best_model']
        validate_stopping_state(stopping, policy, sampler, step, budget['steps_per_pass'])
        expected = {name:p for name,p in model.named_parameters() if p.requires_grad}
        if set(best_model) != set(expected) or any(best_model[name].shape != p.shape
                or not torch.isfinite(best_model[name]).all() for name,p in expected.items()):
            raise ValueError('Invalid selected-best checkpoint weights')
    stop = []
    previous = signal.signal(signal.SIGTERM, lambda *_: stop.append(True))
    started = time.monotonic()
    dirty = False
    def save():
        nonlocal dirty
        save_checkpoint(checkpoint, model, optimizer, device,
            dict(run_identity=run_id, arm=arm, seed=seed, step=step, sampler=sampler.state_dict(),
                 diagnostics=diagnostics, frozen_sha256=frozen, elapsed_seconds=elapsed+time.monotonic()-started,
                 early_stopping=stopping, best_model=best_model))
        dirty = False
    reason = None
    try:
        while not stop and not (stop_after and step >= stop_after):
            if sampler.cursor == sampler.size and stopping['last_validated_epoch'] < sampler.epoch:
                metrics = selection_metrics(model, selection, heads, labels, groups, device,
                                            config['microbatch'], lambda: bool(stop))
                if metrics is None or stop:
                    break
                if record_validation(stopping, metrics, sampler.epoch):
                    best_model = trainable_state(model)
                dirty = True
                save()
                write_json(out/'progress.json', dict(step=step, steps=steps, epoch=sampler.epoch,
                    early_stopping=stopping, diagnostics=diagnostics, run_identity=run_id))
                print(json.dumps(dict(arm=arm, seed=seed, step=step, epoch=sampler.epoch, **metrics)), flush=True)
            if stopping['last_validated_epoch'] == sampler.epoch:
                reason = ('early_stopping' if stopping['bad_epochs'] >= policy['patience'] else
                          'max_epochs' if sampler.epoch == policy['max_epochs'] else None)
                if reason:
                    break
            selected = [entries[i] for i in sampler.next_batch()]
            record, _ = optimization_step(model, optimizer, selected, heads, labels,
                                          groups, fraction, device, config['microbatch'])
            step += 1
            dirty = True
            accumulate_diagnostics(diagnostics, record)
            if step % config['checkpoint_every'] == 0 and sampler.cursor != sampler.size:
                save()
                write_json(out/'progress.json', dict(step=step, steps=steps, epoch=sampler.epoch,
                    last=record, diagnostics=diagnostics, run_identity=run_id))
                print(json.dumps(dict(arm=arm, seed=seed, step=step, loss=record['loss'])), flush=True)
        if dirty:
            save()
        if frozen_digest(model) != frozen:
            raise ValueError('Frozen parameter or BN state changed')
        if not reason or stop or (stop_after and step >= stop_after):
            return dict(complete=False, step=step)
        done = dict(complete=True, stop_reason=reason, run_identity=run_id, steps=step,
                    epochs_completed=sampler.epoch, best_epoch=stopping['best_epoch'],
                    best_loss=stopping['best_loss'], patience_anchor=stopping['patience_anchor'],
                    bad_epochs=stopping['bad_epochs'], last_validated_epoch=stopping['last_validated_epoch'],
                    monitor=policy['monitor'], early_stopping=policy, checkpoint_sha256=sha256(checkpoint),
                    frozen_sha256=frozen, elapsed_seconds=elapsed+time.monotonic()-started,
                    diagnostics=diagnostics, independent_evaluation=False)
        validate_completion(done, run_id, done['checkpoint_sha256'], budget)
        if stop:
            return dict(complete=False, step=step)
        write_json(out/'complete.json', done)
        return done
    finally:
        signal.signal(signal.SIGTERM, previous)


def evaluate(config, arm, seed, device='cuda'):
    import torch
    from panns_adaptation import presence_logits
    from decision_heads import file_score
    from graph_scoring import official_metrics
    setup(device)
    package = validate_config(config)
    rows = {r['ID']: r for r in map(json.loads, Path(config['rows']).read_text().splitlines())}
    entries = [e for e in read_cache_index(config)['entries'] if e['role'] in ('development', 'selection')]
    fixed = load_fixed_predictions(config, device)
    fake = {r['ID']: r for r in fixed['entries']}
    model = model_for_arm(package, arm, seed, device)
    run_id, checkpoint_sha, completion_sha = None, None, None
    if arm != 'R0':
        out = Path(config['output'])/f'train_{arm}_{seed}'
        done = json.loads((out/'complete.json').read_text())
        completion_sha = sha256(out/'complete.json')
        budget, _ = load_budget(config, device)
        run_id, checkpoint_sha = training_identity(config, arm, seed, device), sha256(out/'checkpoint.pt')
        validate_completion(done, run_id, checkpoint_sha, budget)
        saved = torch.load(out/'checkpoint.pt', map_location='cpu', weights_only=True)
        state = saved['early_stopping']
        if (saved['run_identity'] != run_id or saved['step'] != done['steps']
                or any(state[key] != done[key] for key in ('best_epoch', 'best_loss', 'patience_anchor',
                                                          'bad_epochs', 'last_validated_epoch'))):
            raise ValueError('Selected checkpoint metadata differs from its completion receipt')
        expected = {name:p for name,p in model.named_parameters() if p.requires_grad}
        if set(saved['best_model']) != set(expected) or any(saved['best_model'][name].shape != p.shape
                or not torch.isfinite(saved['best_model'][name]).all() for name,p in expected.items()):
            raise ValueError('Invalid selected-best checkpoint weights')
        with torch.no_grad():
            for name, p in model.named_parameters():
                if p.requires_grad:
                    p.copy_(saved['best_model'][name])
    heads, labels, groups = readout_assets(package)
    ensemble = json.loads((package/'model/trained/file_ensemble.json').read_text())
    predictions = []
    with torch.inference_mode():
        for entry in entries:
            row = rows[entry['ID']]
            prior = fake[entry['ID']]
            if prior['audio_sha256'] != row['verified_audio']['sha256']:
                raise ValueError('Fixed fake predictions belong to different audio')
            features = torch.from_numpy(load_cache(entry)['block5'])
            tags = torch.cat([model.tags_from_block5(x.to(device)).cpu() for x in features.split(config['microbatch'])])
            vp, mp = presence_logits(tags, entry['stems'], heads, labels, groups).sigmoid().tolist()
            vf, mf = prior['VOICE_FAKE_PROB'], prior['MUSIC_FAKE_PROB']
            predictions.append(dict(ID=entry['ID'], FILE_FAKE_PROB=file_score(ensemble, dict(Pv=vp,Pm=mp,Fv=vf,Fm=mf)),
                VOICE_FAKE_PROB=vf, MUSIC_FAKE_PROB=mf, VOICE_PRESENT_PROB=vp, MUSIC_PRESENT_PROB=mp))
    reports = {}
    for role in ('development', 'selection'):
        pred = [p for p in predictions if rows[p['ID']]['role'] == role]
        if pred:
            reports[role] = official_metrics([rows[p['ID']] for p in pred], pred)
    result = dict(arm=arm, seed=seed, device=device, reports=reports, predictions=predictions,
                  config_identity=config['identity'], rows_sha256=sha256(config['rows']),
                  run_identity=run_id, checkpoint_sha256=checkpoint_sha,
                  completion_sha256=completion_sha,
                  fixed_predictions_sha256=sha256(config['fixed_predictions']),
                  independent_evaluation=False, adoption_authorized=False)
    write_json(Path(config['output'])/f'evaluation_{arm}_{seed}.json', result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['cache','train','evaluate'])
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--arm', choices=['R0','R1','R2','R3','R5'], default='R3')
    parser.add_argument('--seed', type=int, choices=[41,42,43], default=41)
    parser.add_argument('--device', choices=['cpu','cuda'], default='cuda')
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    config = json.loads(args.run.read_text())
    if args.command == 'cache':
        setup(args.device)
        if config.get('cache_device','cpu') != args.device:
            raise ValueError('Device differs from locked cache configuration')
        cache(config, resume=args.resume)
    elif args.command == 'train':
        if args.arm == 'R0':
            parser.error('R0 is frozen and cannot be trained')
        result = train(config,args.arm,args.seed,args.device,args.resume)
        if result.get('complete') is False:
            raise SystemExit('Training interrupted; checkpoint retained, pipeline must stop')
    else:
        evaluate(config,args.arm,args.seed,args.device)

if __name__ == '__main__':
    main()
