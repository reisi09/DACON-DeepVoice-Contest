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
from panns_training_state import EpochSampler, accumulate_diagnostics
from panns_training_budget import load_budget
from panns_fixed_predictions import configure_numerics, load_fixed_predictions


def setup(device):
    configure_numerics(device)


def training_identity(config, arm, seed, device):
    _, budget_sha = load_budget(config, device)
    return identity(dict(config=config['identity'], arm=arm, seed=seed, device=device,
                         code=sha256(__file__), budget=budget_sha,
                         sampler=sha256(Path(__file__).with_name('panns_training_state.py')),
                         cache=sha256(Path(config['output'])/'cache/index.json')))


def train(config, arm, seed, device='cuda', resume=False, stop_after=None):
    import torch
    setup(device)
    package = validate_config(config)
    if arm not in config['arms'] or seed not in config['seeds']:
        raise ValueError('Training condition outside locked experiment')
    budget, _ = load_budget(config, device)
    steps = budget['training_steps']
    entries = [r for r in read_cache_index(config)['entries'] if r['role'] == 'train' and any(y is not None for y in r['targets'])]
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
        if done['run_identity'] != run_id or done['steps'] != steps or done['checkpoint_sha256'] != sha256(checkpoint):
            raise ValueError('Completed training identity/hash mismatch')
        return done
    if any(out.iterdir()) and not resume:
        raise FileExistsError('Incomplete training exists; explicit --resume required')
    if resume and not checkpoint.exists() and any(out.iterdir()):
        raise FileNotFoundError('No checkpoint for incomplete training')
    model = model_for_arm(package, arm, seed, device)
    frozen = frozen_digest(model)
    optimizer = torch.optim.AdamW(model.optimizer_groups(), weight_decay=1e-4, foreach=False)
    sampler = EpochSampler(len(entries), seed)
    heads, labels, groups = readout_assets(package)
    step, elapsed, diagnostics = 0, 0., {}
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
    stop = []
    previous = signal.signal(signal.SIGTERM, lambda *_: stop.append(True))
    started = time.monotonic()
    def save():
        torch.save(dict(run_identity=run_id, arm=arm, seed=seed, step=step, model=trainable_state(model),
            optimizer=optimizer.state_dict(), sampler=sampler.state_dict(), diagnostics=diagnostics, torch_rng=torch.get_rng_state(),
            cuda_rng=torch.cuda.get_rng_state_all() if device == 'cuda' else [], frozen_sha256=frozen,
            elapsed_seconds=elapsed+time.monotonic()-started), checkpoint.with_suffix('.pt.tmp'))
        checkpoint.with_suffix('.pt.tmp').replace(checkpoint)
    try:
        while step < steps and not stop:
            selected = [entries[i] for i in sampler.next_batch()]
            record, _ = optimization_step(model, optimizer, selected, heads, labels,
                                          groups, fraction, device, config['microbatch'])
            step += 1
            accumulate_diagnostics(diagnostics, record)
            if step % config['checkpoint_every'] == 0 or step == steps or (stop_after and step >= stop_after):
                save()
                write_json(out/'progress.json', dict(step=step, steps=steps, epoch=sampler.epoch,
                    last=record, diagnostics=diagnostics, run_identity=run_id))
                print(json.dumps(dict(arm=arm, seed=seed, step=step, loss=record['loss'])), flush=True)
            if stop_after and step >= stop_after:
                break
        save()
        if frozen_digest(model) != frozen:
            raise ValueError('Frozen parameter or BN state changed')
        if step != steps:
            return dict(complete=False, step=step)
        done = dict(run_identity=run_id, steps=step, checkpoint_sha256=sha256(checkpoint),
                    frozen_sha256=frozen, elapsed_seconds=elapsed+time.monotonic()-started,
                    diagnostics=diagnostics, independent_evaluation=False)
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
    run_id, checkpoint_sha = None, None
    if arm != 'R0':
        out = Path(config['output'])/f'train_{arm}_{seed}'
        done = json.loads((out/'complete.json').read_text())
        budget, _ = load_budget(config, device)
        run_id, checkpoint_sha = training_identity(config, arm, seed, device), sha256(out/'checkpoint.pt')
        if done['run_identity'] != run_id or done['steps'] != budget['training_steps'] or done['checkpoint_sha256'] != checkpoint_sha:
            raise ValueError('Evaluation requires the exact completed final checkpoint')
        saved = torch.load(out/'checkpoint.pt', map_location='cpu', weights_only=True)
        with torch.no_grad():
            for name, p in model.named_parameters():
                if p.requires_grad:
                    p.copy_(saved['model'][name])
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
