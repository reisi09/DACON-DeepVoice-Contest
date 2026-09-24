"""Full-coverage epochs for unchanged WindowMIL and DF-Arena's existing fc5.

Every train row participates once per epoch. Validation selects checkpoints only.
"""
from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
import csv
import gc
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from eight_fixed_io import atomic_json
from full_bank_features import BATCH_CANDIDATES, check_gpu, load_rows, validate_cache, row_identity
from full_bank_batches import Item, epoch_order, loss_weights, groups, duration_balance
from graph_scoring import _roc_metrics
from localization_models import WindowMIL
from v6_data import sha256
from full_bank_contract import config_for, pool_odds, role, feature_contract



def _known(row: Mapping[str, str], name: str) -> bool:
    return row.get(name, '') in ('0', '1', '0.0', '1.0') and row.get(name + '_mask', '1') not in ('0', '0.0', 'False')


def head_labels(row: Mapping[str, str]) -> dict[str, float]:
    """Every head whose component label is known. The route head keeps its legacy semantics; a mixed row
    (song, voice+music mix) additionally trains the other head, since the cache holds both stems' features."""
    heads = {row['route']: float(row['MUSIC_FAKE' if row['route'] == 'music' else 'VOICE_FAKE'])}
    for head, fake, present in (('music', 'MUSIC_FAKE', 'MUSIC_PRESENT'), ('voice', 'VOICE_FAKE', 'VOICE_PRESENT')):
        if head not in heads and _known(row, fake) and row.get(present, '') in ('1', '1.0'):
            heads[head] = float(row[fake])
    return heads


def read_items(manifest: Path, cache: Path, feature_manifest: Path | None = None) -> list[Item]:
    rows = load_rows(manifest)
    index = json.loads((cache / 'index.json').read_text())
    if index['identity']['manifest_sha256'] != sha256(feature_manifest or manifest):
        raise ValueError('Feature manifest mismatch')
    if feature_manifest is not None:
        union = {row['ID']: row for row in load_rows(feature_manifest)}
        for row in rows:
            if union.get(row['ID']) != row:
                raise ValueError(f"Requested row differs from feature manifest: {row['ID']}")
    for name, digest in index['identity']['implementation'].items():
        if sha256(Path(__file__).parent / name) != digest:
            raise ValueError(f'Feature implementation changed: {name}')
    key = hashlib.sha256(json.dumps(index['identity'], sort_keys=True).encode()).hexdigest()
    result = []
    for row in rows:
        entry = index['items'][row['ID']]
        path = cache / entry['path']
        if not path.resolve().is_relative_to(cache.resolve()):
            raise ValueError('Feature path escapes cache')
        validate_cache(path, row_identity(row, key), entry['sha256'])
        with np.load(path, allow_pickle=False) as data:
            arrays = {k: data[k].copy() for k in ('mert_windows', 'voice_embeddings', 'window_truth')}
            silent = bool(data['voice_silent'])
            voice_truth = data['voice_window_truth'].copy() if 'voice_window_truth' in data else np.full(len(arrays['voice_embeddings']), np.nan, np.float32)
        for head, label in head_labels(row).items():
            music = head == 'music'
            if head != row['route'] and not music and silent:
                continue  # labelled vocal but empty separated stem: no evidence for the voice head
            features = arrays['mert_windows' if music else 'voice_embeddings']
            truth = arrays['window_truth'] if music else voice_truth
            if features.ndim != 2 or not len(features) or not np.isfinite(features).all():
                raise ValueError(f"Invalid {head} features: {row['ID']}")
            result.append(Item({**row, 'route': head}, features, truth, label, silent))
    return result



def forward(model: nn.Module, items: Sequence[Item], route: str, fake_index: int,
            device: str, config=None) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Ragged execution is algebraically identical to padded WindowMIL; no truncation."""
    config = config_for(config)
    sizes = [len(item.features) for item in items]
    features = torch.from_numpy(np.concatenate([item.features for item in items])).to(device)
    if route == 'music':
        logits, _ = model.head((features - model.mean) / model.scale)
        files = torch.stack([part.float().logsumexp(0) - np.log(size)
                             for part, size in zip(logits.split(sizes), sizes, strict=True)])
        truth = torch.from_numpy(np.concatenate([item.truth for item in items])).to(device)
        known = torch.isfinite(truth)
        return files, logits[known], truth[known]
    values = model(features).float()
    odds = values[:, fake_index] - values[:, 1-fake_index]
    files = torch.stack([pool_odds(part, config) for part in odds.split(sizes)])
    truth = torch.from_numpy(np.concatenate([item.truth for item in items])).to(device)
    known = torch.isfinite(truth)
    return files, odds[known], truth[known]



def step(model: nn.Module, optimizer: torch.optim.Optimizer, items: Sequence[Item], route: str,
         fake_index: int, weights: Mapping[str, float], microbatch: int, window_limit: int, device: str, config=None) -> float:
    config = config_for(config)
    if any(role(item.row) != 'train' or item.row['split'] != 'train' for item in items):
        raise ValueError('Validation data passed to optimizer')
    optimizer.zero_grad(set_to_none=True)
    local_count = max(1, sum(np.isfinite(item.truth).sum() for item in items))
    total = 0.
    for group in groups(items, microbatch, window_limit):
        with torch.autocast('cuda', dtype=torch.bfloat16, enabled=device == 'cuda'):
            logits, local_logits, truth = forward(model, group, route, fake_index, device, config)
            labels = torch.tensor([item.label for item in group], device=device)
            weight = torch.tensor([weights[item.row['ID']] for item in group], device=device)
            loss = (F.binary_cross_entropy_with_logits(logits.float(), labels, reduction='none') * weight).sum() / len(items)
            if local_logits.numel():
                loss = loss + config[route + '_local_loss_weight'] * F.binary_cross_entropy_with_logits(local_logits.float(), truth.float(), reduction='sum') / local_count
        if not torch.isfinite(loss):
            raise ValueError('Nonfinite training objective')
        loss.backward()
        total += float(loss.detach())
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
    optimizer.step()
    return total


@torch.no_grad()
def predict(model: nn.Module, items: Sequence[Item], route: str, fake_index: int,
            microbatch: int, window_limit: int, device: str, config=None) -> np.ndarray:
    model.eval()
    output = []
    for group in groups(items, microbatch, window_limit):
        with torch.autocast('cuda', dtype=torch.bfloat16, enabled=device == 'cuda'):
            logits, _, _ = forward(model, group, route, fake_index, device, config)
        probabilities = logits.float().sigmoid().cpu().tolist()
        output.extend(0. if route == 'voice' and item.silent else score for item, score in zip(group, probabilities, strict=True))
    return np.asarray(output)


def benchmark(model: nn.Module, items: Sequence[Item], route: str, fake_index: int,
              weights: Mapping[str, float], device: str, config=None) -> tuple[int, int, list[dict]]:
    """Measure real forward/backward on longest training bags, then restore weights."""
    state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    rng = torch.get_rng_state()
    cuda_rng = torch.cuda.get_rng_state() if device == 'cuda' else None
    ordered = sorted(items, key=lambda item: len(item.features), reverse=True)
    measurements = []
    for count in BATCH_CANDIDATES:
        selected = ordered[:count]
        limit = sum(len(x.features) for x in selected)
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=.01)
        try:
            start = time.monotonic()
            step(model, optimizer, selected, route, fake_index, weights, count, limit, device, config)
            torch.cuda.synchronize()
            measurements.append({'batch': count, 'window_limit': limit, 'status': 'ok',
                                 'peak_bytes': torch.cuda.max_memory_allocated(), 'seconds': time.monotonic()-start})
        except torch.cuda.OutOfMemoryError:
            measurements.append({'batch': count, 'status': 'oom'})
        finally:
            model.load_state_dict(state)
            model.zero_grad(set_to_none=True)
            torch.set_rng_state(rng)
            if cuda_rng is not None:
                torch.cuda.set_rng_state(cuda_rng)
            del optimizer
    safe = [m for m in measurements if m['status'] == 'ok' and m['peak_bytes'] <= .8 * torch.cuda.get_device_properties(0).total_memory]
    if not safe:
        raise RuntimeError('No training batch meets measured 80% VRAM headroom')
    chosen = max(safe, key=lambda m: m['batch'])
    return chosen['batch'], chosen['window_limit'], measurements


def train(args: argparse.Namespace) -> None:
    gpu = check_gpu(getattr(args, 'gpu_name', 'H100'))
    if args.max_epochs < 1 or args.patience < 1:
        raise ValueError('Convergence safeguards must be positive')
    config = config_for(json.loads(args.config.read_text()) if args.config else None)
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    args.output.mkdir(parents=True, exist_ok=True)
    feature_manifest = getattr(args, 'feature_manifest', None)
    items = read_items(args.manifest, args.cache, feature_manifest)
    voice = torch.load(args.cache / 'voice_classifier_initial.pt', map_location='cpu', weights_only=True)
    fake_index = voice['fake_label_index']
    run_identity = {'manifest': sha256(args.manifest), 'cache': sha256(args.cache / 'index.json'),
                    'feature_contract': feature_contract(json.loads((args.cache / 'index.json').read_text())['identity']),
                    'trainer': sha256(Path(__file__)), 'model': sha256(Path(__file__).with_name('localization_models.py')),
                    'head': sha256(Path(__file__).with_name('eight_heads.py')),
                    'evaluator': sha256(Path(__file__).with_name('full_bank_evaluate.py')),
                    'batches': sha256(Path(__file__).with_name('full_bank_batches.py')),
                    'voice_initial': sha256(args.cache / 'voice_classifier_initial.pt'),
                    'max_epochs': args.max_epochs, 'patience': args.patience, 'seeds': config['seeds'], 'config': config,
                    'contract': sha256(Path(__file__).with_name('full_bank_contract.py'))}
    if feature_manifest is not None:
        run_identity['feature_manifest'] = sha256(feature_manifest)
    if args.duration_balance:
        run_identity['duration_balance'] = True
        samples = {k: v['samples'] for k, v in json.loads((args.cache / 'index.json').read_text())['items'].items()}
    identity_path = args.output / 'run_identity.json'
    if identity_path.exists() and json.loads(identity_path.read_text()) != run_identity:
        raise ValueError('Output directory belongs to a different training run')
    atomic_json(identity_path, run_identity)
    atomic_json(args.output / 'device.json', gpu)
    results = {}
    for route in ('music', 'voice'):
        training = [x for x in items if x.row['route'] == route and role(x.row) == 'train']
        validation = [x for x in items if x.row['route'] == route and role(x.row) == 'selection']
        if not training or not validation or {x.label for x in validation} != {0., 1.}:
            raise ValueError(f'Incomplete train/validation classes for {route}')
        weights = loss_weights(training)
        if args.duration_balance:
            ids = [x.row['ID'] for x in training]
            factor = duration_balance([x.label for x in training], [samples[i] for i in ids], [weights[i] for i in ids])
            weights = {i: weights[i] * f for i, f in zip(ids, factor, strict=True)}
        means = np.stack([x.features.mean(0) for x in training])
        mean, scale = torch.tensor(means.mean(0)), torch.tensor(np.maximum(means.std(0), 1e-6))
        seed_predictions = []
        for seed in config['seeds']:
            torch.manual_seed(seed)
            np.random.seed(seed)
            model = WindowMIL(mean, scale) if route == 'music' else nn.Linear(1280, 2)
            if route == 'voice':
                model.load_state_dict(voice['state_dict'])
            model.to('cuda')
            microbatch, window_limit, measured = benchmark(model, training, route, fake_index, weights, 'cuda', config)
            atomic_json(args.output / f'{route}_{seed}_batch.json', {'microbatch': microbatch,
                        'window_limit': window_limit, 'effective_batch': 256, 'measurements': measured})
            window_limit = max(window_limit, max(len(x.features) for x in validation))
            optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3 if route == 'music' else 1e-4, weight_decay=.01)
            checkpoint = args.output / f'{route}_{seed}.pt'
            resume = args.output / f'{route}_{seed}_resume.pt'
            best, stale, first_epoch, history = (float('inf'), float('inf')), 0, 1, []
            if resume.exists():
                state = torch.load(resume, map_location='cuda', weights_only=False)
                if state['identity'] != run_identity:
                    raise ValueError('Training resume belongs to a different run')
                model.load_state_dict(state['model'])
                optimizer.load_state_dict(state['optimizer'])
                best, stale, first_epoch, history = state['best'], state['stale'], state['epoch']+1, state['history']
            for epoch in range(first_epoch, args.max_epochs+1):
                if stale >= args.patience:
                    break
                model.train()
                order = epoch_order(training, seed, epoch)
                epoch_loss = 0.
                for offset in range(0, len(order), 256):
                    batch = [training[i] for i in order[offset:offset+256]]
                    epoch_loss += step(model, optimizer, batch, route, fake_index, weights, microbatch, window_limit, 'cuda', config) * len(batch)
                predictions = predict(model, validation, route, fake_index, microbatch, window_limit, 'cuda', config)
                labels = np.asarray([x.label for x in validation])
                metrics = _roc_metrics(labels.tolist(), predictions.tolist())
                eer = float(metrics[0])
                clipped = np.clip(predictions, 1e-7, 1-1e-7)
                bce = float(-(labels*np.log(clipped)+(1-labels)*np.log1p(-clipped)).mean())
                improved = (eer, bce) < best
                if improved:
                    best, stale = (eer, bce), 0
                    payload = {'model': {k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
                               'route': route, 'seed': seed, 'epoch': epoch, 'validation_eer': eer,
                               'validation_bce': bce, 'identity': run_identity, 'fake_label_index': fake_index,
                               'voice_module': voice['module'], 'train_ids': [x.row['ID'] for x in training]}
                    torch.save(payload, checkpoint.with_suffix('.tmp'))
                    checkpoint.with_suffix('.tmp').replace(checkpoint)
                else:
                    stale += 1
                record = {'epoch': epoch, 'loss': epoch_loss/len(training), 'validation_eer': eer,
                          'validation_bce': bce, 'rows_seen': len(order), 'unique_rows_seen': len(set(order.tolist())),
                          'order_sha256': hashlib.sha256(order.tobytes()).hexdigest(), 'improved': improved}
                history.append(record)
                torch.save({'model': model.state_dict(), 'optimizer': optimizer.state_dict(), 'identity': run_identity,
                            'epoch': epoch, 'best': best, 'stale': stale, 'history': history}, resume.with_suffix('.tmp'))
                resume.with_suffix('.tmp').replace(resume)
                atomic_json(args.output / 'progress.json', {'stage': 'train', 'route': route, 'seed': seed, **record})
                atomic_json(args.output / f'{route}_{seed}_history.json', history)
                print(f'{route} seed={seed} epoch={epoch} val_eer={eer:.6f} seen={len(order)}', flush=True)
            state = torch.load(checkpoint, map_location='cuda', weights_only=True)
            model.load_state_dict(state['model'])
            prediction = predict(model, validation, route, fake_index, microbatch, window_limit, 'cuda', config)
            seed_predictions.append(prediction)
            results[f'{route}_{seed}'] = {'epoch': state['epoch'], 'validation_eer': state['validation_eer'],
                                        'checkpoint_sha256': sha256(checkpoint), 'epochs_completed': len(history)}
            del model, optimizer
            gc.collect()
            torch.cuda.empty_cache()
        ensemble = np.mean(seed_predictions, axis=0)
        ensemble_eer, ensemble_auc = _roc_metrics([int(x.label) for x in validation], ensemble.tolist())
        results[route+'_ensemble'] = {'eer': ensemble_eer, 'auc': ensemble_auc}
        with (args.output / f'{route}_validation.csv').open('w', newline='') as stream:
            writer = csv.writer(stream)
            writer.writerow(['ID', 'label', 'probability', *(f'seed{s}' for s in config['seeds'])])
            for i, item in enumerate(validation):
                writer.writerow([item.row['ID'], item.label, ensemble[i], *(p[i] for p in seed_predictions)])
    from full_bank_evaluate import evaluate
    evaluate(args.manifest, args.cache, args.output)
    atomic_json(args.output / 'results.json', results)
    atomic_json(args.output / 'complete.json', {'identity': run_identity, 'results_sha256': sha256(args.output / 'results.json')})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('manifest', 'cache', 'output'):
        parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--feature-manifest', type=Path,
                        help='reuse a cache produced from this union manifest; every requested row must match exactly')
    parser.add_argument('--experiment-config', '--config', dest='config', type=Path)
    parser.add_argument('--max-epochs', type=int, default=50)
    parser.add_argument('--patience', type=int, default=5)
    parser.add_argument('--gpu-name', choices=('A100', 'H100'), default='H100')
    parser.add_argument('--duration-balance', action='store_true',
                        help='reweight train rows so both classes share one log-duration histogram (length shortcut control)')
    train(parser.parse_args())
