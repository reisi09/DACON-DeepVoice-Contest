"""CPU-only, previously exposed cohort diagnostics; never a deployment selector.

Run with the pinned project Python: ads_cached_experiments.py --help.
The three heads share exactly the same train-only standardized pooled features.
"""
from __future__ import annotations

import argparse
from collections import Counter
import csv
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import resource
import sys
import time
from typing import Literal, assert_never

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from eight_fixed_io import atomic_json
from full_bank_features import load_rows, row_identity, validate_cache
from graph_scoring import LABEL_COLUMNS, PREDICTION_COLUMNS, official_metrics
from localization_models import WindowMIL
import decision_heads
from v5_fusion import joint_partial_label_loss, joint_probabilities
from v6_data import sha256

Arm = Literal['independent5', 'shared5', 'joint9']


def peak_rss_gib() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (2**30 if sys.platform == 'darwin' else 2**20)


def check_limits(deadline: float) -> None:
    if time.monotonic() >= deadline:
        raise TimeoutError('CPU diagnostic deadline reached')
    if peak_rss_gib() > 18:
        raise MemoryError('CPU diagnostic exceeded 18 GiB RSS limit')


@dataclass(frozen=True, slots=True)
class Cohort:
    rows: list[dict[str, str]]
    features: torch.Tensor
    labels: torch.Tensor
    presence_names: tuple[str, ...]
    cache_paths: tuple[Path, ...] = ()


def load_cohort(args: argparse.Namespace, deadline: float = float('inf')) -> Cohort:
    """Verify original cache contracts and current lineage before pooling windows."""
    rows = load_rows(args.manifest)
    union = {r['ID']: r for r in load_rows(args.feature_manifest)}
    lineage = {r['ID']: r for r in map(json.loads, args.lineage.read_text().splitlines())}
    index = json.loads((args.cache / 'index.json').read_text())
    if index['identity']['manifest_sha256'] != sha256(args.feature_manifest):
        raise ValueError('Feature union hash mismatch')
    for name, digest in index['identity']['implementation'].items():
        if sha256(Path(__file__).parent / name) != digest:
            raise ValueError('Feature producer changed: ' + name)
    key = hashlib.sha256(json.dumps(index['identity'], sort_keys=True).encode()).hexdigest()
    seen, features, labels, paths, presence_names = {}, [], [], [], ()
    for row in rows:
        check_limits(deadline)
        original = lineage[row['orig_id']]
        if (union.get(row['ID']) != row or original['role'] != row['role']
                or original['component_id'] != row['component_id']
                or original['verified_audio']['sha256'] != row['audio_sha256']):
            raise ValueError('Manifest/lineage mismatch: ' + row['ID'])
        if row['role'] not in ('train', 'selection', 'calibration'):
            raise ValueError('This diagnostic accepts only declared train/selection/calibration roles')
        for group in (*original['group_keys'], row['component_id']):
            if group in seen and seen[group] != row['role']:
                raise ValueError('Source lineage crosses data roles')
            seen[group] = row['role']
        target = []
        for name in LABEL_COLUMNS:
            known = row.get(name + '_mask', '1') == '1' and row.get(name, '') != ''
            value = float(row[name]) if known else np.nan
            if known != (original[name] is not None) or known and value != original[name]:
                raise ValueError('Current labels differ from cached labels')
            target.append(value)
        for fake, present in ((1, 3), (2, 4)):
            if target[present] != 1:
                target[fake] = np.nan
        labels.append(target)
        entry = index['items'][row['ID']]
        path = (args.cache / entry['path']).resolve()
        if not path.is_relative_to(args.cache.resolve()):
            raise ValueError('Cache path escapes its directory')
        validate_cache(path, row_identity(row, key), entry['sha256'])
        paths.append(path)
        with np.load(path, allow_pickle=False) as data:
            stats = json.loads(str(data['presence_features']))
            names = tuple(sorted(stats))
            presence_names = presence_names or names
            if names != presence_names or not len(data['mert_windows']):
                raise ValueError('Missing or inconsistent pooled feature evidence')
            features.append(np.concatenate((data['voice_embeddings'].mean(0),
                data['mert_windows'].mean(0), [stats[name] for name in names])).astype(np.float32))
    matrix = np.stack(features)
    if not np.isfinite(matrix).all():
        raise ValueError('Nonfinite pooled features')
    return Cohort(rows, torch.from_numpy(matrix), torch.tensor(labels), presence_names, tuple(paths))


def load_legacy_cohort(args: argparse.Namespace, deadline: float) -> Cohort:
    """Authenticate old cache rows first; apply current lineage and roles only afterwards."""
    producer = '2ac3a1c133176a7e7ccfa0963a9f00bf6cc3be42cd8e7eddec0f53ff89143f67'
    if sha256(args.legacy_producer) != producer:
        raise ValueError('Legacy producer source does not match the archived cache')
    lock = json.loads(args.reference.read_text())
    current = {r['ID']: r for r in map(json.loads, args.lineage.read_text().splitlines())}
    seen, rows, features, labels, paths, presence_names = {}, [], [], [], [], ()
    specifications = (
        ('cache_tv', 'bank_trainval.csv', '4f427e667cc0deb27d49820fb87a4a2c74edd90496f2e3d7efd20fa7f8641f9f'),
        ('cache_test', 'bank_test_as_val.csv', 'daa2a8bb9ca526be61e225a1c4fac9be2ecc52da1f75cf6d20aced34a52b7129'))
    for directory, manifest_name, digest in specifications:
        cache = args.legacy_cache_root / directory
        manifest = args.legacy_bank_dir / manifest_name
        if sha256(cache / 'index.json') != digest:
            raise ValueError('Legacy cache index differs from the recovered archive')
        index = json.loads((cache / 'index.json').read_text())
        identity = index['identity']
        implementations = {'full_bank_features.py': producer, **{name:
            lock['members']['model/runtime/' + name] for name in
            ('decision_heads.py', 'mert12_submission.py', 'v6_runtime.py')}}
        expected = dict(manifest_sha256=sha256(manifest), implementation=implementations,
            assets=identity['assets'], dtype='float32', duration='all samples, no crop',
            hop=32300, mert_layer=12, torch='2.7.1+cu128', window=64600)
        if identity != expected or any(lock['members'].get('model/assets/' + name) != value
                                      for name, value in identity['assets'].items()):
            raise ValueError('Legacy producer/assets/frontend contract differs')
        with manifest.open(newline='') as stream:
            original_rows = list(csv.DictReader(stream))
        if len(original_rows) != len(index['items']) or {r['ID'] for r in original_rows} != set(index['items']):
            raise ValueError('Legacy manifest/cache coverage mismatch')
        key = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
        for old in original_rows:
            check_limits(deadline)
            entry = index['items'][old['ID']]
            path = (cache / entry['path']).resolve()
            if not path.is_relative_to(cache.resolve()) or entry['identity'] != row_identity(old, key):
                raise ValueError('Legacy cache path or original row identity mismatch')
            validate_cache(path, row_identity(old, key), entry['sha256'])
            source = current[old['orig_id']]
            if source['verified_audio']['sha256'] != old['audio_sha256']:
                raise ValueError('Legacy/current audio identity differs')
            for name in LABEL_COLUMNS:
                known = old.get(name + '_mask', '1') not in ('0', '0.0', 'False') and old.get(name, '') != ''
                if known != (source[name] is not None) or known and float(old[name]) != source[name]:
                    raise ValueError('Legacy/current label knownness or value differs')
            if source['role'] not in ('train', 'selection', 'calibration', 'development'):
                raise ValueError('Unsupported current role in legacy cache')
            for group in (*source['group_keys'], source['component_id']):
                if group in seen and seen[group] != source['role']:
                    raise ValueError('Current source lineage crosses roles')
                seen[group] = source['role']
            target = [float(source[name]) if source[name] is not None else np.nan for name in LABEL_COLUMNS]
            row = dict(old, role=source['role'], component_id=source['component_id'])
            for name in LABEL_COLUMNS:
                row[name] = str(int(source[name])) if source[name] is not None else ''
                row[name + '_mask'] = str(int(source[name] is not None))
            for fake, present in ((1, 3), (2, 4)):
                if target[present] != 1:
                    target[fake] = np.nan
            with np.load(path, allow_pickle=False) as data:
                stats = json.loads(str(data['presence_features']))
                names = tuple(sorted(stats))
                presence_names = presence_names or names
                if names != presence_names or len(names) != 24 or not len(data['mert_windows']):
                    raise ValueError('Legacy pooled feature schema differs')
                features.append(np.concatenate((data['voice_embeddings'].mean(0),
                    data['mert_windows'].mean(0), [stats[name] for name in names])).astype(np.float32))
            rows.append(row)
            labels.append(target)
            paths.append(path)
            if len(rows) % 4096 == 0:
                print(json.dumps(dict(verified_legacy_rows=len(rows))), flush=True)
    if len({r['ID'] for r in rows}) != len(rows) or len({r['orig_id'] for r in rows}) != len(rows):
        raise ValueError('Duplicate legacy audio rows across caches')
    matrix = np.stack(features)
    if not np.isfinite(matrix).all():
        raise ValueError('Nonfinite legacy features')
    print(json.dumps(dict(verified_legacy_rows=len(rows), roles=dict(Counter(r['role'] for r in rows)),
                         load_seconds_remaining=deadline-time.monotonic())), flush=True)
    return Cohort(rows, torch.from_numpy(matrix), torch.tensor(labels), presence_names, tuple(paths))


class Heads(nn.Module):
    """Approximately parameter-matched independent versus shared nonlinear heads."""
    def __init__(self, width: int, arm: Arm) -> None:
        super().__init__()
        self.arm = arm
        match arm:
            case 'independent5':
                self.heads = nn.ModuleList([nn.Sequential(nn.Linear(width, 64), nn.GELU(),
                                                        nn.Linear(64, 1)) for _ in range(5)])
            case 'shared5' | 'joint9':
                self.heads = nn.ModuleList([nn.Sequential(nn.Linear(width, 320), nn.GELU(),
                    nn.Linear(320, 9 if arm == 'joint9' else 5))])
            case unreachable:
                assert_never(unreachable)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return torch.cat([head(features) for head in self.heads], dim=1)


def bce(logits: torch.Tensor, targets: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    """Five-task mean with fixed train class weights and no unknown/absent loss."""
    known = torch.isfinite(targets)
    truth = targets.nan_to_num()
    per_class = torch.where(truth == 1, weights[:, 1], weights[:, 0])
    return (F.binary_cross_entropy_with_logits(logits, truth, reduction='none') * known * per_class).mean()


def probabilities(model: Heads, features: torch.Tensor) -> torch.Tensor:
    logits = model(features)
    return joint_probabilities(logits).clamp(0, 1) if model.arm == 'joint9' else logits.sigmoid()


def report_predictions(data: Cohort, prediction: np.ndarray, output: Path) -> dict:
    np.savez_compressed(output / 'predictions.npz', ID=np.array([r['ID'] for r in data.rows]),
                        probabilities=prediction)
    metrics = {}
    for split in ('selection', 'calibration', 'development'):
        selected = [i for i, row in enumerate(data.rows) if row['role'] == split]
        if not selected:
            continue
        truth = [data.rows[i] for i in selected]
        predicted = [dict(ID=data.rows[i]['ID'], **dict(zip(PREDICTION_COLUMNS,
                     map(float, prediction[i]), strict=True))) for i in selected]
        report = official_metrics(truth, predicted)
        values = report['metrics']
        if all(values.get(name) is not None for name in ('file_eer', 'voice_eer', 'music_eer')):
            report['diagnostic_ads_known_subsets'] = sum(weight * (1 - values[name]) for weight, name in
                ((.5, 'file_eer'), (.2, 'voice_eer'), (.3, 'music_eer')))
        if all(values.get(name) is not None for name in ('voice_presence_auc', 'music_presence_auc')):
            report['diagnostic_cps_known_subsets'] = .5 * (values['voice_presence_auc'] + values['music_presence_auc'])
        metrics[split] = report
    return metrics


def preserved_scores(data: Cohort, args: argparse.Namespace, deadline: float) -> np.ndarray:
    """Replay only verified small heads; static compatibility is not raw-runtime parity."""
    lock = json.loads(args.reference.read_text())
    package = Path(lock['package'])
    runtime = package / 'model/runtime/full_bank_submission.py'
    if sha256(runtime) != 'f0b14ebe7ac0050d77b5fe7c40a69318105ca378ba1f9c844d321b076a9f3211':
        raise ValueError('Reference pooling/frontend runtime differs from the audited legacy recipe')
    caches = ([args.legacy_cache_root / name for name in ('cache_tv', 'cache_test')]
              if args.legacy_cache_root else [args.cache])
    for cache in caches:
        index = json.loads((cache / 'index.json').read_text())
        for name, digest in index['identity']['assets'].items():
            if lock['members'].get('model/assets/' + name) != digest:
                raise ValueError('Reference/cache model assets differ: ' + name)
    for name in ('mert12_submission.py', 'v6_runtime.py', 'decision_heads.py',
                 'localization_models.py', 'eight_heads.py'):
        if lock['members'].get('model/runtime/' + name) != sha256(Path(__file__).with_name(name)):
            raise ValueError('Reference/cache model code differs: ' + name)
    trained = package / 'model/trained'
    for path in trained.iterdir():
        if path.is_file() and sha256(path) != lock['members'].get(str(path.relative_to(package))):
            raise ValueError('Reference head artifact changed: ' + path.name)
    presence_head = json.loads((trained / 'presence_head.json').read_text())
    ensemble = json.loads((trained / 'file_ensemble.json').read_text())
    music_heads, voice_heads, fake_indices = [], [], set()
    for seed in (41, 42, 43):
        for route, heads in (('music', music_heads), ('voice', voice_heads)):
            record = torch.load(trained / f'{route}_{seed}.pt', map_location='cpu', weights_only=True)
            if record['route'] != route or record['seed'] != seed:
                raise ValueError('Reference route/seed mismatch')
            state = record['model']
            model = WindowMIL(state['mean'], state['scale']) if route == 'music' else nn.Linear(1280, 2)
            model.load_state_dict(state, strict=True)
            heads.append(model.eval().requires_grad_(False))
            fake_indices.add(record['fake_label_index'])
    if fake_indices != {0}:
        raise ValueError('Reference fake label mapping differs')
    predicted = []
    with torch.inference_mode():
        for row, path in zip(data.rows, data.cache_paths, strict=True):
            check_limits(deadline)
            with np.load(path, allow_pickle=False) as cached:
                music = torch.from_numpy(cached['mert_windows'])
                voice = torch.from_numpy(cached['voice_embeddings'])
                mf = []
                for head in music_heads:
                    logits, _ = head.head((music - head.mean) / head.scale)
                    mf.append(float((logits.logsumexp(0) - np.log(len(music))).sigmoid()))
                vf = 0. if cached['voice_silent'] else float(np.mean([
                    float(head(voice).softmax(-1)[:, 0].max()) for head in voice_heads]))
                vp, mp = decision_heads.presence(presence_head, json.loads(str(cached['presence_features'])),
                                               tuple(map(float, cached['presence'])))
                values = dict(Pv=vp, Pm=mp, Fv=vf, Fm=float(np.mean(mf)))
                predicted.append([decision_heads.file_score(ensemble, values), vf, values['Fm'], vp, mp])
            if len(predicted) % 4096 == 0:
                print(json.dumps(dict(preserved_head_rows=len(predicted))), flush=True)
    return np.asarray(predicted)


def profile_training(data: Cohort, indices: dict[str, torch.Tensor], deadline: float,
                     max_epochs: int) -> dict:
    train, selection = indices['train'], indices['selection']
    weights = len(train) / (2 * torch.stack([(data.labels[train] == label).sum(0)
                                            for label in (0, 1)], dim=1).float())
    measurements = []
    for arm in ('independent5', 'shared5', 'joint9'):
        torch.manual_seed(41)
        model = Heads(data.features.shape[1], arm)
        optimizer = torch.optim.AdamW(model.parameters(), lr=.001, weight_decay=.0001)
        order = train[torch.randperm(len(train), generator=torch.Generator().manual_seed(41))]
        times = []
        for i, batch in enumerate(order.split(128)):
            if i == 10:
                break
            check_limits(deadline)
            started = time.monotonic()
            optimizer.zero_grad(set_to_none=True)
            logits = model(data.features[batch])
            loss = (joint_partial_label_loss(logits, data.labels[batch]) if arm == 'joint9'
                    else bce(logits, data.labels[batch], weights))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
            optimizer.step()
            if i >= 2:
                times.append(time.monotonic() - started)
        with torch.inference_mode():
            started = time.monotonic()
            probabilities(model, data.features[selection])
            selection_seconds = time.monotonic() - started
            started = time.monotonic()
            probabilities(model, data.features)
            final_seconds = time.monotonic() - started
        measurements.append(dict(arm=arm, timed_update_seconds=times,
            parameters=sum(p.numel() for p in model.parameters()), selection_seconds=selection_seconds,
            final_inference_seconds=final_seconds,
            epoch_seconds=max(times) * math.ceil(len(train)/128) + selection_seconds))
    epoch_seconds_all_conditions = 3 * sum(m['epoch_seconds'] for m in measurements)
    final_seconds = 3 * sum(m['final_inference_seconds'] for m in measurements) + 60
    remaining = deadline - time.monotonic()
    common_epochs = min(max_epochs, math.floor((remaining * .8 - final_seconds) / epoch_seconds_all_conditions))
    if common_epochs < 1:
        raise TimeoutError('Insufficient remaining CPU budget for one common full epoch')
    return dict(measurements=measurements, common_epochs=common_epochs,
        all_conditions_epoch_seconds=epoch_seconds_all_conditions,
        remaining_seconds=remaining, reserve_fraction=.2, final_and_io_reserve_seconds=final_seconds,
        quality_metrics_used=False, profile_parameters_discarded=True)


def file_fusion(data: Cohort, prediction: np.ndarray, output: Path, deadline: float) -> dict:
    """Train-only convex logistic FILE head; frozen components remain bit-identical."""
    component = prediction[:, 1:]
    x = np.column_stack([component] + [component[:, i] * component[:, j]
                                       for i in range(4) for j in range(i + 1, 4)])
    train = np.array([r['role'] == 'train' for r in data.rows]) & np.isfinite(data.labels[:, 0].numpy())
    mean, scale = x[train].mean(0), np.maximum(x[train].std(0), 1e-6)
    x = torch.tensor((x - mean) / scale, dtype=torch.float64)
    y = data.labels[train, 0].double()
    if not torch.isfinite(y).all() or set(y.tolist()) != {0., 1.}:
        raise ValueError('FILE fitting requires known labels and both training classes')
    model = nn.Linear(x.shape[1], 1, dtype=torch.float64)
    with torch.no_grad():
        model.weight.zero_()
        model.bias.zero_()
    weight = torch.where(y == 1, len(y) / (2 * (y == 1).sum()), len(y) / (2 * (y == 0).sum()))
    optimizer = torch.optim.LBFGS(model.parameters(), max_iter=100, line_search_fn='strong_wolfe')
    calls = []
    def closure() -> torch.Tensor:
        check_limits(deadline)
        optimizer.zero_grad(set_to_none=True)
        loss = (F.binary_cross_entropy_with_logits(model(x[train]).squeeze(1), y, reduction='none') * weight).mean()
        loss = loss + .001 * model.weight.square().sum() / 2
        loss.backward()
        calls.append(float(loss.detach()))
        return loss
    optimizer.step(closure)
    with torch.inference_mode():
        result = prediction.copy()
        result[:, 0] = model(x).squeeze(1).sigmoid().numpy()
    if not np.array_equal(result[:, 1:], prediction[:, 1:]) or not torch.isfinite(model.weight).all():
        raise RuntimeError('FILE-only fit changed components or has nonfinite weights')
    torch.save({'model': model.state_dict(), 'mean': torch.tensor(mean), 'scale': torch.tensor(scale)}, output / 'model.pt')
    report = dict(metrics=report_predictions(data, result, output), closure_calls=len(calls),
        initial_loss=calls[0], final_loss=calls[-1], max_weight=float(model.weight.detach().abs().max()),
        limitation='Upstream preserved heads saw training rows: train scores are not OOF; diagnostic only')
    atomic_json(output / 'report.json', report)
    return report


def train_arm(data: Cohort, indices: dict[str, torch.Tensor], config: dict,
              output: Path, deadline: float) -> dict:
    """Full shuffled train passes; selection BCE alone chooses a saved checkpoint."""
    arm, seed = config['arm'], config['seed']
    torch.manual_seed(seed)
    rng = torch.Generator().manual_seed(seed)
    x, y = data.features, data.labels
    train, selection = indices['train'], indices['selection']
    counts = torch.stack([(y[train] == label).sum(0) for label in (0, 1)], dim=1)
    if (counts == 0).any():
        raise ValueError('Every task needs both training classes')
    weights = len(train) / (2 * counts.float())
    model = Heads(x.shape[1], arm)
    optimizer = torch.optim.AdamW(model.parameters(), lr=.001, weight_decay=.0001)
    best, best_state, history, audit = float('inf'), None, [], {}
    updates, epochs = 0, 0
    sampler = hashlib.sha256()
    start = time.monotonic()
    for epoch in range(config['epochs']):
        if peak_rss_gib() > 18:
            raise MemoryError('CPU diagnostic exceeded 18 GiB RSS limit')
        order = train[torch.randperm(len(train), generator=rng)]
        seen = []
        for batch in order.split(128):
            if time.monotonic() >= deadline:
                break
            optimizer.zero_grad(set_to_none=True)
            logits = model(x[batch])
            loss = (joint_partial_label_loss(logits, y[batch]) if arm == 'joint9'
                    else bce(logits, y[batch], weights))
            if not torch.isfinite(loss):
                raise FloatingPointError('Nonfinite training loss')
            loss.backward()
            pending = {}
            for name, p in model.named_parameters():
                if name not in audit and p.grad is not None:
                    norm = float(p.grad.norm())
                    if not np.isfinite(norm):
                        raise RuntimeError('Nonfinite gradient: ' + name)
                    if norm > 0:
                        pending[name] = (p.detach().clone(), norm)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
            optimizer.step()
            for name, p in model.named_parameters():
                if name in pending:
                    before, norm = pending[name]
                    delta = float((p.detach() - before).abs().max())
                    if delta > 0:
                        audit[name] = {'gradient_norm': norm, 'max_update': delta, 'update': updates + 1}
            seen.extend(batch.tolist())
            sampler.update(batch.numpy().tobytes())
            updates += 1
        if len(seen) != len(train):
            break
        if sorted(seen) != sorted(train.tolist()):
            raise RuntimeError('Training pass duplicated or omitted a row')
        missing = set(dict(model.named_parameters())) - set(audit)
        if missing:
            raise RuntimeError('No eligible gradient/update in completed pass: ' + ', '.join(sorted(missing)))
        epochs += 1
        with torch.inference_mode():
            pred = probabilities(model, x[selection]).clamp(1e-7, 1 - 1e-7)
            validation = float(bce(torch.logit(pred), y[selection], weights))
        history.append({'epoch': epochs, 'selection_bce': validation, 'train_rows': len(seen)})
        if epochs == 1 or epochs % 5 == 0:
            print(json.dumps(dict(arm=arm, seed=seed, epoch=epochs, epochs=config['epochs'])), flush=True)
        if validation < best:
            best, best_epoch = validation, epochs
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    record = dict(config, completed=epochs == config['epochs'], epochs_completed=epochs,
        updates=updates, seconds=time.monotonic() - start, parameters=sum(p.numel() for p in model.parameters()),
        sampler_sha256=sampler.hexdigest(), gradient_and_update_audit=audit, history=history)
    torch.save({'model': model.state_dict(), 'optimizer': optimizer.state_dict(),
                'rng': rng.get_state(), 'record': record}, output / 'last.pt')
    if best_state is not None:
        model.load_state_dict(best_state)
        torch.save({'model': best_state, 'config': config, 'best_epoch': best_epoch}, output / 'best.pt')
        with torch.inference_mode():
            prediction = probabilities(model, x).numpy()
        record['best_epoch'] = best_epoch
        record['metrics'] = report_predictions(data, prediction, output)
    atomic_json(output / 'report.json', record)
    return record


def run(args: argparse.Namespace) -> None:
    if not 0 < args.minutes <= 30 or not 1 <= args.epochs <= 20:
        raise ValueError('CPU diagnostic bounds: 0 < minutes <= 30 and 1 <= epochs <= 20')
    legacy = bool(args.legacy_cache_root)
    if legacy and (not all((args.legacy_bank_dir, args.legacy_producer, args.reference)) or
                   any((args.manifest, args.feature_manifest, args.cache))):
        raise ValueError('Legacy mode requires bank directory, producer and reference, with no current-cache arguments')
    if not legacy and (not all((args.manifest, args.feature_manifest, args.cache)) or
                       any((args.legacy_bank_dir, args.legacy_producer))):
        raise ValueError('Current-cache mode requires manifest, feature manifest and cache')
    args.output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    deadline = started + args.minutes * 60
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    data = load_legacy_cohort(args, deadline) if legacy else load_cohort(args, deadline)
    indices = {split: torch.tensor([i for i, row in enumerate(data.rows) if row['role'] == split])
               for split in ('train', 'selection', 'calibration', 'development')
               if any(row['role'] == split for row in data.rows)}
    if any(not len(indices.get(split, [])) for split in ('train', 'selection', 'calibration')):
        raise ValueError('All three declared roles must be nonempty')
    mean = data.features[indices['train']].mean(0)
    scale = data.features[indices['train']].std(0, correction=0).clamp_min(1e-6)
    data = Cohort(data.rows, (data.features - mean) / scale, data.labels, data.presence_names, data.cache_paths)
    torch.save({'mean': mean, 'scale': scale, 'presence_names': data.presence_names}, args.output / 'normalizer.pt')
    states = []
    for values in data.labels.tolist():
        state = [0 if values[p] == 0 else int(values[f]) + 1
                 if values[p] == 1 and np.isfinite(values[f]) else -1 for f, p in ((1, 3), (2, 4))]
        states.append(f'{state[0]}/{state[1]}' if -1 not in state else 'partial')
    input_paths = dict(lineage=args.lineage, runner=Path(__file__), joint=Path(__file__).with_name('v5_fusion.py'))
    if legacy:
        input_paths.update(legacy_producer=args.legacy_producer,
            cache_tv=args.legacy_cache_root/'cache_tv/index.json', cache_test=args.legacy_cache_root/'cache_test/index.json',
            bank_trainval=args.legacy_bank_dir/'bank_trainval.csv', bank_test_as_val=args.legacy_bank_dir/'bank_test_as_val.csv')
    else:
        input_paths.update(manifest=args.manifest, feature_manifest=args.feature_manifest, index=args.cache/'index.json')
    available = {row['orig_id'] for row in data.rows}
    missing_roles, missing_noise, current_count = Counter(), 0, 0
    with args.lineage.open() as stream:
        for line in stream:
            current = json.loads(line)
            current_count += 1
            if current['ID'] not in available:
                missing_roles[current['role']] += 1
                missing_noise += int(current['VOICE_PRESENT'] == 0 and current['MUSIC_PRESENT'] == 0)
    config = dict(arms=['independent5', 'shared5', 'joint9'], seeds=[41, 42, 43], requested_epochs=args.epochs,
        minutes=args.minutes, device='cpu', threads=4, batch=128, lr=.001, weight_decay=.0001,
        feature_policy='voice window mean + backing MERT window mean + sorted 24 presence statistics',
        input_hashes={name: sha256(path) for name, path in input_paths.items()},
        cache_contract='explicit authenticated historical two-cache' if legacy else 'current full-bank cache',
        evaluated_cohort_rows=len(data.rows), current_D0_rows=current_count,
        missing_current_rows_by_role=dict(missing_roles), missing_current_D0_noise_rows=missing_noise,
        role_counts={name: len(index) for name, index in indices.items()},
        class_counts={name: {LABEL_COLUMNS[c]: {'negative': int((data.labels[index, c] == 0).sum()),
            'positive': int((data.labels[index, c] == 1).sum())} for c in range(5)} for name, index in indices.items()},
        joint_states={name: dict(Counter(states[i] for i in index)) for name, index in indices.items()},
        joint_state_order='voice/music: 0=absent, 1=real, 2=fake; partial=not fully known',
        selection='minimum fixed-train-weight balanced five-task BCE, selection role only',
        independent_test=False, adoption_allowed=False,
        file_fusion='fixed 4 components + six pair interactions; train zscore; balanced logistic BCE + .001 L2; LBFGS100',
        reference_sha256=sha256(args.reference) if args.reference else None,
        limitations=['Previously exposed development material; no independent quality claim',
            'Unrepresented joint states and small music subsets must be checked against class counts',
            'Mean pooling discards temporal information; this is not the full MERT/WindowMIL architecture',
            'Legacy reference feature contract is not certified here; no previous-best baseline claim'])
    atomic_json(args.output / 'protocol.json', config)
    baseline = None
    prediction = preserved_scores(data, args, deadline) if args.reference else None
    profile = profile_training(data, indices, deadline, args.epochs)
    config['epochs'] = profile['common_epochs']
    atomic_json(args.output / 'profile.json', profile)
    atomic_json(args.output / 'predeclared.json', config)
    print(json.dumps(dict(common_epochs=config['epochs'], profile=profile)), flush=True)
    if args.reference:
        directory = args.output / 'preserved_heads'
        directory.mkdir()
        baseline = dict(metrics=report_predictions(data, prediction, directory),
            claim='Preserved heads on statically compatible cached features, not certified raw-runtime parity',
            reference_equivalence_to_leaderboard_best=False, historical_cdf_exposure=True)
        atomic_json(directory / 'report.json', baseline)
        directory = args.output / 'file_logistic'
        directory.mkdir()
        file_fusion(data, prediction, directory, deadline)
    records = []
    for seed in config['seeds']:
        for arm in config['arms']:
            if time.monotonic() >= deadline:
                break
            output = args.output / f'{arm}_{seed}'
            output.mkdir()
            result = train_arm(data, indices, dict(arm=arm, seed=seed, epochs=config['epochs']), output, deadline)
            records.append(result)
            print(json.dumps({k: result[k] for k in ('arm', 'seed', 'completed', 'epochs_completed', 'seconds')}), flush=True)
    atomic_json(args.output / 'summary.json', dict(predeclared_sha256=sha256(args.output/'predeclared.json'),
        seconds=time.monotonic()-started, peak_rss_gib=peak_rss_gib(), complete=len(records) == 9 and all(r['completed'] for r in records),
        arms=records, baseline=baseline, winner=None, quality_evidence='diagnostic only; no automatic adoption'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('manifest', 'feature-manifest', 'cache', 'legacy-cache-root', 'legacy-bank-dir', 'legacy-producer', 'lineage', 'output'):
        parser.add_argument('--' + name, type=Path, required=name in ('lineage', 'output'))
    parser.add_argument('--minutes', type=float, default=20.)
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--reference', type=Path)
    run(parser.parse_args())
