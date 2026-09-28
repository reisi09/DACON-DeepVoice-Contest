"""Matched, deadline-bounded cached MERT/MIL/AST comparison on one GPU.

Run with the existing project Python; ``--help`` documents the cache/paths.
AST cache generation is separate, and every candidate is locked before holdout use.
"""
# noqa: SIZE_OK - One bounded experiment entrypoint; no reusable training framework.
from __future__ import annotations

import argparse
from collections import Counter
from collections.abc import Mapping, Sequence
from contextlib import nullcontext
from dataclasses import dataclass
import csv
import gc
import hashlib
import json
from pathlib import Path
import time
from typing import TypeAlias

import numpy as np
import torch
from torch.nn import functional as F

from eight_heads import balanced_draws
from eight_contract import sha256_file
from graph_scoring import _roc_metrics, official_metrics
from localization_models import (ASTLocalizer, FrozenMeanMLP, WindowMIL,
                                 frame_targets_to_patches, masked_logmeanexp)

Json: TypeAlias = None | bool | int | float | str | list['Json'] | dict[str, 'Json']
Model: TypeAlias = FrozenMeanMLP | WindowMIL | ASTLocalizer


class TrainingContractError(ValueError):
    """The cache or comparison violates its registered protocol."""


def write_json(path: Path, payload: Json) -> None:
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(payload, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def check_deadline(deadline: float) -> None:
    if time.time() >= deadline:
        raise TimeoutError('The immutable training deadline has elapsed')


@dataclass(frozen=True, slots=True)
class Item:
    row: Mapping[str, str]
    mert: np.ndarray
    frames: np.ndarray
    frame_mask: np.ndarray
    window_truth: np.ndarray
    patch_truth: np.ndarray
    prefix: Path
    frozen: Mapping[str, float]


def read_items(manifest: Path, cache: Path, ast_assets: Path | None = None) -> list[Item]:
    """Parse truth without filling unknowns, and reject unsafe/misaligned caches."""
    with manifest.open(newline='') as stream:
        rows = list(csv.DictReader(stream))
    receipt = json.loads((cache / 'index.json').read_text())
    if receipt['identity']['manifest_sha256'] != sha256_file(manifest):
        raise TrainingContractError('Feature cache was generated from a different manifest')
    for name in ('localization_models.py', 'ast_multitask.py', 'mert12_submission.py'):
        if receipt['identity']['implementation_sha256'][name] != sha256_file(Path(__file__).parent / name):
            raise TrainingContractError('Feature-generating implementation changed before training')
    index = receipt['items']
    expected_prefix = None
    if ast_assets is not None:
        if set(receipt['identity']['ast_assets_sha256']) != {'config.json', 'preprocessor_config.json', 'model.safetensors'}:
            raise TrainingContractError('Incomplete AST asset binding')
        for name, digest in receipt['identity']['ast_assets_sha256'].items():
            if name not in ('config.json', 'preprocessor_config.json', 'model.safetensors') or sha256_file(ast_assets / name) != digest:
                raise TrainingContractError('AST assets differ from the frozen-prefix cache')
        config = json.loads((ast_assets / 'config.json').read_text())
        patch = config.get('patch_size', 16)
        time_patches = (config.get('max_length', 1024) - patch) // config.get('time_stride', 10) + 1
        frequency_patches = (config.get('num_mel_bins', 128) - patch) // config.get('frequency_stride', 10) + 1
        expected_prefix = (2 + time_patches * frequency_patches, config.get('hidden_size', 768))
    if len({row['ID'] for row in rows}) != len(rows):
        raise TrainingContractError('Duplicate manifest IDs')
    result = []
    for row in rows:
        entry = index[row['ID']]
        paths = [cache / entry[name] for name in ('features', 'ast_prefix')]
        if any(not path.resolve().is_relative_to(cache.resolve()) for path in paths):
            raise TrainingContractError('Cache entry escapes the cache directory')
        if any(sha256_file(path) != entry[key] for path, key in zip(paths, ('features_sha256', 'ast_prefix_sha256'), strict=True)):
            raise TrainingContractError(f"Feature cache bytes changed: {row['ID']}")
        with np.load(paths[0], allow_pickle=False) as data:
            mert = data['mert_windows'].astype(np.float32)
            frames = data['frame_targets'].astype(np.float32)
            mask = data['frame_mask'].astype(bool)
            if (mert.ndim != 2 or not len(mert) or not np.isfinite(mert).all()
                    or frames.shape != mask.shape or frames.shape != (len(mert), 1024)
                    or not mask.any(1).all()):
                raise TrainingContractError(f"Invalid aligned features: {row['ID']}")
            known = np.isfinite(frames) & (frames >= 0) & (frames <= 1)
            if (mask & ~known & ~np.isnan(frames) & (frames != -1)).any():
                raise TrainingContractError(f"Invalid interval truth: {row['ID']}")
            truth = np.where(known & mask, frames, 0).sum(1) / mask.sum(1)
            truth[~(known | ~mask).all(1)] = np.nan
            patches = frame_targets_to_patches(torch.from_numpy(frames), torch.from_numpy(mask)).numpy()
            frozen = {name: float(data[name].item()) for name in
                      ('VOICE_FAKE_PROB', 'VOICE_PRESENT_PROB', 'MUSIC_PRESENT_PROB') if name in data}
        prefix = np.load(paths[1], mmap_mode='r', allow_pickle=False)
        if (prefix.ndim != 3 or len(prefix) != len(mert) or prefix.dtype != np.float16
                or (expected_prefix is not None and prefix.shape[1:] != expected_prefix)):
            raise TrainingContractError(f"Invalid prefix cache: {row['ID']}")
        result.append(Item(row, mert, frames, mask, truth, patches, paths[1], frozen))
    return result


def chunks(items: Sequence[Item], batch_size: int, window_limit: int) -> list[list[Item]]:
    """Keep long evaluation bags below the benchmarked number of AST windows."""
    groups: list[list[Item]] = []
    for item in items:
        if len(item.mert) > window_limit:
            raise TrainingContractError('A single file exceeds the benchmarked window budget')
        if not groups or len(groups[-1]) >= batch_size or sum(len(x.mert) for x in groups[-1]) + len(item.mert) > window_limit:
            groups.append([])
        groups[-1].append(item)
    return groups


def forward_batch(model: Model, items: Sequence[Item], device: str) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return file logits and flattened eligible local logits/truth."""
    sizes = [len(item.mert) for item in items]
    valid = torch.arange(max(sizes), device=device)[None] < torch.tensor(sizes, device=device)[:, None]
    if isinstance(model, ASTLocalizer):
        hidden = torch.from_numpy(np.concatenate([np.load(item.prefix, mmap_mode='r') for item in items])).to(device)
        frame_mask = torch.from_numpy(np.concatenate([item.frame_mask for item in items])).to(device)
        output = model.forward_encoded(hidden, frame_mask)
        windows = torch.nn.utils.rnn.pad_sequence(output.file_logits.split(sizes), batch_first=True)
        logits = masked_logmeanexp(windows, valid)
        truth = torch.from_numpy(np.concatenate([item.patch_truth for item in items])).to(device)
        known = torch.isfinite(truth) & output.patch_mask
        return logits, output.patch_logits[known], truth[known]
    features = torch.nn.utils.rnn.pad_sequence([torch.from_numpy(item.mert) for item in items], batch_first=True).to(device)
    output = model(features, valid)
    truth = torch.nn.utils.rnn.pad_sequence([torch.from_numpy(item.window_truth) for item in items], batch_first=True, padding_value=float('nan')).to(device)
    known = torch.isfinite(truth) & valid
    return output.file_logits, output.window_logits[known], truth[known]


def local_count(items: Sequence[Item], arm: str) -> int:
    return sum(int(np.isfinite(item.patch_truth if arm == 'ast' else item.window_truth).sum()) for item in items)


def train_step(model: Model, optimizer: torch.optim.Optimizer, items: Sequence[Item], arm: str,
               microbatch: int, window_limit: int, device: str, deadline: float) -> float:
    """Normalize both losses over the effective batch, independent of microbatching."""
    optimizer.zero_grad(set_to_none=True)
    count = max(1, local_count(items, arm))
    loss_sum = 0.0
    for group in chunks(items, microbatch, window_limit):
        check_deadline(deadline)
        with torch.autocast('cuda', dtype=torch.bfloat16) if device == 'cuda' else nullcontext():
            logits, local_logits, local_truth = forward_batch(model, group, device)
            target = torch.tensor([float(item.row['MUSIC_FAKE']) for item in group], device=device)
            loss = F.binary_cross_entropy_with_logits(logits.float(), target, reduction='sum') / len(items)
            if arm != 'mean' and local_logits.numel():
                loss = loss + 0.25 * F.binary_cross_entropy_with_logits(local_logits.float(), local_truth.float(), reduction='sum') / count
        if not torch.isfinite(loss):
            raise TrainingContractError('Non-finite training loss')
        loss.backward()
        loss_sum += float(loss.detach())
    torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1, error_if_nonfinite=True)
    optimizer.step()
    return loss_sum


def parameters_to_save(model: Model) -> dict[str, torch.Tensor]:
    names = {name for name, parameter in model.named_parameters() if parameter.requires_grad}
    return {name: value.detach().cpu().clone() for name, value in model.state_dict().items()
            if not isinstance(model, ASTLocalizer) or name in names}


def restore(model: Model, state: Mapping[str, torch.Tensor]) -> None:
    result = model.load_state_dict(state, strict=False)
    trainable = {name for name, parameter in model.named_parameters() if parameter.requires_grad}
    if result.unexpected_keys or trainable.intersection(result.missing_keys):
        raise TrainingContractError('Checkpoint does not cover the exact trainable model')


def optimizer_for(model: Model) -> torch.optim.Optimizer:
    if isinstance(model, ASTLocalizer):
        groups = [{'params': [p for name, p in model.named_parameters() if p.requires_grad and not name.startswith('head.')], 'lr': 2e-5},
                  {'params': list(model.head.parameters()), 'lr': 1e-3}]
        return torch.optim.AdamW(groups, weight_decay=1e-2)
    return torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-2)


def benchmark(model: Model, items: Sequence[Item], arm: str, device: str, deadline: float,
              max_batch: int = 128) -> tuple[int, list[dict[str, Json]]]:
    """Trial real optimizer steps, then restore weights and RNG before training."""
    initial = parameters_to_save(model)
    cpu_rng = torch.get_rng_state()
    gpu_rng = torch.cuda.get_rng_state_all() if device == 'cuda' else []
    records = []
    largest = max(items, key=lambda item: len(item.mert))
    candidates = tuple(size for size in (32, 64, 128, 256) if size <= max_batch) if device == 'cuda' else (128,)
    try:
        for size in candidates:
            optimizer = optimizer_for(model)
            try:
                if device == 'cuda':
                    torch.cuda.reset_peak_memory_stats()
                for _ in range(2):
                    train_step(model, optimizer, [largest] * size, arm, size, size * len(largest.mert), device, deadline)
                if device == 'cuda':
                    torch.cuda.synchronize()
                started = time.perf_counter()
                for _ in range(3):
                    train_step(model, optimizer, [largest] * size, arm, size, size * len(largest.mert), device, deadline)
                if device == 'cuda':
                    torch.cuda.synchronize()
                elapsed = time.perf_counter() - started
                peak = torch.cuda.max_memory_allocated() if device == 'cuda' else 0
                safe = device != 'cuda' or peak <= torch.cuda.get_device_properties(0).total_memory * .8
                records.append({'files': size, 'windows_per_file': len(largest.mert), 'files_per_second': size * 3 / elapsed,
                                'peak_allocated_bytes': peak, 'safe': safe})
            except torch.OutOfMemoryError:
                records.append({'files': size, 'safe': False, 'oom': True})
            finally:
                del optimizer
                model.zero_grad(set_to_none=True)
                restore(model, initial)
                gc.collect()
                if device == 'cuda':
                    torch.cuda.empty_cache()
    finally:
        restore(model, initial)
        torch.set_rng_state(cpu_rng)
        if device == 'cuda':
            torch.cuda.set_rng_state_all(gpu_rng)
    safe = [record for record in records if record['safe']]
    if not safe:
        raise TrainingContractError('No memory-safe benchmark batch')
    fastest = max(float(record['files_per_second']) for record in safe)
    return max(int(record['files']) for record in safe if float(record['files_per_second']) >= .9 * fastest), records


@torch.inference_mode()
def predict(model: Model, items: Sequence[Item], microbatch: int, window_limit: int, device: str, deadline: float) -> np.ndarray:
    model.eval()
    outputs = []
    for group in chunks(items, microbatch, window_limit):
        check_deadline(deadline)
        with torch.autocast('cuda', dtype=torch.bfloat16) if device == 'cuda' else nullcontext():
            logits, _, _ = forward_batch(model, group, device)
        outputs.extend(logits.float().sigmoid().cpu().tolist())
    return np.asarray(outputs, dtype=np.float32)


def music_metrics(items: Sequence[Item], probabilities: np.ndarray) -> dict[str, Json]:
    eligible = [i for i, item in enumerate(items) if item.row.get('MUSIC_PRESENT') == '1' and item.row.get('MUSIC_FAKE') in ('0', '1')]
    targets = [int(items[i].row['MUSIC_FAKE']) for i in eligible]
    scores = np.asarray([probabilities[i] for i in eligible], dtype=np.float64)
    if not np.isfinite(scores).all() or np.any((scores < 0) | (scores > 1)):
        raise TrainingContractError('Predictions are outside finite [0,1]')
    eer, auc = _roc_metrics(targets, scores.tolist()) if len(set(targets)) == 2 else (None, None)
    scores = np.clip(scores, 1e-7, 1 - 1e-7)
    bce = float(np.mean(-np.asarray(targets) * np.log(scores) - (1 - np.asarray(targets)) * np.log1p(-scores))) if targets else None
    return {'music_eer': eer, 'music_auc': auc, 'music_bce': bce, 'known_eligible': len(targets),
            'rows': len(items), 'unknown_relevant': sum(item.row.get('MUSIC_PRESENT') == '' or (item.row.get('MUSIC_PRESENT') == '1' and item.row.get('MUSIC_FAKE') == '') for item in items),
            'real': targets.count(0), 'fake': targets.count(1), 'official_score': None}


def report_metrics(items: Sequence[Item], probabilities: np.ndarray) -> dict[str, Json]:
    report = {}
    for split in sorted({item.row['split'] for item in items}):
        indices = [i for i, item in enumerate(items) if item.row['split'] == split]
        subset = [items[i] for i in indices]
        report[split] = music_metrics(subset, probabilities[indices])
        slices = {}
        for field in ('source', 'generator', 'probe_family', 'duration_seconds'):
            for value in sorted({item.row.get(field, '') for item in subset}):
                selected = [i for i in indices if items[i].row.get(field, '') == value]
                slices[f'{field}:{value}'] = music_metrics([items[i] for i in selected], probabilities[selected])
        report[split]['slices'] = slices
        if all(len(item.frozen) == 3 for item in subset):
            predictions = [dict(ID=item.row['ID'], MUSIC_FAKE_PROB=float(probabilities[i]),
                                FILE_FAKE_PROB=max(item.frozen['VOICE_FAKE_PROB'] * item.frozen['VOICE_PRESENT_PROB'], float(probabilities[i]) * item.frozen['MUSIC_PRESENT_PROB']), **item.frozen)
                           for i, item in zip(indices, subset, strict=True)]
            report[split]['available_component_metrics'] = official_metrics([item.row for item in subset], predictions)
    return report


def original_training_items(train: Sequence[Item]) -> list[Item]:
    original = [item for item in train if item.row['ID'].startswith('core_')]
    if len(original) != 831:
        raise TrainingContractError('The original-data control requires exactly 831 original core training files')
    return original


def sampling_source(source: str) -> str:
    """Balance corpus families, not legacy path tags that already encode generators."""
    return 'SONICS' if source in {'suno', 'udio'} else source.split('/')[0]


def make_model(arm: str, normalizer: tuple[torch.Tensor, torch.Tensor], ast_assets: Path, device: str) -> Model:
    if arm == 'ast':
        return ASTLocalizer.from_pretrained(ast_assets).to(device)
    constructor = WindowMIL if arm == 'mil' else FrozenMeanMLP
    return constructor(*normalizer).to(device)


def run(args: argparse.Namespace) -> None:
    """No held-out prediction is computed until all arms and seeds are locked."""
    output = args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    effective_batch = getattr(args, 'effective_batch', 128)
    if effective_batch < 32 or effective_batch % 2:
        raise TrainingContractError('Effective batch must be even and at least 32')
    check_deadline(args.deadline_epoch)
    if (output / 'config.json').exists():
        raise TrainingContractError('Use a fresh output directory; restarting cannot reset this run')
    items = read_items(args.manifest, args.cache_dir, args.ast_assets)
    train = [item for item in items if item.row['split'] == 'train' and item.row['MUSIC_PRESENT'] == '1' and item.row['MUSIC_FAKE'] in ('0', '1')]
    dev = [item for item in items if item.row['split'] == 'dev']
    if len({item.row['MUSIC_FAKE'] for item in train}) != 2 or music_metrics(dev, np.full(len(dev), .5))['music_eer'] is None:
        raise TrainingContractError('Train and dev require known positive and negative music labels')
    original = original_training_items(train)
    normalizers = {}
    for name, cohort in (('expanded', train), ('original831', original)):
        matrix = np.stack([item.mert.mean(0) for item in cohort])
        deviation = matrix.std(0)
        deviation[deviation < 1e-6] = 1
        mean, scale = torch.from_numpy(matrix.mean(0)), torch.from_numpy(deviation)
        normalizers[name] = (mean, scale)
        torch.save({'mean': mean, 'scale': scale, 'train_ids': [item.row['ID'] for item in cohort]}, output / f'{name}_normalizer.pt')
    device = args.device
    if device == 'cuda' and (not torch.cuda.is_available() or torch.cuda.device_count() != 1):
        raise TrainingContractError('This comparison requires exactly one visible CUDA GPU')
    schedules = {}
    for name, cohort in (('expanded', train), ('original831', original)):
        sampler_rows = [dict(item.row, source=sampling_source(item.row['source']), label=int(item.row['MUSIC_FAKE'])) for item in cohort]
        schedules[name] = {seed: balanced_draws(sampler_rows, steps=args.steps, batch_size=effective_batch, seed=seed, mode='source_generator_parent') for seed in args.seeds}
        exposure = {str(seed): {'draw_sha256': hashlib.sha256('\n'.join(cohort[i].row['ID'] for batch in draws for i in batch).encode()).hexdigest(),
                              'counts': dict(Counter(cohort[i].row['ID'] for batch in draws for i in batch))} for seed, draws in schedules[name].items()}
        for seed, values in exposure.items():
            values['unseen_ids'] = sorted({item.row['ID'] for item in cohort} - set(values['counts']))
            drawn_parents = {cohort[i].row['parent_id'] for batch in schedules[name][int(seed)] for i in batch}
            values['unseen_parents'] = sorted({item.row['parent_id'] for item in cohort} - drawn_parents)
        write_json(output / f'{name}_exposure.json', exposure)
        if any(values['unseen_parents'] for values in exposure.values()):
            raise TrainingContractError('Fixed training budget fails to expose every training parent')
    write_json(output / 'config.json', {'deadline_epoch': args.deadline_epoch, 'effective_batch': effective_batch, 'steps': args.steps,
               'seeds': args.seeds, 'arms': ['mean', 'mil', 'ast'], 'diagnostic_controls': ['original831_mean'], 'selection': 'dev Music EER, tie dev BCE',
               'local_loss_weight': .25, 'ast_learning_rates': {'last_block': 2e-5, 'head': 1e-3},
               'head_learning_rate': 1e-3, 'weight_decay': .01, 'normalizer': 'training file raw-window means only',
               'architecture_and_objective_jointly_changed': True, 'ast_prefix_storage': 'lossy float16',
               'selection_excludes': sorted({item.row['split'] for item in items} - {'train', 'dev'}),
               'evaluate_training': getattr(args, 'evaluate_training', True),
               'manifest_sha256': hashlib.sha256(args.manifest.read_bytes()).hexdigest()})
    locked = []
    controls_locked = []
    selections = {}
    benchmarks = {}
    try:
        for arm in ('original831_mean', 'mean', 'mil', 'ast'):
            cohort_name = 'original831' if arm == 'original831_mean' else 'expanded'
            cohort = original if arm == 'original831_mean' else train
            kind = 'mean' if arm == 'original831_mean' else arm
            for seed in args.seeds:
                check_deadline(args.deadline_epoch)
                torch.manual_seed(seed)
                np.random.seed(seed)
                model = make_model(arm, normalizers[cohort_name], args.ast_assets, device)
                model.train()
                if arm not in benchmarks:
                    benchmarks[arm] = benchmark(model, cohort, kind, device, args.deadline_epoch, effective_batch)
                microbatch, timings = benchmarks[arm]
                window_limit = microbatch * max(len(item.mert) for item in cohort)
                write_json(output / f'{arm}_{seed}_benchmark.json', {'chosen_file_microbatch': microbatch, 'trials': timings})
                optimizer = optimizer_for(model)
                best = (float('inf'), float('inf'))
                history = []
                checkpoint = output / f'{arm}_{seed}.pt'
                for step, draw in enumerate(schedules[cohort_name][seed], 1):
                    model.train()
                    loss = train_step(model, optimizer, [cohort[i] for i in draw], kind, microbatch, window_limit, device, args.deadline_epoch)
                    write_json(output / 'progress.json', {'phase': 'training', 'arm': arm, 'seed': seed, 'update': step,
                               'steps': args.steps, 'loss': loss, 'effective_batch': effective_batch, 'microbatch': microbatch, 'deadline_epoch': args.deadline_epoch})
                    if step % args.eval_every == 0 or step == args.steps:
                        probabilities = predict(model, dev, microbatch, window_limit, device, args.deadline_epoch)
                        metrics = music_metrics(dev, probabilities)
                        history.append(dict(step=step, **metrics))
                        key = (float(metrics['music_eer']), float(metrics['music_bce']))
                        if key < best:
                            best = key
                            torch.save({'model': parameters_to_save(model), 'arm': arm, 'seed': seed, 'selected_update': step,
                                        'dev_metrics': metrics, 'dev_probabilities': torch.from_numpy(probabilities),
                                        'microbatch': microbatch, 'window_limit': window_limit}, checkpoint)
                write_json(output / f'{arm}_{seed}_history.json', history)
                (controls_locked if arm == 'original831_mean' else locked).append(str(checkpoint.name))
                selections[f'{arm}_{seed}'] = torch.load(checkpoint, map_location='cpu', weights_only=True)['selected_update']
                del model, optimizer
                gc.collect()
                if device == 'cuda':
                    torch.cuda.empty_cache()
        dev_keys = {}
        for arm in ('mean', 'mil', 'ast'):
            probabilities = np.stack([torch.load(output / f'{arm}_{seed}.pt', map_location='cpu', weights_only=True)['dev_probabilities'].numpy()
                                      for seed in args.seeds]).mean(0)
            metrics = music_metrics(dev, probabilities)
            dev_keys[arm] = (metrics['music_eer'], metrics['music_bce'])
        winner = min(dev_keys, key=dev_keys.get)
        write_json(output / 'candidate_lock.json', {'locked_epoch': time.time(), 'checkpoints': locked, 'selected_updates': selections,
                   'control_checkpoints': controls_locked,
                   'winner_by_dev_only': winner, 'dev_selection_keys': {arm: list(key) for arm, key in dev_keys.items()},
                   'checkpoint_sha256': {name: sha256_file(output / name) for name in locked + controls_locked}})
        reports = {}
        evaluation = items if getattr(args, 'evaluate_training', True) else [item for item in items if item.row['split'] != 'train']
        for arm in ('original831_mean', 'mean', 'mil', 'ast'):
            seed_predictions = []
            for seed in args.seeds:
                model = make_model(arm, normalizers['original831' if arm == 'original831_mean' else 'expanded'], args.ast_assets, device)
                saved = torch.load(output / f'{arm}_{seed}.pt', map_location='cpu', weights_only=True)
                restore(model, saved['model'])
                probabilities = predict(model, evaluation, saved['microbatch'], saved['window_limit'], device, args.deadline_epoch)
                seed_predictions.append(probabilities)
                np.savez_compressed(output / f'{arm}_{seed}_predictions.npz', ids=np.asarray([item.row['ID'] for item in evaluation]), music_probability=probabilities)
                write_json(output / f'{arm}_{seed}_metrics.json', report_metrics(evaluation, probabilities))
                del model
                gc.collect()
                if device == 'cuda':
                    torch.cuda.empty_cache()
            ensemble = np.stack(seed_predictions).mean(0)
            reports[arm] = report_metrics(evaluation, ensemble)
            with (output / f'{arm}_ensemble.csv').open('w', newline='') as stream:
                writer = csv.DictWriter(stream, fieldnames=[*items[0].row.keys(), 'MUSIC_FAKE_PROB'])
                writer.writeheader()
                writer.writerows(dict(item.row, MUSIC_FAKE_PROB=float(probability)) for item, probability in zip(evaluation, ensemble, strict=True))
        write_json(output / 'results.json', {'status': 'completed', 'winner_by_dev_only': winner,
                   'official_score': None, 'metrics': reports, 'selected_updates': selections})
        write_json(output / 'progress.json', {'phase': 'completed', 'deadline_epoch': args.deadline_epoch})
    except TimeoutError as error:
        write_json(output / 'results.json', {'status': 'partial_deadline', 'reason': str(error), 'locked_candidates': locked,
                   'heldout_evaluation_complete': False, 'official_score': None})
        write_json(output / 'progress.json', {'phase': 'partial_deadline', 'deadline_epoch': args.deadline_epoch})


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('manifest', 'cache-dir', 'ast-assets', 'output-dir'):
        parser.add_argument(f'--{name}', type=Path, required=True)
    parser.add_argument('--deadline-epoch', type=float, required=True, help='Immutable training stop; caller reserves export time')
    parser.add_argument('--steps', type=int, default=125)
    parser.add_argument('--eval-every', type=int, default=25)
    parser.add_argument('--seeds', type=int, nargs='+', default=[41, 42, 43])
    parser.add_argument('--device', choices=('cuda', 'cpu'), default='cuda')
    parser.add_argument('--effective-batch', type=int, default=128)
    parser.add_argument('--evaluate-training', action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()
    if args.steps < 1 or args.eval_every < 1 or not args.seeds or len(set(args.seeds)) != len(args.seeds):
        parser.error('Positive update/evaluation counts and unique seeds are required')
    run(args)


if __name__ == '__main__':
    main()
