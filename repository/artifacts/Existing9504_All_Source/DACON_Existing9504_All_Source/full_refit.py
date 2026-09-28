"""Fixed-epoch final refit on the complete public corpus; no model selection."""
from collections import Counter
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
import csv
import math
import random
import time

import numpy as np
import torch
from transformers import ASTFeatureExtractor

from ast_multitask import ASTMultiTask, compute_pos_weights, masked_multitask_loss
from graph_features import atomic_json, digest
from graph_scoring import LABEL_COLUMNS
from train_v2 import (CONFIGS, SEED, FrozenEmbeddingCache, _batch_inputs,
                      _frozen_logits, _json_hash, _optimizer, _targets,
                      _trainable_state, save_training_checkpoint)


@dataclass(frozen=True, slots=True)
class FullData:
    root: Path
    rows: tuple[dict[str, str], ...]
    manifest_hashes: dict[str, str]


@dataclass(frozen=True, slots=True)
class RefitJob:
    data: FullData
    output: Path
    device: torch.device
    epochs: int
    deadline: float


def read_full_data(root: Path) -> FullData:
    """Preserve original roles while explicitly admitting all four public roles."""
    rows = []
    hashes = {}
    allowed = {'dev_train', 'dev_select', 'test_generator', 'test_source'}
    for folder in ('development', 'locked_test'):
        path = root / folder / 'manifest.csv'
        hashes[folder] = digest(path)
        with path.open(encoding='utf-8-sig', newline='') as stream:
            rows.extend(csv.DictReader(stream))
    if not rows or len({r['ID'] for r in rows}) != len(rows):
        raise ValueError('full-data manifest is empty or has duplicate IDs')
    for row in rows:
        if row['split'] not in allowed:
            raise ValueError('unexpected public-data role')
        path = (root / row['path']).resolve()
        if not path.is_relative_to(root.resolve()) or not path.is_file():
            raise ValueError('missing audio or path outside public corpus')
        if any(row[c] not in {'', '0', '1'} for c in LABEL_COLUMNS):
            raise ValueError('invalid binary or unknown target')
    return FullData(root, tuple(sorted(rows, key=lambda r: r['ID'])), hashes)


def train_stage(model: ASTMultiTask, processor: ASTFeatureExtractor,
                job: RefitJob) -> dict:  # noqa: DICT_OK
    """Consume every ID once per epoch, saving the last epoch without holdout scoring."""
    name = 'AST_FROZEN' if model.mode == 'frozen' else 'AST_LAST_BLOCK'
    config = {**CONFIGS[name], 'epochs': job.epochs, 'seed': SEED,
              'manifest_hashes': job.data.manifest_hashes, 'purpose': 'full_public_data_refit'}
    job.output.mkdir(parents=True, exist_ok=False)
    atomic_json(job.output / 'config.json', config)
    model.to(job.device)
    optimizer = _optimizer(model, config)
    rows = list(job.data.rows)
    weights = compute_pos_weights(_targets(rows, job.device))
    cache = FrozenEmbeddingCache(job.output / 'cache', 'fresh-audioset-backbone',
                                  _json_hash(processor.to_dict())) if model.mode == 'frozen' else None
    exposures = Counter()
    history = []
    head_parameters = list(model.heads.parameters())
    backbone_parameters = [p for n, p in model.named_parameters()
                           if p.requires_grad and not n.startswith('heads.')]
    for epoch in range(1, job.epochs + 1):
        started = time.perf_counter()
        before = _trainable_state(model)
        model.train()
        optimizer.zero_grad(set_to_none=True)
        shuffled = list(rows)
        random.Random(SEED + epoch).shuffle(shuffled)
        consumed, losses, head_gradients, backbone_gradients = [], [], [], []
        batches = math.ceil(len(rows) / config['batch_size'])
        for index, start in enumerate(range(0, len(rows), config['batch_size'])):
            if time.time() > job.deadline:
                raise TimeoutError('full refit deadline; incomplete stage is not a final model')
            batch = shuffled[start:start + config['batch_size']]
            amp = torch.autocast('cuda', dtype=torch.bfloat16) if job.device.type == 'cuda' else nullcontext()
            with amp:
                logits = (_frozen_logits(model, processor, batch, job.data.root, cache, job.device)
                          if cache else model(_batch_inputs(batch, job.data.root, processor, 0, job.device)[0]))
                loss = masked_multitask_loss(logits, _targets(batch, job.device), weights)
            # Preserve the original accumulation rule, including the final partial group.
            (loss / config['accumulation']).backward()
            losses.append(float(loss.detach().cpu()))
            consumed.extend(row['ID'] for row in batch)
            if (index + 1) % config['accumulation'] == 0 or index + 1 == batches:
                head_gradients.append(float(torch.nn.utils.clip_grad_norm_(head_parameters, math.inf)))
                if backbone_parameters:
                    backbone_gradients.append(float(torch.nn.utils.clip_grad_norm_(backbone_parameters, math.inf)))
                norm = torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
                if not math.isfinite(float(norm)):
                    raise FloatingPointError('non-finite full-refit gradient')
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            if (index + 1) % 20 == 0:
                print(f'FULL_BATCH {name} epoch={epoch} batch={index+1}/{batches}', flush=True)
        if Counter(consumed) != Counter(r['ID'] for r in rows):
            raise RuntimeError('epoch did not consume every public file exactly once')
        head_gradient = max(head_gradients)
        backbone_gradient = max(backbone_gradients, default=0.0)
        if head_gradient <= 0 or (backbone_parameters and backbone_gradient <= 0):
            raise FloatingPointError('intended trainable path received no gradient')
        exposures.update(consumed)
        after = _trainable_state(model)
        update = sum(float((value - before[n]).abs().sum()) for n, value in after.items())
        if update <= 0:
            raise FloatingPointError('no parameter update in completed epoch')
        report = {'epoch': epoch, 'mean_loss': float(np.mean(losses)),
                  'unique_files': len(set(consumed)), 'input_count': len(consumed),
                  'original_roles': dict(Counter(r['split'] for r in rows)),
                  'consumed_id_sha256': _json_hash(sorted(consumed)),
                  'head_gradient': head_gradient, 'backbone_gradient': backbone_gradient,
                  'absolute_parameter_update': update, 'seconds': time.perf_counter()-started}
        history.append(report)
        save_training_checkpoint(job.output / 'last.pt', model, optimizer, epoch, _json_hash(config), history)
        if job.device.type == 'mps':
            torch.save(torch.mps.get_rng_state(), job.output / 'mps_rng.pt')
        atomic_json(job.output / f'epoch_{epoch:03d}.json', report)
        print(f'FULL_EPOCH {name} {epoch}/{job.epochs} files={len(consumed)} loss={report["mean_loss"]:.6f}', flush=True)
    result = {'stage': name, 'epochs': history, 'exposures': dict(exposures),
              'epoch_count': job.epochs, 'config': config, 'selection': 'fixed_final_epoch',
              'independent_validation_score': None}
    atomic_json(job.output / 'report.json', result)
    return result
