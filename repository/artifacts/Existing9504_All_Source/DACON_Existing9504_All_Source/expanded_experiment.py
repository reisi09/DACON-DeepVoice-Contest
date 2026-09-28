# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy", "torch", "transformers"]
# ///
# Run through DACON_Expanded_H100_Experiments.ipynb after verified storage preparation.
"""Representative H100 timing gate and the shared notebook subprocess runner."""
from __future__ import annotations

import argparse
from collections import defaultdict
import gc
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

import numpy as np
import torch

from eight_contract import read_manifest, write_manifest
from localization_models import ASTLocalizer
from localization_train import benchmark, read_items


def command(arguments: list[str], log: Path, deadline: float) -> None:
    remaining = int(deadline - time.time())
    if remaining <= 0:
        raise TimeoutError('Original run deadline has elapsed')
    with log.open('w') as stream:
        result = subprocess.run([sys.executable, *arguments], stdout=stream, stderr=subprocess.STDOUT, timeout=remaining)
    if result.returncode:
        raise RuntimeError(f'{arguments[0]} failed: {log.read_text()[-3000:]}')


def feature_arguments(dataset: Path, cache: Path, output: Path, assets: Path, ast: Path, deadline: float) -> list[str]:
    source = Path(__file__).resolve().parent
    return [str(source / 'localization_features.py'), '--dataset-dir', str(dataset), '--cache-dir', str(cache),
            '--output-dir', str(output), '--assets', str(assets), '--ast-assets', str(ast), '--gpu-name', 'H100',
            '--deadline-epoch', str(deadline)]


def canary(inputs: Path, cache: Path, output: Path, assets: Path, deadline: float) -> dict:
    """Measure actual pipeline work and discarded AST optimizer steps before full extraction."""
    rows = read_manifest(inputs / 'dataset/manifest.csv')
    groups = defaultdict(list)
    for row in rows:
        if row['split'] == 'train':
            groups[(row['source'], row['generator'])].append(row)
    train = []
    for index in range(32):
        key = sorted(groups)[index % len(groups)]
        train.append(groups[key][(index // len(groups)) % len(groups[key])])
    if len({r['ID'] for r in train}) != len(train):
        raise ValueError('Canary training rows repeated unexpectedly')
    evaluation = []
    for label in ('0', '1'):
        evaluation += [r for r in rows if r['split'] == 'dev' and r['MUSIC_FAKE'] == label][:4]
        evaluation += [r for r in rows if r['split'] == 'confirmation_presence' and r['VOICE_FAKE'] == label][:4]
    canary_root = output.parent / 'canary'
    data = canary_root / 'dataset'
    data.mkdir(parents=True, exist_ok=False)
    selected = train + evaluation
    for row in selected:
        target = data / row['path']
        target.parent.mkdir(parents=True, exist_ok=True)
        os.link(inputs / 'dataset' / row['path'], target)
    shutil.copyfile(inputs / 'asset_hashes.json', canary_root / 'asset_hashes.json')
    write_manifest(data / 'manifest.csv', selected)
    evidence = output / 'canary'
    evidence.mkdir()
    command(feature_arguments(data, canary_root / 'cache', evidence, assets, inputs / 'ast_assets', deadline), evidence / 'features.log', deadline)
    items = read_items(data / 'manifest.csv', canary_root / 'cache', inputs / 'ast_assets')
    training = [i for i in items if i.row['split'] == 'train']
    torch.manual_seed(41)
    model = ASTLocalizer.from_pretrained(inputs / 'ast_assets').to('cuda').train()
    chosen, trials = benchmark(model, training, 'ast', 'cuda', deadline, max_batch=256)
    del model, items, training
    gc.collect()
    torch.cuda.empty_cache()
    measured = json.loads((evidence / 'feature_cache_receipt.json').read_text())
    full = json.loads((inputs / 'data_preparation.json').read_text())
    timing = measured['stage_seconds']
    setup = max(0, measured['elapsed_seconds'] - sum(timing.values()))
    full_eval = sum(r['split'] != 'train' for r in rows)
    feature_seconds = setup + timing['demucs'] * len(rows) / len(selected)
    feature_seconds += (timing['mert'] + timing['ast_prefix']) * full['windows'] / measured['windows']
    feature_seconds += timing['fixed_branches'] * full_eval / len(evaluation)
    throughput = next(float(t['files_per_second']) for t in trials if t['files'] == chosen)
    training_seconds = 3 * 96 * 256 / throughput
    evaluation_seconds = 3 * (4 * 240 + full_eval) / throughput
    estimate = 1.35 * (feature_seconds + training_seconds + evaluation_seconds) + 300
    remaining = deadline - time.time()
    receipt = {'feature_seconds_estimate': feature_seconds, 'ast_training_seconds_estimate': training_seconds,
               'ast_evaluation_seconds_estimate': evaluation_seconds, 'safety_multiplier': 1.35,
               'total_remaining_seconds_estimate': estimate, 'remaining_seconds': remaining,
               'ast_chosen_microbatch': chosen, 'ast_trials': trials, 'canary_rows': len(selected),
               'canary_windows': measured['windows'], 'full_rows': len(rows), 'full_windows': full['windows'],
               'proceed': estimate < remaining, 'estimate_not_completion_guarantee': True}
    (output / 'capacity_gate.json').write_text(json.dumps(receipt, indent=2) + '\n')
    if estimate >= remaining:
        raise TimeoutError('Measured full-run forecast exceeds original four-hour budget; see capacity_gate.json')
    return receipt
