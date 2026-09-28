# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy", "soundfile", "torch"]
# ///
# How to run: python redesign_submission_smoke.py RUN_ROOT PREVIOUS_ROOT
"""Check the actual offline ZIP runtime against source predictions on real audio."""
from __future__ import annotations

import csv
import gc
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

from redesign_checkpoint import load_checkpoint
from redesign_data_validation import read_rows
from redesign_features_runtime import FrozenEncoders
from redesign_h100_run import atomic_json
from redesign_predictions import _predict_model


class SubmissionCheckError(RuntimeError):
    """A packaged prediction differs from the trained model or file contract."""


def main() -> None:
    """Exercise complete CSV production, silent/long input, and file independence."""
    root, previous = map(Path, sys.argv[1:3])
    output, package = root / 'output', root / 'smoke_package'
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    rows = read_rows(root / 'source/full_training.csv')
    specs = [('mixed', '0'), ('mixed', '1'), ('instrumental', '0'), ('speech', '0'), ('speech', '1')]
    chosen = [next(row for row in rows if row['kind'] == kind and row['FILE_FAKE'] == label) for kind, label in specs]
    inputs = package / 'data/test'
    inputs.mkdir(parents=True, exist_ok=True)
    smoke_rows = []
    for index, row in enumerate(chosen):
        target = inputs / f'probe{index}.wav'
        shutil.copy2(row['path'], target)
        smoke_rows.append({'ID': target.stem, 'path': str(target)})
    silence = inputs / 'silence.wav'
    sf.write(silence, np.zeros(64_000, dtype=np.float32), 16_000)
    music, sr = sf.read(chosen[1]['path'], dtype='float32', always_2d=True)
    long_audio = np.tile(music, (int(np.ceil(60 * sr / len(music))), 1))[:60 * sr]
    stereo = np.repeat(long_audio[:, :1], 2, axis=1)
    long_file = inputs / 'long60_stereo.wav'
    sf.write(long_file, stereo, sr)
    smoke_rows.extend([{'ID': path.stem, 'path': str(path)} for path in (silence, long_file)])
    columns = ['ID', 'FILE_FAKE_PROB', 'VOICE_FAKE_PROB', 'MUSIC_FAKE_PROB', 'VOICE_PRESENT_PROB', 'MUSIC_PRESENT_PROB']
    sample = package / 'data/sample_submission.csv'
    with sample.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows({'ID': row['ID'], **dict.fromkeys(columns[1:], 0)} for row in smoke_rows)
    engine = FrozenEncoders(previous / 'assets', 'cuda')
    records = {row['ID']: engine.extract(Path(row['path'])) for row in smoke_rows}
    predictions = []
    for seed in (41, 42, 43):
        model, _ = load_checkpoint(output / 'models' / f'B_seed{seed}' / 'best.pt', device='cuda')
        predictions.append(_predict_model(model, smoke_rows, records, device='cuda'))
    expected = {}
    for row in smoke_rows:
        values = np.mean([[pred[row['ID']][key] for key in columns[1:]] for pred in predictions], axis=0)
        values[0] = 1.0 - (1.0 - values[1] * values[3]) * (1.0 - values[2] * values[4])
        expected[row['ID']] = values
    atomic_json(output / 'smoke_reference.json', {key: value.tolist() for key, value in expected.items()})
    del engine, model, records
    gc.collect()
    torch.cuda.empty_cache()
    guard = root / 'network_guard'
    guard.mkdir(exist_ok=True)
    (guard / 'sitecustomize.py').write_text(
        "import sys\ndef reject(event, args):\n"
        "    if event in {'socket.connect','socket.getaddrinfo'}:\n"
        "        raise RuntimeError('Offline check rejected network: '+event)\n"
        "sys.addaudithook(reject)\nprint('NETWORK_AUDIT_GUARD_ACTIVE', flush=True)\n")
    env = os.environ | {'PYTHONPATH': str(guard), 'HF_HUB_OFFLINE': '1', 'TRANSFORMERS_OFFLINE': '1'}
    began = time.time()
    with (output / 'packaged_inference.log').open('w') as log:
        subprocess.run([sys.executable, 'script.py'], cwd=package, env=env, stdout=log,
                       stderr=subprocess.STDOUT, check=True, timeout=1200)
    elapsed = time.time() - began
    shutil.copy2(package / 'output/runtime_receipt.json', output / 'runtime_receipt.json')
    actual_path = package / 'output/submission.csv'
    actual = read_rows(actual_path)
    shutil.copy2(actual_path, output / 'smoke_submission.csv')
    if [row['ID'] for row in actual] != [row['ID'] for row in smoke_rows]:
        raise SubmissionCheckError('Package omitted, reordered, or inserted IDs')
    max_error = max(float(np.max(np.abs(np.asarray([float(row[k]) for k in columns[1:]]) - expected[row['ID']]))) for row in actual)
    if not np.isfinite(max_error) or max_error > 1e-5:
        raise SubmissionCheckError(f'Packaged/source numerical disagreement {max_error}')
    if any(not 0 <= float(row[key]) <= 1 for row in actual for key in columns[1:]):
        raise SubmissionCheckError('Package returned a non-finite/out-of-range probability')
    subset = [smoke_rows[1], smoke_rows[0]]
    with sample.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows({'ID': row['ID'], **dict.fromkeys(columns[1:], 0)} for row in subset)
    subset_dir = package / 'data/subset'
    subset_dir.mkdir(exist_ok=True)
    for row in subset:
        shutil.copy2(row['path'], subset_dir / Path(row['path']).name)
    with (output / 'independent_inference.log').open('w') as log:
        subprocess.run([sys.executable, 'script.py', '--test-dir', str(subset_dir)], cwd=package,
                       env=env, stdout=log, stderr=subprocess.STDOUT, check=True, timeout=1200)
    independent = read_rows(actual_path)
    independence_error = max(float(np.max(np.abs(np.asarray([float(row[k]) for k in columns[1:]]) - expected[row['ID']]))) for row in independent)
    if [row['ID'] for row in independent] != [row['ID'] for row in subset] or independence_error > 1e-5:
        raise SubmissionCheckError('Predictions changed with other files or order')
    report = {'passed': True, 'files': len(smoke_rows), 'real_audio_files': len(chosen),
              'edge_inputs': ['4 second silence', '60 second stereo music'],
              'reference_max_abs_error': max_error, 'independence_max_abs_error': independence_error,
              'tolerance': 1e-5, 'network_connections_forbidden': True, 'elapsed_seconds': elapsed,
              'gpu': torch.cuda.get_device_name(0), 'L4_verified': False,
              'official_score_measured': False, 'purpose': 'packaging numerical parity, not model quality evaluation'}
    atomic_json(output / 'submission_smoke.json', report)
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
