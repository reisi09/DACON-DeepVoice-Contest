"""CPU preflight by default; explicitly run the bounded E00/E04 A100/H100 comparison.

No cloud APIs, dependency installs, model imports, or GPU execution in preflight.
The existing feature runner loads only the separately reviewed, hash-pinned assets.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

REPOSITORY = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY))

from eight_fixed_io import atomic_json
from full_bank_contract import config_for, role
from full_bank_features import HOP, WINDOW, load_rows, row_identity
from full_bank_train import head_labels
from v6_data import sha256
import torch


def contained(root, relative):
    path = Path(relative)
    if path.is_absolute() or '..' in path.parts:
        raise ValueError(f'Unsafe relative path: {relative}')
    resolved = (root / path).resolve(strict=True)
    if not resolved.is_relative_to(root.resolve()) or not resolved.is_file():
        raise ValueError(f'Path escapes input root or is not a file: {relative}')
    return resolved


def validate_manifests(feature_manifest, e00_manifest, e04_manifest, config_path):
    config = config_for(json.loads(config_path.read_text()))
    if config['seeds'] != [41]:
        raise ValueError('The controlled E00/E04 experiment requires seed 41 only')
    banks = {name: load_rows(path) for name, path in (
        ('features', feature_manifest), ('E00', e00_manifest), ('E04', e04_manifest))}
    maps = {name: {r['ID']: r for r in rows} for name, rows in banks.items()}
    for name in ('E00', 'E04'):
        for rid, row in maps[name].items():
            if maps['features'].get(rid) != row:
                raise ValueError(f'{name} row differs from feature manifest: {rid}')
            if role(row) not in ('train', 'selection', 'calibration'):
                raise ValueError(f'{name} includes development or sealed rows')
        for assigned in ('train', 'selection', 'calibration'):
            for head in ('music', 'voice'):
                classes = {head_labels(r)[head] for r in banks[name]
                           if role(r) == assigned and head in head_labels(r)}
                if classes != {0., 1.}:
                    raise ValueError(f'{name} lacks both {head} classes in {assigned}')
    if any(maps['E04'].get(rid) != row for rid, row in maps['E00'].items()):
        raise ValueError('E00 is not an unchanged subset of E04')
    added = set(maps['E04']) - set(maps['E00'])
    if not added or any(role(maps['E04'][rid]) != 'train' for rid in added):
        raise ValueError('E04 must add training rows only')
    extras = set(maps['features']) - set(maps['E04'])
    if not extras or any(role(maps['features'][rid]) != 'development' for rid in extras):
        raise ValueError('Feature-only rows must be the development comparison panel')
    audio_roles = {}
    for row in banks['features']:
        audio_roles.setdefault(row['audio_sha256'], set()).add(role(row))
    if any(len(roles) > 1 for roles in audio_roles.values()):
        raise ValueError('Identical audio crosses data roles')
    return banks, config


def preflight(args):
    banks, config = validate_manifests(args.feature_manifest, args.e00_manifest, args.e04_manifest, args.config)
    for destination in (args.cache.resolve(), args.output.resolve()):
        for source in (args.input.resolve(), args.assets.resolve()):
            if destination.is_relative_to(source) or source.is_relative_to(destination):
                raise ValueError('Cache/output must be separate from input audio and assets')
    hashes = json.loads(args.asset_hashes.read_text())
    required = {'baseline/script.py', 'baseline/model/df_arena_1b/pytorch_model.bin',
                'baseline/model/htdemucs/955717e8-8726e21a.th', 'baseline/model/panns/Cnn14_mAP=0.431.pth',
                'mert/pytorch_model.bin', 'mert/config.json', 'mert/preprocessor_config.json',
                'mert/configuration_MERT.py', 'mert/modeling_MERT.py'}
    if not isinstance(hashes, dict) or not required <= set(hashes):
        raise ValueError('Incomplete pinned asset manifest')
    for path in args.assets.rglob('*'):
        if path.is_file() and path.suffix in ('.py', '.pyc', '.so', '.pyd'):
            if path.relative_to(args.assets).as_posix() not in hashes:
                raise ValueError(f'Unpinned executable asset: {path.relative_to(args.assets)}')
    for name, expected in hashes.items():
        if sha256(contained(args.assets, name)) != expected:
            raise ValueError(f'Asset bytes changed: {name}')
    verified = {}
    total_bytes = 0
    for row in banks['features']:
        path = contained(args.input, row['path'])
        if path not in verified:
            verified[path] = sha256(path)
            total_bytes += path.stat().st_size
        if verified[path] != row['audio_sha256']:
            raise ValueError(f'Audio bytes changed: {row["ID"]}')
    report = dict(status='cpu_preflight_passed', gpu_started=False, config=config,
                  max_epochs=50, patience=5, audio_files=len(verified), audio_bytes=total_bytes,
                  pinned_asset_files=len(hashes), asset_hashes_sha256=sha256(args.asset_hashes),
                  manifests={name: sha256(path) for name, path in (
                      ('features', args.feature_manifest), ('E00', args.e00_manifest), ('E04', args.e04_manifest))},
                  roles={name: dict(Counter(role(r) for r in rows)) for name, rows in banks.items()},
                  added_train_rows=len(banks['E04']) - len(banks['E00']),
                  validation_scope='Input bytes and data/config contracts; no model loading or GPU runtime validation')
    args.output.mkdir(parents=True, exist_ok=True)
    atomic_json(args.output / 'preflight.json', report)
    return report


def checked_cache(args):
    """Bind standalone consumers to the assets and union just checked by preflight."""
    index = json.loads((args.cache / 'index.json').read_text())
    expected = dict(manifest_sha256=sha256(args.feature_manifest),
                    assets=json.loads(args.asset_hashes.read_text()),
                    implementation={name: sha256(REPOSITORY / name) for name in (
                        'full_bank_features.py', 'mert12_submission.py', 'v6_runtime.py',
                        'decision_heads.py', 'full_bank_contract.py')},
                    window=WINDOW, hop=HOP, mert_layer=12, dtype='float32',
                    duration='all samples, no crop', torch=torch.__version__)
    if index['identity'] != expected:
        raise ValueError('Cache producer/assets/manifest differ from current preflight')
    rows = load_rows(args.feature_manifest)
    if set(index['items']) != {row['ID'] for row in rows}:
        raise ValueError('Cache coverage differs from complete feature union')
    identity_key = hashlib.sha256(json.dumps(expected, sort_keys=True).encode()).hexdigest()
    for row in rows:
        entry = index['items'][row['ID']]
        contained(args.cache, entry['path'])
        if entry['identity'] != row_identity(row, identity_key):
            raise ValueError('Cache row identity differs: ' + row['ID'])
    receipt = json.loads((args.output / 'features' / 'features_complete.json').read_text())
    if receipt != dict(rows=len(rows), index_sha256=sha256(args.cache / 'index.json'),
                       voice_classifier_sha256=sha256(args.cache / 'voice_classifier_initial.pt')):
        raise ValueError('Feature completion receipt or initial classifier changed')
    # Consumers still verify NPZ bytes and numerical invariants before using each row.
    return index


def check_complete(directory, manifest, config, feature_manifest, cache, index):
    from data_v4.compare_data_admission import checked_run
    identity, _ = checked_run(directory, manifest, feature_manifest, cache, index)
    if (identity['config'] != config or identity['max_epochs'] != 50 or identity['patience'] != 5
            or identity.get('duration_balance', False)):
        raise ValueError(f'Completed training settings differ: {directory.name}')


def run_bounded(command, timeout, log, env):
    if timeout <= 0:
        raise TimeoutError('Experiment time budget exhausted; partial outputs retained')
    with log.open('ab', buffering=0) as stream:
        process = subprocess.Popen(command, cwd=REPOSITORY, env=env, stdout=stream,
                                   stderr=subprocess.STDOUT, start_new_session=True)
        try:
            code = process.wait(timeout=timeout)
            if code:
                raise subprocess.CalledProcessError(code, command)
        except BaseException:
            try:
                os.killpg(process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                pass
            finally:
                # The parent may exit before a task-owned worker; do not leave it running.
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                process.wait()
            raise


def commands(args):
    python = sys.executable
    common = ['--cache', str(args.cache), '--feature-manifest', str(args.feature_manifest),
              '--experiment-config', str(args.config), '--max-epochs', '50', '--patience', '5',
              '--gpu-name', args.gpu_name]
    return {
        'features': [python, str(REPOSITORY / 'full_bank_features.py'), '--manifest', str(args.feature_manifest),
                     '--input', str(args.input), '--assets', str(args.assets), '--asset-hashes', str(args.asset_hashes),
                     '--cache', str(args.cache), '--output', str(args.output / 'features'),
                     '--gpu-name', args.gpu_name],
        'train-e00': [python, str(REPOSITORY / 'full_bank_train.py'), '--manifest', str(args.e00_manifest),
                      '--output', str(args.output / 'E00'), *common],
        'train-e04': [python, str(REPOSITORY / 'full_bank_train.py'), '--manifest', str(args.e04_manifest),
                      '--output', str(args.output / 'E04'), *common],
        'compare': [python, str(REPOSITORY / 'data_v4' / 'compare_data_admission.py'),
                    '--e00-manifest', str(args.e00_manifest), '--e04-manifest', str(args.e04_manifest),
                    '--feature-manifest', str(args.feature_manifest), '--cache', str(args.cache),
                    '--e00-dir', str(args.output / 'E00'), '--e04-dir', str(args.output / 'E04'),
                    '--output', str(args.output / 'comparison'), '--device', 'cpu']}


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument('action', nargs='?', default='preflight',
                        choices=('preflight', 'features', 'train-e00', 'train-e04', 'compare', 'all'))
    for name in ('feature-manifest', 'e00-manifest', 'e04-manifest', 'config', 'input', 'assets',
                 'asset-hashes', 'cache', 'output'):
        result.add_argument('--' + name, type=Path, required=True)
    result.add_argument('--max-seconds', type=int, help='Required for execution; total budget, 1..86400 seconds')
    result.add_argument('--gpu-name', choices=('A100', 'H100'), default='A100',
                        help='Required device family for GPU stages; preflight stays CPU-only')
    return result


def main():
    args = parser().parse_args()
    if args.action != 'preflight' and (args.max_seconds is None or not 1 <= args.max_seconds <= 86400):
        raise ValueError('Explicit execution requires --max-seconds in 1..86400')
    for key, value in vars(args).items():
        if isinstance(value, Path):
            setattr(args, key, value.resolve())
    started = time.monotonic()
    report = preflight(args)
    print(json.dumps(report, sort_keys=True), flush=True)
    if args.action == 'preflight':
        return
    env = {**os.environ, 'HF_HUB_OFFLINE': '1', 'TRANSFORMERS_OFFLINE': '1',
           'HF_DATASETS_OFFLINE': '1', 'PYTHONDONTWRITEBYTECODE': '1',
           'OMP_NUM_THREADS': '4', 'MKL_NUM_THREADS': '4'}
    def interrupted(signum, _frame):
        raise InterruptedError(f'Interrupted by signal {signum}; partial outputs retained')
    signal.signal(signal.SIGTERM, interrupted)
    stages = ('features', 'train-e00', 'train-e04', 'compare') if args.action == 'all' else (args.action,)
    for stage in stages:
        if stage != 'features':
            index = checked_cache(args)
        if stage in ('train-e04', 'compare'):
            check_complete(args.output / 'E00', args.e00_manifest, report['config'],
                           args.feature_manifest, args.cache, index)
        if stage == 'compare':
            check_complete(args.output / 'E04', args.e04_manifest, report['config'],
                           args.feature_manifest, args.cache, index)
        print(json.dumps({'stage': stage, 'status': 'starting'}), flush=True)
        run_bounded(commands(args)[stage], args.max_seconds - (time.monotonic() - started),
                    args.output / f'{stage}.log', env)
        atomic_json(args.output / f'{stage}_exit.json', dict(returncode=0, elapsed_seconds=time.monotonic() - started))


if __name__ == '__main__':
    main()
