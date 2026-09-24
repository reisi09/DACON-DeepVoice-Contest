"""Protect controlled data comparison, CPU-only default, and bounded task cleanup."""
import csv
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from types import SimpleNamespace

import pytest

from data_v4.gpu_experiments import (REPOSITORY, check_complete, checked_cache,
                                     run_bounded, validate_manifests)
from full_bank_contract import config_for, feature_contract
from full_bank_features import HOP, WINDOW, load_rows, row_identity
from v6_data import sha256
import torch


def write_bank(path, rows):
    with path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def experiment(tmp_path):
    audio = tmp_path / 'input'
    audio.mkdir()
    rows = []
    for role in ('train', 'selection', 'calibration', 'development'):
        for head in ('music', 'voice'):
            for label in ('0', '1'):
                rid = f'{role}_{head}_{label}'
                source = audio / f'{rid}.wav'
                # Only byte integrity is under test; preflight must never decode or load a model.
                source.write_bytes(rid.encode())
                rows.append(dict(ID=rid, path=source.name, audio_sha256=sha256(source), component_id=rid,
                                 split='train' if role == 'train' else 'test' if role == 'development' else 'val',
                                 role=role, route=head, MUSIC_FAKE=label if head == 'music' else '',
                                 VOICE_FAKE=label if head == 'voice' else ''))
    new = dict(rows[0], ID='new_train', component_id='new_train', path='new.wav')
    (audio / 'new.wav').write_bytes(b'new training bytes')
    new['audio_sha256'] = sha256(audio / 'new.wav')
    baseline = [r for r in rows if r['role'] != 'development']
    extended = [*baseline, new]
    paths = [tmp_path / name for name in ('union.csv', 'e00.csv', 'e04.csv', 'config.json')]
    for path, bank in zip(paths, ([*rows, new], baseline, extended)):
        write_bank(path, bank)
    paths[-1].write_text(json.dumps({'seeds': [41]}))
    return paths, rows, new


def test_data_only_comparison_rejects_changed_labels_and_added_evaluation(tmp_path):
    paths, rows, new = experiment(tmp_path)
    banks, _ = validate_manifests(*paths)
    assert banks['E04'][:-1] == banks['E00']
    changed = [*banks['E04']]
    changed[0] = dict(changed[0], MUSIC_FAKE='1')
    write_bank(paths[2], changed)
    with pytest.raises(ValueError, match='differs from feature manifest'):
        validate_manifests(*paths)
    # Rewriting both feature and experiment manifests cannot disguise a new validation row.
    new.update(role='selection', split='val')
    write_bank(paths[0], [*rows, new])
    write_bank(paths[2], [r for r in rows if r['role'] != 'development'] + [new])
    with pytest.raises(ValueError, match='training rows only'):
        validate_manifests(*paths)


def test_default_cli_never_loads_models_and_blocks_corrupted_input(tmp_path):
    paths, _, _ = experiment(tmp_path)
    assets = tmp_path / 'assets'
    names = ('baseline/script.py', 'baseline/model/df_arena_1b/pytorch_model.bin',
             'baseline/model/htdemucs/955717e8-8726e21a.th', 'baseline/model/panns/Cnn14_mAP=0.431.pth',
             'mert/pytorch_model.bin', 'mert/config.json', 'mert/preprocessor_config.json',
             'mert/configuration_MERT.py', 'mert/modeling_MERT.py')
    hashes = {}
    for name in names:
        path = assets / name
        path.parent.mkdir(parents=True, exist_ok=True)
        # Importing any model code would fail; this validates the CPU preflight execution boundary.
        path.write_text('raise RuntimeError("MODEL CODE MUST NOT EXECUTE")\n')
        hashes[name] = sha256(path)
    asset_hashes = tmp_path / 'asset_hashes.json'
    asset_hashes.write_text(json.dumps(hashes))
    command = [sys.executable, str(REPOSITORY / 'data_v4/gpu_experiments.py')]
    for name, value in zip(('feature-manifest', 'e00-manifest', 'e04-manifest', 'config'), paths):
        command += ['--' + name, str(value)]
    for name, value in dict(input=tmp_path / 'input', assets=assets, asset_hashes=asset_hashes,
                            cache=tmp_path / 'cache', output=tmp_path / 'output').items():
        command += ['--' + name.replace('_', '-'), str(value)]
    env = {**os.environ, 'CUDA_VISIBLE_DEVICES': '', 'HF_HUB_OFFLINE': '1', 'TRANSFORMERS_OFFLINE': '1'}
    result = subprocess.run(command, env=env, capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stderr
    report = json.loads(result.stdout)
    assert report['audio_files'] == 17 and report['gpu_started'] is False
    (tmp_path / 'input' / 'new.wav').write_bytes(b'corrupted')
    result = subprocess.run(command, env=env, capture_output=True, text=True, timeout=30)
    assert result.returncode != 0 and 'Audio bytes changed: new_train' in result.stderr
    (tmp_path / 'input' / 'new.wav').unlink()
    (tmp_path / 'input' / 'new.wav').symlink_to(paths[-1])
    result = subprocess.run(command, env=env, capture_output=True, text=True, timeout=30)
    assert result.returncode != 0 and 'Path escapes input root' in result.stderr


def test_timeout_terminates_owned_process_and_preserves_partial_output(tmp_path):
    partial = tmp_path / 'partial.txt'
    late = tmp_path / 'late.txt'
    script = tmp_path / 'child.py'
    script.write_text('from pathlib import Path\nimport time\n'
                      f'Path({str(partial)!r}).write_text("reusable")\n'
                      'time.sleep(3)\n'
                      f'Path({str(late)!r}).write_text("outlived budget")\n')
    start = time.monotonic()
    with pytest.raises(subprocess.TimeoutExpired):
        run_bounded([sys.executable, str(script)], 1., tmp_path / 'child.log', os.environ.copy())
    assert time.monotonic() - start < 3
    assert partial.read_text() == 'reusable' and not late.exists()


def test_standalone_consumers_reject_foreign_cache_and_baseline(tmp_path):
    # Numerical NPZ validation belongs to the actual consumers; protect the provenance gate here.
    paths, _, _ = experiment(tmp_path)
    cache, output = tmp_path / 'cache', tmp_path / 'output'
    cache.mkdir()
    (output / 'features').mkdir(parents=True)
    assets = tmp_path / 'assets.json'
    assets.write_text(json.dumps({'reviewed_model': 'a' * 64}))
    identity = dict(manifest_sha256=sha256(paths[0]), assets=json.loads(assets.read_text()),
                    implementation={name: sha256(REPOSITORY / name) for name in (
                        'full_bank_features.py', 'mert12_submission.py', 'v6_runtime.py',
                        'decision_heads.py', 'full_bank_contract.py')},
                    window=WINDOW, hop=HOP, mert_layer=12, dtype='float32',
                    duration='all samples, no crop', torch=torch.__version__)
    key = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    index = {'identity': identity, 'items': {}}
    for row in load_rows(paths[0]):
        path = cache / (row['ID'] + '.npz')
        path.write_bytes(b'consumer validates these bytes separately')
        index['items'][row['ID']] = dict(path=path.name, sha256=sha256(path), identity=row_identity(row, key))
    initial = cache / 'voice_classifier_initial.pt'
    initial.write_bytes(b'initial classifier fixture')
    def save_cache(value):
        (cache / 'index.json').write_text(json.dumps(value))
        (output / 'features' / 'features_complete.json').write_text(json.dumps(dict(
            rows=len(value['items']), index_sha256=sha256(cache / 'index.json'),
            voice_classifier_sha256=sha256(initial))))
    args = SimpleNamespace(cache=cache, output=output, feature_manifest=paths[0], asset_hashes=assets)
    save_cache(index)
    checked_cache(args)
    # A self-consistent cache/receipt produced with other weights must still fail.
    foreign = json.loads(json.dumps(index))
    foreign['identity']['assets']['reviewed_model'] = 'b' * 64
    save_cache(foreign)
    with pytest.raises(ValueError, match='producer/assets/manifest'):
        checked_cache(args)
    foreign['identity'] = dict(identity, manifest_sha256='c' * 64)
    save_cache(foreign)
    with pytest.raises(ValueError, match='producer/assets/manifest'):
        checked_cache(args)
    foreign = json.loads(json.dumps(index))
    foreign['items'].pop('development_music_0')
    save_cache(foreign)
    with pytest.raises(ValueError, match='coverage'):
        checked_cache(args)
    save_cache(index)
    directory = output / 'E00'
    directory.mkdir()
    config = config_for({'seeds': [41]})
    run = dict(manifest=sha256(paths[1]), feature_manifest=sha256(paths[0]), cache=sha256(cache / 'index.json'),
               voice_initial=sha256(initial), feature_contract=feature_contract(identity), config=config,
               max_epochs=50, patience=5)
    for name, filename in {'trainer': 'full_bank_train.py', 'model': 'localization_models.py',
                           'head': 'eight_heads.py', 'evaluator': 'full_bank_evaluate.py',
                           'batches': 'full_bank_batches.py', 'contract': 'full_bank_contract.py'}.items():
        run[name] = sha256(REPOSITORY / filename)
    results = {}
    for head in ('music', 'voice'):
        checkpoint = directory / f'{head}_41.pt'
        checkpoint.write_bytes(head.encode())
        results[f'{head}_41'] = {'checkpoint_sha256': sha256(checkpoint)}
    (directory / 'results.json').write_text(json.dumps(results))
    def save_run(value):
        (directory / 'run_identity.json').write_text(json.dumps(value))
        (directory / 'complete.json').write_text(json.dumps(dict(
            identity=value, results_sha256=sha256(directory / 'results.json'))))
    save_run(run)
    check_complete(directory, paths[1], config, paths[0], cache, index)
    save_run(dict(run, cache='d' * 64))
    with pytest.raises(ValueError, match='cache identity differs'):
        check_complete(directory, paths[1], config, paths[0], cache, index)
    save_run(dict(run, trainer='e' * 64))
    with pytest.raises(ValueError, match='Training implementation changed'):
        check_complete(directory, paths[1], config, paths[0], cache, index)
