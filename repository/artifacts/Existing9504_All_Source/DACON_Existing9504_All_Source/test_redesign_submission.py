"""Submission parity, silence, ID order, and archive-integrity regression checks."""

import ast
import csv
import hashlib
import json
import subprocess
import sys
import types
import zipfile
from pathlib import Path

import numpy as np
import pytest
import torch

from redesign_checkpoint import save_checkpoint, sha256
from redesign_features_cache import FeatureRecord
from redesign_models import MusicResidual, make_control
from redesign_predictions import _predict_model


def _fixture() -> FeatureRecord:
    rng = np.random.default_rng(123)
    voice = np.asarray([.2, .7, .4, .99], dtype=np.float32)
    presence = np.asarray([[.6, .1], [.3, .8], [.4, .5], [.99, .99]], dtype=np.float32)
    aux = rng.normal(size=(4, 6)).astype(np.float32)
    aux[:, 4] = [1, 0, 1, 1]
    aux[:, 5] = [1, 1, 1, 0]
    return FeatureRecord(
        stem=rng.normal(size=(4, 1024)).astype(np.float32),
        raw=rng.normal(size=(4, 1024)).astype(np.float32), aux=aux,
        starts=np.arange(4, dtype=np.int64) * 64600,
        valid=np.asarray([1, 1, 1, 0], dtype=bool),
        stem_audible=aux[:, 4].astype(bool),
        voice_embedding=np.zeros((4, 1280), dtype=np.float32),
        voice_logits=np.log(voice / (1 - voice)), voice_probability=voice,
        presence=presence, evidence=np.asarray([voice.max(), *presence.max(0)], dtype=np.float32),
    )


def _checkpoints(root: Path) -> list[Path]:
    paths = []
    for seed, epochs in zip((41, 42, 43), (17, 17, 8), strict=True):
        model = MusicResidual(make_control(seed))
        with torch.no_grad():
            model.delta_out.weight.normal_(0, .05)
        report = dict(arm='B', seed=seed, epochs=epochs, selected_epoch=epochs,
                      epochs_completed=epochs, mode='fixed_epochs', fixed_epochs=True,
                      earlystop_ids=[], train_ids=[f'train-{i:04}' for i in range(1696)], validation_loss=[],
                      raw_equal_stem='False', status='complete')
        path = root / f'B_seed{seed}.pt'
        save_checkpoint(path, model, 'B', f'synthetic-{seed}', report)
        paths.append(path)
    return paths


def test_submission_matches_reference_and_preserves_music_mask(tmp_path: Path) -> None:
    # Given: real saved B heads with nonzero raw residual and masked voice/presence sentinels.
    from redesign_submission import load_heads, score_ensemble

    torch.set_num_threads(2)
    heads = load_heads(_checkpoints(tmp_path))
    record = _fixture()
    references = [_predict_model(head, [{'ID': 'x'}], {'x': record}, device='cpu')['x']
                  for head in heads]
    expected_music = np.mean([p['MUSIC_FAKE_PROB'] for p in references])
    expected = [1 - (1 - .6 * .7) * (1 - .8 * expected_music), .7, expected_music, .6, .8]
    # When: deployment scores one file and then a copy whose masked features are adversarial.
    direct = score_ensemble(heads, record, 'cpu')
    masked = {name: value.copy() for name, value in record.items()}
    masked['raw'][[1, 3]] = 1e8
    masked['stem'][[1, 3]] = -1e8
    altered = score_ensemble(heads, masked, 'cpu')
    # Then: all component scores match reference and invalid windows cannot affect the bag.
    np.testing.assert_allclose(direct, expected, atol=1e-7, rtol=0)
    np.testing.assert_array_equal(direct, altered)
    silent = {name: value.copy() for name, value in record.items()}
    silent['stem_audible'][:] = False
    silent['aux'][:, 4] = 0
    assert score_ensemble(heads, silent, 'cpu')[2] < 1e-40
    raw_equal = {name: value.copy() for name, value in record.items()}
    raw_equal['raw'] = raw_equal['stem'].copy()
    assert abs(score_ensemble(heads, raw_equal, 'cpu')[2] - direct[2]) > 1e-7


def _baseline_io() -> types.ModuleType:
    # Execute the original IO functions, avoiding irrelevant encoder imports in this CPU check.
    tree = ast.parse(Path(__file__).with_name('baseline_original').joinpath('script.py').read_text())
    names = {'parse_arguments', 'find_audio_files', 'read_sample_submission',
             'order_audio_files', 'save_submission'}
    import argparse
    module = types.ModuleType('baseline_io')
    module.__dict__.update(argparse=argparse, Path=Path, csv=csv,
        DEFAULT_TEST_DIR=Path('data/test'), DEFAULT_SAMPLE_SUBMISSION=Path('data/sample_submission.csv'),
        DEFAULT_OUTPUT_PATH=Path('output/submission.csv'), SUPPORTED_AUDIO_EXTENSIONS={'.wav'})
    from graph_scoring import PREDICTION_COLUMNS
    module.PREDICTION_COLUMNS = list(PREDICTION_COLUMNS)
    selected = ast.Module(body=[node for node in tree.body
                               if isinstance(node, ast.FunctionDef) and node.name in names], type_ignores=[])
    exec(compile(selected, 'baseline_original/script.py', 'exec'), module.__dict__)
    return module


def test_runtime_preserves_official_id_order_and_file_independence(tmp_path: Path, monkeypatch) -> None:
    # Given: official baseline IO, opposite sample/audio sort orders and deterministic per-file features.
    import redesign_submission as runtime
    from graph_scoring import SUBMISSION_COLUMNS

    heads = runtime.load_heads(_checkpoints(tmp_path))
    base = _baseline_io()
    audio = tmp_path / 'data/test'
    audio.mkdir(parents=True)
    for name in ('a', 'b'):
        (audio / f'{name}.wav').write_bytes(b'fixture boundary')
    sample = tmp_path / 'data/sample_submission.csv'
    with sample.open('w', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(SUBMISSION_COLUMNS)
        writer.writerows([[name, 0, 0, 0, 0, 0] for name in ('b', 'a')])
    records = {'a': _fixture(), 'b': _fixture()}
    records['b']['raw'] = -records['b']['raw']
    calls = []
    class Encoder:
        def __init__(self, assets, device):
            self.base = base
        def extract(self, path):
            calls.append(path.stem)
            return records[path.stem]
    manifest = {'choice': {'checkpoints': ['fake.pt']}}
    (tmp_path / 'model').mkdir()
    (tmp_path / 'model/manifest.json').write_text(json.dumps(manifest))
    monkeypatch.setattr(runtime, 'verify_package', lambda root: manifest)
    monkeypatch.setattr(runtime, '_load_baseline', lambda path: base)
    monkeypatch.setattr(runtime, 'FrozenEncoders', Encoder)
    monkeypatch.setattr(runtime, 'load_heads', lambda paths, device: heads)
    monkeypatch.setattr(sys, 'argv', ['script.py', '--device', 'cpu'])
    monkeypatch.chdir(tmp_path)
    # When: the same entrypoint used by script.py processes the sample submission.
    runtime.main(tmp_path)
    with (tmp_path / 'output/submission.csv').open() as stream:
        rows = list(csv.DictReader(stream))
    # Then: calls and output follow official IDs, each score equals isolated-file inference.
    assert calls == ['b', 'a']
    assert [row['ID'] for row in rows] == ['b', 'a']
    for row in rows:
        expected = runtime.score_ensemble(heads, records[row['ID']], 'cpu')
        np.testing.assert_allclose([float(row[k]) for k in SUBMISSION_COLUMNS[1:]], expected,
                                   atol=5e-11, rtol=0)


def test_packaged_runtime_imports_and_hashes_match_written_bytes(tmp_path: Path) -> None:
    # Given: the exact approved asset-name set with tiny stand-ins and real trained-head serialization.
    from package_redesign_submission import ASSET_NAMES, package
    from redesign_features_cache import receipt_payload
    from redesign_submission import SubmissionError, verify_package

    root = Path(__file__).resolve().parent
    assets = tmp_path / 'assets'
    asset_hashes = {}
    for name in ASSET_NAMES:
        path = assets / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(f'packaging-only asset fixture: {name}'.encode())
        asset_hashes[name] = sha256(path)
    (assets / 'asset_hashes.json').write_text(json.dumps(asset_hashes))
    checkpoints = _checkpoints(tmp_path)
    receipt = dict(training_mode='full_data_refit', arm='B', seeds=[41, 42, 43], epochs=[17, 17, 8],
                   checkpoints=[dict(seed=seed, epochs=epochs, checkpoint_sha256=sha256(path))
                                for path, seed, epochs in zip(checkpoints, (41, 42, 43), (17, 17, 8), strict=True)],
                   train_ids=[f'train-{i:04}' for i in range(1696)], label_eligible_music=1216,
                   independent_validation=False, file_fusion='noisy_or',
                   feature_receipt=receipt_payload(assets, asset_hashes))
    receipt_path = tmp_path / 'fullrefit_receipt.json'
    receipt_path.write_text(json.dumps(receipt))
    # When: package, reopen and import from the extracted runtime in a separate process.
    archive_path = tmp_path / 'B.zip'
    report = package(root, archive_path, assets=assets, checkpoints=checkpoints, fullrefit_receipt=receipt_path)
    extracted = tmp_path / 'unpacked'
    with zipfile.ZipFile(archive_path) as archive:
        archive.extractall(extracted)
        hashes = {name: hashlib.sha256(archive.read(name)).hexdigest() for name in archive.namelist()}
        assert archive.testzip() is None
    env = dict(__import__('os').environ, PYTHONPATH=str(extracted / 'model/runtime'))
    command = "from redesign_submission import load_heads; from pathlib import Path; load_heads([Path(f'model/trained/B_seed{s}.pt') for s in (41,42,43)]); print('head-runtime-ok')"
    result = subprocess.run([sys.executable, '-c', command], cwd=extracted, env=env,
                            capture_output=True, text=True, check=True)
    # Then: every archived byte is externally hashed and the isolated dependency closure reloads the heads.
    assert 'head-runtime-ok' in result.stdout
    assert hashes == report['member_sha256']
    manifest = verify_package(extracted)
    assert set(hashes) == set(manifest['sha256']) | {'model/manifest.json'}
    assert not any(name.startswith(('data/', 'output/')) for name in hashes)
    (extracted / 'model/trained/B_seed41.pt').write_bytes(b'corrupted')
    with pytest.raises(SubmissionError, match='digest mismatch'):
        verify_package(extracted)
    # A stale full-refit receipt must fail before creating a replacement artifact.
    receipt['checkpoints'][0]['checkpoint_sha256'] = '0' * 64
    receipt_path.write_text(json.dumps(receipt))
    with pytest.raises(SubmissionError, match='receipt'):
        package(root, tmp_path / 'rejected.zip', assets=assets, checkpoints=checkpoints,
                fullrefit_receipt=receipt_path)
