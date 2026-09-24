"""Protect label alignment, optimizer role boundaries, and train/runtime score parity."""
import json

import numpy as np
import pytest
import torch
from torch import nn

from full_bank_batches import Item
from full_bank_contract import config_for, pool_voice
from full_bank_features import voice_window_truth
from full_bank_train import forward, step


def test_repeated_short_voice_uses_repeated_labels_and_unknown_stays_masked():
    row = {'event_intervals': json.dumps([
        dict(component='voice', start=0, end=1, label=0),
        dict(component='voice', start=1, end=2, label=1)])}
    truth = voice_window_truth(row, np.array([0]), 32000, 64600)
    assert abs(truth[0] - 32000 / 64600) < 1e-7
    assert np.isnan(voice_window_truth({'event_intervals': '[]'}, np.array([0]), 32000, 64600)).all()


def test_voice_local_supervision_changes_real_classifier_update_and_microbatch_is_equivalent():
    torch.manual_seed(4)
    base = nn.Linear(1280, 2)
    items = [Item({'ID': str(i), 'split': 'train', 'role': 'train'},
                  np.random.default_rng(i).normal(size=(3, 1280)).astype('float32'),
                  np.array([0., 1., np.nan], dtype='float32'), float(i % 2)) for i in range(2)]
    weights = {str(i): 1. for i in range(2)}
    states = []
    for local_weight, batch in ((0., 2), (.5, 2), (.5, 1)):
        model = nn.Linear(1280, 2); model.load_state_dict(base.state_dict())
        optimizer = torch.optim.SGD(model.parameters(), lr=.01)
        step(model, optimizer, items, 'voice', 1, weights, batch, 6, 'cpu',
             config=config_for({'voice_local_loss_weight': local_weight}))
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
        states.append(model.weight.detach().clone())
    assert not torch.allclose(states[0], states[1], atol=1e-7, rtol=0)
    torch.testing.assert_close(states[1], states[2], atol=1e-7, rtol=1e-6)
    forbidden = [Item({**items[0].row, 'role': 'calibration'}, items[0].features, items[0].truth, 0.)]
    with pytest.raises(ValueError, match='Validation|role'):
        step(base, torch.optim.SGD(base.parameters(), lr=.01), forbidden, 'voice', 1, weights, 1, 3, 'cpu')


def test_pooling_training_matches_runtime_and_file_order_is_irrelevant():
    model = nn.Linear(1280, 2)
    items = [Item({'ID': str(i)}, np.random.default_rng(i).normal(size=(i+2, 1280)).astype('float32'),
                  np.full(i+2, np.nan, dtype='float32'), 0.) for i in range(2)]
    for pooling in ('max', 'lme', 'topk'):
        config = config_for({'voice_pooling': pooling})
        actual = forward(model, items, 'voice', 1, 'cpu', config=config)[0].sigmoid()
        expected = torch.stack([pool_voice(model(torch.from_numpy(x.features)), 1, config).sigmoid() for x in items])
        torch.testing.assert_close(actual, expected, atol=1e-7, rtol=1e-6)
        reverse = forward(model, list(reversed(items)), 'voice', 1, 'cpu', config=config)[0].sigmoid().flip(0)
        torch.testing.assert_close(actual, reverse, atol=1e-7, rtol=1e-6)


def test_configured_seed_loading_agrees_between_evaluator_and_package(tmp_path):
    from full_bank_evaluate import load_selected_heads
    from full_bank_submission import load_heads
    from localization_models import WindowMIL
    for seeds in ([41], [41, 42, 43]):
        config = config_for({'seeds': seeds, 'voice_pooling': 'topk'})
        (tmp_path / 'run_identity.json').write_text(json.dumps({'config': config}))
        for seed in seeds:
            torch.manual_seed(seed)
            for route, model in [('music', WindowMIL(torch.zeros(1024), torch.ones(1024))), ('voice', nn.Linear(1280, 2))]:
                torch.save(dict(model=model.state_dict(), route=route, seed=seed, fake_label_index=1,
                                identity={'config': config}), tmp_path / f'{route}_{seed}.pt')
        evaluated = load_selected_heads(tmp_path, 'cpu')
        packaged = load_heads(tmp_path, 'cpu')
        x = torch.randn(5, 1280)
        a = torch.stack([pool_voice(h(x), 1, config).sigmoid() for h in evaluated[1]]).mean()
        b = torch.stack([pool_voice(h(x), 1, config).sigmoid() for h in packaged[1]]).mean()
        torch.testing.assert_close(a, b, atol=0, rtol=0)
    (tmp_path / 'run_identity.json').write_text(json.dumps({'config': config_for({'seeds': seeds})}))
    with pytest.raises(ValueError, match='identity differs|settings differ'):
        load_heads(tmp_path, 'cpu')


def test_packager_carries_configured_heads_contract_and_preserves_existing_output(tmp_path):
    import hashlib
    import zipfile
    from package_datav4_submission import package, REQUIRED_BASE
    from full_bank_submission import load_heads
    from localization_models import WindowMIL
    config = config_for({'seeds': [41], 'voice_pooling': 'lme'})
    from pathlib import Path
    members = {name: (Path(name.removeprefix('model/runtime/')).read_bytes() if name.startswith('model/runtime/') and name.endswith('.py') else b'packaging fixture bytes') for name in REQUIRED_BASE}
    members['model/manifest.json'] = b'{}'
    producer = {'assets': {n.removeprefix('model/assets/'): hashlib.sha256(v).hexdigest() for n,v in members.items() if n.startswith('model/assets/')},
                'implementation': {'mert12_submission.py': hashlib.sha256(members['model/runtime/mert12_submission.py']).hexdigest()}}
    identity = {'config': config, 'feature_contract': producer}
    heads = tmp_path / 'heads'; heads.mkdir()
    hashes = {}
    for route, model in [('music', WindowMIL(torch.zeros(1024), torch.ones(1024))), ('voice', nn.Linear(1280, 2))]:
        path = heads / f'{route}_41.pt'
        torch.save(dict(model=model.state_dict(), route=route, seed=41, fake_label_index=1,
                        identity=identity), path)
        hashes[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
    for name, content in {'candidate_lock.json': {'checkpoints': hashes, 'config': config},
                          'run_identity.json': identity, 'validation_metrics.json': {}, 'results.json': {},
                          'file_ensemble.json': {'fitted_on': 'calibration', 'weights': {'max': 1}}}.items():
        (heads / name).write_text(json.dumps(content))
    presence = tmp_path / 'presence.json'; presence.write_text('{}')
    base = tmp_path / 'base.zip'
    with zipfile.ZipFile(base, 'w') as z:
        for name, data in members.items(): z.writestr(name, data)
    base.with_suffix('.json').write_text(json.dumps(dict(sha256=hashlib.sha256(base.read_bytes()).hexdigest(),
        member_sha256={k: hashlib.sha256(v).hexdigest() for k,v in members.items()})))
    output = tmp_path / 'candidate.zip'
    receipt = package(heads/'candidate_lock.json', base, output, presence, 'cpu-contract-fixture')
    with zipfile.ZipFile(output) as z:
        extracted = tmp_path / 'extracted'; z.extractall(extracted)
        assert z.read('model/runtime/full_bank_contract.py') == __import__('pathlib').Path('full_bank_contract.py').read_bytes()
        assert {n for n in z.namelist() if n.endswith('.pt')} == {'model/trained/music_41.pt', 'model/trained/voice_41.pt'}
    expected = load_heads(heads, 'cpu')[1][0]
    packaged = load_heads(extracted/'model/trained', 'cpu')[1][0]
    x = torch.randn(5, 1280)
    torch.testing.assert_close(pool_voice(expected(x), 1, config), pool_voice(packaged(x), 1, config), atol=0, rtol=0)
    before = output.read_bytes()
    with pytest.raises(FileExistsError):
        package(heads/'candidate_lock.json', base, output, presence, 'same')
    assert output.read_bytes() == before
    assert receipt['official_l4_validated'] is False

    with zipfile.ZipFile(base, 'w') as z:
        for name, data in members.items():
            z.writestr(name, b'wrong frozen encoder' if name == 'model/assets/mert/pytorch_model.bin' else data)
    altered = dict(members); altered['model/assets/mert/pytorch_model.bin'] = b'wrong frozen encoder'
    base.with_suffix('.json').write_text(json.dumps(dict(sha256=hashlib.sha256(base.read_bytes()).hexdigest(),
        member_sha256={k: hashlib.sha256(v).hexdigest() for k,v in altered.items()})))
    with pytest.raises(ValueError, match='differ from training'):
        package(heads/'candidate_lock.json', base, tmp_path/'wrong.zip', presence, 'wrong')


def test_changed_encoder_cache_is_rejected_even_if_its_row_hashes_are_consistent():
    from full_bank_contract import feature_contract, validate_feature_contract
    identity = dict(manifest_sha256='old', assets={'encoder': 'trusted'}, window=64600, dtype='float32')
    run = {'feature_contract': feature_contract(identity)}
    validate_feature_contract(run, {**identity, 'manifest_sha256': 'new-role-panel'})
    with pytest.raises(ValueError, match='provenance'):
        validate_feature_contract(run, {**identity, 'assets': {'encoder': 'different'}})
    with pytest.raises(ValueError, match='provenance'):
        validate_feature_contract({}, identity)


def test_explicit_union_cache_reuses_exact_rows_and_evaluates_only_requested_selection(tmp_path):
    """An ablation may omit new data, but cannot change its cached labels, roles or audio."""
    import csv
    import hashlib
    from pathlib import Path
    from full_bank_contract import feature_contract
    from full_bank_evaluate import evaluate
    from full_bank_features import row_identity
    from full_bank_train import read_items
    from localization_models import WindowMIL
    from mert12_submission import WINDOW
    from v6_data import sha256

    rows = [dict(ID=name, split='val' if name == 'selection' else 'train',
                 role='selection' if name == 'selection' else 'train', route='music',
                 component_id=name, path=name + '.wav', audio_sha256=hashlib.sha256(name.encode()).hexdigest(),
                 MUSIC_FAKE=str(i % 2), MUSIC_PRESENT='1', VOICE_FAKE='', VOICE_PRESENT='0', FILE_FAKE=str(i % 2))
            for i, name in enumerate(('old', 'selection', 'new'))]

    def write_rows(path, records):
        with path.open('w', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader(); writer.writerows(records)

    union, subset = tmp_path / 'union.csv', tmp_path / 'subset.csv'
    write_rows(union, rows); write_rows(subset, rows[:2])
    cache = tmp_path / 'cache'; cache.mkdir()
    identity = dict(manifest_sha256=sha256(union), implementation={'full_bank_features.py': sha256(Path('full_bank_features.py'))})
    key = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    index = dict(identity=identity, items={})
    arrays = {}
    for i, row in enumerate(rows):
        arrays[row['ID']] = dict(identity=np.array(row_identity(row, key)), samples=np.array(WINDOW),
            starts=np.array([0]), valid_samples=np.array([WINDOW]),
            mert_windows=np.full((1, 1024), i + .25, np.float32),
            voice_embeddings=np.full((1, 1280), i + .5, np.float32),
            window_truth=np.array([np.nan], np.float32), voice_silent=np.array(False), presence=np.array([.1, .9]))
        path = cache / (row['ID'] + '.npz')
        np.savez(path, **arrays[row['ID']])
        index['items'][row['ID']] = dict(path=path.name, sha256=sha256(path))
    (cache / 'index.json').write_text(json.dumps(index))

    with pytest.raises(ValueError, match='Feature manifest mismatch'):
        read_items(subset, cache)
    items = read_items(subset, cache, union)
    assert [x.row['ID'] for x in items] == ['old', 'selection']
    for item in items:
        np.testing.assert_array_equal(item.features, arrays[item.row['ID']]['mert_windows'])
        assert item.label == float(item.row['MUSIC_FAKE'])
        assert np.isnan(item.truth).all()
    assert [x.row['ID'] for x in read_items(union, cache)] == [row['ID'] for row in rows]

    output = tmp_path / 'heads'; output.mkdir()
    config = config_for({'seeds': [41]})
    run = dict(config=config, manifest=sha256(subset), feature_manifest=sha256(union), feature_contract=feature_contract(identity))
    (output / 'run_identity.json').write_text(json.dumps(run))
    for route, model in [('music', WindowMIL(torch.zeros(1024), torch.ones(1024))), ('voice', nn.Linear(1280, 2))]:
        torch.save(dict(model=model.state_dict(), route=route, seed=41, fake_label_index=1, identity=run), output / f'{route}_41.pt')
    evaluate(subset, cache, output, 'cpu')
    with (output / 'validation_predictions.csv').open() as stream:
        assert [row['ID'] for row in csv.DictReader(stream)] == ['selection']
    assert json.loads((output / 'candidate_lock.json').read_text())['validation_manifest_sha256'] == sha256(subset)

    for change in ({'ID': 'missing'}, {'MUSIC_FAKE': '1'}, {'role': 'selection', 'split': 'val'},
                   {'component_id': 'another'}, {'path': 'other.wav'}, {'audio_sha256': 'different'}):
        write_rows(subset, [{**rows[0], **change}])
        with pytest.raises(ValueError, match='Requested row differs'):
            read_items(subset, cache, union)
    write_rows(subset, [rows[0], rows[0]])
    with pytest.raises(ValueError, match='unique'):
        read_items(subset, cache, union)
    write_rows(subset, [rows[0]])
    with pytest.raises(ValueError, match='Feature manifest mismatch'):
        read_items(subset, cache, subset)
    path = cache / 'old.npz'
    arrays['old']['mert_windows'][0, 0] = np.inf
    np.savez(path, **arrays['old'])
    with pytest.raises(ValueError, match='Cache hash changed'):
        read_items(subset, cache, union)
    index['items']['old']['sha256'] = sha256(path)
    (cache / 'index.json').write_text(json.dumps(index))
    with pytest.raises(ValueError, match='Invalid MERT windows'):
        read_items(subset, cache, union)
