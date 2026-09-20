"""v2 regressions: synthetic cache/policy wiring, NOT real audio/GPU parity.

Defects prevented: gain-policy mixing, manifest/cache disagreement, whole-file
measurements incorrectly treated as a global policy, and policy bypass at public
production entrypoints. Synthetic records deliberately have production-shaped
metadata solely to exercise these guards; they are never performance evidence.
"""
import copy
import json
import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from dacon_stacking.common import ContractError, digest, read_json, sha256, write_json
from dacon_stacking.features import SCHEMA, write_extended, read_extended, from_measurements
from dacon_stacking.workflow import load_records

ROOT = Path(os.environ.get('STACKING_INPUT_ROOT', '/mnt/data/input_v2'))
POLICY = {'sr': 16000, 'mono': True, 'default_peak_cap': .6, 'preserve_gain': False,
          'demucs': 'shifts=0;split=True;overlap=.25;trim/right-pad; source snapshot hashed'}


def cache_fixture(directory, flags=(False, False), split='meta_dev', count=None, tag='dev'):
    """Real NPZ/JSON I/O with explicitly synthetic scores and metadata."""
    directory = Path(directory); directory.mkdir(parents=True, exist_ok=True)
    flags = list(flags) if count is None else [False] * count
    rows, records = [], []
    index = {'schema': 'dacon.stacking.cache_index.v1', 'items': {}}
    for i, flag in enumerate(flags):
        identifier = f'v2_TEST_ONLY_{tag}_{i}'
        parent = f'v2_TEST_ONLY_{tag}_parent_{i}'
        audio_hash = digest({'tag': tag, 'row': i})
        y = i % 2
        exposure = {'head_train': 'no', 'head_selection': 'no', 'historical_development': 'no',
                    'upstream': 'unknown', 'evidence': ['SYNTHETIC TEST WIRING; NOT ACQUIRED AUDIO'],
                    'reviewed_parent_ids': [parent]}
        row = {'ID': identifier, 'split': split, 'category': 'Voice-only',
               'parents': [{'parent_id': parent, 'audio_sha256': audio_hash,
                            'speaker': parent, 'track': '', 'source': 'TEST_SOURCE_' + tag,
                            'generator_family': 'TEST_GENERATOR_' + tag if y else 'real'}],
               'audio_sha256': audio_hash, 'preserve_gain': flag, 'base_exposure': exposure,
               'FILE_FAKE': y, 'FILE_FAKE_mask': 1, 'VOICE_FAKE': y, 'VOICE_FAKE_mask': 1,
               'MUSIC_FAKE': None, 'MUSIC_FAKE_mask': 0,
               'VOICE_PRESENT': 1, 'VOICE_PRESENT_mask': 1, 'MUSIC_PRESENT': 0, 'MUSIC_PRESENT_mask': 1}
        pv, pm, fv, fm = .85, .1, .15 + .6*y, .2
        record = {'ID': identifier, 'baseline': {'FILE_FAKE_PROB': max(pv*fv, pm*fm),
                  'VOICE_FAKE_PROB': fv, 'MUSIC_FAKE_PROB': fm,
                  'VOICE_PRESENT_PROB': pv, 'MUSIC_PRESENT_PROB': pm},
                  'features': {'Pv': pv, 'Pm': pm, 'Fv': fv, 'Fm': fm}}
        peak = .3 + .4*(i % 3)
        provenance = {'audio_sha256': audio_hash, 'code_sha256': {'test': 'b'*64},
                      'checkpoint_sha256': {'test': 'c'*64}, 'numeric_mode': 'synthetic_cpu_test',
                      'feature_schema': SCHEMA, 'timebase': {'unit': 'samples', 'sr': 16000},
                      'preprocessing': {**POLICY, 'preserve_gain': flag, 'source_peak': peak,
                                        'normalization_gain': .6/peak if not flag and peak > .6 else 1.},
                      'base_exposure': exposure}
        path = directory / (identifier + '.npz')
        side_hash = write_extended(path, record, provenance)
        index['items'][identifier] = {'path': path.name, 'sidecar_sha256': side_hash,
                                     'identity': digest(provenance)}
        rows.append(row); records.append(record)
    write_json(directory/'index.json', index)
    with (directory/'manifest.jsonl').open('w', encoding='utf8') as stream:
        for row in rows: stream.write(json.dumps(row) + '\n')
    return rows, records, directory/'index.json', directory/'manifest.jsonl'


def mutate_cache(index_path, identifier, change):
    """Re-sign a deliberately changed test cache: detect semantics, not just hashes."""
    index = read_json(index_path); item = index['items'][identifier]
    side = Path(index_path).parent / Path(item['path']).with_suffix('.json')
    meta = read_json(side); change(meta['provenance']['preprocessing'])
    meta['identity'] = digest(meta['provenance']); write_json(side, meta)
    item['sidecar_sha256'] = sha256(side); item['identity'] = meta['identity']
    write_json(index_path, index)


def test_same_policy_accepts_distinct_per_file_peak_and_gain(tmp_path):
    rows, _, index, _ = cache_fixture(tmp_path)
    records = load_records(index, rows)
    assert records[0]['provenance']['preprocessing']['source_peak'] != records[1]['provenance']['preprocessing']['source_peak']
    assert records[0]['provenance']['preprocessing']['normalization_gain'] != records[1]['provenance']['preprocessing']['normalization_gain']
    assert len(records) == 2


@pytest.mark.parametrize('case', ['mixed', 'manifest_false_cache_true', 'all_true', 'wrong_cap', 'wrong_sr', 'wrong_demucs', 'missing_policy'])
def test_load_records_rejects_incompatible_preprocessing(tmp_path, case):
    flags = (False, True) if case == 'mixed' else ((True, True) if case == 'all_true' else (False, False))
    rows, _, index, _ = cache_fixture(tmp_path, flags)
    mutations = {'manifest_false_cache_true': lambda p: p.update(preserve_gain=True),
                 'wrong_cap': lambda p: p.update(default_peak_cap=.9),
                 'wrong_sr': lambda p: p.update(sr=24000),
                 'wrong_demucs': lambda p: p.update(demucs='shifts=1'),
                 'missing_policy': lambda p: p.pop('preserve_gain')}
    if case in mutations: mutate_cache(index, rows[1]['ID'], mutations[case])
    with pytest.raises(ContractError, match='preprocessing|preserve_gain'):
        load_records(index, rows)


def test_manifest_rejects_gain_preservation_and_non_boolean_flag(tmp_path):
    from dacon_stacking.provenance import validate_manifest
    rows, _, _, _ = cache_fixture(tmp_path)
    validate_manifest(rows, registry=set())
    for invalid in (True, 'False', 0, None):
        bad = copy.deepcopy(rows); bad[1]['preserve_gain'] = invalid
        with pytest.raises(ContractError, match='preprocessing|preserve_gain'):
            validate_manifest(bad, registry=set())


def test_extract_rejects_every_bad_row_before_loading_engine(tmp_path, monkeypatch):
    from dacon_stacking import gpu
    rows, _, _, _ = cache_fixture(tmp_path/'input', flags=(False, True))
    def forbidden(*args, **kwargs):
        pytest.fail('Native GPU loader reached before preprocessing rejection')
    monkeypatch.setattr(gpu, 'load_local_engine', forbidden)
    with pytest.raises(ContractError, match='preprocessing|preserve_gain'):
        gpu.extract_audio(rows, tmp_path/'no-assets', tmp_path/'output')
    assert not (tmp_path/'output').exists()


def test_single_file_adapter_rejects_true_before_predict(tmp_path):
    from dacon_stacking.gpu import record_from_engine
    def forbidden(*args, **kwargs):
        pytest.fail('Engine.predict reached for unsupported gain policy')
    with pytest.raises(ContractError, match='preprocessing|preserve_gain'):
        record_from_engine(SimpleNamespace(predict=forbidden), tmp_path/'unused.wav', 'TEST', preserve_gain=True)


def test_deployment_and_ood_lock_require_explicit_production_policy(tmp_path, monkeypatch):
    from dacon_stacking import gpu, workflow
    base = {'schema': 'dacon.stacking.lock.v1', 'purpose': 'production_candidate',
            'feature_set': 'F0', 'feature_extraction_identity': {}, 'ood_manifest_sha256': 'a'*64}
    def forbidden(*args, **kwargs):
        pytest.fail('Runtime identity inspected before rejecting invalid/missing preprocessing')
    monkeypatch.setattr(gpu, 'runtime_identity', forbidden)
    for policy in (None, {**POLICY, 'preserve_gain': True}):
        lock = copy.deepcopy(base)
        if policy is not None: lock['preprocessing_policy'] = policy
        path = tmp_path/'candidate_lock.json'; write_json(path, lock)
        with pytest.raises(ContractError, match='preprocessing|preserve_gain'):
            workflow.verify_ood_lock(path, sha256(path), 'a'*64)
        with pytest.raises(ContractError, match='preprocessing|preserve_gain'):
            gpu.load_deployment(path, sha256(path), tmp_path/'no-assets')


def test_production_cache_inference_cannot_bypass_policy(tmp_path):
    from dacon_stacking.models import fit_estimator
    from dacon_stacking.features import feature_frame
    from dacon_stacking.workflow import infer_record
    rows, plain, index, _ = cache_fixture(tmp_path, count=12)
    x = feature_frame(plain, 'F0')
    model, _ = fit_estimator('lr', {'C': 1.}, x, np.arange(12)%2, np.arange(12).astype(str))
    model['provenance'] = {'purpose': 'production_candidate', 'preprocessing_policy': POLICY}
    good = load_records(index, rows)[0]
    assert set(infer_record(good, model)) == {'ID', *good['baseline']}
    for policy in (None, {**POLICY, 'preserve_gain': True}):
        bad = copy.deepcopy(good)
        if policy is None: bad['provenance'].pop('preprocessing')
        else: bad['provenance']['preprocessing'] = policy
        with pytest.raises(ContractError, match='preprocessing|preserve_gain'):
            infer_record(bad, model)


def workflow_fixture(directory, corrupt_ood=False):
    """Exercise real dev/final-fit/JSON-bundle/lock code, on synthetic scores only."""
    from dacon_stacking.workflow import run_dev, lock_candidate
    pytest.importorskip('xgboost')
    directory = Path(directory)
    rows, _, index, manifest = cache_fixture(directory/'dev', count=36, tag='dev')
    ood_rows, _, ood_index, ood_manifest = cache_fixture(directory/'ood', count=12, tag='ood', split='locked_ood')
    if corrupt_ood:
        mutate_cache(ood_index, ood_rows[0]['ID'], lambda p: p.update(preserve_gain=True))
    config = {'seed': 41, 'outer_folds': 3, 'inner_folds': 2, 'bootstrap': 40,
              'weighting': 'parent_equal', 'group_mode': 'parent', 'feature_sets': ['F0'],
              'grids': {'xgb': [{'max_depth': 2, 'n_estimators': 8, 'min_child_weight': 1}]}}
    config_path = directory/'config.json'; write_json(config_path, config)
    report = run_dev(manifest, index, config_path, ROOT, directory/'dev_result')
    assert report['systems']['F0/xgb']['status'] == 'OK'
    receipt = lock_candidate(manifest, index, config_path, ROOT,
                             directory/'dev_result/dev_report.json', 'xgb', 'F0',
                             sha256(ood_manifest), sha256(ood_index), directory/'candidate')
    return {'rows': rows, 'index': index, 'manifest': manifest, 'config': config_path,
            'ood_rows': ood_rows, 'ood_index': ood_index, 'ood_manifest': ood_manifest,
            'lock': directory/'candidate/candidate_lock.json', 'receipt': receipt,
            'report': directory/'dev_result/dev_report.json'}


class MeasuredTestEngine:
    """CPU test double for the unavailable native encoder boundary; no audio decode."""
    def __init__(self): self.calls = []
    def predict(self, path, *, preserve_gain=False, collect_features=False):
        self.calls.append((str(path), preserve_gain, collect_features))
        peak = .3 if Path(path).name.startswith('low') else 1.2
        baseline = {'FILE_FAKE_PROB': .85*.3, 'VOICE_FAKE_PROB': .3, 'MUSIC_FAKE_PROB': .2,
                    'VOICE_PRESENT_PROB': .85, 'MUSIC_PRESENT_PROB': .1}
        measurements = {'voice_seed': [.3]*3, 'music_seed': [.2]*3,
                        'voice_window': np.full((3, 1), .3), 'music_window': np.full((3, 1), .2),
                        'df_original': .4, 'samples': 32000, 'source_peak': peak,
                        'normalization_gain': .6/peak if peak > .6 and not preserve_gain else 1.,
                        'normalized_rms': .1, 'voice_silent': False}
        return baseline, measurements


def test_dev_lock_ood_and_deployment_bind_policy_not_file_measurements(tmp_path, monkeypatch):
    from dacon_stacking import workflow, gpu
    from dacon_stacking.models import load_bundle, save_bundle
    f = workflow_fixture(tmp_path)
    lock = read_json(f['lock'])
    meta = read_json(f['lock'].parent/'bundle/manifest.json')
    assert read_json(f['report'])['preprocessing_policy'] == POLICY
    assert lock['preprocessing_policy'] == meta['provenance']['preprocessing_policy'] == POLICY
    assert 'source_peak' not in lock['preprocessing_policy']
    assert 'normalization_gain' not in lock['preprocessing_policy']
    result = workflow.evaluate_ood(f['ood_manifest'], f['ood_index'], ROOT, f['lock'],
                                   f['receipt']['lock_sha256'], tmp_path/'ood_result')
    assert result['preprocessing_policy'] == POLICY
    assert read_json(f['lock'].parent/'OOD_OPENED.json')['status'] == 'COMPLETE'
    # Full native asset verification is explicitly not executed in this CPU test.
    monkeypatch.setattr(gpu, 'runtime_identity', lambda root: lock['feature_extraction_identity'])
    model, contract = gpu.load_deployment(f['lock'], f['receipt']['lock_sha256'], tmp_path/'NO_REAL_ASSETS')
    engine = MeasuredTestEngine()
    prediction = gpu.predict_engine_file(engine, tmp_path/'low.synthetic', 'synthetic', model, contract)
    assert engine.calls == [(str(tmp_path/'low.synthetic'), False, True)]
    baseline, _ = MeasuredTestEngine().predict(tmp_path/'low.synthetic')
    for key in baseline:
        if key != 'FILE_FAKE_PROB': assert prediction[key] == baseline[key]
    bad = copy.deepcopy(contract); bad['preprocessing_policy']['preserve_gain'] = True
    with pytest.raises(ContractError, match='preprocessing|preserve_gain'):
        gpu.predict_engine_file(engine, tmp_path/'high.synthetic', 'synthetic', model, bad)
    assert len(engine.calls) == 1
    # Production bundle I/O must not accept pre-v2 missing policy, even with valid hashes.
    bad_provenance = copy.deepcopy(model['provenance']); bad_provenance.pop('preprocessing_policy')
    with pytest.raises(ContractError, match='preprocessing'):
        save_bundle(tmp_path/'bad_bundle', model, bad_provenance)
    meta['provenance'].pop('preprocessing_policy')
    bundle = f['lock'].parent/'bundle'; write_json(bundle/'manifest.json', meta)
    with pytest.raises(ContractError, match='preprocessing'):
        load_bundle(bundle, sha256(bundle/'manifest.json'))


def test_actual_dev_lock_and_ood_commands_refuse_policy_gaps(tmp_path):
    from dacon_stacking import workflow
    f = workflow_fixture(tmp_path, corrupt_ood=True)
    # The OOD digest was locked BEFORE opening its intentionally bad cache.
    with pytest.raises(ContractError, match='preprocessing|preserve_gain'):
        workflow.evaluate_ood(f['ood_manifest'], f['ood_index'], ROOT, f['lock'],
                              f['receipt']['lock_sha256'], tmp_path/'bad_ood_result')
    assert not (tmp_path/'bad_ood_result').exists()
    assert read_json(f['lock'].parent/'OOD_OPENED.json')['status'].startswith('OPENED;')
    report = read_json(f['report']); report.pop('preprocessing_policy'); write_json(f['report'], report)
    with pytest.raises(ContractError, match='preprocessing'):
        workflow.lock_candidate(f['manifest'], f['index'], f['config'], ROOT, f['report'],
                                'xgb', 'F0', sha256(f['ood_manifest']), sha256(f['ood_index']), tmp_path/'bad_lock')
    assert not (tmp_path/'bad_lock').exists()
    mutate_cache(f['index'], f['rows'][0]['ID'], lambda p: p.update(preserve_gain=True))
    with pytest.raises(ContractError, match='preprocessing|preserve_gain'):
        workflow.run_dev(f['manifest'], f['index'], f['config'], ROOT, tmp_path/'bad_dev')
    assert not (tmp_path/'bad_dev').exists()


def test_extraction_roundtrip_allows_file_measurements_to_differ(tmp_path, monkeypatch):
    from dacon_stacking import gpu
    rows, _, _, _ = cache_fixture(tmp_path/'input')
    for i, row in enumerate(rows):
        path = tmp_path / (('low' if i == 0 else 'high') + '.synthetic')
        path.write_bytes(f'SYNTHETIC TEST BYTES, NOT AUDIO: {i}'.encode())
        row['audio_path'] = str(path); row['audio_sha256'] = sha256(path)
    engine = MeasuredTestEngine()
    monkeypatch.setattr(gpu, 'load_local_engine', lambda root: engine)
    monkeypatch.setattr(gpu, 'runtime_identity', lambda root: {
        'code_sha256': {'test': 'b'*64}, 'checkpoint_sha256': {'test': 'c'*64},
        'numeric_mode': 'synthetic_cpu_test', 'feature_schema': SCHEMA})
    gpu.extract_audio(rows, tmp_path/'NO_REAL_ASSETS', tmp_path/'extracted')
    records = load_records(tmp_path/'extracted/index.json', rows)
    assert all(call[1:] == (False, True) for call in engine.calls)
    assert [r['provenance']['preprocessing']['source_peak'] for r in records] == [.3, 1.2]
    assert [r['provenance']['preprocessing']['normalization_gain'] for r in records] == [1., .5]
