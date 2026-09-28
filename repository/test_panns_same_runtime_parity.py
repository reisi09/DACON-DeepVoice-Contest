"""Reject substituted references and broadcastable mismatches before a costly gate."""
import json
from pathlib import Path

import numpy as np
import pytest

import panns_cloud_pipeline as pipeline
from decision_heads import SINGING, VOICE_CORE
from panns_adaptation_cache import cache_key
from panns_adaptation_data import cache_provenance, identity, sha256, write_json


def parity_fixture(tmp_path):
    labels = list(dict.fromkeys([*VOICE_CORE, *SINGING, 'Music']))
    groups = dict(voice=list(VOICE_CORE)+list(SINGING), music=['Music'])
    head = dict(features=['tag_voice_max'], mean=[0.], scale=[1.], coef=[1.], intercept=0.)
    assets = (dict(VOICE_PRESENT=head, MUSIC_PRESENT=head), labels, groups)
    tags = np.full((1, len(labels)), .5, np.float32)
    row = dict(ID='source:a', role='train', component_id='parent:a', VOICE_PRESENT=1,
               MUSIC_PRESENT=0, verified_audio=dict(sha256='audio-sha'))
    rows = tmp_path/'rows.jsonl'; rows.write_text(json.dumps(row)+'\n')
    reference = tmp_path/'reference.json'; write_json(reference, dict(package=str(tmp_path), members={}))
    gold_path = tmp_path/'gold/a.npz'; gold_path.parent.mkdir()
    np.savez(gold_path, identity='preserved-mac', block5=np.zeros((1, 2), np.float32),
             tags=tags+np.float32(2e-5), starts=np.array([0]))
    base_entry = dict(ID=row['ID'], role=row['role'], component_id=row['component_id'],
                      targets=[1, 0], stems={}, segments=1, seconds=1., tag_error=0.,
                      stats_error=0., presence_error=0.)
    gold_entry = dict(base_entry, path=str(gold_path), identity='preserved-mac', sha256=sha256(gold_path))
    write_json(gold_path.parent/'index.json', dict(entries=[gold_entry]))
    manifest = tmp_path/'bundle_manifest.json'
    write_json(manifest, dict(sha256={'gold/index.json': sha256(gold_path.parent/'index.json'),
                                    'gold/a.npz': sha256(gold_path)}))
    config = dict(output=str(tmp_path/'cuda_preflight'), cache_device='cuda', rows=str(rows),
                  reference=str(reference), panel_ids=[row['ID']], parity_runtime=pipeline.parity_runtime(),
                  inputs={str(path): sha256(path) for path in (rows, reference, manifest)})
    config['identity'] = identity(config)
    for current in (config, pipeline.cpu_reference_config(config)):
        path = Path(current['output'])/'cache/a.npz'; path.parent.mkdir(parents=True)
        key = cache_key(current, row, [0])
        np.savez(path, identity=key, block5=np.zeros((1, 2), np.float32), tags=tags, starts=np.array([0]))
        entry = dict(base_entry, path=str(path), identity=key, sha256=sha256(path))
        write_json(path.parent/'index.json', dict(config_identity=current['identity'],
                   provenance=cache_provenance(), entries=[entry], stem_reused_files=0))
        (path.parent/'index.sha256').write_text(sha256(path.parent/'index.json')+'\n')
    return config, assets


def test_separate_mac_drift_does_not_change_same_runtime_reference(tmp_path):
    # Given authenticated artifacts with Mac drift above the unchanged device tolerance.
    config, assets = parity_fixture(tmp_path)
    before = sha256(tmp_path/'gold/index.json'), sha256(tmp_path/'gold/a.npz')
    # When both independently bound cache indexes and their real NPZ payloads are read.
    cpu_config, cpu, cuda, gold = pipeline.parity_indexes(config, tmp_path)
    same_runtime = pipeline.prediction_errors(cpu, cuda, assets)
    cross_platform = pipeline.cross_platform_errors(gold, cpu, assets)
    # Then device parity is exact while the historical cross-platform failure stays visible.
    assert same_runtime == dict(tag_max_abs=0., presence_max_abs=0.)
    assert cross_platform['tag_max_abs'] > 1e-5
    assert cross_platform['comparable'] is True and cross_platform['within_legacy_tolerance'] is False
    assert cpu_config['identity'] != config['identity']
    assert before == (sha256(tmp_path/'gold/index.json'), sha256(tmp_path/'gold/a.npz'))


@pytest.mark.parametrize('mismatch', ['cuda_substitution', 'runtime', 'base_forward', 'gold_hash'])
def test_reference_mismatch_is_rejected_before_comparison(tmp_path, mismatch):
    config, _ = parity_fixture(tmp_path)
    cpu_config = pipeline.cpu_reference_config(config)
    cpu_path = Path(cpu_config['output'])/'cache/index.json'
    if mismatch == 'runtime':
        config['parity_runtime']['architecture'] = 'different-architecture'
        expected = 'producer platform'
    elif mismatch == 'gold_hash':
        (tmp_path/'gold/a.npz').write_bytes(b'changed historical gold')
        expected = 'gold activation hash'
    else:
        saved = json.loads(cpu_path.read_text())
        if mismatch == 'cuda_substitution':
            saved = json.loads((Path(config['output'])/'cache/index.json').read_text())
            expected = 'Cache/config identity'
        else:
            saved['entries'][0]['tag_error'] = 2e-6
            expected = 'original-forward/readout parity'
        write_json(cpu_path, saved); cpu_path.with_suffix('.sha256').write_text(sha256(cpu_path)+'\n')
    with pytest.raises(ValueError, match=expected):
        pipeline.parity_indexes(config, tmp_path)


def test_broadcastable_segment_mismatch_cannot_pass_zero_tag_error(tmp_path):
    config, assets = parity_fixture(tmp_path)
    _, cpu, cuda, _ = pipeline.parity_indexes(config, tmp_path)
    entry = cuda['entries'][0]; path = Path(entry['path'])
    with np.load(path, allow_pickle=False) as saved:
        values = {key: saved[key].copy() for key in saved.files}
    # Broadcasting would report zero error for this duplicated segment.
    values['tags'] = np.repeat(values['tags'], 2, axis=0)
    np.savez(path, **values); entry['sha256'] = sha256(path)
    with pytest.raises(ValueError, match='segment starts, counts or tensor shapes'):
        pipeline.prediction_errors(cpu, cuda, assets)


def test_panns_gate_fixes_frozen_stems_but_still_detects_tag_drift(tmp_path):
    config, _ = parity_fixture(tmp_path)
    _, cpu, cuda, _ = pipeline.parity_indexes(config, tmp_path)
    labels = list(dict.fromkeys([*VOICE_CORE, *SINGING, 'Music']))
    head = dict(features=['tag_voice_max', 'stem_vocal_ratio_max'], mean=[0., 0.],
                scale=[1., 1.], coef=[1., 1.], intercept=0.)
    assets = (dict(VOICE_PRESENT=head, MUSIC_PRESENT=head), labels,
              dict(voice=list(VOICE_CORE)+list(SINGING), music=['Music']))
    cpu['entries'][0]['stems'] = {'stem_vocal_ratio_max': .2}
    cuda['entries'][0]['stems'] = {'stem_vocal_ratio_max': .201}
    before = json.dumps([cpu, cuda], sort_keys=True)
    assert pipeline.prediction_errors(cpu, cuda, assets)['presence_max_abs'] > 1e-5
    assert pipeline.prediction_errors(cpu, cuda, assets, shared_actual_stems=True) == dict(
        tag_max_abs=0., presence_max_abs=0.)
    assert json.dumps([cpu, cuda], sort_keys=True) == before
    entry = cuda['entries'][0]; path = Path(entry['path'])
    with np.load(path, allow_pickle=False) as saved:
        values = {key: saved[key].copy() for key in saved.files}
    values['tags'] += np.float32(.01)
    np.savez(path, **values); entry['sha256'] = sha256(path)
    errors = pipeline.prediction_errors(cpu, cuda, assets, shared_actual_stems=True)
    assert errors['tag_max_abs'] > 1e-5 and errors['presence_max_abs'] > 1e-5


def test_mac_layout_mismatch_blocks_even_when_same_runtime_tags_match(tmp_path):
    config, assets = parity_fixture(tmp_path)
    _, cpu, cuda, gold = pipeline.parity_indexes(config, tmp_path)
    entry = gold['entries'][0]; path = Path(entry['path'])
    with np.load(path, allow_pickle=False) as saved:
        values = {key: saved[key].copy() for key in saved.files}
    assert pipeline.prediction_errors(cpu, cuda, assets)['tag_max_abs'] == 0.
    for starts in (np.array([1]), np.array([0, 1])):
        values['starts'] = starts
        np.savez(path, **values); entry['sha256'] = sha256(path)
        with pytest.raises(pipeline.ParityLayoutMismatch, match='source:a'):
            pipeline.cross_platform_errors(gold, cpu, assets)
    entry['sha256'] = '0'*64
    with pytest.raises(ValueError, match='Cache hash mismatch'):
        pipeline.cross_platform_errors(gold, cpu, assets)
