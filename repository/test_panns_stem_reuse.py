from typing import Literal, assert_never

import numpy as np
import pytest

import panns_adaptation_cache as producer
from decision_heads import stem_features
from panns_adaptation_data import identity, sha256, write_json


def stem_artifact(tmp_path):
    rows = {'a': dict(ID='a', verified_audio=dict(sha256='a'*64, local_path='/local/a.wav'))}
    asset = 'model/assets/baseline/model/htdemucs/weights.th'
    reference = tmp_path/'reference.json'
    write_json(reference, dict(members={asset: 'b'*64}))
    unit = np.ones(1, np.float32)
    entry = dict(audio_sha256='a'*64, stems=stem_features(unit, unit, 16000),
                 source_sha256='c'*64, source_bank_id='original:a', source_cache='original.npz')
    payload = dict(version=1, rows_audio_identity=identity([['a', 'a'*64]]),
                   source_receipt=dict(assets={asset: 'b'*64}), entries={'a': entry})
    path = tmp_path/'stems.json'
    write_json(path, payload)
    config = dict(stem_cache=str(path), reference=str(reference), inputs={str(path): sha256(path)})
    return config, rows, payload


def test_authenticated_stems_survive_path_remap_with_values_unchanged(tmp_path):
    config, rows, payload = stem_artifact(tmp_path)
    rows['a']['verified_audio']['local_path'] = '/shared/audio/a.wav'
    assert producer.read_stem_cache(config, rows) == payload['entries']


@pytest.mark.parametrize('fault', ['unbound_input', 'audio', 'rowset', 'assets', 'feature', 'nonfinite', 'source_sha'])
def test_stem_reuse_rejects_untrusted_or_incompatible_records(
        tmp_path, fault: Literal['unbound_input', 'audio', 'rowset', 'assets', 'feature', 'nonfinite', 'source_sha']):
    config, rows, payload = stem_artifact(tmp_path)
    match fault:
        case 'unbound_input': config['inputs'].clear()
        case 'audio': payload['entries']['a']['audio_sha256'] = 'd'*64
        case 'rowset': rows['noise'] = dict(ID='noise', verified_audio=dict(sha256='e'*64))
        case 'assets': payload['source_receipt']['assets'][next(iter(payload['source_receipt']['assets']))] = 'f'*64
        case 'feature': payload['entries']['a']['stems'].pop('mix_rms_db')
        case 'nonfinite': payload['entries']['a']['stems']['mix_rms_db'] = float('nan')
        case 'source_sha': payload['entries']['a']['source_sha256'] = 'partial'
        case unreachable: assert_never(unreachable)
    write_json(config['stem_cache'], payload)
    if fault != 'unbound_input':
        config['inputs'][config['stem_cache']] = sha256(config['stem_cache'])
    with pytest.raises(ValueError):
        producer.read_stem_cache(config, rows)
