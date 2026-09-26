import copy

import numpy as np
import pytest

import panns_adaptation_cache as producer
from panns_adaptation_data import load_cache, sha256
from test_panns_cache_resume import unpublished_cache


def test_shard_preserves_values_and_rejects_cross_file_substitution(tmp_path, monkeypatch):
    config, entry, _ = unpublished_cache(tmp_path, packed=False)
    values = load_cache(entry)
    other = dict(entry, ID='audio:b', identity='different-audio')
    changed = {k: v.copy() for k, v in values.items()}
    changed['identity'] = np.array(other['identity'])
    changed['block5'] *= 2
    original = copy.deepcopy(entry)
    entries = producer.write_cache_shard(config, 0, [(entry, values), (other, changed)], False)
    assert entry == original
    for actual, expected in zip(entries, (values, changed), strict=True):
        for key, value in load_cache(actual).items():
            np.testing.assert_array_equal(value, expected[key])
    monkeypatch.setenv('PANNS_LOCAL_CACHE', str(tmp_path/'local'))
    np.testing.assert_array_equal(load_cache(entries[0])['block5'], values['block5'])
    with pytest.raises(ValueError, match='hash'):
        load_cache(entries[0] | {'member': entries[1]['member'], 'sha256': 'a'*64})


def test_interrupted_shard_publication_is_preserved_and_recomputed(tmp_path, monkeypatch):
    config, entry, _ = unpublished_cache(tmp_path, packed=False)
    values = load_cache(entry)
    original_write = producer.write_json
    def interrupted(*_):
        raise OSError('interrupted before shard receipt')
    monkeypatch.setattr(producer, 'write_json', interrupted)
    with pytest.raises(OSError, match='interrupted'):
        producer.write_cache_shard(config, 0, [(entry, values)], False)
    archive = tmp_path/'cache/part_00000.zip'
    previous = sha256(archive)
    assert not archive.with_suffix('.json').exists()
    monkeypatch.setattr(producer, 'write_json', original_write)
    with pytest.raises(FileExistsError, match='resume'):
        producer.write_cache_shard(config, 0, [(entry, values)], False)
    recovered = producer.write_cache_shard(config, 0, [(entry, values)], True)
    assert any(sha256(p) == previous for p in archive.parent.glob('part_00000.orphan.*.zip'))
    np.testing.assert_array_equal(load_cache(recovered[0])['block5'], values['block5'])
