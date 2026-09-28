import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

import panns_adaptation_cache as producer
from panns_adaptation_data import cache_provenance, identity, load_cache, read_cache_index, sha256, write_json


def unpublished_cache(tmp_path: Path, packed=True):
    row = dict(ID='audio:a', role='train', component_id='parent:a', VOICE_PRESENT=1,
               MUSIC_PRESENT=0, verified_audio=dict(sha256='audio-hash'))
    rows = tmp_path/'rows.jsonl'
    rows.write_text(json.dumps(row)+'\n')
    reference = tmp_path/'reference.json'
    write_json(reference, dict(package=str(tmp_path), members={}))
    archive = tmp_path/'audio.tar'
    archive.write_bytes(b'')
    config = dict(output=str(tmp_path), rows=str(rows), reference=str(reference),
                  archive=str(archive), panel_ids=[row['ID']], inputs={str(rows): sha256(rows)})
    config['identity'] = identity(config)
    path = tmp_path/'cache'/f"{hashlib.sha256(row['ID'].encode()).hexdigest()}.npz"
    path.parent.mkdir()
    key = producer.cache_key(config, row, [0])
    np.savez(path, identity=key, block5=np.array([[.25, -.4]], np.float32),
             tags=np.array([[.1, .7]], np.float32), starts=np.array([0]))
    entry = dict(ID=row['ID'], role=row['role'], component_id=row['component_id'],
                 targets=[1, 0], identity=key, path=str(path), sha256=sha256(path),
                 stems=dict(mix_rms_db=-23.), tag_error=0., stats_error=0., presence_error=0.)
    if packed:
        entry = producer.write_cache_shard(config, 0, [(entry, load_cache(entry))], False)[0]
    index = path.parent/'index.json'
    write_json(index, dict(config_identity=config['identity'], provenance=cache_provenance(), entries=[entry]))
    return config, entry, index


@pytest.mark.parametrize('marker', [None, '', 'abc123'])
def test_resume_rebuilds_unpublished_index_without_recomputing(tmp_path, monkeypatch, marker):
    config, entry, index = unpublished_cache(tmp_path)
    if marker is not None:
        index.with_suffix('.sha256').write_text(marker)
    before = {p.name: (sha256(p), p.stat().st_mtime_ns) for p in index.parent.iterdir() if p != index and p.suffix in ('.npz', '.zip', '.json')}
    def forbid_model_load(_package):
        pytest.fail('Verified receipts must recover without loading extraction models')
    monkeypatch.setattr(producer, 'load_base', forbid_model_load)
    recovered = producer.cache(config, resume=True)
    assert recovered['entries'] == [entry]
    assert read_cache_index(config) == recovered
    assert before == {p.name: (sha256(p), p.stat().st_mtime_ns) for p in index.parent.iterdir() if p != index and p.suffix in ('.npz', '.zip', '.json')}


def test_resume_rejects_finalized_index_hash_mismatch(tmp_path):
    config, _entry, index = unpublished_cache(tmp_path)
    index.with_suffix('.sha256').write_text('0'*64+'\n')
    with pytest.raises(ValueError, match='index hash mismatch'):
        producer.cache(config, resume=True)


def test_resume_rejects_hash_valid_but_contradictory_final_index(tmp_path):
    config, _entry, index = unpublished_cache(tmp_path)
    payload = json.loads(index.read_text())
    payload['entries'][0]['targets'] = [0, 1]
    write_json(index, payload)
    index.with_suffix('.sha256').write_text(sha256(index)+'\n')
    with pytest.raises(ValueError, match='target mismatch'):
        producer.cache(config, resume=True)


def test_resume_cannot_recover_corrupt_activation(tmp_path):
    config, entry, _index = unpublished_cache(tmp_path)
    Path(entry['path']).write_bytes(b'interrupted payload')
    with pytest.raises(ValueError, match='Cache hash mismatch'):
        producer.cache(config, resume=True)


def test_unpublished_index_requires_explicit_resume(tmp_path):
    config, _entry, _index = unpublished_cache(tmp_path)
    with pytest.raises(FileExistsError, match='resume'):
        producer.cache(config)
