import hashlib
import io
import json
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest

from panns_adaptation_data import load_cache, sha256
from panns_prefetch_cache import prefetch


def test_prefetch_preserves_values_and_concurrent_reader_temporary(tmp_path, monkeypatch):
    values = dict(identity=np.array('locked-audio'), block5=np.arange(24, dtype=np.float32).reshape(2, 3, 4),
                  tags=np.array([[.125, .75]], dtype=np.float32), starts=np.array([0., 3.]))
    stream = io.BytesIO()
    np.savez(stream, **values)
    payload = stream.getvalue()
    digest = hashlib.sha256(payload).hexdigest()
    shared = tmp_path/'shared'
    shared.mkdir()
    source = shared/'part_00000.zip'
    with zipfile.ZipFile(source, 'w') as archive:
        archive.writestr('sample.npz', payload)
    entry = dict(path=str(source), member='sample.npz', sha256=digest, identity='locked-audio')
    index = shared/'index.json'
    index.write_text(json.dumps(dict(entries=[entry])))
    local = tmp_path/'local'
    local.mkdir()
    reader_temporary = local/(digest+'.npz.tmp')
    reader_temporary.write_bytes(b'in-flight reader data')
    # Two publishers may race with the production reader; neither owns its temporary.
    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(lambda _: prefetch(index, sha256(index), local, time.time()+60), range(2)))
    assert sum(result.copied for result in results) == 1
    assert reader_temporary.read_bytes() == b'in-flight reader data'
    source.rename(source.with_suffix('.offline'))
    monkeypatch.setenv('PANNS_LOCAL_CACHE', str(local))
    for key, actual in load_cache(entry).items():
        np.testing.assert_array_equal(actual, values[key])


@pytest.mark.parametrize('corruption', ['index', 'payload', 'existing'])
def test_prefetch_rejects_corruption_without_replacing_existing_data(tmp_path, corruption):
    shared = tmp_path/'shared'
    shared.mkdir()
    source = shared/'part_00000.zip'
    payload = b'authenticated payload'
    digest = hashlib.sha256(payload).hexdigest()
    with zipfile.ZipFile(source, 'w') as archive:
        archive.writestr('sample.npz', b'changed payload' if corruption == 'payload' else payload)
    index = shared/'index.json'
    index.write_text(json.dumps(dict(entries=[dict(path=str(source), member='sample.npz', sha256=digest)])))
    expected = sha256(index)
    local = tmp_path/'local'
    local.mkdir()
    existing = local/(digest+'.npz')
    if corruption == 'index':
        index.write_text(index.read_text()+' ')
    if corruption == 'existing':
        existing.write_bytes(b'preserve conflicting bytes')
    with pytest.raises(ValueError, match='hash'):
        prefetch(index, expected, local, time.time()+60)
    if corruption == 'existing':
        assert existing.read_bytes() == b'preserve conflicting bytes'
    else:
        assert not existing.exists()
