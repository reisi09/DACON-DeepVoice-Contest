"""Ensure a hash-valid environment archive cannot escape the restore directory."""
import io
import json
from pathlib import Path
import tarfile
import pytest
from panns_restore_environment import restore, sha


def test_restore_rejects_external_symlink_even_with_valid_archive_hash(tmp_path):
    shared=tmp_path/'shared';work=tmp_path/'work'
    source=shared/'panns-cnn-v1-20260925/environment'
    source.mkdir(parents=True);(work/'bundle').mkdir(parents=True)
    lock=work/'bundle/requirements.lock.txt';lock.write_text('pinned runtime\n')
    archive=source/'linux-env.tar.gz'
    with tarfile.open(archive,'w:gz') as tar:
        link=tarfile.TarInfo('env/escape');link.type=tarfile.SYMTYPE
        link.linkname='../../../outside';tar.addfile(link)
        payload=b'forbidden write'
        item=tarfile.TarInfo('env/escape/payload');item.size=len(payload)
        tar.addfile(item,io.BytesIO(payload))
    record=dict(producer_sha256=sha(Path(__file__).with_name('panns_cpu_environment.py')),
        lock_sha256=sha(lock),actual_gpu_used=False,bytes=archive.stat().st_size,archive_sha256=sha(archive))
    bundle=shared/'panns-cnn-v1-20260925/inputs-v3/run_bundle.tar'
    bundle.parent.mkdir(parents=True);bundle.write_bytes(b'fixed bundle fixture')
    record['bundle_sha256']=sha(bundle)
    (source/'environment.json').write_text(json.dumps(record))
    with pytest.raises(tarfile.FilterError):restore(shared,work)
    assert not (tmp_path/'outside/payload').exists()


def test_matching_marker_cannot_hide_corrupted_runtime(tmp_path):
    shared=tmp_path/'shared';work=tmp_path/'work'
    source=shared/'panns-cnn-v1-20260925/environment'
    source.mkdir(parents=True);(work/'bundle').mkdir(parents=True)
    lock=work/'bundle/requirements.lock.txt';lock.write_text('pinned runtime\n')
    runtime=work/'env/venv/bin/python';runtime.parent.mkdir(parents=True)
    runtime.write_bytes(b'expected runtime content')
    record=dict(producer_sha256=sha(Path(__file__).with_name('panns_cpu_environment.py')),
        lock_sha256=sha(lock),actual_gpu_used=False,members={'venv/bin/python':{'sha256':sha(runtime)}})
    bundle=shared/'panns-cnn-v1-20260925/inputs-v3/run_bundle.tar'
    bundle.parent.mkdir(parents=True);bundle.write_bytes(b'fixed bundle fixture')
    record['bundle_sha256']=sha(bundle)
    (source/'environment.json').write_text(json.dumps(record))
    (work/'env/.panns_environment.json').write_text(json.dumps(record))
    restore(shared,work)
    runtime.write_bytes(b'corrupted runtime content')
    with pytest.raises(ValueError,match='Environment member changed'):restore(shared,work)
