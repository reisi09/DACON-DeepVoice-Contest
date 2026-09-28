"""Protect runtime reuse identity, member integrity, and archive containment."""
import io
import json
from pathlib import Path
import tarfile
import pytest
import panns_restore_environment as runtime


@pytest.fixture
def verified_environment(tmp_path, monkeypatch):
    shared=tmp_path/'shared';work=tmp_path/'application-v4';environment=tmp_path/'verified-prefix'
    source=shared/'panns-cnn-v1-20260925/environment'
    source.mkdir(parents=True);(work/'bundle').mkdir(parents=True)
    contents={'requirements.lock.txt':b'pinned runtime\n','reference.json':b'fixed reference',
        'package/model/labels.csv':b'fixed labels','gold/reference.npz':b'fixed numeric gold'}
    hashes={}
    for name,payload in contents.items():
        path=work/'bundle'/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(payload)
        hashes[name]=runtime.sha(path)
    archive=shared/'panns-cnn-v1-20260925/inputs-v3/run_bundle.tar'
    archive.parent.mkdir(parents=True)
    with tarfile.open(archive,'w') as tar:
        payload=json.dumps({'sha256':hashes}).encode()
        member=tarfile.TarInfo('bundle_manifest.json');member.size=len(payload)
        tar.addfile(member,io.BytesIO(payload))
    binary=environment/'env/venv/bin/python';binary.parent.mkdir(parents=True)
    binary.write_bytes(b'expected runtime content')
    record=dict(producer_sha256=runtime.sha(Path(__file__).with_name('panns_cpu_environment.py')),
        lock_sha256=hashes['requirements.lock.txt'],actual_gpu_used=False,python_prefix=str(environment/'env'),
        bundle_sha256=runtime.sha(archive),members={'venv/bin/python':{'sha256':runtime.sha(binary)}})
    receipt=source/'environment.json';receipt.write_text(json.dumps(record))
    (environment/'env/.panns_environment.json').write_text(json.dumps(record))
    monkeypatch.setattr(runtime,'ENVIRONMENT_ROOT',environment,raising=False)
    monkeypatch.setattr(runtime,'RECEIPT_SHA256',runtime.sha(receipt),raising=False)
    monkeypatch.setattr(runtime,'HISTORICAL_BUNDLE_SHA256',runtime.sha(archive),raising=False)
    return shared,work,environment,source


def test_code_only_revision_reuses_original_environment_prefix(verified_environment):
    shared,work,environment,_=verified_environment
    (work/'bundle/code').mkdir();(work/'bundle/code/revised.py').write_text('revised application\n')
    (work/'bundle/gold/index.json').write_text('{"metadata": "derived audio IDs"}')
    runtime.restore(shared,work)
    assert (environment/'env/venv/bin/python').read_bytes()==b'expected runtime content'
    assert not (work/'env').exists()


@pytest.mark.parametrize('changed',['receipt','requirements.lock.txt','reference.json','package/model/labels.csv','gold/reference.npz'])
def test_reuse_rejects_changed_verification_inputs(verified_environment,changed):
    shared,work,_,source=verified_environment
    target=source/'environment.json' if changed=='receipt' else work/'bundle'/changed
    if changed=='receipt':
        record=json.loads(target.read_text());record['probe_stdout']='fabricated verification'
        target.write_text(json.dumps(record))
    else:target.write_bytes(b'changed input')
    with pytest.raises(ValueError,match='receipt|runtime|input'):
        runtime.restore(shared,work)


def test_matching_marker_cannot_hide_corrupted_runtime(verified_environment):
    shared,work,environment,_=verified_environment
    (environment/'env/venv/bin/python').write_bytes(b'corrupted runtime content')
    with pytest.raises(ValueError,match='Environment member changed'):
        runtime.restore(shared,work)


def test_unverified_environment_prefix_relocation_is_rejected(verified_environment,monkeypatch):
    shared,work,_,source=verified_environment
    receipt=source/'environment.json';record=json.loads(receipt.read_text())
    record['python_prefix']=str(work/'env');receipt.write_text(json.dumps(record))
    monkeypatch.setattr(runtime,'RECEIPT_SHA256',runtime.sha(receipt))
    with pytest.raises(ValueError,match='prefix relocation'):
        runtime.restore(shared,work)


def test_restore_rejects_external_symlink_even_with_valid_archive_hash(verified_environment,monkeypatch):
    shared,work,environment,source=verified_environment
    for path in (environment/'env').rglob('*'):
        if path.is_file():path.unlink()
    for path in sorted((environment/'env').rglob('*'),reverse=True):path.rmdir()
    (environment/'env').rmdir()
    archive=source/'linux-env.tar.gz'
    with tarfile.open(archive,'w:gz') as tar:
        link=tarfile.TarInfo('env/escape');link.type=tarfile.SYMTYPE
        link.linkname='../../../outside';tar.addfile(link)
        payload=b'forbidden write'
        item=tarfile.TarInfo('env/escape/payload');item.size=len(payload)
        tar.addfile(item,io.BytesIO(payload))
    receipt=source/'environment.json';record=json.loads(receipt.read_text())
    record.update(bytes=archive.stat().st_size,archive_sha256=runtime.sha(archive))
    receipt.write_text(json.dumps(record));monkeypatch.setattr(runtime,'RECEIPT_SHA256',runtime.sha(receipt),raising=False)
    with pytest.raises(tarfile.FilterError):runtime.restore(shared,work)
    assert not (environment.parent/'outside/payload').exists()
