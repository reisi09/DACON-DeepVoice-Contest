import errno
from pathlib import Path
from types import SimpleNamespace

import pytest

import panns_cloud_prepare as prepare_module
from panns_adaptation_data import sha256, write_json


@pytest.fixture
def staged_inputs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    bundle, volume, work = (tmp_path/name for name in ('bundle', 'volume', 'work'))
    bundle.mkdir()
    source = volume/prepare_module.WEIGHTS/'weight.bin'
    source.parent.mkdir(parents=True)
    source.write_bytes(bytes(range(256))*8)
    audio = volume/prepare_module.AUDIO
    audio.parent.mkdir(parents=True)
    audio.write_bytes(b'locked audio fixture')
    monkeypatch.setattr(prepare_module, 'AUDIO_SHA', sha256(audio))
    write_json(bundle/'reference.json', {'members': {'model/assets/weight.bin': sha256(source)}})
    (bundle/'presence_rows.jsonl').write_text('')
    (bundle/'spec.md').write_text('fixture specification')
    (bundle/'workflow.md').write_text('fixture workflow')
    write_json(bundle/'bundle_manifest.json', {'sha256': {p.name: sha256(p) for p in bundle.iterdir()}})
    return bundle, volume, work, source


@pytest.mark.parametrize('failure', ['interrupted', 'corrupt'])
def test_failed_asset_copy_keeps_source_and_allows_clean_retry(staged_inputs, monkeypatch, failure):
    # Given: authenticated input bytes and a filesystem copy that fails after writing.
    bundle, volume, work, source = staged_inputs
    original = source.read_bytes()
    real_copy = prepare_module.shutil.copyfile

    def bad_copy(source_path, destination_path):
        Path(destination_path).write_bytes(b'partial bytes')
        if failure == 'interrupted':
            raise OSError(errno.ENOSPC, 'Injected disk exhaustion')

    monkeypatch.setattr(prepare_module.shutil, 'copyfile', bad_copy)
    # When: prepare receives an interrupted or silently corrupt copy.
    with pytest.raises((OSError, ValueError)):
        prepare_module.prepare(bundle, volume, work)
    # Then: no partial asset is published and retry authenticates the complete bytes.
    assert source.read_bytes() == original
    assert not list((work/'package').rglob('*.bin'))
    assert not list(work.rglob('*.tmp'))
    monkeypatch.setattr(prepare_module.shutil, 'copyfile', real_copy)
    prepare_module.prepare(bundle, volume, work)
    assert (work/'package/model/assets/weight.bin').read_bytes() == original
    prepare_module.prepare(bundle, volume, work)
    assert source.read_bytes() == original


@pytest.mark.parametrize('existing', ['mismatch', 'dangling_symlink'])
def test_existing_asset_is_not_replaced_or_followed(staged_inputs, existing):
    # Given: an existing unrelated asset or a dangling destination symlink.
    bundle, volume, work, source = staged_inputs
    destination = work/'package/model/assets/weight.bin'
    destination.parent.mkdir(parents=True)
    outside = work/'unrelated.bin'
    if existing == 'mismatch':
        destination.write_bytes(b'preserve existing bytes')
    else:
        destination.symlink_to(outside)
    # When: staging reaches that destination.
    with pytest.raises(ValueError):
        prepare_module.prepare(bundle, volume, work)
    # Then: no other file is created and the existing destination remains intact.
    assert not outside.exists()
    if existing == 'mismatch':
        assert destination.read_bytes() == b'preserve existing bytes'
    else:
        assert destination.is_symlink()
    assert source.read_bytes() == bytes(range(256))*8


def test_insufficient_space_rejects_before_creating_asset(staged_inputs, monkeypatch):
    bundle, volume, work, source = staged_inputs
    monkeypatch.setattr(prepare_module.shutil, 'disk_usage',
                        lambda path: SimpleNamespace(free=source.stat().st_size-1))
    with pytest.raises(OSError) as error:
        prepare_module.prepare(bundle, volume, work)
    assert error.value.errno == errno.ENOSPC
    assert not list(work.rglob('*.tmp'))
    assert not list((work/'package').rglob('*.bin'))
    assert source.read_bytes() == bytes(range(256))*8
