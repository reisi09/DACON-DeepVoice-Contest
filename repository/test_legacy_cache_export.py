import hashlib
import io
import json
from pathlib import Path
import struct
import tarfile
import zipfile

import pytest

import legacy_cache_export as exporter


def make_archive(root: Path, bad: str | None = None, corrupt: bool = False) -> tuple[Path, Path, str]:
    header = "{'descr': '<f4', 'fortran_order': False, 'shape': (2,), }"
    header = (header + ' ' * ((64 - (10 + len(header) + 1) % 64) % 64) + '\n').encode()
    array = b'\x93NUMPY\x01\x00' + struct.pack('<H', len(header)) + header + struct.pack('<ff', .125, -.25)
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, 'w') as bundle:
        bundle.writestr('mert_windows.npy', array)
    raw = buffer.getvalue()
    items = {'row1': dict(path='row1.npz', sha256=hashlib.sha256(raw).hexdigest(), identity='example')}
    metadata = json.dumps(dict(identity=dict(mert_layer=12, dtype='float32'), items=items)).encode()
    source = root.resolve() / 'source/data_v4/feature_cache_v4.tar'
    source.parent.mkdir(parents=True)
    with tarfile.open(source, 'w') as archive:
        for key in ('cache_tv', 'cache_test'):
            for name, content in [('index.json', metadata), ('row1.npz', raw + b'changed' if corrupt else raw)]:
                info = tarfile.TarInfo('./data_v4/' + key + '/' + name)
                info.size = len(content)
                archive.addfile(info, io.BytesIO(content))
        if bad:
            info = tarfile.TarInfo(bad)
            info.size = 4
            archive.addfile(info, io.BytesIO(b'evil'))
    return source, root.resolve() / 'output/feature_cache_v4.tar', hashlib.sha256(metadata).hexdigest()


def test_export_preserves_archive_metadata_and_idempotent_retry(tmp_path: Path):
    source, target, expected = make_archive(tmp_path)
    before = source.read_bytes()

    result = exporter.export(source, target, expected_tv=expected)

    assert target.read_bytes() == before == source.read_bytes()
    assert result['destination_readback_sha256'] == hashlib.sha256(before).hexdigest()
    assert result['npz_count'] == 2 and result['caches']['cache_tv']['rows'] == 1
    for key in ('cache_tv', 'cache_test'):
        assert hashlib.sha256((target.parent / f'{key}_index.json').read_bytes()).hexdigest() == expected
    assert exporter.export(source, target, expected_tv=expected)['reused_existing'] is True


@pytest.mark.parametrize('partial', [False, True])
def test_export_refuses_conflicting_destination_bytes(tmp_path: Path, partial: bool):
    source, target, expected = make_archive(tmp_path)
    target.parent.mkdir()
    conflict = target.with_suffix('.tar.partial') if partial else target
    conflict.write_bytes(b'existing-unrelated-output')

    with pytest.raises(exporter.ExportError):
        exporter.export(source, target, expected_tv=expected)

    assert conflict.read_bytes() == b'existing-unrelated-output'
    assert not (target if partial else target.with_suffix('.tar.partial')).exists()


@pytest.mark.parametrize(('bad', 'corrupt'), [('../escape', False), (None, True)])
def test_export_rejects_unsafe_path_or_corrupt_payload_before_copy(tmp_path: Path, bad: str | None, corrupt: bool):
    source, target, expected = make_archive(tmp_path, bad=bad, corrupt=corrupt)
    before = source.read_bytes()

    with pytest.raises(exporter.ExportError):
        exporter.export(source, target, expected_tv=expected)

    assert source.read_bytes() == before
    assert not target.exists() and not target.with_suffix('.tar.partial').exists()


def test_interrupted_copy_resumes_only_from_verified_prefix(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    source, target, expected = make_archive(tmp_path)
    before = source.read_bytes()
    original_read = exporter.HashedReader.read
    copied_reads = 0

    def interrupted_read(reader: exporter.HashedReader, size: int = exporter.CHUNK) -> bytes:
        nonlocal copied_reads
        if target.with_suffix('.tar.partial').exists():
            copied_reads += 1
            if copied_reads > 1:
                raise InterruptedError('Controlled copy interruption')
            return original_read(reader, min(size, 2048))
        return original_read(reader, size)

    with monkeypatch.context() as patch:
        patch.setattr(exporter.HashedReader, 'read', interrupted_read)
        with pytest.raises(InterruptedError):
            exporter.export(source, target, expected_tv=expected)

    assert not target.exists() and source.read_bytes() == before
    assert json.loads((target.parent / 'export_attempt.json').read_text())['status'] == 'failed'
    assert 0 < target.with_suffix('.tar.partial').stat().st_size < len(before)
    exporter.export(source, target, expected_tv=expected)
    assert target.read_bytes() == before == source.read_bytes()


def test_failed_later_attempt_preserves_completed_receipt(tmp_path: Path):
    source, target, expected = make_archive(tmp_path)
    exporter.export(source, target, expected_tv=expected)
    completed = (target.parent / 'export_receipt.json').read_bytes()
    before = target.read_bytes()

    with pytest.raises(exporter.ExportError):
        exporter.export(source, target, expected_tv='0' * 64)

    assert (target.parent / 'export_receipt.json').read_bytes() == completed
    assert target.read_bytes() == before
