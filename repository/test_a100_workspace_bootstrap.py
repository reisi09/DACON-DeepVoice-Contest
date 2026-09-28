"""Protect the real deployment from corrupted copies and archive path escapes."""
import hashlib
import io
import tarfile

import pytest

from data_v4.a100_workspace_bootstrap import copy_checked, extract_regular


def test_deployment_preserves_existing_files_and_rejects_corrupt_or_escaping_inputs(tmp_path):
    source = tmp_path / 'source'
    source.write_bytes(b'original audio bytes')
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    target = tmp_path / 'copied'
    copy_checked(source, target, digest)
    assert target.read_bytes() == source.read_bytes()
    source.write_bytes(b'corrupted')
    with pytest.raises(FileExistsError):
        copy_checked(source, target, digest)
    assert target.read_bytes() == b'original audio bytes'
    with pytest.raises(ValueError, match='hash mismatch'):
        copy_checked(source, tmp_path / 'bad_copy', digest)
    assert not (tmp_path / 'bad_copy').exists()
    archive = tmp_path / 'unsafe.tar'
    with tarfile.open(archive, 'w') as packed:
        for name in ('audio/valid.wav', '../escape'):
            item = tarfile.TarInfo(name)
            item.size = 1
            packed.addfile(item, io.BytesIO(b'x'))
    with pytest.raises(ValueError, match='Unsafe'):
        extract_regular(archive, tmp_path / 'input')
    assert not (tmp_path / 'input').exists()
    assert not (tmp_path / 'escape').exists()
