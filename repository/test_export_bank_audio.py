"""Protect exact cross-cluster audio transfer and preservation of existing artifacts."""
import csv
import hashlib
import json
import tarfile
import wave

import pytest

from data_v4.export_bank_audio import export, sha256


def test_export_preserves_actual_audio_and_rejects_corruption_or_overwrite(tmp_path):
    root = tmp_path / 'input'
    (root / 'audio').mkdir(parents=True)
    rows = []
    for i in range(2):
        path = root / f'audio/{i}.wav'
        with wave.open(str(path), 'wb') as stream:
            stream.setparams((1, 2, 16000, 0, 'NONE', 'not compressed'))
            stream.writeframes(bytes([i, 0]) * 100)
        rows.append(dict(path=f'audio/{i}.wav', audio_sha256=sha256(path)))
    manifest = tmp_path / 'bank.csv'
    with manifest.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    original = {r['path']: (root / r['path']).read_bytes() for r in rows}
    output = tmp_path / 'export.tar'
    receipt = export(manifest, sha256(manifest), root, output)
    with tarfile.open(output) as archive:
        hashes = json.load(archive.extractfile('SHA256SUMS.json'))
        assert set(archive.getnames()) == set(original) | {'SHA256SUMS.json'}
        for name, content in original.items():
            actual = archive.extractfile(name).read()
            assert actual == content == (root / name).read_bytes()
            assert hashes[name]['sha256'] == hashlib.sha256(actual).hexdigest()
    assert receipt['archive_sha256'] == sha256(output)
    with pytest.raises(FileExistsError):
        export(manifest, sha256(manifest), root, output)
    assert receipt['archive_sha256'] == sha256(output)
    (root / rows[-1]['path']).write_bytes(b'corruption')
    with pytest.raises(ValueError, match='Audio hash mismatch'):
        export(manifest, sha256(manifest), root, tmp_path / 'corrupt.tar')
    assert not (tmp_path / 'corrupt.tar').exists()
    (root / rows[-1]['path']).unlink()
    (root / rows[-1]['path']).symlink_to(manifest)
    with pytest.raises(ValueError, match='Unsafe audio path'):
        export(manifest, sha256(manifest), root, tmp_path / 'escape.tar')
