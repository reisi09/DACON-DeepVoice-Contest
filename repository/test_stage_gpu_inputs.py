"""Archive integrity and no-clobber staging through the real tar/filesystem path."""
import hashlib
import io
import json
from pathlib import Path
import tarfile
import wave

import pytest

from data_v4.stage_gpu_inputs import sha256_file, stage


def test_cpu_setup_preserves_pinned_code_and_assets_on_rerun(tmp_path):
    """A stale setup directory or changed cloud weight must fail before executing model code."""
    import subprocess
    import sys
    script = Path('data_v4/prepare_gpu_cpu.sh').read_text().split("<<'PY'\n", 1)[1].split('\nPY\n', 1)[0]
    base, shared = tmp_path / 'local', tmp_path / 'cloud'
    base.mkdir()
    asset = shared / 'weights/baseline-assets/model.py'
    asset.parent.mkdir(parents=True)
    asset.write_bytes(b'raise RuntimeError("must never execute asset")\n')
    members = {'code/entry.py': b'# reviewed project code\n',
               'inputs/asset_hashes.json': json.dumps({'model.py': sha256_file(asset)}).encode()}
    members['SHA256SUMS.json'] = json.dumps({n: hashlib.sha256(b).hexdigest() for n, b in members.items()}).encode()
    bundle = tmp_path / 'ready.tar.gz'
    with tarfile.open(bundle, 'w:gz') as tar:
        for name, data in members.items():
            info = tarfile.TarInfo(name); info.size = len(data)
            tar.addfile(info, io.BytesIO(data))
    def run(digest):
        return subprocess.run([sys.executable, '-c', script, str(bundle), digest, str(base), str(shared)],
                              text=True, capture_output=True)
    assert run('0' * 64).returncode != 0
    assert not (base / 'prepared-v1').exists()
    outside = tmp_path / 'outside'; outside.mkdir()
    (base / 'assets').symlink_to(outside, target_is_directory=True)
    assert run(sha256_file(bundle)).returncode != 0
    assert list(outside.iterdir()) == []
    (base / 'assets').unlink()
    result = run(sha256_file(bundle))
    assert result.returncode == 0, result.stderr
    for name, data in members.items():
        assert (base / 'prepared-v1' / name).read_bytes() == data
    assert (base / 'assets/model.py').read_bytes() == asset.read_bytes()
    before = (base / 'assets/model.py').stat().st_ino
    assert run(sha256_file(bundle)).returncode == 0
    assert (base / 'assets/model.py').stat().st_ino == before
    extra = base / 'prepared-v1/code/unreviewed.py'
    extra.write_bytes(b'unknown executable')
    assert run(sha256_file(bundle)).returncode != 0
    assert extra.read_bytes() == b'unknown executable'
    extra.unlink()
    (base / 'assets/model.py').write_bytes(b'preexisting changed asset')
    assert run(sha256_file(bundle)).returncode != 0
    assert (base / 'assets/model.py').read_bytes() == b'preexisting changed asset'


def test_verified_audio_staging_rejects_unsafe_archives_and_preserves_existing(tmp_path):
    root = tmp_path / 'bank'
    root.mkdir()
    original = root / 'historical.wav'
    original.write_bytes(b'valuable original')
    wav = io.BytesIO()
    with wave.open(wav, 'wb') as stream:
        stream.setparams((1, 2, 16000, 0, 'NONE', 'not compressed'))
        stream.writeframes(b'\x34\x12\xcc\xed' * 100)
    payload = wav.getvalue()
    name = 'audio/additional-v2/' + 'a' * 64 + '.wav'
    digest = hashlib.sha256(payload).hexdigest()

    def archive(label, members, string_hash=False, audio_digest=digest):
        path = tmp_path / (label + '.tar.gz')
        hashes = {name: audio_digest if string_hash else dict(sha256=audio_digest, bytes=len(payload))}
        with tarfile.open(path, 'w:gz') as tar:
            for member_name, data, kind in members:
                info = tarfile.TarInfo(member_name)
                if kind:
                    info.type, info.linkname = kind, '../../historical.wav'
                    tar.addfile(info)
                else:
                    info.size = len(data)
                    tar.addfile(info, io.BytesIO(data))
            data = json.dumps(hashes).encode()
            info = tarfile.TarInfo(label + '/SHA256SUMS.json')
            info.size = len(data)
            tar.addfile(info, io.BytesIO(data))
        return path, sha256_file(path)

    cases = [
        ('traversal', [('../historical.wav', payload, None), (name, payload, None)]),
        ('absolute', [('/tmp/escaped.wav', payload, None), (name, payload, None)]),
        ('symlink', [(name, b'', tarfile.SYMTYPE)]),
        ('hardlink', [(name, b'', tarfile.LNKTYPE)]),
        ('duplicate', [(name, payload, None), (name, payload, None)]),
        ('unexpected_audio', [('audio/other.wav', payload, None), (name, payload, None)]),
    ]
    for label, members in cases:
        spec = archive(label, members)
        with pytest.raises(ValueError):
            stage(root, [spec])
        assert original.read_bytes() == b'valuable original'
        assert not (root / name).exists()
    valid = archive('valid', [(name, payload, None)])
    with pytest.raises(ValueError, match='Archive hash'):
        stage(root, [(valid[0], '0' * 64)])
    incorrect_payload = archive('bad_payload_hash', [(name, payload, None)], audio_digest='0' * 64)
    with pytest.raises(ValueError, match='Audio bytes'):
        stage(root, [incorrect_payload])
    assert not (root / name).exists(), 'unverified bytes must never appear at final destination'
    target = root / name
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(b'existing conflict')
    with pytest.raises(ValueError, match='Existing audio conflicts'):
        stage(root, [valid])
    assert target.read_bytes() == b'existing conflict' and original.read_bytes() == b'valuable original'
    target.unlink()  # Only the task-owned fixture is removed to exercise a fresh install.
    installed = stage(root, [valid])
    assert target.read_bytes() == payload and installed['items'][0]['sha256'] == digest
    assert installed['installed'] == 1
    before = target.stat().st_ino
    reused = stage(root, [archive('string_manifest', [(name, payload, None)], string_hash=True)])
    assert target.read_bytes() == payload and target.stat().st_ino == before and reused['reused'] == 1
    assert original.read_bytes() == b'valuable original'
