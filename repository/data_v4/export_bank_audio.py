"""CPU-only, no-clobber export of exactly the pinned bank audio for another cluster."""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import os
from pathlib import Path
import re
import tarfile
import tempfile


def sha256(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def export(manifest, expected, input_root, output):
    if sha256(manifest) != expected:
        raise ValueError('Manifest hash mismatch')
    if output.exists() or output.is_symlink():
        raise FileExistsError('Refuse to replace an existing export')
    root = input_root.resolve(strict=True)
    if output.resolve().is_relative_to(root):
        raise ValueError('Export output must be outside input audio')
    rows = list(csv.DictReader(manifest.open()))
    if not rows or len({row['path'] for row in rows}) != len(rows):
        raise ValueError('Expected unique nonempty audio paths')
    files = []
    for row in rows:
        relative = Path(row['path'])
        path = root / relative
        if (relative.is_absolute() or '..' in relative.parts or '\\' in row['path']
                or relative.as_posix() != row['path'] or not row['path'].startswith('audio/')
                or any(p.is_symlink() for p in (path, *path.parents) if p.is_relative_to(root))
                or not path.resolve(strict=True).is_relative_to(root) or not path.is_file()
                or not re.fullmatch('[0-9a-f]{64}', row['audio_sha256'])):
            raise ValueError('Unsafe audio path or digest')
        files.append((row, path))
    output.parent.mkdir(parents=True, exist_ok=True)
    handle = tempfile.NamedTemporaryFile(prefix='.' + output.name + '.', dir=output.parent, delete=False)
    temporary = Path(handle.name)
    handle.close()
    try:
        hashes = {}
        total = 0
        with tarfile.open(temporary, 'w') as archive:
            for index, (row, path) in enumerate(files, 1):
                # Each clip is read once; add the same verified bytes, eliminating a hash/copy race.
                with path.open('rb') as stream:
                    data = stream.read(64 * 1024**2 + 1)
                if len(data) > 64 * 1024**2:
                    raise ValueError('Clip exceeds the 64 MiB CPU export memory bound')
                if hashlib.sha256(data).hexdigest() != row['audio_sha256']:
                    raise ValueError('Audio hash mismatch: ' + row['path'])
                member = tarfile.TarInfo(row['path'])
                member.size, member.mode = len(data), 0o644
                archive.addfile(member, io.BytesIO(data))
                hashes[row['path']] = dict(sha256=row['audio_sha256'], bytes=len(data))
                total += len(data)
                if index % 1000 == 0 or index == len(files):
                    print(json.dumps(dict(phase='export_audio', files=index, total=len(files))), flush=True)
            data = json.dumps(hashes, sort_keys=True).encode()
            member = tarfile.TarInfo('SHA256SUMS.json')
            member.size, member.mode = len(data), 0o644
            archive.addfile(member, io.BytesIO(data))
        digest = sha256(temporary)
        # Hard link publishes atomically without overwriting an existing artifact.
        os.link(temporary, output)
        return dict(status='audio_export_complete', files=len(files), audio_bytes=total,
                    archive_bytes=output.stat().st_size, archive_sha256=digest,
                    manifest_sha256=expected, gpu_used=False)
    finally:
        temporary.unlink(missing_ok=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('manifest', 'input', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--manifest-sha256', required=True)
    args = parser.parse_args()
    print(json.dumps(export(args.manifest, args.manifest_sha256, args.input, args.output)), flush=True)
