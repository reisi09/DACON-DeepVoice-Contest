#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = []
# ///
# Run: python data_v4/package_acquired_audio.py PAYLOAD OUTPUT.tar
"""Package explicit verified originals and provenance; verify every archived member byte."""
import hashlib
import io
import json
from pathlib import Path
import sys
import tarfile


def main(payload: Path, target: Path) -> None:
    """Exclude download locks, partial files and unrelated metadata from the delivery."""
    rows = [json.loads(line) for line in (payload/'verified_sources.jsonl').open()]
    files = {row['path']: row['sha256'] for row in rows}
    for name in ('sources.jsonl', 'verified_sources.jsonl', 'verification.json', 'ATTRIBUTION.md',
                 'README.upstream.md', 'README.TXT', 'LICENSE.TXT', 'pilot_receipt.json', 'selection_receipt.json'):
        path = payload/name
        if path.is_file():
            files[name] = hashlib.sha256(path.read_bytes()).hexdigest()
    for name, expected in files.items():
        path = payload/name
        if path.is_symlink() or not path.resolve().is_relative_to(payload.resolve()):
            raise ValueError('Escaped or linked package member')
        with path.open('rb') as stream:
            if hashlib.file_digest(stream, 'sha256').hexdigest() != expected:
                raise ValueError('Payload changed after verification')
    target.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(target, 'x') as archive:
        for name in sorted(files):
            path = payload/name
            info = archive.gettarinfo(str(path), arcname=payload.name + '/' + name)
            info.uid = info.gid = 0
            info.uname = info.gname = ''
            with path.open('rb') as stream:
                archive.addfile(info, stream)
        content = (json.dumps(files, sort_keys=True, indent=2) + '\n').encode()
        info = tarfile.TarInfo(payload.name + '/SHA256SUMS.json')
        info.size = len(content)
        archive.addfile(info, io.BytesIO(content))
    with tarfile.open(target) as archive:
        actual = {m.name for m in archive.getmembers()}
        if actual != {payload.name + '/' + n for n in files} | {payload.name + '/SHA256SUMS.json'}:
            raise ValueError('Archive member set differs from explicit delivery list')
        for name, expected in files.items():
            with archive.extractfile(payload.name + '/' + name) as stream:
                if hashlib.file_digest(stream, 'sha256').hexdigest() != expected:
                    raise ValueError('Archive does not preserve the original member bytes')
    with target.open('rb') as stream:
        digest = hashlib.file_digest(stream, 'sha256').hexdigest()
    report = dict(archive=target.name, bytes=target.stat().st_size, sha256=digest,
        audio_files=len(rows), members=len(files) + 1, all_member_hashes_verified=True,
        explicit_member_list=True, training_admitted=False)
    target.with_suffix('.receipt.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report))


if __name__ == '__main__':
    main(*(Path(arg) for arg in sys.argv[1:]))
