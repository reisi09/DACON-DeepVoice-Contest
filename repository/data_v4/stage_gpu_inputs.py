"""Stage verified supplemental audio onto an existing bank root using CPU and stdlib only."""
from __future__ import annotations

import argparse
from contextlib import ExitStack
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import tarfile
import tempfile

AUDIO = re.compile(r'audio/additional-v2/[0-9a-f]{64}\.wav\Z')
DIGEST = re.compile(r'[0-9a-f]{64}\Z')


def sha256_file(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def unique_json(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError('Duplicate key in hash manifest')
        result[key] = value
    return result


def destination(root, name):
    path = root / name
    if not path.resolve().is_relative_to(root):
        raise ValueError('Destination escapes input root')
    for part in (path, *path.parents):
        if part == root:
            break
        if part.is_symlink():
            raise ValueError('Destination contains a symbolic link')
    return path


def stage(input_root, archives):
    root = Path(input_root).resolve(strict=True)
    if not root.is_dir():
        raise ValueError('Input must be an existing historical audio directory')
    seen, plans, archive_receipts = set(), [], []
    with ExitStack() as opened:
        # Validate every archive and existing destination before installing any new audio.
        for archive, expected in archives:
            archive = Path(archive)
            if not DIGEST.fullmatch(expected) or sha256_file(archive) != expected:
                raise ValueError(f'Archive hash mismatch: {archive.name}')
            tar = opened.enter_context(tarfile.open(archive, 'r:gz'))
            members = tar.getmembers()
            names = set()
            for member in members:
                name = PurePosixPath(member.name)
                if (not name.parts or name.is_absolute() or '..' in name.parts or '\\' in member.name
                        or str(name) != member.name or not member.isfile() or member.name in names):
                    raise ValueError(f'Unsafe or duplicate archive member: {member.name}')
                names.add(member.name)
                if name.parts[0] == 'audio' and not AUDIO.fullmatch(member.name):
                    raise ValueError(f'Unexpected audio member: {member.name}')
            manifests = [m for m in members if PurePosixPath(m.name).name == 'SHA256SUMS.json']
            if len(manifests) != 1 or manifests[0].size > 10_000_000:
                raise ValueError('Expected one bounded SHA256SUMS.json')
            with tar.extractfile(manifests[0]) as stream:
                hashes = json.load(stream, object_pairs_hook=unique_json)
            audio = [m for m in members if AUDIO.fullmatch(m.name)]
            if not audio or {m.name for m in audio} != {name for name in hashes if name.startswith('audio/')}:
                raise ValueError('Audio members differ from the hash manifest')
            for member in audio:
                if member.name in seen:
                    raise ValueError('Duplicate audio destination across archives')
                seen.add(member.name)
                record = hashes[member.name]
                digest = record.get('sha256') if isinstance(record, dict) else record
                if not isinstance(digest, str) or not DIGEST.fullmatch(digest):
                    raise ValueError('Invalid audio digest')
                if isinstance(record, dict) and record.get('bytes') != member.size:
                    raise ValueError('Audio size differs from manifest')
                target = destination(root, member.name)
                if target.exists() and (not target.is_file() or sha256_file(target) != digest):
                    raise ValueError(f'Existing audio conflicts: {member.name}')
                plans.append((tar, member, target, digest, archive.name))
            archive_receipts.append(dict(name=archive.name, sha256=expected, bytes=archive.stat().st_size,
                                         verified_audio_members=len(audio)))
        items = []
        for tar, member, target, digest, archive_name in plans:
            destination(root, member.name)
            target.parent.mkdir(parents=True, exist_ok=True)
            destination(root, member.name)
            temporary = None
            try:
                existing = target.exists()
                if existing:
                    if not target.is_file() or sha256_file(target) != digest:
                        raise ValueError(f'Existing audio changed: {member.name}')
                else:
                    handle = tempfile.NamedTemporaryFile(prefix='.' + target.name + '.', dir=target.parent, delete=False)
                    temporary = Path(handle.name)
                computed, count = hashlib.sha256(), 0
                with tar.extractfile(member) as stream:
                    while block := stream.read(1024 * 1024):
                        computed.update(block)
                        count += len(block)
                        if not existing:
                            handle.write(block)
                if not existing:
                    handle.flush()
                    os.fsync(handle.fileno())
                    handle.close()
                if computed.hexdigest() != digest or count != member.size:
                    raise ValueError(f'Audio bytes differ from manifest: {member.name}')
                if not existing:
                    # Link installs atomically without replacing a destination created concurrently.
                    try:
                        os.link(temporary, target)
                    except FileExistsError:
                        destination(root, member.name)
                        if not target.is_file() or sha256_file(target) != digest:
                            raise ValueError(f'Concurrent audio conflict: {member.name}')
                        existing = True
                if sha256_file(target) != digest:
                    raise ValueError(f'Staged audio changed: {member.name}')
                items.append(dict(path=member.name, sha256=digest, bytes=count,
                                  status='reused' if existing else 'installed', archive=archive_name))
            finally:
                if temporary is not None:
                    if not handle.closed:
                        handle.close()
                    temporary.unlink(missing_ok=True)
    return dict(input_root=str(root), archives=archive_receipts, files=len(items),
                installed=sum(i['status'] == 'installed' for i in items),
                reused=sum(i['status'] == 'reused' for i in items),
                bytes=sum(i['bytes'] for i in items), gpu_used=False, items=items)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', required=True, type=Path)
    parser.add_argument('--archive', required=True, action='append', nargs=2, metavar=('PATH', 'SHA256'))
    parser.add_argument('--receipt', required=True, type=Path)
    args = parser.parse_args()
    if args.receipt.exists():
        raise FileExistsError(args.receipt)
    result = stage(args.input, args.archive)
    with args.receipt.open('x') as stream:
        json.dump(result, stream, indent=2)
        stream.write('\n')
    print(json.dumps({k: v for k, v in result.items() if k != 'items'}))


if __name__ == '__main__':
    main()
