# /// script
# requires-python = ">=3.11"
# dependencies = []
# ///
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import hashlib
import io
import json
import math
import os
from pathlib import Path
import re
import tempfile
import time
from typing import Final
import zipfile

MAX_MEMBER_BYTES: Final = 64 * 1024**2
MAX_SHARD_BYTES: Final = 128 * 1024**2


@dataclass(frozen=True, slots=True)
class PrefetchResult:
    copied: int
    verified_existing: int
    shards_read: int
    seconds: float


def verify_existing(path: Path, expected: str) -> bool:
    if path.is_symlink():
        raise ValueError('Local cache must not be a symbolic link')
    if not path.exists():
        return False
    if not path.is_file() or not 0 < path.stat().st_size <= MAX_MEMBER_BYTES:
        raise ValueError('Local cache exceeds payload bounds')
    with path.open('rb') as stream:
        if hashlib.file_digest(stream, 'sha256').hexdigest() != expected:
            raise ValueError('Existing local cache hash mismatch; preserved')
    return True


def prefetch(index: Path, expected_sha256: str, destination: Path, deadline_epoch: float) -> PrefetchResult:
    started = time.monotonic()
    remaining = deadline_epoch-time.time()
    if not math.isfinite(remaining) or not 0 < remaining <= 1800:
        raise ValueError('Prefetch requires a future deadline within 30 minutes')
    stop = started+remaining
    with index.open('rb') as stream:
        raw = stream.read(96*1024**2+1)
    if len(raw) > 96*1024**2 or hashlib.sha256(raw).hexdigest() != expected_sha256:
        raise ValueError('Pinned cache index hash mismatch')
    source_root = index.resolve().parent
    destination = destination.resolve()
    if destination == source_root or destination.is_relative_to(source_root):
        raise ValueError('Local cache must be separate from the source cache')
    destination.mkdir(parents=True, exist_ok=True)
    groups: dict[Path, list[dict[str, str]]] = {}
    for entry in json.loads(raw)['entries']:
        path = Path(entry['path'])
        if (path.parent != source_root
                or not re.fullmatch(r'part_[0-9]{5}\.zip', path.name)
                or not re.fullmatch('[0-9a-f]{64}', entry['sha256'])
                or not isinstance(entry['member'], str)):
            raise ValueError('Cache entry outside the pinned shard layout')
        groups.setdefault(path, []).append(dict(member=entry['member'], sha256=entry['sha256']))
    copied = existing = shards = 0
    for path, entries in groups.items():
        if time.monotonic() >= stop or time.time() >= deadline_epoch:
            raise TimeoutError('Prefetch deadline; verified local copies retained')
        if path.is_symlink() or path.resolve().parent != source_root:
            raise ValueError('Cache shard outside the pinned source directory')
        missing = []
        for entry in entries:
            if verify_existing(destination/(entry['sha256']+'.npz'), entry['sha256']):
                existing += 1
            else:
                missing.append(entry)
        if not missing:
            continue
        with path.open('rb') as stream:
            raw = stream.read(MAX_SHARD_BYTES+1)
        if len(raw) > MAX_SHARD_BYTES:
            raise ValueError('Cache shard exceeds bounded staging memory')
        with zipfile.ZipFile(io.BytesIO(raw)) as archive:
            for entry in missing:
                if time.monotonic() >= stop or time.time() >= deadline_epoch:
                    raise TimeoutError('Prefetch deadline; verified local copies retained')
                info = archive.getinfo(entry['member'])
                if (info.compress_type != zipfile.ZIP_STORED or not 0 < info.file_size <= MAX_MEMBER_BYTES
                        or archive.namelist().count(entry['member']) != 1):
                    raise ValueError('Invalid stored cache member')
                payload = archive.read(info)
                if hashlib.sha256(payload).hexdigest() != entry['sha256']:
                    raise ValueError('Cache payload hash mismatch')
                target = destination/(entry['sha256']+'.npz')
                with tempfile.NamedTemporaryFile(dir=destination, prefix='.prefetch-', suffix='.tmp') as temporary:
                    temporary.write(payload)
                    temporary.flush()
                    try:
                        os.link(temporary.name, target)
                        copied += 1
                    except FileExistsError:
                        verify_existing(target, entry['sha256'])
                        existing += 1
        shards += 1
        if shards % 16 == 0:
            print(json.dumps(asdict(PrefetchResult(copied, existing, shards, time.monotonic()-started))), flush=True)
    return PrefetchResult(copied, existing, shards, time.monotonic()-started)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Prefetch authenticated cache bytes without model execution or overwriting reader temporaries.')
    parser.add_argument('--index', type=Path, required=True)
    parser.add_argument('--sha256', required=True)
    parser.add_argument('--destination', type=Path, required=True)
    parser.add_argument('--deadline-epoch', type=float, required=True)
    args = parser.parse_args()
    print(json.dumps(asdict(prefetch(args.index, args.sha256, args.destination, args.deadline_epoch))), flush=True)
