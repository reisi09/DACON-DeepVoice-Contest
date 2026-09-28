#!/usr/bin/env python3
# /// script
# requires-python = ">=3.11"
# dependencies = []
# ///
# Run: python3 legacy_cache_export.py --seconds 600
"""Inspect and preserve the legacy feature archive without extracting or loading models."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path, PurePosixPath
import signal
import sys
import tarfile
import time
from typing import BinaryIO, Final

MAX_BYTES: Final = 12 * 2**30
MAX_INDEX: Final = 64 * 2**20
CHUNK: Final = 4 * 2**20
EXPECTED_TV: Final = '4f427e667cc0deb27d49820fb87a4a2c74edd90496f2e3d7efd20fa7f8641f9f'
SOURCE: Final = Path('/source/data_v4/feature_cache_v4.tar')
DESTINATION: Final = Path('/shared/lb-improvement-20260925/legacy-cache-v1/feature_cache_v4.tar')


class ExportError(RuntimeError):
    """Unsafe archive, incomplete provenance, or conflicting destination."""


class HashedReader:
    """Hash every input byte while tarfile walks headers and payloads once."""
    def __init__(self, stream: BinaryIO, deadline: float):
        self.stream, self.deadline = stream, deadline
        self.digest = hashlib.sha256()
        self.bytes = 0

    def read(self, size: int = CHUNK) -> bytes:
        if time.monotonic() >= self.deadline:
            raise TimeoutError('Export deadline reached')
        data = self.stream.read(size)
        self.bytes += len(data)
        if self.bytes > MAX_BYTES:
            raise ExportError('Archive exceeds 12 GiB')
        self.digest.update(data)
        return data


def digest_file(path: Path, deadline: float) -> str:
    if path.is_symlink() or not path.is_file():
        raise ExportError('Only regular files may be hashed')
    with path.open('rb') as stream:
        reader = HashedReader(stream, deadline)
        while reader.read():
            pass
    return reader.digest.hexdigest()


def write_receipt(path: Path, value) -> None:
    temporary = path.with_name(f'{path.name}.{time.time_ns()}.partial')
    if path.is_symlink() or (path.exists() and not path.is_file()):
        raise ExportError('Receipt symlink is forbidden')
    with temporary.open('x') as stream:
        stream.write(json.dumps(value, sort_keys=True, indent=2) + '\n')
    temporary.replace(path)


def inspect_archive(source: Path, deadline: float, expected_tv: str):
    """Verify index identity and every referenced NPZ hash before any archive copy."""
    hashes, indices, prefixes, members = {}, {}, {}, set()
    counts, size = Counter(), 0
    with source.open('rb') as stream:
        reader = HashedReader(stream, deadline)
        with tarfile.open(fileobj=reader, mode='r|') as archive:
            for member in archive:
                path = PurePosixPath(member.name)
                if path.is_absolute() or '..' in path.parts or '\\' in member.name:
                    raise ExportError('Unsafe archive member path')
                if not (member.isfile() or member.isdir()) or member.issparse():
                    raise ExportError('Links, devices, and sparse members are forbidden')
                if member.name in members:
                    raise ExportError('Duplicate archive member')
                members.add(member.name)
                if len(members) > 200000:
                    raise ExportError('Excessive archive member count')
                if len(members) % 5000 == 0:
                    print(json.dumps(dict(phase='inspect', members=len(members), bytes_read=reader.bytes)), flush=True)
                if member.isdir():
                    continue
                roots = [i for i, part in enumerate(path.parts) if part in ('cache_tv', 'cache_test')]
                if len(roots) != 1 or roots[0] != len(path.parts) - 2:
                    raise ExportError('Regular member is outside a supported cache directory')
                key = path.parts[roots[0]]
                prefix = str(PurePosixPath(*path.parts[:roots[0] + 1]))
                if key in prefixes and prefixes[key] != prefix:
                    raise ExportError('Multiple prefixes contain the same cache key')
                prefixes[key] = prefix
                if path.suffix not in ('.npz', '.json') and path.name != 'voice_classifier_initial.pt':
                    raise ExportError('Unexpected non-feature member')
                size += member.size
                if size > MAX_BYTES or member.size < 0:
                    raise ExportError('Excessive archive payload size')
                is_index = path.name == 'index.json'
                if is_index and member.size > MAX_INDEX:
                    raise ExportError('Index exceeds 64 MiB')
                payload = archive.extractfile(member)
                if payload is None:
                    raise ExportError('Unreadable regular member')
                digest, chunks, read_bytes = hashlib.sha256(), [], 0
                with payload:
                    while block := payload.read(CHUNK):
                        if time.monotonic() >= deadline:
                            raise TimeoutError('Export deadline reached')
                        digest.update(block)
                        read_bytes += len(block)
                        if is_index:
                            chunks.append(block)
                if read_bytes != member.size:
                    raise ExportError('Truncated member')
                name = f'{key}/{path.name}'
                if name in hashes:
                    raise ExportError('Duplicate normalized cache member')
                hashes[name] = digest.hexdigest()
                if path.suffix == '.npz':
                    counts[key] += 1
                if is_index:
                    indices[key] = b''.join(chunks)
        while reader.read():
            pass
    if hashes.get('cache_tv/index.json') != expected_tv:
        raise ExportError('Legacy train/validation index SHA256 differs from pinned identity')
    if set(indices) != {'cache_tv', 'cache_test'}:
        raise ExportError('Both legacy cache indices are required')
    summaries = {}
    for key, raw in indices.items():
        index = json.loads(raw)
        if not isinstance(index, dict) or not isinstance(index.get('items'), dict):
            raise ExportError('Malformed cache index')
        expected = {}
        for identifier, entry in index['items'].items():
            if not isinstance(entry, dict) or entry.get('path') != f'{identifier}.npz':
                raise ExportError('Index ID/path mismatch')
            expected[f'{key}/{identifier}.npz'] = entry.get('sha256')
        actual = {name: value for name, value in hashes.items() if name.startswith(key + '/') and name.endswith('.npz')}
        if not expected or expected != actual:
            raise ExportError('NPZ membership or payload SHA256 differs from its index')
        identity = index.get('identity', {})
        if not isinstance(identity, dict):
            raise ExportError('Malformed cache identity')
        summaries[key] = dict(rows=len(expected), npz_count=counts[key], prefix=prefixes[key],
            index_sha256=hashes[f'{key}/index.json'], identity_sha256=hashlib.sha256(
                json.dumps(identity, sort_keys=True).encode()).hexdigest(),
            identity_fields=sorted(identity), metadata_bytes=len(raw))
    return dict(archive_sha256=reader.digest.hexdigest(), archive_bytes=reader.bytes,
                member_count=len(members), npz_count=sum(counts.values()), caches=summaries), indices


def export(source: Path, destination: Path, seconds: float = 600, expected_tv: str = EXPECTED_TV):
    """Preserve source bytes; refuse conflicting targets and leave interrupted copies unpromoted."""
    started, deadline = time.monotonic(), time.monotonic() + seconds
    if not 0 < seconds <= 1200:
        raise ExportError('Deadline must be within 1,200 seconds')
    if source.resolve().parent != source.parent or source.is_symlink() or destination.is_symlink() or destination.parent.resolve() != destination.parent:
        raise ExportError('Source/destination symlink is forbidden')
    destination.parent.mkdir(parents=True, exist_ok=True)
    receipt = destination.parent / 'export_attempt.json'
    status = dict(status='started', gpu_used=False, source_modified=False, destination=destination.name)
    write_receipt(receipt, status)
    try:
        if not source.is_file():
            directories = []
            if source.parent.is_dir():
                for first in sorted(source.parent.iterdir()):
                    if first.is_symlink() or not first.is_dir() or any(x in first.name.lower() for x in ('audio', 'raw', 'weight')):
                        continue
                    directories.append(first.name)
                    for second in sorted(first.iterdir()):
                        if second.is_dir() and not second.is_symlink() and not any(x in second.name.lower() for x in ('audio', 'raw', 'weight')):
                            directories.append(f'{first.name}/{second.name}')
                        if len(directories) >= 100:
                            break
                    if len(directories) >= 100:
                        break
            status.update(source_exists=False, scoped_directories=directories[:100])
            raise ExportError('Expected legacy archive is missing')
        before = source.stat()
        if before.st_size > MAX_BYTES:
            raise ExportError('Source archive exceeds 12 GiB')
        report, indices = inspect_archive(source, deadline, expected_tv)
        status.update(report, status='archive_validated')
        write_receipt(receipt, status)
        if destination.exists():
            if digest_file(destination, deadline) != report['archive_sha256']:
                raise ExportError('Existing destination has different bytes; refusing overwrite')
            status['reused_existing'] = True
        else:
            partial = destination.with_suffix('.tar.partial')
            if partial.is_symlink():
                raise ExportError('Partial archive symlink is forbidden')
            status['status'] = 'copying'
            write_receipt(receipt, status)
            with source.open('rb') as original:
                copied = HashedReader(original, deadline)
                if partial.exists():
                    with partial.open('rb') as previous:
                        while block := previous.read(CHUNK):
                            if copied.read(len(block)) != block:
                                raise ExportError('Existing partial differs from source; refusing overwrite')
                with partial.open('ab' if partial.exists() else 'xb') as target:
                    while block := copied.read():
                        target.write(block)
                    target.flush()
            after = source.stat()
            if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns) or copied.digest.hexdigest() != report['archive_sha256']:
                raise ExportError('Source changed during export')
            if digest_file(partial, deadline) != report['archive_sha256']:
                raise ExportError('Destination full readback SHA256 mismatch')
            if destination.exists():
                raise ExportError('Destination appeared during copy; refusing overwrite')
            partial.rename(destination)
            status['reused_existing'] = False
        for key, raw in indices.items():
            metadata = destination.parent / f'{key}_index.json'
            if metadata.exists():
                if digest_file(metadata, deadline) != hashlib.sha256(raw).hexdigest():
                    raise ExportError('Existing exported index differs; refusing overwrite')
            else:
                temporary = metadata.with_name(f'{metadata.name}.{time.time_ns()}.partial')
                with temporary.open('xb') as stream:
                    stream.write(raw)
                if digest_file(temporary, deadline) != hashlib.sha256(raw).hexdigest():
                    raise ExportError('Exported index readback differs')
                if metadata.exists():
                    raise ExportError('Exported index appeared during write')
                temporary.rename(metadata)
        status.update(status='complete', destination_readback_sha256=report['archive_sha256'],
                      elapsed_seconds=time.monotonic() - started)
        write_receipt(receipt, status)
        write_receipt(destination.parent / 'export_receipt.json', status)
        return status
    except (OSError, ExportError, tarfile.TarError, ValueError, KeyboardInterrupt) as error:
        status.update(status='failed', error_type=type(error).__name__, error=str(error),
                      elapsed_seconds=time.monotonic() - started)
        write_receipt(receipt, status)
        raise


def stop_export(_signal, _frame) -> None:
    raise InterruptedError('Export interrupted; source preserved and partial copy not promoted')


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seconds', type=float, default=600)
    parser.add_argument('--expected-tv-index-sha256', default=EXPECTED_TV)
    args = parser.parse_args()
    if SOURCE.resolve().parent != SOURCE.parent or DESTINATION.parent.resolve() != DESTINATION.parent:
        raise ExportError('Mount paths must not traverse symlinks')
    signal.signal(signal.SIGTERM, stop_export)
    signal.signal(signal.SIGALRM, stop_export)
    signal.alarm(max(1, min(1200, int(args.seconds))))
    try:
        print(json.dumps(export(SOURCE, DESTINATION, args.seconds, args.expected_tv_index_sha256)), flush=True)
        return 0
    except (OSError, ExportError, tarfile.TarError, ValueError) as error:
        print(json.dumps(dict(status='failed', error_type=type(error).__name__, error=str(error))), flush=True)
        return 1
    finally:
        signal.alarm(0)


if __name__ == '__main__':
    sys.exit(main())
