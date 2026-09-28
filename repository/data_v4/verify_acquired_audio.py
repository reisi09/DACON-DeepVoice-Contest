#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy", "imageio-ffmpeg"]
# ///
# Run: PYTHONPATH=. python data_v4/verify_acquired_audio.py PAYLOAD PRIOR_JSONL...
"""Decode originals on CPU and compare byte/PCM identities before staging any training data."""
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import sys
import unicodedata

import numpy as np

from v2_data import decode_audio, pcm_sha256, sha256_file


def main(payload: Path, previous: list[Path]) -> None:
    """Write measured provenance without promoting labels, quality, or split assignments."""
    known, previous_hashes, prior_artist_names = set(), {}, set()
    def normalized_name(value: str) -> str:
        return ''.join(c for c in unicodedata.normalize('NFKD', value).casefold() if c.isalnum())
    for path in previous:
        previous_hashes[str(path)] = sha256_file(path)
        with path.open() as stream:
            for line in stream:
                row = json.loads(line)
                if row.get('artist_name'):
                    prior_artist_names.add(normalized_name(row['artist_name']))
                for key, value in row.items():
                    if 'sha256' in key and isinstance(value, str) and len(value) == 64:
                        known.add(value)
                known.update(group.split(':', 1)[1] for group in row.get('group_keys', [])
                             if group.startswith(('bytes:', 'pcm:')))
    rows = [json.loads(line) for line in (payload/'sources.jsonl').open()]
    if len({row['path'] for row in rows}) != len(rows):
        raise ValueError('Multiple source rows refer to one payload path')

    def measure(row):
        path = (payload/row['path']).resolve()
        if not path.is_relative_to(payload.resolve()) or not path.is_file():
            raise ValueError('Missing or escaped original audio')
        digest = sha256_file(path)
        if path.stat().st_size != row['bytes'] or (row.get('sha256') and row['sha256'] != digest):
            raise ValueError('Original bytes do not match source identity')
        audio = decode_audio(path, limit_seconds=None)
        duration = len(audio) / 16000
        expected = row.get('duration_metadata_seconds', row.get('duration_seconds'))
        flags = []
        if row.get('artist_name') and normalized_name(row['artist_name']) in prior_artist_names:
            flags.append('prior_artist_name_match_review')
        if not 0 < duration <= 600:
            raise ValueError('Decoded duration outside collection resource bounds')
        if expected is not None and abs(duration - expected) > max(.25, expected * .02):
            flags.append('duration_metadata_mismatch')
        peak = float(np.abs(audio).max())
        clipped = float((np.abs(audio) >= .999).mean())
        if clipped > .01:
            flags.append('clipping_review')
        if peak < .001:
            flags.append('near_silence_review')
        pcm = pcm_sha256(audio)
        if digest in known or pcm in known:
            flags.append('prior_exact_duplicate')
        return row | dict(sha256=digest, pcm_sha256=pcm, duration_seconds=duration,
            canonical_pcm='ffmpeg mono 16000Hz float32 full duration', peak=peak,
            clipping_fraction=clipped, full_audio_decode_verified=True, quality_flags=flags,
            training_admitted=False, role='unassigned', listening_verified=False)

    with ThreadPoolExecutor(max_workers=4) as pool:
        measured = list(pool.map(measure, rows))
    seen = set()
    for row in measured:
        if row['sha256'] in seen or row['pcm_sha256'] in seen:
            row['quality_flags'].append('within_collection_exact_duplicate')
        seen.update((row['sha256'], row['pcm_sha256']))
    report = dict(files=len(measured), bytes=sum(row['bytes'] for row in measured),
        duration_seconds=sum(row['duration_seconds'] for row in measured),
        duration_range_seconds=[min(row['duration_seconds'] for row in measured),
                                max(row['duration_seconds'] for row in measured)],
        flags=dict(Counter(flag for row in measured for flag in row['quality_flags'])),
        prior_input_sha256=previous_hashes, source_manifest_sha256=sha256_file(payload/'sources.jsonl'),
        original_byte_and_full_decode_verified=True, gpu_used=False, training_admitted=False,
        limitations=['perceptual quality unreviewed', 'transformed duplicates and cross-corpus aliases unreviewed',
                     'component labels unreviewed', 'role assignment pending'])
    (payload/'verified_sources.jsonl').write_text(''.join(json.dumps(row, ensure_ascii=False) + '\n' for row in measured))
    (payload/'verification.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report))


if __name__ == '__main__':
    main(Path(sys.argv[1]), [Path(arg) for arg in sys.argv[2:]])
