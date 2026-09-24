"""Extract a bounded, speaker-balanced original-audio pilot; never infer training admission."""
from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import io
import json
from pathlib import Path, PurePosixPath
import tarfile

import numpy as np
import soundfile as sf


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--archive', type=Path, required=True)
    parser.add_argument('--receipt', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--limit', type=int, default=200)
    parser.add_argument('--min-seconds', type=float, default=8)
    parser.add_argument('--max-seconds', type=float, default=35)
    parser.add_argument('--exclude-sources', type=Path, action='append', default=[])
    parser.add_argument('--source-split', choices=('dev-other', 'dev-clean', 'test-other'), default='dev-other')
    args = parser.parse_args()
    if args.out.exists() or not 1 <= args.limit <= 500:
        raise ValueError('Use a fresh namespace and limit in [1,500]')
    if not 0 < args.min_seconds <= args.max_seconds <= 60:
        raise ValueError('Duration bounds must satisfy 0 < min <= max <= 60')
    seen = set()
    for previous in args.exclude_sources:
        with previous.open() as stream:
            seen.update(json.loads(line)['sha256'] for line in stream if line.strip())
    receipt = json.loads(args.receipt.read_text().splitlines()[-1])
    digest = hashlib.file_digest(args.archive.open('rb'), 'sha256').hexdigest()
    if receipt.get('status') != 'original_checksum_verified' or receipt.get('sha256') != digest:
        raise ValueError('Archive does not match its verified acquisition receipt')
    candidates = defaultdict(list)
    with tarfile.open(args.archive) as archive:
        members = archive.getmembers()
        names = [m.name for m in members]
        if len(names) != len(set(names)) or any(PurePosixPath(n).is_absolute() or '..' in PurePosixPath(n).parts for n in names):
            raise ValueError('Unsafe archive path or duplicate member')
        if any(m.issym() or m.islnk() or (not m.isfile() and not m.isdir()) for m in members):
            raise ValueError('Links or special files are not accepted')
        if sum(m.size for m in members) > 2 * 2**30:
            raise ValueError('Pilot archive exceeds extraction budget')
        for member in members:
            p = PurePosixPath(member.name)
            if member.isfile() and p.suffix == '.flac' and len(p.parts) == 5 and p.parts[:2] == ('LibriSpeech', args.source_split):
                candidates[p.parts[2]].append(member)
        for values in candidates.values():
            values.sort(key=lambda m: hashlib.sha256(('20260924|' + m.name).encode()).hexdigest())
        audio_members = sorted((m for group in candidates.values() for m in group), key=lambda m: m.offset)
        if sum(m.size for m in audio_members) > 512 * 2**20:
            raise ValueError('Pilot audio exceeds the 512 MiB memory budget')
        # Read in archive order once; random seeks into gzip repeatedly decompress the prefix.
        content = {m.name: archive.extractfile(m).read() for m in audio_members}
        args.out.mkdir(parents=True)
        rows, rejected = [], []
        # Round robin caps speaker concentration without using model scores.
        while candidates and len(rows) < args.limit:
            for speaker in sorted(list(candidates)):
                member = candidates[speaker].pop()
                if not candidates[speaker]:
                    del candidates[speaker]
                raw = content.pop(member.name)
                audio_digest = hashlib.sha256(raw).hexdigest()
                if audio_digest in seen:
                    rejected.append(dict(member=member.name, reason='previously collected or duplicate audio'))
                    continue
                audio, rate = sf.read(io.BytesIO(raw), dtype='float32')
                duration = len(audio) / rate
                if rate != 16000 or audio.ndim != 1 or not np.isfinite(audio).all() or not args.min_seconds <= duration <= args.max_seconds:
                    rejected.append(dict(member=member.name, reason='rate/channels/finite/duration eligibility'))
                    continue
                path = 'audio/' + PurePosixPath(member.name).name
                (args.out/'audio').mkdir(exist_ok=True)
                (args.out/path).write_bytes(raw)
                seen.add(audio_digest)
                rows.append(dict(ID='librispeech:' + args.source_split + ':' + Path(path).stem, path=path,
                    archive_member=member.name, sha256=audio_digest, bytes=len(raw),
                    source='LibriSpeech', source_split=args.source_split, source_url=receipt['url'], archive_sha256=digest,
                    license='https://creativecommons.org/licenses/by/4.0/', speaker_id=speaker,
                    group_keys=['speaker:librispeech:' + speaker, 'chapter:librispeech:' + PurePosixPath(member.name).parts[3]],
                    duration_seconds=duration, sample_rate=rate, peak=float(np.abs(audio).max()),
                    clipping_fraction=float((np.abs(audio) >= .999).mean()),
                    FILE_FAKE=0, VOICE_FAKE=0, VOICE_PRESENT=1, MUSIC_FAKE=None, MUSIC_PRESENT=0,
                    label_basis='original LibriSpeech read-speech corpus; no synthetic transformation',
                    role='unassigned', training_admitted=False, listening_verified=False,
                    corpus_overlap_review='pending', backbone_pretraining_exposure='unknown'))
                if len(rows) == args.limit:
                    break
        for name in ('LibriSpeech/LICENSE.TXT', 'LibriSpeech/README.TXT'):
            if name in names:
                (args.out/PurePosixPath(name).name).write_bytes(archive.extractfile(name).read())
    (args.out/'sources.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
    (args.out/'ATTRIBUTION.md').write_text('LibriSpeech ASR corpus, Vassil Panayotov, Daniel Povey et al.\n'
        'Source: https://www.openslr.org/12/\nLicense: https://creativecommons.org/licenses/by/4.0/\n'
        'Original FLAC files retained without transformation. Selection only; see sources.jsonl.\n')
    report = dict(selected=len(rows), speakers=len({r['speaker_id'] for r in rows}), bytes=sum(r['bytes'] for r in rows),
                  duration_seconds=sum(r['duration_seconds'] for r in rows), archive_sha256=digest,
                  source_checksum=receipt['checksum'], rejected=len(rejected), gpu_used=False,
                  full_audio_decode_verified=True, training_admitted=False, sealed=False,
                  requested=args.limit, duration_bounds_seconds=[args.min_seconds, args.max_seconds],
                  excluded_sources_sha256={str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in args.exclude_sources},
                  remaining=['cross-corpus identity review', 'listening quality and component labels',
                             'long-speech and fake-speech coverage', 'role assignment before any construction'])
    (args.out/'pilot_receipt.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report))


if __name__ == '__main__':
    main()
