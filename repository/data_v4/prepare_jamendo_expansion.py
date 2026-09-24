#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = []
# ///
# Run: python data_v4/prepare_jamendo_expansion.py ORIGINAL_PROJECT_ROOT OUTPUT_DIRECTORY
"""Plan original Jamendo tracks from new artists, preserving held-out artist exclusions."""
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import unicodedata


def main(root: Path, output: Path) -> None:
    """Select at most 1,000 public CC-BY/CC-BY-SA originals under a 6 GiB cap."""
    bank = root/'_workspace/data_expansion_20260918/jamendo'
    prior = [json.loads(line) for line in (bank/'sources.jsonl').open()]
    known = {str(row.get('track_id')) for row in prior}
    artists = {str(row.get('artist_id')) for row in prior}
    locked = json.loads((root/'results/data_expansion_20260918/jamendo_selection.json').read_text())
    held = set(locked['reserved_eval_artist_ids'])
    excluded_names = set(locked['excluded_fma_normalized_names'])
    manifest = root/'_workspace/lb_cpu_payload_20260924/datasets/roles-v3/manifest.jsonl'
    for line in manifest.open():
        row = json.loads(line)
        for group in row.get('group_keys', []):
            if group.startswith('reference:jamendo:'):
                known.add(group.split(':')[-1])
            if group.startswith('artist:jamendo:') and row['role'] != 'train':
                held.add(group.split(':')[-1])
    candidates, metadata_hashes = {}, {}
    for path in sorted((bank/'metadata').glob('*.jsonl')):
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        metadata_hashes[path.name] = digest
        for line in path.open():
            row = json.loads(line)
            artist, track = str(row.get('artist_id', '')), str(row.get('id', ''))
            name = ''.join(c for c in unicodedata.normalize('NFKD', row.get('artist_name', '')).casefold() if c.isalnum())
            license_url = row.get('license_ccurl', '').replace('http:', 'https:')
            if not (row.get('audiodownload_allowed') is True and 30 <= row.get('duration', 0) <= 480
                    and license_url.startswith(('https://creativecommons.org/licenses/by/',
                                               'https://creativecommons.org/licenses/by-sa/'))
                    and track not in known and artist not in held | artists and name not in excluded_names):
                continue
            url = f'https://prod-1.storage.jamendo.com/download/track/{track}/mp32/'
            if not track.isdigit() or not artist.isdigit() or row.get('audiodownload') != url:
                continue
            candidates[track] = dict(ID='jamendo:' + track, path='audio/jamendo_' + track + '.mp3',
                source='Jamendo', track_id=track, artist_id=artist, artist_name=row['artist_name'],
                title=row.get('name'), source_url=url, track_url='https://www.jamendo.com/track/' + track,
                license=license_url, releasedate=row.get('releasedate'), metadata_file=path.name,
                metadata_sha256=digest, duration_metadata_seconds=row['duration'],
                group_keys=['artist:jamendo:' + artist, 'reference:jamendo:' + track],
                FILE_FAKE=0, MUSIC_FAKE=0, MUSIC_PRESENT=1, VOICE_FAKE=None, VOICE_PRESENT=None,
                label_basis='historical music catalogue metadata; no independent listening verification',
                role='unassigned', training_admitted=False, listening_verified=False,
                rights_review='retain attribution and ShareAlike terms; competition review pending',
                corpus_overlap_review='exact prior artist IDs and FMA names excluded; aliases pending')
    by_artist = {}
    for row in sorted(candidates.values(), key=lambda r: hashlib.sha256(r['ID'].encode()).hexdigest()):
        by_artist.setdefault(row['artist_id'], row)
    chosen = sorted(by_artist.values(), key=lambda r: hashlib.sha256(('20260924|' + r['ID']).encode()).hexdigest())[:1000]

    def probe(row):
        result = subprocess.run(['curl', '-sSI', '--proto', '=https', '--connect-timeout', '10',
                                 '--max-time', '25', row['source_url']], capture_output=True, text=True)
        lines = result.stdout.splitlines()
        statuses = [line.split()[1] for line in lines if line.startswith('HTTP/')]
        sizes = [line.split(':', 1)[1].strip() for line in lines if line.lower().startswith('content-length:')]
        status = statuses[-1] if statuses else 'connection_error'
        size = int(sizes[-1]) if sizes and sizes[-1].isdigit() else 0
        return row, status, size

    output.mkdir(parents=True, exist_ok=False)
    sources, plan, failures = [], [], []
    total = 0
    with ThreadPoolExecutor(max_workers=6) as pool:
        for row, status, size in pool.map(probe, chosen):
            if status != '200' or not 0 < size <= 25 * 2**20 or total + size > 6 * 2**30:
                failures.append(dict(track_id=row['track_id'], status=status, bytes=size))
                continue
            total += size
            sources.append(row | dict(bytes=size))
            plan.append(dict(dataset='Jamendo_CC', path=row['path'], url=row['source_url'], size=size,
                checksum='size-only:unpublished', license=row['license'], source_split='unassigned'))
    (output/'sources.jsonl').write_text(''.join(json.dumps(row, ensure_ascii=False) + '\n' for row in sources))
    (output/'plan.json').write_text(json.dumps(plan, indent=2) + '\n')
    report = dict(selected=len(sources), bytes=total, unique_artists=len(sources), failures=failures,
        prior_journal_sha256=hashlib.sha256((bank/'sources.jsonl').read_bytes()).hexdigest(),
        roles_manifest_sha256=hashlib.sha256(manifest.read_bytes()).hexdigest(), metadata_sha256=metadata_hashes,
        metadata_revision='13090d785d071dcd617636deaa4a9775936ac995', training_admitted=False)
    (output/'selection_receipt.json').write_text(json.dumps(report, indent=2) + '\n')
    (output/'ATTRIBUTION.md').write_text('Original music from Jamendo. Per-track author, title, URL and license: sources.jsonl.\n'
        'Original MP3 files retained without modification. CC BY or CC BY-SA as specified per track.\n'
        'Metadata provenance: https://huggingface.co/datasets/amaai-lab/JamendoMaxCaps\n'
        'Competition admission and cross-catalogue alias review remain pending.\n')
    print(json.dumps(dict(selected=len(sources), bytes=total, failures=len(failures))))


if __name__ == '__main__':
    main(*(Path(arg) for arg in sys.argv[1:]))
