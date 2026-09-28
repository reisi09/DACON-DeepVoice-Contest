"""Plan a bounded CC-BY FMA pilot from original metadata; exclude previously catalogued artists."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import quote
from urllib.request import Request, urlopen


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--metadata', type=Path, required=True)
    ap.add_argument('--previous', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--limit', type=int, default=200)
    args = ap.parse_args()
    if args.out.exists() or not 1 <= args.limit <= 500:
        raise ValueError('Use a fresh output directory and limit in [1, 500]')
    prior = [json.loads(line) for line in args.previous.open()]
    used = {r['artist_id'] for r in prior}
    chosen = []
    for row in csv.DictReader(args.metadata.open()):
        if row['artist_id'] in used or row['license_url'].replace('http:', 'https:') not in (
                'https://creativecommons.org/licenses/by/4.0/', 'https://creativecommons.org/licenses/by/3.0/'):
            continue
        try:
            mins, secs = map(int, row['track_duration'].split(':'))
        except ValueError:
            continue
        if not 30 <= mins * 60 + secs <= 180 or not row['track_file'].startswith('music/'):
            continue
        used.add(row['artist_id'])
        chosen.append(row)
    chosen.sort(key=lambda r: hashlib.sha256(('20260924|' + r['track_id']).encode()).hexdigest())
    args.out.mkdir(parents=True)
    plan, sources, failures = [], [], []
    def probe(row):
        url = 'https://files.freemusicarchive.org/storage-freemusicarchive-org/' + quote(row['track_file'], safe='/')
        try:
            with urlopen(Request(url, method='HEAD'), timeout=15) as response:
                size = int(response.headers.get('Content-Length', 0))
            return row, url, size, None
        except HTTPError as exc:
            return row, url, 0, 'pending_access' if exc.code in (401, 403) else f'HTTP {exc.code}'
        except (URLError, TimeoutError, ValueError):
            return row, url, 0, 'metadata_unavailable'
    with ThreadPoolExecutor(max_workers=4) as pool:
        for offset in range(0, min(len(chosen), 800), 20):
            for row, url, size, error in pool.map(probe, chosen[offset:offset+20]):
                if len(plan) >= args.limit:
                    break
                if error or not 0 < size <= 8 * 2**20:
                    failures.append(dict(track_id=row['track_id'], status=error or 'size_limit'))
                    continue
                path = 'audio/fma_' + row['track_id'] + '.mp3'
                plan.append(dict(dataset='FMA_CC_BY_pilot', path=path, url=url, size=size,
                                 checksum='size-only:unpublished', license=row['license_url'], source_split='unassigned'))
                sources.append(dict(ID='fma_pilot:' + row['track_id'], path=path, source_url=url,
                                    artist=row['artist_name'], title=row['track_title'], track_url=row['track_url'],
                                    artist_id=row['artist_id'], track_id=row['track_id'], license=row['license_url'],
                                    group_id='artist:fma:' + row['artist_id'], role='unassigned',
                                    prior_catalog_artist=False, voice_label=None, training_admitted=False,
                                    rights_scope='CC-BY: retain attribution and license; mark any later transformations',
                                    source_date=row['track_date_created']))
            print(json.dumps({'probed': min(offset+20, len(chosen)), 'selected': len(plan)}), flush=True)
            if len(plan) >= args.limit or (not plan and failures and all(r['status'] == 'pending_access' for r in failures)):
                break
    (args.out/'plan.json').write_text(json.dumps(plan, indent=2)+'\n')
    (args.out/'sources.jsonl').write_text(''.join(json.dumps(r, ensure_ascii=False)+'\n' for r in sources))
    (args.out/'selection_receipt.json').write_text(json.dumps(dict(
        target=args.limit, selected=len(plan), bytes=sum(r['size'] for r in plan), failures=failures,
        metadata_sha256=hashlib.file_digest(args.metadata.open('rb'), 'sha256').hexdigest(),
        previous_sha256=hashlib.file_digest(args.previous.open('rb'), 'sha256').hexdigest(),
        status='candidate_only; cross-corpus canonical matching and audio quality gate pending'), indent=2)+'\n')


if __name__ == '__main__':
    main()
