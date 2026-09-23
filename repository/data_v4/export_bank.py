"""data_v4 manifest + splits -> CSV consumed by full_bank_features.py / full_bank_train.py (unchanged formats).

  python data_v4/export_bank.py --manifest OUT/manifest.jsonl --splits OUT/splits.csv --out OUT/bank
writes bank_trainval.csv (train+val: feature extraction + head training) and bank_test.csv (held out).
Rows with no trainable head label (noise, unknown authenticity) are listed in skipped.json, not dropped silently.
Mixed rows get route = their music head; full_bank_train.head_labels also trains the voice head from them.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from build_manifest import LABELS, components  # noqa: E402

COLUMNS = ['ID', 'orig_id', 'split', 'route', 'path', 'audio_sha256', 'source_sha256', 'group_id', 'component_id',
           *[c for n in LABELS for c in (n, n + '_mask')], 'source', 'generator', 'language', 'codec_chain',
           'event_intervals', 'parent_ids', 'variant', 'license', 'historical_development_exposure']


def safe_id(rid: str) -> str:
    return 'v4_' + hashlib.sha256(rid.encode()).hexdigest()[:24]   # cache filenames must be plain names


def route(r: dict) -> str | None:
    if r['MUSIC_PRESENT'] == 1 and r['MUSIC_FAKE'] is not None:
        return 'music'
    if r['VOICE_PRESENT'] == 1 and r['VOICE_FAKE'] is not None:
        return 'voice'
    return None


def music_events(r: dict) -> list[dict]:
    """HAIM mixset segments (original timeline) -> intervals relative to the materialized crop."""
    if not r.get('segments'):
        return []
    s0, n = r.get('crop_start_s', 0.), r.get('crop_len_s', 0.)
    out = []
    for a, b, fake in r['segments']:
        lo, hi = max(0., a - s0), min(n, b - s0)
        if hi > lo:
            out.append({'component': 'music', 'start': round(lo, 3), 'end': round(hi, 3), 'label': int(fake)})
    return out


def export(rows: list[dict], split: dict[str, str]) -> tuple[list[dict], list[dict], list[dict]]:
    comp = components(rows)
    out, test, skipped = [], [], []
    for r in rows:
        s = split.get(r['ID'])
        head = route(r)
        if s is None or head is None:
            skipped.append({'ID': r['ID'], 'split': s, 'reason': 'no split' if s is None else 'no trainable head label'})
            continue
        row = {'ID': safe_id(r['ID']), 'orig_id': r['ID'], 'split': s, 'route': head, 'path': r['path'],
               'audio_sha256': r['sha256'], 'source_sha256': r['sha256'], 'group_id': comp[r['ID']],
               'component_id': comp[r['ID']], 'source': r['source'], 'generator': r.get('generator') or '',
               'language': r.get('language') or '', 'codec_chain': r.get('codec_chain', ''),
               'event_intervals': json.dumps(music_events(r)), 'parent_ids': json.dumps(r.get('parent_ids', [])),
               'variant': '', 'license': r.get('license', ''), 'historical_development_exposure': r.get('prior_seen') or ''}
        for n in LABELS:
            row[n] = '' if r[n] is None else str(int(r[n]))
            row[n + '_mask'] = '0' if r[n] is None else '1'
        (test if s == 'test' else out).append(row)
    return out, test, skipped


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--manifest', type=Path, required=True)
    ap.add_argument('--splits', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    a = ap.parse_args()
    rows = [json.loads(l) for l in a.manifest.open()]
    split = {r['ID']: r['split'] for r in csv.DictReader(a.splits.open())}
    trainval, test, skipped = export(rows, split)
    a.out.mkdir(parents=True, exist_ok=True)
    for name, data in (('bank_trainval.csv', trainval), ('bank_test.csv', test)):
        with (a.out / name).open('w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=COLUMNS)
            w.writeheader()
            w.writerows(data)
    (a.out / 'skipped.json').write_text(json.dumps(skipped, indent=1) + '\n')
    both = sum(1 for r in trainval if r['MUSIC_FAKE'] and r['VOICE_FAKE'] and r['VOICE_PRESENT'] == '1')
    print(json.dumps({'trainval': len(trainval), 'test': len(test), 'skipped': len(skipped), 'rows_training_both_heads': both}))


def self_check() -> None:
    from build_manifest import lab
    base = dict(sha256='a' * 64, path='audio/x.flac', source='s', license='l', crop_start_s=5., crop_len_s=10.)
    rows = [dict(ID='song', group_keys=['trk:1'], **base, **lab(1, 1, 1, 1)),
            dict(ID='speech', group_keys=['spk:1'], **base, **lab(1, 0, 0, None)),
            dict(ID='noise', group_keys=['n:1'], **base, **lab(0, 0, None, None)),
            dict(ID='mixset', group_keys=['trk:2'], segments=[(0, 8, 0), (8, 30, 1)], **base, **lab(None, 1, None, 1))]
    tv, te, sk = export(rows, {'song': 'train', 'speech': 'val', 'noise': 'train', 'mixset': 'test'})
    by = {r['orig_id']: r for r in tv + te}
    assert by['song']['route'] == 'music' and by['song']['VOICE_FAKE'] == '1', 'song keeps its voice label for the second head'
    assert by['speech']['route'] == 'voice' and by['speech']['MUSIC_FAKE_mask'] == '0'
    assert [s['ID'] for s in sk] == ['noise'], 'rows without trainable label are reported, not silently dropped'
    assert json.loads(by['mixset']['event_intervals']) == [{'component': 'music', 'start': 0.0, 'end': 3.0, 'label': 0},
                                                           {'component': 'music', 'start': 3.0, 'end': 10.0, 'label': 1}]
    assert te and te[0]['orig_id'] == 'mixset' and all(r['split'] != 'test' for r in tv)
    assert all('/' not in r['ID'] and ':' not in r['ID'] for r in tv + te)
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from full_bank_train import head_labels
    assert head_labels(by['song']) == {'music': 1., 'voice': 1.}, 'export -> trainer contract'
    print('export_bank self-check ok')


if __name__ == '__main__':
    self_check() if sys.argv[1:] == ['--self-check'] else main()
