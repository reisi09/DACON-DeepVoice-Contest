"""Locked set v2 (review round 3, before any metric): drop locked rows whose audio sha256 equals a bank row or a new row,
group OpenBMAT rows by connected components of their music tracks, and mark the OpenBMAT part as conditional
(separated by music track only; broadcast speech sources may recur). Rewrites locked_rows.csv / manifest, writes LOCK_v2.json.
  python locked_v2.py LOCKED_DIR BANK_DIR NEW_HASHES_CSV"""
import csv
import hashlib
import json
import sys
import zipfile
from pathlib import Path

L, BANK, NEWH = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
rows = list(csv.DictReader((L / 'locked_rows.csv').open()))
cols = list(rows[0])
used = {r['audio_sha256'] for f in ('bank_trainval.csv', 'bank_test.csv') for r in csv.DictReader((BANK / f).open()) if r.get('audio_sha256')}
used |= {r['audio_sha256'] for r in csv.DictReader(NEWH.open()) if r['audio_sha256']}
dup = [r['ID'] for r in rows if r['audio_sha256'] in used]
rows = [r for r in rows if r['audio_sha256'] not in used]

z = zipfile.ZipFile('/root/newdata/ai_openbmat/AI-OpenBMAT.zip')
parent = {}


def find(x):
    parent.setdefault(x, x)
    while parent[x] != x:
        parent[x] = parent[parent[x]]
        x = parent[x]
    return x


for r in rows:
    if r['source'] == 'openbmat':
        art = 'ai' if '/ai/' in r['path'] else 'human'
        m = json.loads(z.read(f"AI-OpenBMAT/{art}/{Path(r['path']).stem}.json"))
        for seg in m['segments']:
            for s in seg.get('sources', []):
                if s.get('type') == 'music':
                    parent[find(('row', r['ID']))] = find(('track', Path(s['file']).stem))
comp = {}
for r in rows:
    if r['source'] == 'openbmat':
        root = find(('row', r['ID']))
        r['group_id'] = r['component_id'] = f"locked:openbmat:c{comp.setdefault(root, len(comp))}"
        r['variant'] = 'conditional: separated from used rows by music track only'
with (L / 'locked_rows.csv').open('w', newline='') as f:
    w = csv.DictWriter(f, cols)
    w.writeheader()
    w.writerows(rows)
with (L / 'locked_manifest.csv').open('w', newline='') as f:
    w = csv.writer(f)
    w.writerow(['ID', 'path'])
    w.writerows([r['ID'], r['path']] for r in rows)
v1 = json.loads((L / 'LOCK.json').read_text())
label_cols = ['ID', 'VOICE_PRESENT', 'VOICE_PRESENT_mask', 'MUSIC_PRESENT', 'MUSIC_PRESENT_mask', 'VOICE_FAKE', 'VOICE_FAKE_mask',
              'MUSIC_FAKE', 'MUSIC_FAKE_mask', 'FILE_FAKE', 'FILE_FAKE_mask', 'group_id']
from datetime import datetime  # noqa: E402
from zoneinfo import ZoneInfo  # noqa: E402
lock = {'version': 2, 'created_kst': datetime.now(ZoneInfo('Asia/Seoul')).strftime('%Y-%m-%d %H:%M'), 'v1_ids_sha256': v1['ids_sha256'],
        'n': len(rows), 'dropped_hash_duplicates': dup,
        'ids_sha256': hashlib.sha256('\n'.join(sorted(r['ID'] for r in rows)).encode()).hexdigest(),
        'labels_sha256': hashlib.sha256(json.dumps([[r[c] for c in label_cols] for r in sorted(rows, key=lambda r: r['ID'])]).encode()).hexdigest(),
        'openbmat_components': len(comp), 'openbmat_note': 'conditional evaluation: music-track separated, not fully independent',
        'rule': 'score once after the final candidate is fixed; no metric computed before'}
(L / 'LOCK_v2.json').write_text(json.dumps(lock, indent=1))
print(json.dumps({k: v for k, v in lock.items() if k != 'dropped_hash_duplicates'} | {'dropped': len(dup)}, indent=1))
