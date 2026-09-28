"""R2 data integrity (research/top15/work5/debate_20260928, design 2026-09-28): content hashes and full lineage for the new rows.
  1. sha256 of every new-row audio file; exact duplicates across splits (train / newval / newtest) and against the bank
     (bank audio_sha256), with the split pair of every duplicate
  2. full lineage components: AI-OpenBMAT excerpts joined by EVERY music track in their metadata (not only the first),
     remix rows joined to EVERY donor in parent_ids; a component that spans train and an evaluation split is a leak
  3. rows of each evaluation split that are clean under both checks (the E2 / newval rows the gates may use)
  python integrity_check.py NEW_ROWS BANK_DIR OUT_JSON   (VESSL; reads audio, writes only OUT_JSON and OUT_JSON.hashes.csv)"""
import csv
import hashlib
import json
import sys
import zipfile
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

EVAL = ('newval', 'newtest')


def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()


class DSU:
    def __init__(self):
        self.p = {}

    def find(self, x):
        self.p.setdefault(x, x)
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]
            x = self.p[x]
        return x

    def union(self, a, b):
        self.p[self.find(a)] = self.find(b)


def openbmat_tracks(rows):
    z = zipfile.ZipFile('/root/newdata/ai_openbmat/AI-OpenBMAT.zip')
    names = set(z.namelist())
    out = {}
    for r in rows:
        stem = Path(r['path']).stem
        art = 'ai' if '/ai/' in r['path'] else 'human'
        meta = f'AI-OpenBMAT/{art}/{stem}.json'
        if meta in names:
            m = json.loads(z.read(meta))
            out[r['ID']] = {Path(s['file']).stem for seg in m['segments'] for s in seg.get('sources', []) if s.get('type') == 'music'}
    return out


def main(new_rows, bank_dir, out):
    rows = list(csv.DictReader(open(new_rows)))
    with ThreadPoolExecutor(16) as ex:
        hashes = list(ex.map(lambda r: sha(r['path']) if Path(r['path']).exists() else None, rows))
    with open(str(out) + '.hashes.csv', 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['ID', 'split', 'audio_sha256'])
        w.writerows([r['ID'], r['split'], h] for r, h in zip(rows, hashes))
    bank = {}
    for name in ('bank_trainval.csv', 'bank_test.csv'):
        for r in csv.DictReader(open(Path(bank_dir) / name)):
            if r.get('audio_sha256'):
                bank.setdefault(r['audio_sha256'], set()).add('bank_' + r['split'])
    by_hash = defaultdict(set)
    for r, h in zip(rows, hashes):
        if h:
            by_hash[h].add(r['split'])
    dup_pairs = defaultdict(int)
    dirty = set()
    for r, h in zip(rows, hashes):
        if not h:
            continue
        splits = by_hash[h] | bank.get(h, set())
        if len(splits) > 1:
            dup_pairs['|'.join(sorted(splits))] += 1
            if r['split'] in EVAL:
                dirty.add(r['ID'])
    # lineage components: group_id, every OpenBMAT music track, every remix donor
    dsu, ids = DSU(), {r['ID']: r for r in rows}
    for r in rows:
        dsu.union(('id', r['ID']), ('g', r['group_id']))
        for p in json.loads(r.get('parent_ids') or '[]'):
            dsu.union(('id', r['ID']), ('id', p) if p in ids else ('g', 'parent:' + p))
    for i, tracks in openbmat_tracks([r for r in rows if r['source'] == 'openbmat']).items():
        for t in tracks:
            dsu.union(('id', i), ('track', t))
    comp_splits = defaultdict(set)
    for r in rows:
        comp_splits[dsu.find(('id', r['ID']))].add(r['split'])
    leak = [r for r in rows if r['split'] in EVAL and 'train' in comp_splits[dsu.find(('id', r['ID']))]]
    for r in leak:
        dirty.add(r['ID'])
    by_source = defaultdict(lambda: defaultdict(int))
    for r in leak:
        by_source[r['split']][r['source']] += 1
    res = {'rows': len(rows), 'missing_audio': sum(h is None for h in hashes),
           'exact_duplicates_by_split_set': dict(dup_pairs),
           'lineage_leak_rows': {k: dict(v) for k, v in by_source.items()},
           'dirty_eval_rows': len(dirty), 'clean_eval_rows': {s: sum(1 for r in rows if r['split'] == s and r['ID'] not in dirty) for s in EVAL}}
    json.dump({**res, 'dirty_ids': sorted(dirty)}, open(out, 'w'), indent=1)
    print(json.dumps(res, indent=1))


if __name__ == '__main__':
    main(*sys.argv[1:4])
