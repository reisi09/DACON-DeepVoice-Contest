"""Splits of the new rows by lineage group (research/top15/work5/newdata), fixed before any training:
  AT-ADD      eval pool -> newtest (generators unseen in train/dev); train pool: 15 % of groups -> newval, rest train
  open data   30 % of groups -> newtest, next 15 % -> newval, rest train (u = sha1(group) mapped to [0, 1))
  remix_d1    train only, and only mixes whose donors are all train rows (no held-out lineage inside a training mix)
  python assign_splits.py [ROWS_DIR]   -> ROWS_DIR/new_rows.csv (every column of the inputs, 'split' reassigned)"""
import csv
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

D = Path(sys.argv[1]) if len(sys.argv) > 1 else Path('/root/newdata/rows')


def u(group: str) -> float:
    return int(hashlib.sha1(group.encode()).hexdigest()[:8], 16) / 2 ** 32


def main():
    atadd = list(csv.DictReader((D / 'atadd_rows.csv').open()))
    open_ = list(csv.DictReader((D / 'open_rows.csv').open()))
    out = []
    for r in atadd:
        split = 'newtest' if r['split'] == 'newtest' else ('newval' if u(r['group_id']) < .15 else 'train')
        out.append({**r, 'split': split})
    for r in open_:
        if r['source'] == 'remix_d1':
            continue
        x = u(r['group_id'])
        out.append({**r, 'split': 'newtest' if x < .3 else ('newval' if x < .45 else 'train')})
    split_of = {r['ID']: r['split'] for r in out}
    dropped = 0
    for r in open_:
        if r['source'] != 'remix_d1':
            continue
        if all(split_of.get(p) == 'train' for p in json.loads(r['parent_ids'])):
            out.append({**r, 'split': 'train'})
        else:
            dropped += 1
    with (D / 'new_rows.csv').open('w', newline='') as f:
        w = csv.DictWriter(f, list(atadd[0].keys()))
        w.writeheader()
        w.writerows(out)
    c = Counter((r['source'], r['split']) for r in out)
    for s in sorted({k[0] for k in c}):
        print(f"{s:20s}", {k: c[(s, k)] for k in ('train', 'newval', 'newtest')})
    print('remix dropped (held-out donor)', dropped, 'total', len(out))


if __name__ == '__main__':
    main()
