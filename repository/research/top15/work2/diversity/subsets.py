"""Deterministic train-row subsets for the diversity study (DB protocol: only TRAIN rows are subsampled; val and held-out
test are never touched).

Universe: backbone-cache rows whose npz existed before 2026-09-27 12:00 KST = the 31,270-row R2 row set (the cache is
being extended while this study runs; frozen once into universe.txt).
Cell = the split cell of data_v4/split.py: type|FILE_FAKE|source|generator (type = presence combination).
Stratum = type|FILE_FAKE, so both arms keep the route/label mix of train exactly.
  quantity(f, draw):  every cell keeps ceil(f * n_cell) rows, taken group by group in a random group order
                      (last group trimmed) -> all cells kept, fewer groups per cell.
  diversity(f, draw): per stratum, whole cells in random order until the stratum holds exactly the quantity arm's row
                      count (last cell trimmed group-wise) -> fewer cells, same rows.
  lofo(F):            train minus the family's relative cells (relatives.json).
  control(F):         same rows per stratum as lofo(F), but every cell thinned proportionally (no cell dropped).
All subsets are nested in f (prefix construction).     python subsets.py  -> checks + subsets_summary.json
"""
from __future__ import annotations

import csv
import json
import math
import os
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path('/Users/choiwuseck/Documents/ChatGPT/데이콘')
BANK = ROOT / 'results/data_v4_20260923/data_v4_results_20260923T1639Z/data/bank'
CACHE = ROOT / '_workspace/backbone_cache'
HERE = Path(__file__).resolve().parent
FRACS, Q_DRAWS, D_DRAWS = (.25, .5, .75), (0,), (1, 2, 3)


def universe() -> set[str]:
    p = HERE / 'universe.txt'
    if not p.exists():
        cut = time.mktime(time.strptime('2026-09-27 12:00', '%Y-%m-%d %H:%M'))
        ids = sorted(e.name[:-4] for e in os.scandir(CACHE) if e.name.endswith('.npz') and e.stat().st_mtime < cut)
        p.write_text('\n'.join(ids) + '\n')
    return set(p.read_text().split())


def load_rows() -> dict[str, list[dict]]:
    u = universe()
    return {s: [r for r in csv.DictReader((BANK / f).open()) if r['split'] == s and r['ID'] in u]
            for s, f in (('train', 'bank_trainval.csv'), ('val', 'bank_trainval.csv'), ('test', 'bank_test.csv'))}


def rtype(r) -> str:
    v = r['VOICE_PRESENT'] if r['VOICE_PRESENT_mask'] == '1' else ''
    m = r['MUSIC_PRESENT'] if r['MUSIC_PRESENT_mask'] == '1' else ''
    if v == '' or m == '':
        return 'unknown_presence'
    return {(1, 0): 'voice', (0, 1): 'music', (1, 1): 'mixed', (0, 0): 'none'}[(int(float(v)), int(float(m)))]


def stratum(r) -> str:
    return f"{rtype(r)}|{r['FILE_FAKE'] if r['FILE_FAKE_mask'] == '1' else 'None'}"


def cell(r) -> str:
    return f"{stratum(r)}|{r['source']}|{r['generator'] or 'None'}"


def _group_order(rows, draw):
    """rows ordered group by group (random group order for this draw, rows by ID inside a group)."""
    by = defaultdict(list)
    for r in rows:
        by[r['group_id']].append(r)
    keys = sorted(by)
    rng = np.random.default_rng(draw)
    return [r for k in (keys[i] for i in rng.permutation(len(keys))) for r in sorted(by[k], key=lambda r: r['ID'])]


def _cells(train):
    c = defaultdict(list)
    for r in train:
        c[cell(r)].append(r)
    return c


def quantity(train, f, draw=0) -> set[str]:
    return {r['ID'] for name, rs in _cells(train).items() for r in _group_order(rs, draw)[:math.ceil(f * len(rs))]}


def diversity(train, f, draw) -> set[str]:
    cells, keep = _cells(train), set()
    by_stratum = defaultdict(list)
    for name in sorted(cells):
        by_stratum[name.rsplit('|', 2)[0]].append(name)
    rng = np.random.default_rng(1000 + draw)
    for s, names in sorted(by_stratum.items()):
        target = sum(math.ceil(f * len(cells[n])) for n in names)
        taken = 0
        for n in (names[i] for i in rng.permutation(len(names))):
            rs = _group_order(cells[n], draw)
            take = rs[:target - taken]
            keep |= {r['ID'] for r in take}
            taken += len(take)
            if taken >= target:
                break
    return keep


def relatives() -> dict:
    return json.loads((HERE / 'relatives.json').read_text())


def lofo(train, fam) -> set[str]:
    rel = set(relatives()[fam]['cells'])
    return {r['ID'] for r in train if cell(r) not in rel}


def control(train, fam, draw=7) -> set[str]:
    """LOFO's row count per stratum, but every cell of the stratum (relatives included) is thinned proportionally
    (largest-remainder allocation, group order) -> same rows as LOFO, no cell dropped."""
    rel = set(relatives()[fam]['cells'])
    cells, keep = _cells(train), set()
    by_stratum = defaultdict(list)
    for name in sorted(cells):
        by_stratum[name.rsplit('|', 2)[0]].append(name)
    for s, names in by_stratum.items():
        n = sum(len(cells[c]) for c in names)
        target = sum(len(cells[c]) for c in names if c not in rel)
        quota = {c: len(cells[c]) * target / n for c in names}
        alloc = {c: int(q) for c, q in quota.items()}
        for c in sorted(names, key=lambda c: alloc[c] - quota[c])[:target - sum(alloc.values())]:
            alloc[c] += 1
        for c in names:
            keep |= {r['ID'] for r in _group_order(cells[c], draw)[:alloc[c]]}
    return keep


def configs(train) -> dict[str, set[str]]:
    out = {'full': {r['ID'] for r in train}}
    for f in FRACS:
        for d in Q_DRAWS:
            out[f'Q{int(f * 100)}_d{d}'] = quantity(train, f, d)
        for d in D_DRAWS:
            out[f'D{int(f * 100)}_d{d}'] = diversity(train, f, d)
    if (HERE / 'relatives.json').exists():
        for fam in relatives():
            out[f'LOFO_{fam}'] = lofo(train, fam)
            out[f'CTRL_{fam}'] = control(train, fam)
    return out


def describe(train, ids):
    rs = [r for r in train if r['ID'] in ids]
    gens = lambda lab: {r['generator'] or r['source'] for r in rs if r['FILE_FAKE'] == lab}
    return {'rows': len(rs), 'cells': len({cell(r) for r in rs}), 'sources': len({r['source'] for r in rs}),
            'groups': len({r['group_id'] for r in rs}), 'fake_generators': len(gens('1')),
            'per_stratum': {s: n for s, n in sorted(__import__('collections').Counter(stratum(r) for r in rs).items())}}


if __name__ == '__main__':
    train = load_rows()['train']
    C = configs(train)
    full = describe(train, C['full'])
    for f in FRACS:   # self-checks: equal rows per stratum across arms, nesting in f, Q keeps every cell
        q = describe(train, C[f'Q{int(f * 100)}_d0'])
        assert q['cells'] == full['cells'], f
        for d in D_DRAWS:
            assert describe(train, C[f'D{int(f * 100)}_d{d}'])['per_stratum'] == q['per_stratum'], (f, d)
    for a, b in zip(FRACS, FRACS[1:]):
        for arm in ['Q%d_d0'] + [f'D%d_d{d}' for d in D_DRAWS]:
            assert C[arm % int(a * 100)] <= C[arm % int(b * 100)], (arm, a, b)
    for k in C:
        if k.startswith('LOFO_'):
            assert describe(train, C[k])['per_stratum'] == describe(train, C['CTRL_' + k[5:]])['per_stratum'], k
    summary = {k: describe(train, v) for k, v in C.items()}
    (HERE / 'subsets_summary.json').write_text(json.dumps(summary, indent=1))
    for k, v in summary.items():
        print(k, {x: v[x] for x in ('rows', 'cells', 'sources', 'groups', 'fake_generators')})
