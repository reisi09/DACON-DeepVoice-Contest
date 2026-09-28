"""Supplement (no presence needed): MUSIC EER on every newval row with a known music label, s7_new vs W_new music heads
(the mix changes only the music member), lineage-linked rows removed, paired group bootstrap (2,000)."""
import csv, json, sys
import numpy as np
sys.path.insert(0, '/root/proj/research/top15/work3/integration')
sys.path.insert(0, '/root/proj')
from fit_ft_fusion import eer
exec(open('/root/nd_eval/select_newval.py').read().split("print('linked newval'")[0])
rows = {r['ID']: r for r in csv.DictReader(open('/root/newdata/rows/new_rows.csv'))}
def music(cfg):
    r = json.load(open(f'/root/nd_newval/{cfg}/music_new__mert6_12+eat16+sslampt5.json'))['every']['newval']
    return dict(zip(r['ids'], np.mean(r['preds'], 0)))
a, b = music('s7_new'), music('W_new')
ids = [i for i in a if i in b and i not in linked and rows[i]['MUSIC_PRESENT'] == '1' and rows[i]['MUSIC_FAKE_mask'] == '1']
y = np.array([int(rows[i]['MUSIC_FAKE']) for i in ids]); pa = np.array([a[i] for i in ids]); pb = np.array([b[i] for i in ids])
g = np.array([rows[i]['group_id'] for i in ids]); ug = np.unique(g); members = {x: np.flatnonzero(g == x) for x in ug}
rng = np.random.default_rng(0); d = []
for _ in range(2000):
    idx = np.concatenate([members[x] for x in rng.choice(ug, len(ug))])
    if len(set(y[idx])) == 2:
        d.append(eer(y[idx], pa[idx]) - eer(y[idx], pb[idx]))   # > 0 = W_new music better
d = np.array(d)
print(f'newval music rows {len(ids)} (fake {int(y.sum())}) | MUSIC EER s7_new {eer(y, pa):.4f}  W_new {eer(y, pb):.4f} | '
      f'EER gain {eer(y, pa) - eer(y, pb):+.4f} CI [{np.percentile(d, 2.5):+.4f},{np.percentile(d, 97.5):+.4f}] p>0 {(d > 0).mean():.3f} '
      f'| ADS-equivalent x0.3 {0.3 * (eer(y, pa) - eer(y, pb)):+.4f}')
by = {}
for i, yy, x1, x2 in zip(ids, y, pa, pb):
    by.setdefault(rows[i]['source'], []).append((yy, x1, x2))
for s, v in sorted(by.items()):
    yy = np.array([t[0] for t in v])
    if len(set(yy)) == 2:
        print(f'   {s:14s} n={len(v):5d} MUSIC EER s7_new {eer(yy, np.array([t[1] for t in v])):.4f}  W_new {eer(yy, np.array([t[2] for t in v])):.4f}')
