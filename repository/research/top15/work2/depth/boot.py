"""Group bootstrap (group_id, 1000 reps, models fixed) of held-out official ADS / component EER deltas of every ads.py
configuration vs A (R2 both-MLP) and vs B (depth all-row reference), plus per-source voice-route errors (each run at its own
route-EER threshold) for reference vs chosen voice heads.   python boot.py  (needs ads_test_preds.json)"""
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_curve

ROOT = Path('/Users/choiwuseck/Documents/ChatGPT/데이콘')
sys.path.insert(0, str(ROOT / 'data_v4'))
import train_backbone_heads as H  # noqa: E402

HERE = Path(__file__).resolve().parent
frozen = set(json.loads((HERE / 'rows_r2.json').read_text())['ids'])
test = [r for r in csv.DictReader((H.BANK / 'bank_test.csv').open()) if r['ID'] in frozen]
P = json.loads((HERE / 'ads_test_preds.json').read_text())
rep = json.loads((HERE / 'ads_report.json').read_text())
name = {'|'.join(c): n for n, c in rep['configs'].items()}
BASES = {'A': '|'.join(rep['configs']['A R2 both-MLP']), 'B': '|'.join(rep['configs']['B depth all-row reference'])}


def lab(r, c):
    return None if r[c + '_mask'] != '1' or r[c] == '' else int(r[c])


def eer(t, s):
    fpr, tpr, thr = roc_curve(t, s, drop_intermediate=False)
    i = np.argmin(np.abs(fpr - (1 - tpr)))
    return (fpr[i] + 1 - tpr[i]) / 2, thr[i]


C3 = (('file', 'FILE_FAKE'), ('voice', 'VOICE_FAKE'), ('music', 'MUSIC_FAKE'))
groups = np.array([r['group_id'] for r in test])
el = {'file': np.array([lab(r, 'FILE_FAKE') is not None for r in test]),
      'voice': np.array([lab(r, 'VOICE_PRESENT') == 1 and lab(r, 'VOICE_FAKE') is not None for r in test]),
      'music': np.array([lab(r, 'MUSIC_PRESENT') == 1 and lab(r, 'MUSIC_FAKE') is not None for r in test])}
y = {k: np.array([lab(r, c) if e else -1 for r, e in zip(test, el[k])]) for k, c in C3}
S = {n: {k: np.array([p[c + '_PROB'] for p in v]) for k, c in C3} for n, v in P.items()}
assert all([p['ID'] for p in v] == [r['ID'] for r in test] for v in P.values())


def ads(n, idx):
    e = {k: eer(y[k][idx][el[k][idx]], S[n][k][idx][el[k][idx]])[0] for k, _ in C3}
    return np.array([.5 * (1 - e['file']) + .2 * (1 - e['voice']) + .3 * (1 - e['music']), e['file'], e['voice'], e['music']])


ug = np.unique(groups)
members = {g: np.flatnonzero(groups == g) for g in ug}
rng = np.random.default_rng(0)
point = {n: ads(n, np.arange(len(test))) for n in P}
draws = defaultdict(list)
for _ in range(1000):
    idx = np.concatenate([members[g] for g in rng.choice(ug, len(ug))])
    a = {n: ads(n, idx) for n in P}
    for b, bn in BASES.items():
        for n in P:
            if n != bn:
                draws[(b, n)].append(a[n] - a[bn])
out = {'n_rows': len(test), 'n_groups': len(ug),
       'point': {name[n]: dict(zip(('ads', 'file_eer', 'voice_eer', 'music_eer'), v.round(5).tolist())) for n, v in point.items()},
       'delta': {}}
for (b, n), d in draws.items():
    d = np.array(d)
    out['delta'][f'{name[n]} - {b}'] = {
        'ads_delta': round(float(point[n][0] - point[BASES[b]][0]), 5), 'ads_ci95': np.percentile(d[:, 0], [2.5, 97.5]).round(5).tolist(),
        'p_ads_le_0': float((d[:, 0] <= 0).mean()),
        **{f'{k}_eer_delta': round(float(point[n][i] - point[BASES[b]][i]), 5) for i, k in ((1, 'file'), (2, 'voice'), (3, 'music'))},
        **{f'{k}_ci95': np.percentile(d[:, i], [2.5, 97.5]).round(5).tolist() for i, k in ((1, 'file'), (2, 'voice'), (3, 'music'))}}
lock = json.loads((HERE / 'choice_lock.json').read_text())['choice']
per = {}
src = {x['ID']: x['source'] for x in test}
for g, ref in (('fullvoice_db', 'mlp256'), ('voice_ad', 'resmlp')):
    for v in sorted({ref, lock[g]}):
        r = json.loads((HERE / 'runs' / f'{g}__{v}.json').read_text())
        keep = [i for i, x in enumerate(r['test_ids']) if x in frozen]
        t, s = np.array(r['test_labels'])[keep], np.mean(r['test_preds'], 0)[keep]
        e, thr = eer(t, s)
        cells = defaultdict(lambda: [0, 0])
        for i, lbl, sc in zip(np.array(r['test_ids'])[keep], t, s):
            c = cells[(src[i], int(lbl))]
            c[0] += 1
            c[1] += int((sc >= thr) != bool(lbl))
        per[f'{g}__{v}'] = {'eer_frozen_rows': round(e, 5), 'cells': {f'{k[0]}|{"fake" if k[1] else "real"}': [n, round(m / n, 4)]
                                                                     for k, (n, m) in sorted(cells.items()) if n >= 30}}
out['voice_route_error_by_source'] = per
(HERE / 'boot.json').write_text(json.dumps(out, indent=1))
print(json.dumps({k: out[k] for k in ('n_rows', 'n_groups', 'point', 'delta')}, indent=1))
for n, v in per.items():
    print(n, v['eer_frozen_rows'], v['cells'])
