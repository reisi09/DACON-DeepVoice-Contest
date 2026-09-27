"""Q: are the held-out gains of the val-chosen MLP voice heads robust to resampling independent groups, and which held-out
sources do they come from?  Group bootstrap (group_id, 1000 reps, models fixed) of official ADS / voice EER deltas vs the
S3-like baseline (ads_test_preds.json), plus per-source error rates at each model's own route-EER threshold."""
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
cache = ROOT / '_workspace/backbone_cache'
test = [r for r in csv.DictReader((H.BANK / 'bank_test.csv').open()) if (cache / f"{r['ID']}.npz").exists()]
P = json.loads((HERE / 'ads_test_preds.json').read_text())
BASE = 'voice_df|voice_ad|music_6_12'


def lab(r, c):
    return None if r[c + '_mask'] != '1' or r[c] == '' else int(r[c])


def eer(t, s):
    fpr, tpr, thr = roc_curve(t, s, drop_intermediate=False)
    i = np.argmin(np.abs(fpr - (1 - tpr)))
    return (fpr[i] + 1 - tpr[i]) / 2, thr[i]


groups = np.array([r['group_id'] for r in test])
el = {'file': np.array([lab(r, 'FILE_FAKE') is not None for r in test]),
      'voice': np.array([lab(r, 'VOICE_PRESENT') == 1 and lab(r, 'VOICE_FAKE') is not None for r in test]),
      'music': np.array([lab(r, 'MUSIC_PRESENT') == 1 and lab(r, 'MUSIC_FAKE') is not None for r in test])}
y = {k: np.array([lab(r, c) if e else -1 for r, e in zip(test, el[k])]) for k, c in (('file', 'FILE_FAKE'), ('voice', 'VOICE_FAKE'), ('music', 'MUSIC_FAKE'))}
S = {n: {k: np.array([p[c] for p in v]) for k, c in (('file', 'FILE_FAKE_PROB'), ('voice', 'VOICE_FAKE_PROB'), ('music', 'MUSIC_FAKE_PROB'))} for n, v in P.items()}
assert all([p['ID'] for p in v] == [r['ID'] for r in test] for v in P.values())


def ads(n, idx):
    e = {k: eer(y[k][idx][el[k][idx]], S[n][k][idx][el[k][idx]])[0] for k in ('file', 'voice', 'music')}
    return .5 * (1 - e['file']) + .2 * (1 - e['voice']) + .3 * (1 - e['music']), e['voice']


ug = np.unique(groups)
members = {g: np.flatnonzero(groups == g) for g in ug}
rng = np.random.default_rng(0)
point = {n: ads(n, np.arange(len(test))) for n in P}
draws = defaultdict(list)
for _ in range(1000):
    idx = np.concatenate([members[g] for g in rng.choice(ug, len(ug))])
    b = ads(BASE, idx)
    for n in P:
        if n != BASE:
            a = ads(n, idx)
            draws[n].append((a[0] - b[0], a[1] - b[1]))
out = {'n_groups': len(ug), 'point': {n: {'ads': round(v[0], 5), 'voice_eer': round(v[1], 5)} for n, v in point.items()}, 'delta_vs_base': {}}
for n, d in draws.items():
    d = np.array(d)
    out['delta_vs_base'][n] = {'ads_delta': round(point[n][0] - point[BASE][0], 5), 'ads_ci95': np.percentile(d[:, 0], [2.5, 97.5]).round(5).tolist(),
                               'p_ads_le_0': float((d[:, 0] <= 0).mean()), 'voice_eer_delta': round(point[n][1] - point[BASE][1], 5),
                               'voice_ci95': np.percentile(d[:, 1], [2.5, 97.5]).round(5).tolist()}
# per-source voice-route errors at each run's own EER threshold (3-seed mean route scores from the R2 run files)
per = {}
for name in ('voice_df__base', 'voice_df__mlp_long', 'voice_ad__base', 'voice_ad__resmlp_long'):
    r = json.loads((HERE / 'runs' / f'{name}.json').read_text())
    t, s = np.array(r['test_labels']), np.mean(r['test_preds'], 0)
    e, thr = eer(t, s)
    src = {x['ID']: x['source'] for x in test}
    cells = defaultdict(lambda: [0, 0])
    for i, lbl, sc in zip(r['test_ids'], t, s):
        c = cells[(src[i], int(lbl))]
        c[0] += 1
        c[1] += int((sc >= thr) != bool(lbl))
    per[name] = {'eer': round(e, 5), 'cells': {f'{k[0]}|{"fake" if k[1] else "real"}': [n, round(m / n, 4)] for k, (n, m) in sorted(cells.items()) if n >= 30}}
out['voice_route_error_by_source'] = per
(HERE / 'boot.json').write_text(json.dumps(out, indent=1))
print(json.dumps({k: out[k] for k in ('n_groups', 'point', 'delta_vs_base')}, indent=1))
for name, v in per.items():
    print(name, v['eer'], v['cells'])
