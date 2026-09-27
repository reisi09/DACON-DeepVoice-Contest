"""Skeptic check (read-only, held-out preds already computed by ads.py test): is the S4+SA held-out gain one source?
Question: does RT held-out dADS(S4+SA - S4) survive (a) dropping each source, (b) a source-cluster bootstrap?"""
import csv
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from ads import Boot, eer, lab, load_rows  # noqa: E402

rows = load_rows()['test']
P = json.loads((HERE / 'test_preds.json').read_text())
base, sa = P['S4|RT'], P['S4+SA|RT']
assert [p['ID'] for p in base] == [r['ID'] for r in rows] == [p['ID'] for p in sa]
B = Boot(rows, reps=1)
Sb, Ss = B.vec(base), B.vec(sa)
src = np.array([r['source'] for r in rows])
grp = np.array([r['group_id'] for r in rows])


def d(idx):
    return B.ads(Ss, idx) - B.ads(Sb, idx)


def veer(S, idx):
    m = B.el['voice'][idx]
    return eer(B.y['voice'][idx][m], S['voice'][idx][m])[0]


out = {'all': {'n': len(rows), 'dADS': round(d(np.arange(len(rows))), 5)}}
vm = B.el['voice']
out['sources'] = {s: {'rows': int((src == s).sum()), 'groups': len(set(grp[src == s])),
                      'voice_rows_fake': int(((src == s) & vm & (B.y['voice'] == 1)).sum()),
                      'voice_rows_real': int(((src == s) & vm & (B.y['voice'] == 0)).sum())}
                  for s, _ in Counter(src).most_common()}
out['drop_one_source'] = {}
for s in out['sources']:
    idx = np.flatnonzero(src != s)
    out['drop_one_source'][s] = {'dADS': round(d(idx), 5), 'dVoiceEER': round(veer(Ss, idx) - veer(Sb, idx), 5)}
# source-cluster bootstrap: resample whole sources (the held-out split is source-held-out, so sources are the unit)
us = np.array(list(out['sources']))
mem = {s: np.flatnonzero(src == s) for s in us}
rng = np.random.default_rng(0)
dr = []
for _ in range(1000):
    idx = np.concatenate([mem[s] for s in rng.choice(us, len(us))])
    dr.append(d(idx))
dr = np.array(dr)
ok = np.isfinite(dr)
out['source_bootstrap'] = {'n_sources': len(us), 'n_valid_reps': int(ok.sum()), 'ci95': np.percentile(dr[ok], [2.5, 97.5]).round(5).tolist(),
                           'p_le_0': float((dr[ok] <= 0).mean())}
(HERE / 'critique_check.json').write_text(json.dumps(out, indent=1))
print(json.dumps(out, indent=1))
