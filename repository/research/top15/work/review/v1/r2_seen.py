"""Q: is R2's held-out ADS gain (MLP voice heads, screen semantics, r2/ads_test_preds.json) concentrated on held-out rows
whose source/parents were in AntiDeepfake/DF-Arena training data (fusion_fit.member_seen), like S3's AD voice fusion?
Also per-component EER deltas on seen/unseen. Group bootstrap (group_id, 1000 reps) as eval_semantics.py.
  python r2_seen.py"""
import csv, json, sys
from pathlib import Path
import numpy as np
ROOT = Path('/Users/choiwuseck/Documents/ChatGPT/데이콘')
sys.path.insert(0, str(ROOT / 'data_v4')); sys.path.insert(0, str(ROOT))
import train_backbone_heads as H
from fusion_fit import member_seen

HERE = Path(__file__).resolve().parent
P = json.loads((ROOT / 'research/top15/work/r2/ads_test_preds.json').read_text())
rows = {r['ID']: r for r in csv.DictReader((H.BANK / 'bank_test.csv').open())}
lab = lambda r, c: -1 if r[c] == '' else int(r[c])


def eer(y, s, w):
    o = np.argsort(-s, kind='stable'); y, s, w = y[o], s[o], w[o]
    tp, fp = np.cumsum(w * y), np.cumsum(w * (1 - y))
    last = np.r_[s[1:] != s[:-1], True]
    tp, fp = tp[last], fp[last]
    fpr, tpr = np.r_[0., fp / fp[-1]], np.r_[0., tp / tp[-1]]
    k = int(np.argmin(np.abs(fpr - (1 - tpr))))
    return (fpr[k] + 1 - tpr[k]) / 2


def arrays(rs, pr):
    out = {}
    for n, t, c, m in (('file', 'FILE_FAKE', 'FILE_FAKE_PROB', None), ('voice', 'VOICE_FAKE', 'VOICE_FAKE_PROB', 'VOICE_PRESENT'),
                       ('music', 'MUSIC_FAKE', 'MUSIC_FAKE_PROB', 'MUSIC_PRESENT')):
        idx = [k for k, r in enumerate(rs) if lab(r, t) >= 0 and (m is None or lab(r, m) == 1)]
        out[n] = (np.array(idx), np.array([lab(rs[k], t) for k in idx], float), np.array([pr[k][c] for k in idx], float))
    return out


def comps(a, w):
    e = {n: eer(y, s, w[i]) for n, (i, y, s) in a.items()}
    return e | {'ads': .5 * (1 - e['file']) + .2 * (1 - e['voice']) + .3 * (1 - e['music'])}


base = 'voice_df|voice_ad|music_6_12'
ids = [p['ID'] for p in P[base]]
rng = np.random.default_rng(0)
rep = {}
for sub in ('all', 'unseen', 'seen'):
    keep = [k for k, i in enumerate(ids) if sub == 'all' or member_seen(rows[i]) == (sub == 'seen')]
    rs = [rows[ids[k]] for k in keep]
    A = {n: arrays(rs, [P[n][k] for k in keep]) for n in P}
    assert all([P[n][k]['ID'] for k in keep] == [r['ID'] for r in rs] for n in P)
    gid = np.unique([r['group_id'] for r in rs], return_inverse=True)[1]
    ng = gid.max() + 1
    one = np.ones(len(rs))
    pt = {n: comps(A[n], one) for n in P}
    boot = {n: [] for n in P if n != base}
    for _ in range(1000):
        w = np.bincount(rng.integers(0, ng, ng), minlength=ng)[gid].astype(float)
        b = comps(A[base], w)['ads']
        for n in boot:
            boot[n].append(comps(A[n], w)['ads'] - b)
    rep[sub] = {'n': len(rs), 'groups': int(ng), 'point': {n: {k: round(v, 5) for k, v in pt[n].items()} for n in P},
                'delta_vs_S3like': {n: {'ads': round(pt[n]['ads'] - pt[base]['ads'], 5),
                                        'voice_eer': round(pt[n]['voice'] - pt[base]['voice'], 5),
                                        'file_eer': round(pt[n]['file'] - pt[base]['file'], 5),
                                        'ci95': [round(float(np.quantile(d, q)), 5) for q in (.025, .975)],
                                        'p_le_0': round(float((np.array(d) <= 0).mean()), 3)} for n, d in boot.items()}}
    print(sub, rep[sub]['n'], rep[sub]['groups'], flush=True)
    for n, v in rep[sub]['delta_vs_S3like'].items():
        print('  ', n, v, flush=True)
(HERE / 'r2_seen.json').write_text(json.dumps(rep, indent=1))
