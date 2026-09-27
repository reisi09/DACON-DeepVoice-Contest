"""Question: the official EERs pool files of different types (speech-only, music-only, mixed) under ONE threshold.
Held-out within-type FILE EERs are 0.050/0.161/0.185 vs pooled 0.109 (composition_check.out.json). Can a per-file
recalibration conditioned on the RUNTIME presence type (Pv, Pm >= .5; LB CPS 0.9977 says this type is reliable on the
official test) and on log2(duration), fitted on val only, lower pooled FILE/VOICE/MUSIC EER on the source-held-out test?
Oracle (fit on test) gives the ceiling. 'llr' variants drop the per-type prior (composition-robust).
DB-head predictions (read-only). CPU, seconds."""
import json
import numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve

R = '/Users/choiwuseck/Documents/ChatGPT/데이콘/'
K = R + '_keep_best_20260926/train_duration_balanced/'
B = R + 'results/data_v4_20260923/data_v4_results_20260923T1639Z/data/bank/'
bank = pd.concat([pd.read_csv(B + 'bank_trainval.csv'), pd.read_csv(B + 'bank_test.csv')])
dur = {}
for c in ('cache_tv', 'cache_test'):
    for k, v in json.load(open(R + f'_workspace/cache_v4/{c}/index.json'))['items'].items():
        dur[k] = v['samples'] / 16000
S = {s: bank[bank.split == s].merge(pd.read_csv(K + f'{s}_predictions_decision.csv'), on='ID') for s in ('val', 'test')}
for d in S.values():
    d['dur'] = d.ID.map(dur)
    d['t'] = (d.VOICE_PRESENT_PROB >= .5).astype(int) * 2 + (d.MUSIC_PRESENT_PROB >= .5).astype(int)
    assert d.dur.notna().all()
M = {'file': ('FILE_FAKE', 'FILE_FAKE_PROB', lambda d: d.FILE_FAKE_mask == 1),
     'voice': ('VOICE_FAKE', 'VOICE_FAKE_PROB', lambda d: (d.VOICE_PRESENT == 1) & (d.VOICE_FAKE_mask == 1)),
     'music': ('MUSIC_FAKE', 'MUSIC_FAKE_PROB', lambda d: (d.MUSIC_PRESENT == 1) & (d.MUSIC_FAKE_mask == 1))}


def eer(y, s):
    fpr, tpr, _ = roc_curve(y, s, drop_intermediate=False)
    i = np.argmin(np.abs(fpr - (1 - tpr)))
    return (fpr[i] + 1 - tpr[i]) / 2


def lg(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def X(d, s, use_dur):
    return np.c_[lg(d[s]), np.log2(d.dur)] if use_dur else lg(d[s]).to_numpy()[:, None]


def calib(fit, app, y, s, by_type, use_dur, llr):
    out = np.zeros(len(app))
    for t in (sorted(app.t.unique()) if by_type else [None]):
        f = fit if t is None else fit[fit.t == t]
        a = np.ones(len(app), bool) if t is None else (app.t == t).to_numpy()
        if f[y].nunique() < 2 or len(f) < 30:  # too few to fit: keep raw logit
            out[a] = lg(app[s][a]); continue
        m = LogisticRegression(C=1e3, max_iter=1000).fit(X(f, s, use_dur), f[y])
        z = m.decision_function(X(app[a], s, use_dur))
        if llr:
            pr = f[y].mean(); z = z - np.log(pr / (1 - pr))
        out[a] = z
    return out


VAR = {'type_platt': (1, 0, 0), 'type_llr': (1, 0, 1), 'dur_only': (0, 1, 0), 'type_dur_llr': (1, 1, 1)}
rng = np.random.default_rng(0)
res = {}
for name, (y, s, mk) in M.items():
    v, t = S['val'][mk(S['val'])], S['test'][mk(S['test'])].reset_index(drop=True)
    sc = {'base': t[s].to_numpy()}
    for vn, (bt, ud, ll) in VAR.items():
        sc[vn] = calib(v, t, y, s, bt, ud, ll)
        sc['ORACLE_' + vn] = calib(t, t, y, s, bt, ud, ll)
    r = {k: round(float(eer(t[y], x)), 5) for k, x in sc.items()}
    g = t.group_id.to_numpy(); ug = np.unique(g); idx = {u: np.flatnonzero(g == u) for u in ug}
    boot = {k: [] for k in VAR}
    for _ in range(300):
        ii = np.concatenate([idx[u] for u in rng.choice(ug, len(ug))])
        yb = t[y].to_numpy()[ii]
        e0 = eer(yb, sc['base'][ii])
        for k in VAR:
            boot[k].append(e0 - eer(yb, sc[k][ii]))
    r['gain_ci95_vs_base'] = {k: [round(float(np.percentile(b, q)), 4) for q in (2.5, 97.5)] for k, b in boot.items()}
    r['n_test'] = len(t); r['type_counts_test'] = t.t.value_counts().to_dict()
    res[name] = r
w = {'file': .5, 'voice': .2, 'music': .3}
res['ADS'] = {k: round(sum(w[m] * (1 - res[m][k]) for m in w), 5) for k in res['file'] if isinstance(res['file'][k], float)}
print(json.dumps(res, indent=1, default=int))
