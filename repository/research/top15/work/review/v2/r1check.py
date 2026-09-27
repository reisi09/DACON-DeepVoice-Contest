"""Q6: is R1's retrained-pooling gain distinguishable from seed noise and held-out sampling noise?
Uses R1's own CPU-retrained window logits (r1/retrain_*/windows_*.npz, seeds 41/42/43) and R1's scoring/FILE-refit
(agg.score, agg.fit on val, agg.file_vec); held-out = all 6,450 test rows (DB protocol, cache_v4).
 (a) 3-seed ensemble ADS per config (must reproduce r1/grid.json)
 (b) single-seed ADS per config (voice seed j with music seed j), paired deltas vs control
 (c) test-group bootstrap (fits fixed) of the ensemble ADS delta vs control: CI, P(delta>0)
Read-only on project files and r1/; writes r1check.json here."""
import sys
sys.dont_write_bytecode = True
import json, time
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parents[1] / 'r1')]
import core
import agg, eval_grid as G

t0 = time.time()
b, _, _ = agg.load()
m = agg.masks(b)
val, test = (b.split == 'val').values, (b.split == 'test').values
fk = m['file'].values
V = {p: G.seeds('voice', p) for p in ('max', 'lme')}
M = {p: G.seeds('music', p) for p in ('lme', 'mean')}
y = {k: b[k.upper() + '_FAKE'].where(m[k], '-1').astype(int).values for k in m}
el = {k: m[k].values for k in m}
pv, pm = b.VOICE_PRESENT_PROB.values, b.MUSIC_PRESENT_PROB.values
CFG = {'control': ('max', 'lme'), 'max|mean': ('max', 'mean'), 'lme|lme': ('lme', 'lme'), 'lme|mean': ('lme', 'mean')}
GRID = {'control': .86348, 'max|mean': .86743, 'lme|lme': .86442, 'lme|mean': .86837}


def sel(zs, j):
    return zs if j is None else [z[:, [j]] for z in zs]


def scores(cfg, j=None):
    vp, mp = CFG[cfg]
    fv, fm = agg.score(sel(V[vp], j), sel(M[mp], j), b.silent.values, G.VA[vp], mp)
    k = val & fk
    ens, _ = agg.fit(agg.files(fv[k], fm[k], b[k]), b.FILE_FAKE[k].astype(int).values)
    return {'file': agg.file_vec(ens, fv, fm, pv, pm), 'voice': fv, 'music': fm}


def ads(sc, idx):
    e = {k: core.eer(y[k][idx][el[k][idx]], sc[k][idx][el[k][idx]]) for k in el}
    return .5 * (1 - e['file']) + .2 * (1 - e['voice']) + .3 * (1 - e['music']), e


ti = np.flatnonzero(test)
out = {'n_test': len(ti)}
SC = {c: scores(c) for c in CFG}
out['a_ensemble_test_ads'] = {c: round(ads(SC[c], ti)[0], 5) for c in CFG}
for c in CFG:
    assert abs(out['a_ensemble_test_ads'][c] - GRID[c]) < 6e-6, (c, out['a_ensemble_test_ads'][c])
print('ensemble reproduces grid.json', out['a_ensemble_test_ads'], f'{time.time() - t0:.0f}s', flush=True)
single = {}
for j in range(3):
    s = {c: ads(scores(c, j), ti) for c in CFG}
    single[41 + j] = {'ads': {c: round(v[0], 5) for c, v in s.items()},
                      **{f'{c}-control': round(s[c][0] - s['control'][0], 5) for c in CFG if c != 'control'},
                      'music_eer_mean_minus_lme': round(s['max|mean'][1]['music'] - s['control'][1]['music'], 5),
                      'voice_eer_lme_minus_max': round(s['lme|lme'][1]['voice'] - s['control'][1]['voice'], 5)}
    print('seed', 41 + j, single[41 + j], flush=True)
out['b_single_seed'] = single
out['b_summary'] = {k: {'mean': round(float(np.mean(v)), 5), 'sd': round(float(np.std(v, ddof=1)), 5), 'n_gt0': int(sum(x > 0 for x in v))}
                    for k in single[41] if k != 'ads' for v in [[single[s][k] for s in single]]}
tg = core.group_index(b.group_id.values, ti)
rng = np.random.default_rng(20260927)
d = {c: [] for c in CFG if c != 'control'}
for _ in range(1000):
    idx = core.resample(tg, rng)
    a0 = ads(SC['control'], idx)[0]
    for c in d:
        d[c].append(ads(SC[c], idx)[0] - a0)
pt = out['a_ensemble_test_ads']
out['c_test_group_bootstrap'] = {c: core.summ(v, pt[c] - pt['control']) for c, v in d.items()}
out['test_groups'] = len(tg)
out['seconds'] = round(time.time() - t0)
(HERE / 'r1check.json').write_text(json.dumps(out, indent=1, default=float))
for c, v in out['c_test_group_bootstrap'].items():
    print('boot', c, v, flush=True)
print('summary', out['b_summary'], 'seconds', out['seconds'])
