"""Q: does R1's retrained pooling gain (voice LME + music mean vs control voice max + music LME, DB protocol, 3 seeds,
runtime semantics, r1/eval_grid.py) hold on held-out rows outside AntiDeepfake/DF-Arena/SONICS training sources
(fusion_fit.member_seen)? Same R1 scores and val-refit FILE ensemble; ADS group bootstrap per subset.
  python r1_seen.py"""
import json, sys
from pathlib import Path
import numpy as np
ROOT = Path('/Users/choiwuseck/Documents/ChatGPT/데이콘')
R1 = ROOT / 'research/top15/work/r1'
sys.path.insert(0, str(R1)); sys.path.insert(0, str(ROOT / 'data_v4'))
import agg, eval_grid as G
from fusion_fit import member_seen

b, _, _ = agg.load()
m = agg.masks(b)
val, test = (b.split == 'val').values, (b.split == 'test').values
fk = m['file'].values
seen = np.array([member_seen(r) for r in b.to_dict('records')])
V = {p: G.seeds('voice', p) for p in ('max', 'lme')}
M = {p: G.seeds('music', p) for p in ('lme', 'mean')}
S = {}
for name, vp, mp in (('control', 'max', 'lme'), ('voice_lme|music_mean', 'lme', 'mean'), ('voice_lme|music_lme', 'lme', 'lme'),
                     ('voice_max|music_mean', 'max', 'mean')):
    fv, fm = agg.score(V[vp], M[mp], b.silent.values, G.VA[vp], mp)
    ens, _ = agg.fit(agg.files(fv[val & fk], fm[val & fk], b[val & fk]), b.FILE_FAKE[val & fk].astype(int).values)
    S[name] = {'file': agg.file_vec(ens, fv, fm, b.VOICE_PRESENT_PROB.values, b.MUSIC_PRESENT_PROB.values), 'voice': fv, 'music': fm}


def ads(sc, sel, w=None):
    e = {}
    for k in m:
        mk = sel & m[k].values
        y, s = b.loc[mk, k.upper() + '_FAKE'].astype(int).values, sc[k][mk]
        if w is None:
            e[k] = agg.eer(y, s)
        else:   # weighted EER via replication
            ww = w[mk].astype(int)
            e[k] = agg.eer(np.repeat(y, ww), np.repeat(s, ww))
    return .5 * (1 - e['file']) + .2 * (1 - e['voice']) + .3 * (1 - e['music']), e


rng = np.random.default_rng(0)
rep = {}
for sub, sel in (('all', test), ('unseen', test & ~seen), ('seen', test & seen), ('val', val)):
    gid = np.unique(b.group_id.values, return_inverse=True)[1]
    gs = np.unique(gid[sel])
    base, be = ads(S['control'], sel)
    res = {'n': int(sel.sum()), 'groups': len(gs), 'control': {'ads': round(base, 5)} | {k: round(v, 5) for k, v in be.items()}}
    for name in S:
        if name == 'control':
            continue
        a, e = ads(S[name], sel)
        d = []
        for _ in range(300 if sub != 'val' else 0):
            cnt = np.zeros(gid.max() + 1)
            np.add.at(cnt, rng.choice(gs, len(gs)), 1)
            w = cnt[gid]
            d.append(ads(S[name], sel & (w > 0), w)[0] - ads(S['control'], sel & (w > 0), w)[0])
        res[name] = {'ads': round(a, 5), 'delta': round(a - base, 5)} | {k: round(v, 5) for k, v in e.items()}
        if d:
            res[name] |= {'ci95': [round(float(np.quantile(d, q)), 5) for q in (.025, .975)], 'p_le_0': round(float((np.array(d) <= 0).mean()), 3)}
    rep[sub] = res
    print(sub, json.dumps(res), flush=True)
(Path(__file__).resolve().parent / 'r1_seen.json').write_text(json.dumps(rep, indent=1))
