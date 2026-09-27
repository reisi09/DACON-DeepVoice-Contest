"""R1 (a)+(b): window-aggregation pilot on DB heads, from windows.npz (CPU, read-only on project files).
  python agg.py a   -> a.json   true held-out files: route EER, FILE EER (val-refit ensemble), ADS, length correlation
  python agg.py b   -> b.json   synthetic long files (same source+label, group-disjoint partners), fixed k=1 threshold
"""
import json, sys
from pathlib import Path
import numpy as np, pandas as pd
from scipy.special import expit, logsumexp
from scipy.stats import spearmanr

ROOT = Path('/Users/choiwuseck/Documents/ChatGPT/데이콘')
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / 'data_v4'))
import decision_heads as dh  # noqa: E402
from file_ensemble import fit, eer  # noqa: E402  (DACON EER formula + val grid/equal-weight rule)

OUT = Path(__file__).parent
BANK = ROOT / 'results/data_v4_20260923/data_v4_results_20260923T1639Z/data/bank'
DB = ROOT / '_keep_best_20260926/train_duration_balanced'
VMAX, MMAX = 15, 29   # windows in a 60 s file (voice 64,600 non-overlap + tail; music hop 32,300 + tail)


def topk_mean(z, k):  # z [n, 3] logits -> mean prob of top-k windows per seed
    return expit(np.sort(z, 0)[-min(k, len(z)):]).mean(0)


VAGG = {'max': lambda z: expit(z.max(0)),
        'top3': lambda z: topk_mean(z, 3),
        'top30pct': lambda z: topk_mean(z, int(np.ceil(.3 * len(z)))),
        'mean_logit': lambda z: expit(z.mean(0)),
        'lme': lambda z: expit(logsumexp(z, 0) - np.log(len(z)))}
MAGG = {'lme': lambda z: expit(logsumexp(z, 0) - np.log(len(z))),
        'max': lambda z: expit(z.max(0)),
        'mean': lambda z: expit(z.mean(0))}


def load():
    w = dict(np.load(OUT / 'windows.npz'))   # materialise once: NpzFile re-reads the array on every key access
    vz = np.split(w['vz'], w['vo'][1:-1])
    mz = np.split(w['mz'], w['mo'][1:-1])
    ids = w['ids']
    bank = pd.concat([pd.read_csv(BANK / f, dtype=str, keep_default_na=False) for f in ('bank_trainval.csv', 'bank_test.csv')])
    bank = bank.set_index('ID').loc[ids].reset_index()
    dur = {}
    for c in ('cache_tv', 'cache_test'):
        dur.update({k: v['samples'] / 16000 for k, v in json.loads((ROOT / '_workspace/cache_v4' / c / 'index.json').read_text())['items'].items()})
    bank['dur'] = bank.ID.map(dur)
    bank['silent'] = w['silent']
    pred = pd.concat([pd.read_csv(DB / f'{s}_predictions_decision.csv') for s in ('val', 'test')]).set_index('ID')
    bank = bank.join(pred, on='ID')
    return bank, vz, mz


def score(vz, mz, silent, va, ma):
    fv = np.array([0. if s else VAGG[va](z).mean() for z, s in zip(vz, silent)])
    fm = np.array([MAGG[ma](z).mean() for z in mz])
    return fv, fm


def masks(b):
    k = lambda c: (b[c + '_mask'] == '1') & b[c].isin(['0', '1'])
    fk = k('FILE_FAKE')
    return {'file': fk,
            'voice': fk & k('VOICE_PRESENT') & (b.VOICE_PRESENT == '1') & k('VOICE_FAKE'),
            'music': fk & k('MUSIC_PRESENT') & (b.MUSIC_PRESENT == '1') & k('MUSIC_FAKE')}


def files(fv, fm, b):
    return [dict(Pv=p, Pm=q, Fv=v, Fm=m) for p, q, v, m in zip(b.VOICE_PRESENT_PROB, b.MUSIC_PRESENT_PROB, fv, fm)]


def file_scores(ens, fv, fm, b):
    return np.array([dh.file_score(ens, f) for f in files(fv, fm, b)])


def within_source_rho(x, s, d, min_n=20):
    """n-weighted mean over sources of Spearman(score, duration); pooled sources with >=min_n rows."""
    num = den = 0.
    for src in np.unique(s):
        m = s == src
        if m.sum() >= min_n and len(np.unique(d[m])) > 5 and len(np.unique(x[m])) > 5:
            num += m.sum() * spearmanr(x[m], d[m]).statistic; den += m.sum()
    return round(num / den, 3) if den else None


def eer_thr(y, s):
    from sklearn.metrics import roc_curve
    fpr, tpr, thr = roc_curve(y, s, drop_intermediate=False)
    i = int(np.argmin(np.abs(fpr - (1 - tpr))))
    return float(thr[i]), float((fpr[i] + 1 - tpr[i]) / 2)


def boot_delta(y, s0, s1, g, reps=1000, seed=0):
    """Group-cluster bootstrap of EER(s1) - EER(s0)."""
    rng = np.random.default_rng(seed)
    idx = pd.Series(np.arange(len(g))).groupby(g).indices
    keys = list(idx)
    out = []
    for _ in range(reps):
        j = np.concatenate([idx[k] for k in rng.choice(keys, len(keys))])
        if len(np.unique(y[j])) == 2:
            out.append(eer(y[j], s1[j]) - eer(y[j], s0[j]))
    out = np.array(out)
    return {'ci95': [round(float(q), 5) for q in np.percentile(out, [2.5, 97.5])], 'p_le_0': round(float((out <= 0).mean()), 3)}


def part_a():
    b, vz, mz = load()
    m = masks(b)
    val, test = (b.split == 'val').values, (b.split == 'test').values
    res = {'n': {s: {k: int((v & (b.split == s)).sum()) for k, v in m.items()} for s in ('val', 'test')}}
    # reproduction of the DB decision CSV with the DB's own fitted file_ensemble.json
    ens_db = json.loads((DB / 'file_ensemble.json').read_text())
    fv, fm = score(vz, mz, b.silent.values, 'max', 'lme')
    fk = m['file'].values
    res['repro'] = {'max_abs_dFv': float(np.nanmax(np.abs(fv[fk] - b.VOICE_FAKE_PROB.values[fk]))),
                    'max_abs_dFm': float(np.nanmax(np.abs(fm[fk] - b.MUSIC_FAKE_PROB.values[fk]))),
                    'max_abs_dFILE': float(np.nanmax(np.abs(file_scores(ens_db, fv, fm, b)[fk] - b.FILE_FAKE_PROB.values[fk])))}
    for s, sm in (('val', val), ('test', test)):
        f_db = file_scores(ens_db, fv, fm, b)
        res['repro'][s] = {r: round(eer(b.loc[sm & m[r], r.upper() + '_FAKE'].astype(int).values,
                                         {'file': f_db, 'voice': fv, 'music': fm}[r][sm & m[r].values]), 5) for r in m}
    combos = [(v, 'lme') for v in VAGG] + [('max', a) for a in MAGG if a != 'lme']
    base = None
    for va, ma in combos:
        fv, fm = score(vz, mz, b.silent.values, va, ma)
        vm = val & fk
        ens, rep = fit(files(fv[vm], fm[vm], b[vm]), b.FILE_FAKE[vm].astype(int).values)
        ff = file_scores(ens, fv, fm, b)
        sc = {'file': ff, 'voice': fv, 'music': fm}
        r = {'file_ensemble': {k: rep[k] for k in ('chosen_weights', 'equal_weights_eer', 'grid_best_eer')}}
        for s, sm in (('val', val), ('test', test)):
            e = {k: round(eer(b.loc[sm & m[k], k.upper() + '_FAKE'].astype(int).values, sc[k][sm & m[k].values]), 5) for k in m}
            e['ADS'] = round(.5 * (1 - e['file']) + .2 * (1 - e['voice']) + .3 * (1 - e['music']), 5)
            r[s] = e
        # length dependence among REAL files on held-out test (within source, n-weighted)
        for k, lab in (('voice', 'VOICE_FAKE'), ('file', 'FILE_FAKE'), ('music', 'MUSIC_FAKE')):
            mm = (test & m[k].values)
            for cls in ('0', '1'):
                c = mm & (b[lab] == cls).values
                r.setdefault('len_rho_within_source_test', {})[f'{k}_{"real" if cls == "0" else "fake"}'] = \
                    within_source_rho(sc[k][c], b.source.values[c], b.dur.values[c])
        # FPR on reals / FNR on fakes by duration bin at the test EER threshold (FILE and voice)
        for k, lab in (('file', 'FILE_FAKE'), ('voice', 'VOICE_FAKE')):
            mm = test & m[k].values
            y, s_ = b[lab].values[mm].astype(int), sc[k][mm]
            thr, _ = eer_thr(y, s_)
            bins = pd.cut(b.dur.values[mm], [0, 4, 8, 15, 30, 61], right=False, labels=['<4', '4-8', '8-15', '15-30', '30-60'])
            t = pd.DataFrame({'y': y, 'hit': s_ >= thr, 'bin': bins})
            r.setdefault('err_by_duration_test', {})[k] = {
                'fpr_real': {str(kk): [round(float(g.hit.mean()), 3), len(g)] for kk, g in t[t.y == 0].groupby('bin', observed=True)},
                'fnr_fake': {str(kk): [round(float(1 - g.hit.mean()), 3), len(g)] for kk, g in t[t.y == 1].groupby('bin', observed=True)}}
        if base is None:
            base = sc
        else:  # group bootstrap of delta vs current (max/lme) on the held-out test
            for k, lab in (('voice', 'VOICE_FAKE'), ('music', 'MUSIC_FAKE'), ('file', 'FILE_FAKE')):
                mm = test & m[k].values
                if np.allclose(sc[k][mm], base[k][mm]):
                    continue
                r.setdefault('boot_delta_test_vs_current', {})[k] = boot_delta(b[lab].values[mm].astype(int), base[k][mm], sc[k][mm], b.group_id.values[mm])
        res[f'voice={va}|music={ma}'] = r
        print(va, ma, r['val'], r['test'], r['len_rho_within_source_test'], flush=True)
    # test window-count / duration profile
    res['test_duration_q'] = {q: round(float(b.dur[test].quantile(q)), 1) for q in (.1, .25, .5, .75, .9, .99)}
    res['test_voice_windows_q'] = {q: float(np.quantile([len(z) for z, t in zip(vz, test) if t], q)) for q in (.5, .9, .99)}
    json.dump(res, open(OUT / 'a.json', 'w'), indent=1)
    print(json.dumps(res['repro'], indent=1), res['test_duration_q'], res['test_voice_windows_q'])


if __name__ == '__main__':
    if sys.argv[1] == 'a': part_a()


def file_vec(ens, fv, fm, pv, pm):
    """Vectorised decision_heads.file_score (same candidates, tables, weights)."""
    cand = {'max': np.maximum(pv * fv, pm * fm), 'noisy_or': 1 - (1 - pv * fv) * (1 - pm * fm), 'max_nopresence': np.maximum(fv, fm)}
    lv = lambda t: np.linspace(0, 1, len(t))
    num = sum(w * np.interp(cand[n], ens['tables'][n], lv(ens['tables'][n])) for n, w in ens['weights'].items() if w)
    return num / sum(w for w in ens['weights'].values() if w)


def partners(cell_idx, groups, k, rng):
    """For each anchor in a cell: k-1 partners from distinct groups other than the anchor's (fallback: any other file)."""
    out, fallback = {}, 0
    by_g = list(pd.Series(cell_idx).groupby(groups[cell_idx]).apply(list).items())
    gpos = {gid: j for j, (gid, _) in enumerate(by_g)}
    for a in cell_idx:
        ga = gpos[groups[a]]
        pick = [j for j in rng.choice(len(by_g), min(k, len(by_g)), replace=False) if j != ga][:k - 1]
        p = [by_g[j][1][rng.integers(len(by_g[j][1]))] for j in pick]
        if len(p) < k - 1:   # cell has too few groups: fill with other files (group reuse)
            pool = np.setdiff1d(cell_idx, [a] + p)
            p += list(rng.permutation(pool)[:k - 1 - len(p)])
            fallback += 1
        out[a] = [a] + p
    return out, fallback


_SETS = {}


def synth_sets(route, cells, g, k, r):
    """Partner sets shared by all aggregators of one route (paired comparison)."""
    key = (route, k, r)
    if key not in _SETS:
        rng = np.random.default_rng(1000 * r + k)
        sets, fb = {}, 0
        for idx in cells.values():
            s_, f_ = partners(np.array(idx), g, k, rng); sets.update(s_); fb += f_
        _SETS[key] = (sets, fb)
    return _SETS[key]


def part_b(R=3, K=8, vz=None, tasks=None, out_name='b.json', mz=None, magg='lme', keep=None):
    b, vz0, mz0 = load()
    vz = vz0 if vz is None else vz
    mz = mz0 if mz is None else mz
    m = masks(b)
    test, val = (b.split == 'test').values, (b.split == 'val').values
    g = b.group_id.values; src = b.source.values; sil = b.silent.values
    pv, pm = b.VOICE_PRESENT_PROB.values, b.MUSIC_PRESENT_PROB.values
    fk = m['file'].values
    ens = {}
    for va in (VAGG if tasks is None or any(t[0] == 'file' for t in tasks) else ()):   # val refit, identical to part a
        fv, fm = score(vz, mz, sil, va, magg)
        ens[va] = fit(files(fv[val & fk], fm[val & fk], b[val & fk]), b.FILE_FAKE[val & fk].astype(int).values)[0]
    tasks = tasks or [('voice', 'VOICE_FAKE', a) for a in VAGG] + [('music', 'MUSIC_FAKE', a) for a in MAGG] + [('file', 'FILE_FAKE', a) for a in VAGG]
    res = {}
    for route, lab, agg in tasks:
        mm = np.where(test & m[route].values & (True if keep is None else keep))[0]
        y = b[lab].values.astype(object)
        cells = pd.Series(mm).groupby([src[mm], y[mm]]).apply(list).to_dict()
        per_k = {}
        for r in range(R):
            rng = np.random.default_rng(100 + r)
            scores = {}
            for k in range(1, K + 1):
                sets, fb = synth_sets(route, cells, g, k, r)
                ids = mm
                def cat(zs, cap, members): return np.concatenate([zs[j] for j in members])[:cap]
                if route == 'music':
                    s = np.array([MAGG[agg](cat(mz, MMAX, sets[i])).mean() for i in ids]); nw = [min(MMAX, sum(len(mz[j]) for j in sets[i])) for i in ids]
                else:
                    fv = np.array([0. if sil[i] else VAGG[agg](cat(vz, VMAX, sets[i])).mean() for i in ids])
                    nw = [min(VMAX, sum(len(vz[j]) for j in sets[i])) for i in ids]
                    if route == 'voice':
                        s = fv
                    else:
                        fm = np.array([MAGG[magg](cat(mz, MMAX, sets[i])).mean() for i in ids])
                        s = file_vec(ens[agg], fv, fm, np.array([pv[sets[i]].max() for i in ids]), np.array([pm[sets[i]].max() for i in ids]))
                scores[k] = (s, np.array(nw), fb)
            yy = b[lab].values[mm].astype(int)
            thr, e1 = eer_thr(yy, scores[1][0])
            kk = rng.integers(1, K + 1, len(mm))
            mixed = np.array([scores[k_][0][j] for j, k_ in enumerate(kk)])
            for k in range(1, K + 1):
                s, nw, fb = scores[k]
                d = per_k.setdefault(k, {'fpr_real': [], 'fnr_fake': [], 'eer_k': [], 'mean_windows': [], 'cells_group_fallback': []})
                d['fpr_real'].append(float((s[yy == 0] >= thr).mean())); d['fnr_fake'].append(float((s[yy == 1] < thr).mean()))
                d['eer_k'].append(eer(yy, s)); d['mean_windows'].append(float(nw.mean())); d['cells_group_fallback'].append(fb / max(1, len(mm)))
            per_k.setdefault('mixed_k_uniform_eer', []).append(eer(yy, mixed))
            adverse = np.r_[scores[K][0][yy == 0], scores[1][0][yy == 1]]; ya = np.r_[np.zeros((yy == 0).sum()), np.ones((yy == 1).sum())]
            favour = np.r_[scores[1][0][yy == 0], scores[K][0][yy == 1]]
            per_k.setdefault('eer_real_k8_fake_k1', []).append(eer(ya, adverse)); per_k.setdefault('eer_real_k1_fake_k8', []).append(eer(ya, favour))
        out = {str(k): {n: round(float(np.mean(v)), 4) for n, v in d.items()} for k, d in per_k.items() if isinstance(k, int)}
        out.update({n: round(float(np.mean(per_k[n])), 5) for n in ('mixed_k_uniform_eer', 'eer_real_k8_fake_k1', 'eer_real_k1_fake_k8')})
        out['n'] = [int((b[lab].values[mm] == '0').sum()), int((b[lab].values[mm] == '1').sum())]
        res[f'{route}|{agg}'] = out
        print(route, agg, 'k1 eer', out['1']['eer_k'], 'mixed', out['mixed_k_uniform_eer'], 'r8f1', out['eer_real_k8_fake_k1'],
              'FPR k1..8', [out[str(k)]['fpr_real'] for k in range(1, K + 1)], 'FNR', [out[str(k)]['fnr_fake'] for k in range(1, K + 1)], flush=True)
    json.dump(res, open(OUT / out_name, 'w'), indent=1)
    return res


if __name__ == '__main__' and sys.argv[1] == 'b':
    part_b()
