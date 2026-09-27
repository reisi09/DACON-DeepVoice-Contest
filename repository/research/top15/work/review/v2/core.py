"""Shared, vectorised re-implementation of the ads_compare.py evaluation (voice CDF fusion, FILE ensemble refit on val,
official EER/ADS) plus resampling helpers.  Every piece is checked against the project code in selfcheck()."""
import itertools, json, sys
sys.dont_write_bytecode = True
import numpy as np

LIN = np.linspace(0, 1, 1001)
NAMES = ('max', 'noisy_or', 'max_nopresence')          # dh.CANDIDATES order, 'prev' absent (no prev_file)
GRID = [dict(zip(NAMES, c)) for c in itertools.product(range(11), repeat=3) if sum(c) == 10]


def eer(y, s, w=None):
    """DACON EER == sklearn roc_curve(drop_intermediate=False), argmin|fpr-fnr| (first), (fpr+fnr)/2; optional weights."""
    o = np.argsort(s, kind='mergesort')[::-1]
    s, y = s[o], y[o]
    w = np.ones(len(s)) if w is None else w[o]
    tp, fp = np.cumsum(w * y), np.cumsum(w * (1 - y))
    last = np.r_[np.flatnonzero(np.diff(s)), len(s) - 1]
    tp, fp = np.r_[0, tp[last]], np.r_[0, fp[last]]
    fpr, fnr = fp / fp[-1], 1 - tp / tp[-1]
    i = int(np.argmin(np.abs(fpr - fnr)))
    return float((fpr[i] + fnr[i]) / 2)


def qtable(v):   # == decision_heads.quantile_table
    t = np.quantile(np.asarray(v, float), LIN)
    return np.maximum.accumulate(t) + np.arange(len(LIN)) * 1e-12


def cands(pv, pm, fv, fm):
    return {'max': np.maximum(pv * fv, pm * fm), 'noisy_or': 1 - (1 - pv * fv) * (1 - pm * fm), 'max_nopresence': np.maximum(fv, fm)}


def fit_file(pv, pm, fv, fm, y, min_gain=.002, force=None):
    """== file_ensemble.fit: 0.1-step simplex grid on val FILE EER, equal weights unless grid wins by >= min_gain.
    force: fixed weights (tables still fit on val) for rule-sensitivity checks."""
    c = cands(pv, pm, fv, fm)
    tab = {n: qtable(c[n]) for n in NAMES}
    if force:
        return {'weights': force, 'tables': tab}
    cdf = {n: np.interp(c[n], tab[n], LIN) for n in NAMES}
    sc = lambda w: eer(y, sum(w[n] * cdf[n] for n in NAMES) / sum(w.values()))
    best = min((sc(w), json.dumps(w, sort_keys=True)) for w in GRID)
    eq = {n: 1 for n in NAMES}
    return {'weights': json.loads(best[1]) if sc(eq) - best[0] >= min_gain else eq, 'tables': tab}


def file_vec(ens, pv, pm, fv, fm):
    c = cands(pv, pm, fv, fm)
    w = {n: x for n, x in ens['weights'].items() if x}
    return sum(x * np.interp(c[n], ens['tables'][n], LIN) for n, x in w.items()) / sum(w.values())


class Data:
    """rows = val rows then test rows; comp = {key: {ID: prob}}; P = {ID: (Pv, Pm)}."""
    def __init__(self, val, test, comp, P):
        self.rows = val + test
        self.nv = len(val)
        ids = [r['ID'] for r in self.rows]
        self.comp = {k: np.array([v.get(i, 0.) for i in ids]) for k, v in comp.items()}
        self.has = {k: np.array([i in v for i in ids]) for k, v in comp.items()}
        self.pv, self.pm = (np.array([P[i][j] for i in ids]) for j in (0, 1))
        lab = lambda r, c: -1 if r[c + '_mask'] != '1' or r[c] == '' else int(r[c])
        self.y = {k: np.array([lab(r, c) for r in self.rows]) for k, c in (('file', 'FILE_FAKE'), ('voice', 'VOICE_FAKE'), ('music', 'MUSIC_FAKE'))}
        self.el = {'file': self.y['file'] >= 0,
                   'voice': (np.array([lab(r, 'VOICE_PRESENT') for r in self.rows]) == 1) & (self.y['voice'] >= 0),
                   'music': (np.array([lab(r, 'MUSIC_PRESENT') for r in self.rows]) == 1) & (self.y['music'] >= 0)}
        self.fitmask = np.array([r['FILE_FAKE_mask'] == '1' for r in self.rows])
        self.group = np.array([r['group_id'] for r in self.rows])
        self.source = np.array([r['source'] for r in self.rows])
        self.vi, self.ti = np.arange(self.nv), np.arange(self.nv, len(self.rows))

    def scores(self, cfg, vidx=None):
        """cfg = dict(vdf=key, vad=key|None, w=float, m=key, clip=bool); vidx = val row indices used for fitting."""
        vidx = self.vi if vidx is None else vidx
        fv = self.comp[cfg['vdf']]
        if cfg.get('vad'):
            to = qtable(fv[vidx][self.has[cfg['vdf']][vidx]])
            ad = self.comp[cfg['vad']]
            ta = qtable(ad[vidx][self.has[cfg['vad']][vidx]])
            c = (np.interp(fv, to, LIN) + cfg['w'] * np.interp(ad, ta, LIN)) / (1 + cfg['w'])
            fv = np.interp(c, np.linspace(0, 1, len(to)), to)
            if cfg.get('clip'):
                fv = np.minimum(fv, 1.)
        fm = self.comp[cfg['m']]
        k = vidx[self.fitmask[vidx]]
        ens = fit_file(self.pv[k], self.pm[k], fv[k], fm[k], self.y['file'][k], force=cfg.get('force_w'))
        return {'file': file_vec(ens, self.pv, self.pm, fv, fm), 'voice': fv, 'music': fm, 'ens_w': ens['weights']}

    def ads(self, sc, idx, w=None):
        e = {}
        for k in ('file', 'voice', 'music'):
            m = self.el[k][idx]
            e[k] = eer(self.y[k][idx][m], sc[k][idx][m], None if w is None else w[m])
        return .5 * (1 - e['file']) + .2 * (1 - e['voice']) + .3 * (1 - e['music']), e


def group_index(groups, idx):
    """{group: row indices} restricted to idx."""
    order = np.argsort(groups[idx], kind='mergesort')
    g = groups[idx][order]
    cut = np.flatnonzero(g[1:] != g[:-1]) + 1
    return np.split(idx[order], cut)


def resample(members, rng):
    return np.concatenate([members[j] for j in rng.integers(0, len(members), len(members))])


def summ(v, point):
    v = np.asarray(v)
    lo, hi = np.percentile(v, [2.5, 97.5])
    return {'point': round(point, 5), 'boot_mean': round(float(v.mean()), 5), 'sd': round(float(v.std()), 5),
            'pct_ci95': [round(lo, 5), round(hi, 5)], 'basic_ci95': [round(2 * point - hi, 5), round(2 * point - lo, 5)],
            'p_gt0': round(float((v > 0).mean()), 4), 'n': len(v)}


def selfcheck():
    from sklearn.metrics import roc_curve
    rng = np.random.default_rng(1)
    for n in (50, 400, 3000):
        y = rng.integers(0, 2, n)
        s = np.round(rng.random(n) + .3 * y, 2)            # ties on purpose
        w = rng.random(n) * 3
        for ww in (None, w):
            fpr, tpr, _ = roc_curve(y, s, drop_intermediate=False, sample_weight=ww)
            i = np.argmin(np.abs(fpr - (1 - tpr)))
            assert abs(eer(y, s, ww) - (fpr[i] + 1 - tpr[i]) / 2) < 1e-12
    print('core.eer matches sklearn (ties, weights)')


if __name__ == '__main__':
    selfcheck()
