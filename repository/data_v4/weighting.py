"""Training-row weights for the heads (research/top15/work5/w/SYNTHESIS_R3.md, pre-registered 2026-09-28).
  s7  current: full_bank_batches.loss_weights (equal class, equal (label, source, generator) cell) x global duration_balance
  C   s7 weights + final caps (row <= 5x class mean, lineage component <= 2% of class mass)
  W   voice: strata KS/XS/SM/SG with target mass q per class (within-stratum 50:50), base max(n_cell, 20)^-1/2 inside
      (class, stratum), duration balance inside each stratum (clip [0.5, 2]), then the caps; music: global 50:50, the same
      base and a global duration balance, then the caps.
Every class keeps total mass N/2 (the scale of loss_weights). Selected with env WEIGHTING (default s7).
  python data_v4/weighting.py   # self-check
"""
from __future__ import annotations

import json
import os
from collections import Counter, defaultdict

import numpy as np

from full_bank_batches import duration_balance, loss_weights

SCHEME = os.environ.get('WEIGHTING', 's7')
Q_VOICE = {'KS': .25, 'XS': .45, 'SM': .05, 'SG': .25}
ROW_CAP, GROUP_CAP, CELL_FLOOR = 5., .02, 20
SPEECH = {'mlaad', 'speechfake', 'zeroth', 'fleurs_ko', 'musan_speech', 'existing:librispeech_test-clean', 'asvspoof2021_la',
          'asvspoof2021_df', 'existing:asvspoof2019_la', 'constructed_enhanced', 'constructed_splice', 'singmos_pro', 'atadd_speech',
          'gtsinger_ko_speech'}
SPEECH_PARENTS = {'mlaad', 'speechfake', 'fleurs_ko', 'zeroth', 'musan_speech'}
ALIASES = {'existing:SONICS': 'sonics', 'existing:HAIM': 'haim', 'existing:FMA/FMA': 'existing:FMA', 'existing:suno': 'sonics',
           'existing:udio': 'sonics'}


def source_family(source: str) -> str:
    return 'echoes' if source.startswith('existing:Echoes/') else ALIASES.get(source, source)


def voice_stratum(row: dict) -> str:
    """KS Korean speech, XS other/unknown-language speech, SM speech mixed onto music, SG singing (a cappella or in songs)."""
    s = row.get('source', '')
    if s == 'constructed_mix':
        parents = {p.split(':')[0] for p in json.loads(row.get('parent_ids') or '[]')}
        return 'SM' if parents & SPEECH_PARENTS else 'SG'
    if s in SPEECH:
        return 'KS' if row.get('language') == 'ko' else 'XS'
    return 'SG'


def capped(pre: np.ndarray, groups: list, mass: float, iters: int = 50) -> np.ndarray:
    """Proportional allocation of `mass` over rows with weights proportional to `pre`, subject to row <= ROW_CAP * mass / n and
    every group's sum <= GROUP_CAP * mass (water-filling: violators are fixed at their caps, the rest re-scaled)."""
    n = len(pre)
    row_cap, group_cap = ROW_CAP * mass / n, GROUP_CAP * mass
    g = np.unique(np.asarray(groups, dtype=object).astype(str), return_inverse=True)[1]
    w, fixed = np.zeros(n), np.zeros(n, bool)
    for _ in range(iters):
        free_mass = mass - w[fixed].sum()
        if free_mass <= 0 or not (~fixed).any():
            raise ValueError('caps infeasible: no free mass left')
        w[~fixed] = pre[~fixed] * free_mass / pre[~fixed].sum()
        changed = False
        over = (~fixed) & (w > row_cap * (1 + 1e-12))
        if over.any():
            w[over], fixed[over], changed = row_cap, True, True
        sums = defaultdict(float)
        for gi, wi in zip(g, w):
            sums[gi] += wi
        for gi, total in sums.items():
            if total > group_cap * (1 + 1e-9):   # scale the whole group down to its cap (never breaks a row cap)
                m = g == gi
                w[m] *= group_cap / total
                fixed[m], changed = True, True
        if not changed:
            return w
    raise ValueError('caps did not converge')


DURATION_CHECK: dict = {}


def _separability(y, d, w) -> float:
    from sklearn.metrics import roc_auc_score
    a = roc_auc_score(y, d, sample_weight=w)
    return max(a, 1 - a)


def weights(items, samples, route: str, scheme: str = SCHEME) -> dict:
    """ID -> weight for training `items` (full_bank_batches.Item-like: .label, .row) with `samples` (lengths, same order)."""
    ids = [x.row['ID'] for x in items]
    y = np.array([float(x.label) for x in items])
    base = loss_weights(items)
    if scheme == 's7' or scheme == 'C':
        w = np.array([base[i] for i in ids]) * duration_balance(y, samples, [base[i] for i in ids])
        if scheme == 's7':
            return dict(zip(ids, w))
    else:
        if scheme != 'W':
            raise ValueError(f'unknown WEIGHTING {scheme!r}')
        strata = [voice_stratum(x.row) if route == 'voice' else 'all' for x in items]
        cells = Counter((yy, st, source_family(x.row.get('source', '')), x.row.get('generator', ''))
                        for yy, st, x in zip(y, strata, items))
        b = np.array([max(cells[(yy, st, source_family(x.row.get('source', '')), x.row.get('generator', ''))], CELL_FLOOR) ** -.5
                      for yy, st, x in zip(y, strata, items)])
        w = np.zeros(len(items))
        for c in (0., 1.):
            present = {st for st, yy in zip(strata, y) if yy == c}
            q = {st: (Q_VOICE[st] if route == 'voice' else 1.) for st in present}
            total_q = sum(q.values())
            for st in present:
                m = np.array([(yy == c) and (s == st) for yy, s in zip(y, strata)])
                w[m] = b[m] * (q[st] / total_q) * len(y) / 2 / b[m].sum()
        s7w = np.array([base[i] for i in ids]) * duration_balance(y, samples, [base[i] for i in ids])
        logd = np.log(np.maximum(np.asarray(samples, float), 1.))
        for st in sorted(set(strata)):   # duration balance inside each stratum (both classes needed), clipped [0.5, 2]
            m = np.array([s == st for s in strata])
            if len(set(y[m])) == 2:
                def balanced(clip):
                    f = duration_balance(y[m], np.asarray(samples)[m], w[m], clip=clip)
                    f = np.clip(f, .5, 2.) if clip == 2. else f
                    out = w[m].copy()
                    for c in (0., 1.):   # restore each (class, stratum) mass
                        mc = y[m] == c
                        before = out[mc].sum()
                        out[mc] *= f[mc]
                        out[mc] *= before / out[mc].sum()
                    return out
                trial = balanced(2.)
                # pre-registered shortcut check: widen to [0.1, 10] if duration alone separates the classes more than
                # max(0.60, the S7 weights' separability) in this stratum
                sep_w, sep_s7 = _separability(y[m], logd[m], trial), _separability(y[m], logd[m], s7w[m])
                widened = sep_w > max(.60, sep_s7)
                w[m] = balanced(10.) if widened else trial
                DURATION_CHECK[(route, st)] = {'sep_W': round(sep_w, 4), 'sep_s7': round(sep_s7, 4), 'widened': bool(widened)}
    out = np.zeros(len(items))
    for c in (0., 1.):
        m = y == c
        out[m] = capped(w[m], [items[i].row.get('group_id', ids[i]) for i in np.flatnonzero(m)], len(y) / 2)
    return dict(zip(ids, out))


def describe(items, w: dict, route: str) -> dict:
    """Per class: Kish effective rows / lineage groups, max group share, mass by stratum and top sources."""
    out = {}
    for c in (0., 1.):
        xs = [x for x in items if float(x.label) == c]
        v = np.array([w[x.row['ID']] for x in xs])
        g, st, src = defaultdict(float), defaultdict(float), defaultdict(float)
        for x, wi in zip(xs, v):
            g[x.row.get('group_id', x.row['ID'])] += wi
            st[voice_stratum(x.row) if route == 'voice' else 'all'] += wi
            src[source_family(x.row.get('source', ''))] += wi
        tot = v.sum()
        out['fake' if c else 'real'] = {'rows': len(xs), 'kish_rows': round(tot ** 2 / (v ** 2).sum(), 1),
                                        'kish_groups': round(tot ** 2 / sum(u * u for u in g.values()), 1),
                                        'max_group_share': round(max(g.values()) / tot, 4), 'max_row_over_mean': round(v.max() / v.mean(), 2),
                                        'stratum_share': {k: round(u / tot, 3) for k, u in sorted(st.items())},
                                        'top_sources': {k: round(u / tot, 3) for k, u in sorted(src.items(), key=lambda t: -t[1])[:6]}}
    return out


if __name__ == '__main__':
    class It:
        def __init__(self, i, label, source, generator, group, language=''):
            self.label, self.row = label, {'ID': str(i), 'source': source, 'generator': generator, 'group_id': group, 'language': language}
    rng = np.random.default_rng(0)
    items = [It(i, float(i % 3 == 0), ['mlaad', 'zeroth', 'sonics', 'vocalset'][i % 4], f'g{i % 50}', f'grp{i % 97 if i % 5 else 0}',
                'ko' if i % 7 == 0 else '') for i in range(3000)]
    samples = rng.integers(16000, 16000 * 60, len(items))
    for scheme in ('s7', 'C', 'W'):
        w = weights(items, samples, 'voice', scheme)
        v = np.array([w[x.row['ID']] for x in items]); y = np.array([x.label for x in items])
        assert abs(v[y == 1].sum() - len(items) / 2) < 1e-6 * len(items) and abs(v[y == 0].sum() - len(items) / 2) < 1e-6 * len(items), scheme
        if scheme != 's7':
            d = describe(items, w, 'voice')
            for k in ('real', 'fake'):
                assert d[k]['max_group_share'] <= GROUP_CAP + 1e-9 and d[k]['max_row_over_mean'] <= ROW_CAP + 1e-9, (scheme, d[k])
    print('self-check ok')
