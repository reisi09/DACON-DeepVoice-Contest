"""X2 (broadcast low-SNR rows) pre-registered gates, T-type (research/top15/work5/debate_20260928 design, DESIGN_KO.md 5.2),
fixed before any X2 result. Candidate CAND vs control CTRL, both DB protocol (same jobs, WEIGHTING=s7, NEW_VAL=0).
  T1  broadcast target: J = 0.5 FILE EER + 0.3 MUSIC EER on E2 AI-OpenBMAT rows that are clean (leak_check + integrity):
      dJ <= -0.02 and bootstrap CI upper < 0 over music-track components (< 20 components -> HOLD)
  T2  synthetic held-out-donor panel (bpanel, 360 rows, never trained on, NOT the locked set): dJ <= -0.02 (point only:
      the panel shares music donors, 3 lineage components, so no group CI exists; decided before any result)
  N1  natural distribution E1 + clean E2: dADS CI lower >= -0.002
  N2  45-point grid (type share x phone share x unseen-generator share, row-weighted ROC): every point dADS >= -0.002
  P   protected slices (increase allowed, candidate-specific thresholds at old-val TPR 95 %):
      music-absent real FILE FPR <= +0.01 | mixed (voice+music) FILE EER <= +0.010 | music-present fake rows with
      Pm <= 0.1 and 0.1 < Pm <= 0.5: FILE FNR <= +0.01 | realV+fakeM: FILE FNR, MUSIC FNR, VOICE FPR <= +0.01 |
      music-only FILE EER, AT-ADD singing FILE and VOICE EER, phone-channel FILE EER <= +0.010
      a slice with rows but < 20 groups per class -> HOLD (not pass); a slice with no rows anywhere -> N/A (reported)
  S   seeds: per-seed dJ (T1) negative for >= 2 of 3 seeds
  (E3 Korean real FPR is scored on the Mac with score_e3.py and checked there: FPR increase <= 0.01.)
  MODE=x1 (X1 FILE rule, candidate 'CONFIG@B'): the target is the music-absent real FILE FPR at the candidate-specific
  old-val TPR-95 threshold, decrease >= 0.01 with group-bootstrap CI upper < 0, instead of T1/T2 (no bpanel needed).
  python x2_gates.py RUNS CTRL CAND NEW_ROWS BPANEL_RUNS BPANEL_ROWS INTEGRITY_JSON LINKED_JSON OUT_JSON"""
import csv
import itertools
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
RUNS, CTRL, CAND, NEW_ROWS, BP_RUNS, BP_ROWS, INTEG, LINKED, OUT = sys.argv[1:10]


def load(runs, split, rows_csv):
    sys.argv = ['new_eval.py', runs, CTRL, CTRL, CAND]
    os.environ['NEW_ROWS'], os.environ['EVAL_SPLITS'] = rows_csv, split
    g = {'__file__': str(HERE / 'new_eval.py'), '__name__': 'x'}
    src = open(g['__file__']).read()
    cut = "preds = {c: predictions(c) for c in CONFIGS}"
    exec(compile(src[:src.index(cut)] + cut + "\n", g['__file__'], 'exec'), g)
    return g


g = load(RUNS, 'newtest', NEW_ROWS)
F, lab, val = g['F'], g['lab'], g['val']
dirty = set(json.load(open(INTEG))['dirty_ids']) | set(json.load(open(LINKED)))
e1 = list(zip(g['test'], g['preds'][CTRL]['test'], g['preds'][CAND]['test']))
e2 = [t for t in zip(g['newtest'], g['preds'][CTRL]['newtest'], g['preds'][CAND]['newtest']) if t[0]['ID'] not in dirty]
MODE = os.environ.get('MODE', 'x2')
if MODE == 'x2':
    gb = load(BP_RUNS, 'bpanel', f'{NEW_ROWS},{BP_ROWS}')
    bp = list(zip(gb['newtest'], gb['preds'][CTRL]['newtest'], gb['preds'][CAND]['newtest']))
else:
    bp = []
COMP = {'FILE': ('FILE_FAKE', None), 'VOICE': ('VOICE_FAKE', 'VOICE_PRESENT'), 'MUSIC': ('MUSIC_FAKE', 'MUSIC_PRESENT')}


def keep(r, comp):
    fake, present = COMP[comp]
    return lab(r, fake) is not None and (present is None or lab(r, present) == 1)


def eer(rows, preds, comp, w=None):
    ii = [k for k, r in enumerate(rows) if keep(r, comp)]
    y = np.array([lab(rows[k], COMP[comp][0]) for k in ii])
    if len(set(y.tolist())) < 2:
        return float('nan')
    from sklearn.metrics import roc_curve
    s = np.array([preds[k][f'{comp}_FAKE_PROB'] for k in ii])
    fpr, tpr, _ = roc_curve(y, s, sample_weight=None if w is None else np.asarray(w)[ii], drop_intermediate=False)
    i = np.argmin(np.abs(fpr - (1 - tpr)))
    return float((fpr[i] + 1 - tpr[i]) / 2)


def J(rows, preds):
    return .5 * eer(rows, preds, 'FILE') + .3 * eer(rows, preds, 'MUSIC')


def ads(rows, preds, w=None):
    return .5 * (1 - eer(rows, preds, 'FILE', w)) + .2 * (1 - eer(rows, preds, 'VOICE', w)) + .3 * (1 - eer(rows, preds, 'MUSIC', w))


def boot(triples, stat, reps=2000, seed=0):
    """Paired group bootstrap of stat(cand) - stat(ctrl); NaN resamples dropped and counted."""
    rows = [t[0] for t in triples]
    groups = defaultdict(list)
    for k, r in enumerate(rows):
        groups[r['group_id']].append(k)
    keys = sorted(groups)
    rng = np.random.default_rng(seed)
    a, b = [t[1] for t in triples], [t[2] for t in triples]
    point = stat(rows, b) - stat(rows, a)
    d = []
    for _ in range(reps):
        idx = [k for x in rng.choice(len(keys), len(keys)) for k in groups[keys[x]]]
        rr = [rows[k] for k in idx]
        d.append(stat(rr, [b[k] for k in idx]) - stat(rr, [a[k] for k in idx]))
    d = np.array(d)
    ok = np.isfinite(d)
    return {'delta': None if not np.isfinite(point) else round(point, 5), 'n_rows': len(rows), 'n_groups': len(keys),
            'ci95': np.percentile(d[ok], [2.5, 97.5]).round(5).tolist() if ok.any() else None, 'undefined_resamples': int((~ok).sum())}


def thresholds(preds_val, comp):
    """Old-val threshold at TPR 95 % for comp (candidate-specific)."""
    s = [p[f'{comp}_FAKE_PROB'] for r, p in zip(val, preds_val) if keep(r, comp) and lab(r, COMP[comp][0]) == 1]
    return float(np.quantile(s, .05))


def rate(triples, comp, cls, thr_a, thr_b):
    """FPR (cls 0) or FNR (cls 1) at the given thresholds for control / candidate, and class / group counts."""
    sel = [t for t in triples if keep(t[0], comp) and lab(t[0], COMP[comp][0]) == cls]
    if not sel:
        return None
    k = f'{comp}_FAKE_PROB'
    fa = np.mean([t[1][k] >= thr_a for t in sel]) if cls == 0 else np.mean([t[1][k] < thr_a for t in sel])
    fb = np.mean([t[2][k] >= thr_b for t in sel]) if cls == 0 else np.mean([t[2][k] < thr_b for t in sel])
    return {'ctrl': round(float(fa), 4), 'cand': round(float(fb), 4), 'increase': round(float(fb - fa), 4), 'n': len(sel),
            'groups': len({t[0]['group_id'] for t in sel})}


def slice_eer(triples, comp):
    rows = [t[0] for t in triples]
    a, b = eer(rows, [t[1] for t in triples], comp), eer(rows, [t[2] for t in triples], comp)
    groups = defaultdict(set)
    for r in rows:
        if keep(r, comp):
            groups[lab(r, COMP[comp][0])].add(r['group_id'])
    return {'ctrl': a, 'cand': b, 'increase': None if not (np.isfinite(a) and np.isfinite(b)) else round(b - a, 4),
            'n': len(rows), 'groups_per_class': {k: len(v) for k, v in groups.items()}}


pv = {c: g['preds'][c]['val'] for c in (CTRL, CAND)}
thr = {c: {k: thresholds(pv[c], k) for k in COMP} for c in (CTRL, CAND)}
nat = e1 + e2
V = lambda r: lab(r, 'VOICE_PRESENT') == 1
Mu = lambda r: lab(r, 'MUSIC_PRESENT') == 1
res = {'ctrl': CTRL, 'cand': CAND, 'n_e1': len(e1), 'n_e2_clean': len(e2), 'n_bpanel': len(bp)}
ob = [t for t in e2 if t[0]['source'] == 'openbmat']   # regrouped by music-track components below
def track_components(triples):
    """OpenBMAT rows regrouped by connected components of their music tracks (review round 3: a track-set string splits
    rows that share only some tracks into different groups)."""
    import zipfile
    z = zipfile.ZipFile('/root/newdata/ai_openbmat/AI-OpenBMAT.zip')
    parent = {}

    def find(x):
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    for t in triples:
        r = t[0]
        art = 'ai' if '/ai/' in r['path'] else 'human'
        m = json.loads(z.read(f"AI-OpenBMAT/{art}/{Path(r['path']).stem}.json"))
        for seg in m['segments']:
            for src in seg.get('sources', []):
                if src.get('type') == 'music':
                    parent[find(('row', r['ID']))] = find(('track', Path(src['file']).stem))
    return [({**t[0], 'group_id': 'obc:%s' % str(find(('row', t[0]['ID'])))}, t[1], t[2]) for t in triples]


ob = track_components(ob)
remap = {t[0]['ID']: t[0]['group_id'] for t in ob}
nat = [({**t[0], 'group_id': remap[t[0]['ID']]}, t[1], t[2]) if t[0]['ID'] in remap else t for t in nat]
res['N1'] = boot(nat, ads)
res['T1'] = boot(ob, J)
cls_groups = defaultdict(set)
for t in ob:
    cls_groups[lab(t[0], 'FILE_FAKE')].add(t[0]['group_id'])
share = defaultdict(int)
for t in ob:
    share[t[0]['group_id']] += 1
res['T1'].update(groups_per_class={str(k): len(v) for k, v in cls_groups.items()},
                 max_group_share=round(max(share.values()) / len(ob), 3) if ob else None)
# T2: the synthetic panel shares music donors across pairs (3 lineage components), so no group CI exists: point only
res['T2'] = ({'delta': round(J([t[0] for t in bp], [t[2] for t in bp]) - J([t[0] for t in bp], [t[1] for t in bp]), 5),
              'n_rows': len(bp), 'n_components': len({t[0]['group_id'] for t in bp})} if bp else None)
# N2 grid: row weights = product over dimensions of target share / observed share of the row's stratum
def strata(r):
    ty = 'mixed' if V(r) and Mu(r) else 'voice' if V(r) else 'music'
    return ty, r.get('codec_chain', '').startswith('tel'), r.get('dataset') == 'atadd' or r['source'].startswith('atadd')
S = [strata(t[0]) for t in nat]
obs = [defaultdict(float) for _ in range(3)]
for s in S:
    for d in range(3):
        obs[d][s[d]] += 1 / len(S)
TYPES = [(.6, .1, .3), (.45, .2, .35), (.3, .3, .4), (.3, .2, .5), (.5, .3, .2)]
grid, worst_margin, infeasible = [], 0., 0


def err_point(w, target):
    tot = w.sum()
    e = 0.
    for d in range(3):
        cur = defaultdict(float)
        for x, s_ in zip(w, S):
            cur[s_[d]] += x / tot
        e = max(e, max(abs(cur[k] - v) for k, v in target[d].items()))
    return e
for (tv, tm, tx), tel, un in itertools.product(TYPES, (0., .15, .3), (0., .4, .8)):
    target = [{'voice': tv, 'music': tm, 'mixed': tx}, {True: tel, False: 1 - tel}, {True: un, False: 1 - un}]
    # raking (iterative proportional fitting): the weighted marginals match every target share (review round 3)
    w = np.ones(len(S))
    for _ in range(100):
        for d in range(3):
            tot = w.sum()
            cur = defaultdict(float)
            for x, s_ in zip(w, S):
                cur[s_[d]] += x / tot
            w = np.array([x * (target[d][s_[d]] / cur[s_[d]] if cur[s_[d]] else 0.) for x, s_ in zip(w, S)])
    if err_point(w, target) >= .01:
        infeasible += 1
        continue
    rows = [t[0] for t in nat]
    grid.append(round(ads(rows, [t[2] for t in nat], w) - ads(rows, [t[1] for t in nat], w), 5))
res['N2'] = {'min': min(grid) if grid else None, 'max': max(grid) if grid else None, 'n_feasible': len(grid),
             'n_infeasible': infeasible, 'finite': bool(grid) and bool(np.all(np.isfinite(grid)))}
no_music_real = [t for t in nat if lab(t[0], 'MUSIC_PRESENT') == 0]
mixed = [t for t in nat if V(t[0]) and Mu(t[0])]
low = lambda lo, hi: [t for t in nat if Mu(t[0]) and lo < t[1]['MUSIC_PRESENT_PROB'] <= hi]
rvfm = [t for t in nat if lab(t[0], 'VOICE_FAKE') == 0 and lab(t[0], 'MUSIC_FAKE') == 1]
res['P'] = {
    'music_absent_real_FILE_FPR': rate(no_music_real, 'FILE', 0, thr[CTRL]['FILE'], thr[CAND]['FILE']),
    'mixed_FILE_EER': slice_eer(mixed, 'FILE'),
    'lowPm_le0.1_FILE_FNR': rate(low(-1, .1), 'FILE', 1, thr[CTRL]['FILE'], thr[CAND]['FILE']),
    'lowPm_0.1_0.5_FILE_FNR': rate(low(.1, .5), 'FILE', 1, thr[CTRL]['FILE'], thr[CAND]['FILE']),
    'realV_fakeM_FILE_FNR': rate(rvfm, 'FILE', 1, thr[CTRL]['FILE'], thr[CAND]['FILE']),
    'realV_fakeM_MUSIC_FNR': rate(rvfm, 'MUSIC', 1, thr[CTRL]['MUSIC'], thr[CAND]['MUSIC']),
    'realV_fakeM_VOICE_FPR': rate(rvfm, 'VOICE', 0, thr[CTRL]['VOICE'], thr[CAND]['VOICE']),
    'music_only_FILE_EER': slice_eer([t for t in nat if Mu(t[0]) and lab(t[0], 'VOICE_PRESENT') == 0], 'FILE'),
    'singing_FILE_EER': slice_eer([t for t in e2 if t[0]['source'] == 'atadd_singing'], 'FILE'),
    'singing_VOICE_EER': slice_eer([t for t in e2 if t[0]['source'] == 'atadd_singing'], 'VOICE'),
    'phone_FILE_EER': slice_eer([t for t in nat if t[0].get('codec_chain', '').startswith('tel')], 'FILE'),
    'sequential': 'N/A (no rows in any development set)'}


def verdict_rate(x, lim=.01):
    if x is None:          # rows vanished after filtering: not measured, so the gate cannot pass (review round 3)
        return 'HOLD'
    if x['groups'] < 20:
        return 'HOLD'
    return 'PASS' if x['increase'] <= lim else 'FAIL'


def verdict_eer(x, lim=.010):
    if x['increase'] is None:
        return 'HOLD'
    if min(x['groups_per_class'].values() or [0]) < 20:
        return 'HOLD'
    return 'PASS' if x['increase'] <= lim else 'FAIL'


V_ = {}
if MODE == 'x2':
    if (res['T1']['ci95'] is None or res['T1']['delta'] is None or min(res['T1']['groups_per_class'].values() or [0]) < 20
            or (res['T1']['max_group_share'] or 1) > .1):
        V_['T1'] = 'HOLD'
    else:
        V_['T1'] = 'PASS' if res['T1']['delta'] <= -.02 and res['T1']['ci95'][1] < 0 else 'FAIL'
    V_['T2'] = 'PASS' if res['T2'] and res['T2']['delta'] is not None and res['T2']['delta'] <= -.02 else 'FAIL'
else:
    ta, tb = thr[CTRL]['FILE'], thr[CAND]['FILE']
    nm = [t for t in no_music_real if lab(t[0], 'FILE_FAKE') == 0]
    groups = defaultdict(list)
    for k, t in enumerate(nm):
        groups[t[0]['group_id']].append(k)
    keys, rng, d = sorted(groups), np.random.default_rng(1), []
    fa = np.array([t[1]['FILE_FAKE_PROB'] >= ta for t in nm], float)
    fb = np.array([t[2]['FILE_FAKE_PROB'] >= tb for t in nm], float)
    for _ in range(2000):
        idx = [k for x in rng.choice(len(keys), len(keys)) for k in groups[keys[x]]]
        d.append(fb[idx].mean() - fa[idx].mean())
    res['X1_target'] = {'ctrl_fpr': round(float(fa.mean()), 4), 'cand_fpr': round(float(fb.mean()), 4),
                        'delta': round(float(fb.mean() - fa.mean()), 4), 'ci95': np.percentile(d, [2.5, 97.5]).round(4).tolist(),
                        'n_rows': len(nm), 'n_groups': len(keys)}
    V_['X1_target'] = 'PASS' if res['X1_target']['delta'] <= -.01 and res['X1_target']['ci95'][1] < 0 else 'FAIL'
V_['N1'] = 'PASS' if res['N1']['ci95'] and res['N1']['ci95'][0] >= -.002 else 'FAIL'
V_['N2'] = ('HOLD' if res['N2']['n_feasible'] < 20 else 'PASS' if res['N2']['finite'] and res['N2']['min'] >= -.002 else 'FAIL')
for k, x in res['P'].items():
    if isinstance(x, str):
        V_[k] = 'N/A'
    elif 'groups_per_class' in x:
        V_[k] = verdict_eer(x)
    else:
        V_[k] = verdict_rate(x)


# S: per-seed paired comparison (each seed's members alone, same CDF/FILE procedure), target statistic of the mode
def seed_dir(k):
    out = Path(f'/root/x2/seed_runs/{MODE}/{k}')
    for c in (CTRL, CAND):
        src = Path(RUNS) / c
        (out / c).mkdir(parents=True, exist_ok=True)
        for f in src.glob('*.json'):
            j = json.loads(f.read_text())
            for split in j.get('every', {}):
                j['every'][split]['preds'] = [j['every'][split]['preds'][k]]
            (out / c / f.name).write_text(json.dumps(j))
    return str(out)


seed_d = []
for k in range(3):
    gk = load(seed_dir(k), 'newtest', NEW_ROWS)
    e1k = list(zip(gk['test'], gk['preds'][CTRL]['test'], gk['preds'][CAND]['test']))
    e2k = [t for t in zip(gk['newtest'], gk['preds'][CTRL]['newtest'], gk['preds'][CAND]['newtest']) if t[0]['ID'] not in dirty]
    if MODE == 'x2':
        obk = [t for t in e2k if t[0]['source'] == 'openbmat']
        rows_k = [t[0] for t in obk]
        seed_d.append(J(rows_k, [t[2] for t in obk]) - J(rows_k, [t[1] for t in obk]))
    else:
        g_prev = g
        g = gk
        thr_k = {c: thresholds(gk['preds'][c]['val'], 'FILE') for c in (CTRL, CAND)}
        g = g_prev
        nm_k = [t for t in e1k + e2k if lab(t[0], 'MUSIC_PRESENT') == 0 and lab(t[0], 'FILE_FAKE') == 0]
        seed_d.append(float(np.mean([t[2]['FILE_FAKE_PROB'] >= thr_k[CAND] for t in nm_k]) - np.mean([t[1]['FILE_FAKE_PROB'] >= thr_k[CTRL] for t in nm_k])))
res['S'] = {'per_seed_delta': [round(x, 5) for x in seed_d]}
V_['S'] = 'PASS' if sum(x < 0 for x in seed_d) >= 2 else 'FAIL'
res['verdicts'] = V_
res['offline_gates_pass'] = all(v in ('PASS', 'N/A') for v in V_.values())
res['adoption'] = 'pending the E3 Korean-real FPR check on the Mac' if res['offline_gates_pass'] else 'rejected'
json.dump(res, open(OUT, 'w'), indent=1)
print(json.dumps(res, indent=1))
