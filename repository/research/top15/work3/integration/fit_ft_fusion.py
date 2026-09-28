"""Fit and gate the fine-tuned AntiDeepfake member 'ft' (research/top15/work3/DESIGN.md) in the S4 voice CDF fusion.
ft = window-max fake probability of the fine-tuned detector on the voice stem (member CSV column ft_max).
  option R: VOICE = fuse(ours 1, ft w)            (ft replaces the frozen-AD ResMLP member 'ad'; package size unchanged)
  option A: VOICE = fuse(ours 1, ad 0.5, ft w)    (ft added; needs the fp16 MERT bin to stay under 10 GB)

  db  DB-protocol gate on the S4-equivalent member scores (work2/stacking: runs/*.json, presence_*.json, frozen rows).
      w in {0.5, 1, 2} per option by val official ADS (RT protocol), FILE ensemble refit on val for every configuration;
      option = higher val RT ADS (tie -> R). The choice uses val only; held-out ADS + group bootstrap vs S4 are then
      reported. Gate (DESIGN.md): held-out RT dADS >= +0.01 and P(dADS > 0) >= 0.95. The S4 baseline must reproduce
      the recorded val/held-out ADS (RT 0.93773/0.88457, AC 0.93845/0.88643) or the run aborts.
    python fit_ft_fusion.py db --ft FT_VAL.csv FT_TEST.csv --out DIR
  p1  package fit: S4 P1 heads dir + local caches (every val row with a backbone npz) + ft val CSV -> heads dir with
      component_fusion.json and the refitted file_ensemble.json (train_backbone_heads.final semantics), usable as both
      --heads and --fusion of package_datav4_submission.py. --option S4 must reproduce the S4 tables.
    python fit_ft_fusion.py p1 --option R --w 1 --ft FT_VAL.csv --out DIR
Paths default to this checkout (script at research/top15/work3/integration/); pass them explicitly elsewhere. --code:
dirs with decision_heads, component_fusion, file_ensemble (data_v4/), graph_scoring (+ voice_heads, localization_models for p1).
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[4]
S4_EXPECTED = {'RT': {'val': .93773, 'test': .88457}, 'AC': {'val': .93845, 'test': .88643}}
WEIGHTS = (.5, 1, 2)
OPTIONS = {'R': lambda w, ad: {'ours': 1, 'ft': w}, 'A': lambda w, ad: {'ours': 1, 'ad': ad, 'ft': w}}


def read_ft(paths, col):
    out = {}
    for p in paths:
        for r in csv.DictReader(open(p)):
            v = float(r[col])
            if not math.isfinite(v) or not 0 <= v <= 1 or out.get(r['ID'], v) != v:
                raise ValueError(f'bad or conflicting ft value for {r["ID"]} in {p}')
            out[r['ID']] = v
    return out


def bank_rows(bank: Path, split: str, keep=None):
    f = bank / {'train': 'bank_trainval.csv', 'val': 'bank_trainval.csv', 'test': 'bank_test.csv'}[split]
    rows = [r for r in csv.DictReader(f.open()) if r['split'] == split and (keep is None or r['ID'] in keep)]
    if not rows:   # 'train' used to read bank_test.csv and silently return []
        raise ValueError(f'no {split} rows in {f}')
    return rows


def lab(r, c):
    return None if r[c + '_mask'] != '1' or r[c] == '' else int(r[c])


def eer(t, s):
    """EER; NaN when t holds a single class (EER is undefined there)."""
    if len(np.unique(t)) < 2:
        return float('nan')
    from sklearn.metrics import roc_curve
    fpr, tpr, _ = roc_curve(t, s, drop_intermediate=False)
    i = np.argmin(np.abs(fpr - (1 - tpr)))
    return (fpr[i] + 1 - tpr[i]) / 2


def ads_of(rows, preds):
    from graph_scoring import official_metrics
    m = official_metrics(rows, preds)['metrics']
    return {k: round(m[k], 5) for k in ('file_eer', 'voice_eer', 'music_eer')} | {
        'ads': round(.5 * (1 - m['file_eer']) + .2 * (1 - m['voice_eer']) + .3 * (1 - m['music_eer']), 5)}


def file_preds(files, keep_rows):
    """FILE ensemble refit on the val rows with a FILE label (file_ensemble.fit default: equal unless grid gains .002)."""
    import decision_heads as dh
    from file_ensemble import fit
    keep = [(f, r) for f, r in zip(files['val'], keep_rows) if r['FILE_FAKE_mask'] == '1']
    ens, report = fit([f for f, _ in keep], np.array([int(r['FILE_FAKE']) for _, r in keep]))
    return ens, report, {s: [{'ID': f['ID'], 'FILE_FAKE_PROB': dh.file_score(ens, f), 'VOICE_FAKE_PROB': f['Fv'],
                              'MUSIC_FAKE_PROB': f['Fm'], 'VOICE_PRESENT_PROB': f['Pv'], 'MUSIC_PRESENT_PROB': f['Pm']}
                             for f in fs] for s, fs in files.items()}


class Boot:
    """Group bootstrap of official ADS with models and fits fixed (work2/stacking/ads.py Boot semantics)."""
    def __init__(self, rows, reps=1000, seed=0):
        self.el = {'file': np.array([lab(r, 'FILE_FAKE') is not None for r in rows]),
                   'voice': np.array([lab(r, 'VOICE_PRESENT') == 1 and lab(r, 'VOICE_FAKE') is not None for r in rows]),
                   'music': np.array([lab(r, 'MUSIC_PRESENT') == 1 and lab(r, 'MUSIC_FAKE') is not None for r in rows])}
        self.y = {k: np.array([lab(r, c) if e else -1 for r, e in zip(rows, self.el[k])])
                  for k, c in (('file', 'FILE_FAKE'), ('voice', 'VOICE_FAKE'), ('music', 'MUSIC_FAKE'))}
        g = np.array([r['group_id'] for r in rows])
        ug = np.unique(g)
        members = {x: np.flatnonzero(g == x) for x in ug}
        rng = np.random.default_rng(seed)
        self.idx = [np.concatenate([members[x] for x in rng.choice(ug, len(ug))]) for _ in range(reps)]
        self.n_groups, self.ids = len(ug), [r['ID'] for r in rows]

    def ads(self, preds, idx=None):
        assert [p['ID'] for p in preds] == self.ids
        S = {k: np.array([p[c] for p in preds]) for k, c in (('file', 'FILE_FAKE_PROB'), ('voice', 'VOICE_FAKE_PROB'), ('music', 'MUSIC_FAKE_PROB'))}
        idx = np.arange(len(self.ids)) if idx is None else idx
        e = {k: eer(self.y[k][idx][self.el[k][idx]], S[k][idx][self.el[k][idx]]) for k in S}
        return .5 * (1 - e['file']) + .2 * (1 - e['voice']) + .3 * (1 - e['music'])

    def delta(self, a, b):
        """ADS(b) - ADS(a) with its group-bootstrap distribution. Resamples where an EER is undefined (one class) are
        dropped and counted; an undefined point estimate is reported as None, never as p_gt_0 = 0."""
        d = np.array([self.ads(b, i) - self.ads(a, i) for i in self.idx])
        ok, point = np.isfinite(d), self.ads(b) - self.ads(a)
        if not np.isfinite(point) or not ok.any():
            return {'delta': None, 'ci95': None, 'p_gt_0': None, 'undefined_resamples': int((~ok).sum())}
        return {'delta': round(point, 5), 'ci95': np.percentile(d[ok], [2.5, 97.5]).round(5).tolist(),
                'p_gt_0': float((d[ok] > 0).mean()), 'undefined_resamples': int((~ok).sum())}


# ------------------------------------------------------------------------------------------------ DB protocol gate
def db(a):
    from component_fusion import fuse
    import decision_heads as dh
    frozen = set(json.loads((a.stacking / 'frozen_rows.json').read_text())['ids'])
    rows = {s: bank_rows(a.bank, s, frozen) for s in ('val', 'test')}
    P = {**json.loads((a.stacking / 'presence_val.json').read_text()), **json.loads((a.stacking / 'presence_test.json').read_text())}
    ft = read_ft(a.ft, a.ft_col)
    C, silent = {'AC': {}, 'RT': {}}, {}
    for n, run in (('ours', 'voice_df__mlp_long'), ('ad', 'voice_ad__resmlp_long'), ('mu', 'music__L6_12')):
        r = json.loads((a.stacking / 'runs' / f'{run}.json').read_text())
        for s in rows:   # AC: label-conditioned seed means (R2 semantics); RT: every row (runtime semantics)
            C['AC'].setdefault(n, {}).update(zip(r[f'{s}_ids'], np.mean(r[f'{s}_preds'], 0).tolist()))
            C['RT'].setdefault(n, {}).update(zip(r['every'][s]['ids'], np.mean(r['every'][s]['preds'], 0).tolist()))
            if n == 'ours':
                silent.update(zip(r['every'][s]['ids'], r['every'][s]['silent']))
    need = [r['ID'] for s in rows for r in rows[s] if not silent[r['ID']] and r['ID'] not in ft]
    if need:
        raise SystemExit(f'ft missing for {len(need)} non-silent val/test rows (RT needs every row scored), e.g. {need[:3]}')
    C['RT']['ft'] = ft
    C['AC']['ft'] = {i: ft[i] for i in C['AC']['ours'] if i in ft}   # label-conditioned like the other AC members
    val = rows['val']
    ns = [r['ID'] for r in val if not silent[r['ID']]]
    tables = {'AC': {n: dh.quantile_table([c[r['ID']] for r in val if r['ID'] in c]) for n, c in C['AC'].items()},
              'RT': {n: dh.quantile_table([c[i] for i in ns]) for n, c in C['RT'].items()}}

    def run(weights, prot, splits):
        c = C[prot]
        spec = {'weights': weights, 'tables': tables[prot]}
        files = {s: [{'ID': r['ID'], 'Pv': P[r['ID']][0], 'Pm': P[r['ID']][1], 'Fm': c['mu'].get(r['ID'], 0.),
                      'Fv': 0. if prot == 'RT' and silent[r['ID']] else fuse(spec, {n: c[n].get(r['ID'], 0.) for n in weights})}
                     for r in rows[s]] for s in splits}
        ens, _, preds = file_preds(files, val)
        return {'metrics': {s: ads_of(rows[s], preds[s]) for s in splits}, 'file_weights': ens['weights'], 'preds': preds}

    s4w = {'ours': 1, 'ad': .5}
    rep = {'n_rows': {s: len(v) for s, v in rows.items()}, 'ft_col': a.ft_col, 'ft_files': [str(p) for p in a.ft],
           'member_voice_eer_AC': {}, 'val': {}}
    for s in rows:   # each member alone on its labelled voice rows (quality of the new member at a glance)
        vi = [r for r in rows[s] if lab(r, 'VOICE_PRESENT') == 1 and lab(r, 'VOICE_FAKE') is not None]
        y = [lab(r, 'VOICE_FAKE') for r in vi]
        rep['member_voice_eer_AC'][s] = {n: round(eer(y, [C['AC'][n].get(r['ID'], 0.) for r in vi]), 5) for n in ('ours', 'ad', 'ft')}
    # ---- val only: S4 reproduction, w per option, option choice
    for prot in ('RT', 'AC'):
        rep['val'][f'S4|{prot}'] = run(s4w, prot, ('val',))['metrics']['val']
        if rep['val'][f'S4|{prot}']['ads'] != S4_EXPECTED[prot]['val']:
            raise SystemExit(f'S4 val reproduction failed ({prot}): {rep["val"][f"S4|{prot}"]}')
        for o, mk in OPTIONS.items():
            for w in WEIGHTS:
                rep['val'][f'{o}{w}|{prot}'] = run(mk(w, .5), prot, ('val',))['metrics']['val']
    best = {o: max(WEIGHTS, key=lambda w: (rep['val'][f'{o}{w}|RT']['ads'], -WEIGHTS.index(w))) for o in OPTIONS}
    option = max(OPTIONS, key=lambda o: (rep['val'][f'{o}{best[o]}|RT']['ads'], o == 'R'))
    rep['choice'] = {'w': best, 'option': option, 'weights': OPTIONS[option](best[option], .5),
                     'rule': 'w per option and then the option by val RT official ADS (FILE refit on val); ties -> smaller w, R'}
    # ---- held-out: reported for the fixed choices, never revises them
    boot = Boot(rows['test'], a.reps)
    rep['n_test_groups'], rep['test'], rep['boot_vs_S4'] = boot.n_groups, {}, {}
    for prot in ('RT', 'AC'):
        base = run(s4w, prot, ('val', 'test'))
        rep['test'][f'S4|{prot}'] = base['metrics']
        if base['metrics']['test']['ads'] != S4_EXPECTED[prot]['test']:
            raise SystemExit(f'S4 held-out reproduction failed ({prot}): {base["metrics"]}')
        assert round(boot.ads(base['preds']['test']), 5) == S4_EXPECTED[prot]['test'], 'bootstrap ADS != official ADS'
        for o, mk in OPTIONS.items():
            x = run(mk(best[o], .5), prot, ('val', 'test'))
            rep['test'][f'{o}{best[o]}|{prot}'] = x['metrics'] | {'file_weights': x['file_weights']}
            rep['boot_vs_S4'][f'{o}{best[o]}|{prot}'] = boot.delta(base['preds']['test'], x['preds']['test'])
    g = rep['boot_vs_S4'][f'{option}{best[option]}|RT']
    rep['gate'] = {'rule': 'held-out RT dADS >= +0.01 and P(dADS > 0) >= 0.95 (group bootstrap, test group_id, rng 0)',
                   'option': option, 'delta': g['delta'], 'p_gt_0': g['p_gt_0'],
                   'pass': bool(g['delta'] >= .01 and g['p_gt_0'] >= .95)}
    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / 'ft_fusion_db.json').write_text(json.dumps(rep, indent=1) + '\n')
    print(json.dumps({k: rep[k] for k in ('n_rows', 'member_voice_eer_AC', 'choice', 'test', 'boot_vs_S4', 'gate')}, indent=1))


# ------------------------------------------------------------------------------------------------ P1 package fit
def p1(a):
    import torch
    import decision_heads as dh
    import voice_heads
    from component_fusion import fuse
    from graph_scoring import official_metrics
    from localization_models import WindowMIL
    torch.set_num_threads(4)
    s4 = json.loads((a.s4 / 'component_fusion.json').read_text())
    adw = s4['voice']['weights']['ad']
    weights = {'ours': 1, 'ad': adw} if a.option == 'S4' else OPTIONS[a.option](a.w, adw)
    layers = json.loads((a.s4 / 'music_layers.json').read_text())
    val = [r for r in bank_rows(a.bank, 'val') if (a.backbone_cache / f"{r['ID']}.npz").exists()]
    heads = {k: [torch.load(a.s4 / f'{k}_{s}.pt', map_location='cpu', weights_only=True) for s in (41, 42, 43)]
             for k in ('voice', 'voice_ad', 'music')}
    fake = {x['fake_label_index'] for x in heads['voice']}
    assert len(fake) == 1 and all(x['fake_label_index'] == 0 for x in heads['voice_ad']), 'fake-label index mismatch'
    fake = fake.pop()
    models = {k: [voice_heads.load(x).eval() for x in heads[k]] for k in ('voice', 'voice_ad')}
    models['music'] = []
    for x in heads['music']:
        m = WindowMIL(x['model']['mean'], x['model']['scale'])
        m.load_state_dict(x['model'], strict=True)
        models['music'].append(m.eval())
    presence = json.loads(a.presence_head.read_text())
    if 'ft' in weights and a.ft_col != 'ft_max':
        raise SystemExit('the runtime ft member is the window max: fit the package on ft_max')
    ft = read_ft(a.ft, a.ft_col) if 'ft' in weights else {}
    miss = [r['ID'] for r in val if 'ft' in weights and r['ID'] not in ft]
    if miss:
        raise SystemExit(f'ft missing for {len(miss)} val rows, e.g. {miss[:3]}')
    sc = {'ours': {}, 'ad': {}, 'mu': {}, 'ft': ft}
    base = {}
    with torch.no_grad():   # the runtime's fp32 head math (full_bank_submission.Engine.predict)
        for r in val:
            i = r['ID']
            with np.load(a.old_cache / f'{i}.npz') as o:
                emb, silent = torch.from_numpy(o['voice_embeddings'].astype(np.float32)), bool(o['voice_silent'])
                legacy, feats = o['presence'], json.loads(str(o['presence_features']))
            with np.load(a.backbone_cache / f'{i}.npz') as d:
                ade = torch.from_numpy(d['ad_embeddings'].astype(np.float32))
                L = list(d['layers'])
                f = torch.from_numpy(d['mert_layers'][:, [L.index(k) for k in layers]].reshape(len(d['mert_layers']), -1).astype(np.float32))
            sc['ours'][i] = 0. if silent else float(np.mean([float(m(emb).softmax(-1)[:, fake].max()) for m in models['voice']]))
            sc['ad'][i] = float(np.mean([float(m(ade).softmax(-1)[:, 0].max()) for m in models['voice_ad']]))
            sc['mu'][i] = float(np.mean([float((m.head((f - m.mean) / m.scale)[0].logsumexp(0) - np.log(len(f))).sigmoid())
                                         for m in models['music']]))
            vp, mp = dh.presence(presence, feats, (float(legacy[0]), float(legacy[1])))
            base[i] = {'ID': i, 'Pv': vp, 'Pm': mp, 'Fm': sc['mu'][i]}
    ns = [r['ID'] for r in val if sc['ours'][r['ID']] > 0]
    spec = {'weights': weights, 'tables': {n: dh.quantile_table([sc[n][i] for i in ns]) for n in weights}}
    files = {'val': [{**base[r['ID']], 'Fv': 0. if sc['ours'][r['ID']] == 0. else
                      fuse(spec, {n: sc[n][r['ID']] for n in weights})} for r in val]}
    ens, ens_report, preds = file_preds(files, val)
    metrics = official_metrics(val, preds['val'])
    a.out.mkdir(parents=True, exist_ok=True)
    keep = ['voice', 'music'] + (['voice_ad'] if 'ad' in weights else [])
    lock = json.loads((a.s4 / 'candidate_lock.json').read_text())
    lock['checkpoints'] = {n: h for n, h in lock['checkpoints'].items() if n.rsplit('_', 1)[0] in keep}
    for n in lock['checkpoints']:
        shutil.copy(a.s4 / n, a.out / n)
    results = {k: v for k, v in json.loads((a.s4 / 'results.json').read_text()).items() if 'ad' in weights or k != 'voice_ad_ensemble'}
    identity = json.loads((a.s4 / 'run_identity.json').read_text()) | {
        'voice_fusion': weights, 'ft': None if 'ft' not in weights else {'files': [str(p) for p in a.ft], 'column': a.ft_col,
        'definition': 'fine-tuned AntiDeepfake-1B window-max fake prob on the voice stem (DB-protocol checkpoint)'},
        'file_ensemble': ens_report, 'fit': 'research/top15/work3/integration/fit_ft_fusion.py p1'}
    if 'ad' not in weights:
        identity.pop('ad_weight', None)
    for name, obj in (('component_fusion.json', {'voice': spec, 'music': {'weights': {'ours': 1}, 'tables': {}}, 'fitted_on': 'data v4 val'}),
                      ('file_ensemble.json', ens), ('music_layers.json', layers), ('results.json', results),
                      ('candidate_lock.json', lock), ('run_identity.json', identity), ('validation_metrics.json', metrics)):
        (a.out / name).write_text(json.dumps(obj, **({} if name in ('component_fusion.json', 'file_ensemble.json', 'music_layers.json') else {'indent': 1})) + '\n')
    report = {'option': a.option, 'weights': weights, 'val_rows': len(val), 'non_silent': len(ns), 'file_ensemble': ens_report,
              'val_official': {k: round(v, 5) for k, v in metrics['metrics'].items() if v is not None},
              'val_ads': ads_of(val, preds['val'])['ads']}
    if a.option == 'S4':   # the P1 plumbing must reproduce the shipped S4 fit
        f4 = json.loads((a.s4 / 'file_ensemble.json').read_text())
        report['s4_reproduction_max_abs'] = {
            **{f'voice_table_{n}': float(np.abs(np.subtract(spec['tables'][n], s4['voice']['tables'][n])).max()) for n in weights},
            **{f'file_table_{n}': float(np.abs(np.subtract(ens['tables'][n], f4['tables'][n])).max()) for n in f4['tables']},
            'file_weights_equal': ens['weights'] == f4['weights']}
    (a.out / 'p1_report.json').write_text(json.dumps(report, indent=1) + '\n')
    print(json.dumps(report, indent=1))
    if a.option == 'S4' and not (report['s4_reproduction_max_abs'].pop('file_weights_equal')
                                 and max(report['s4_reproduction_max_abs'].values()) < 1e-9):
        raise SystemExit('P1 plumbing does not reproduce the S4 fit')


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest='mode', required=True)
    bank = ROOT / 'results/data_v4_20260923/data_v4_results_20260923T1639Z/data/bank'
    for m in ('db', 'p1'):
        p = sub.add_parser(m)
        p.add_argument('--ft', type=Path, nargs='+', help='member CSVs with ID and --ft-col (val and/or test rows)')
        p.add_argument('--ft-col', default='ft_max')
        p.add_argument('--bank', type=Path, default=bank, help='dir with bank_trainval.csv and bank_test.csv')
        p.add_argument('--code', type=Path, nargs='*', default=[ROOT, ROOT / 'data_v4'])
        p.add_argument('--out', type=Path, required=True)
    d = sub.choices['db']
    d.add_argument('--stacking', type=Path, default=ROOT / 'research/top15/work2/stacking')
    d.add_argument('--reps', type=int, default=1000)
    q = sub.choices['p1']
    q.add_argument('--option', choices=('S4', 'R', 'A'), required=True)
    q.add_argument('--w', type=float, help='ft fusion weight (the db choice)')
    q.add_argument('--s4', type=Path, default=ROOT / '_workspace/final_s4_20260927')
    q.add_argument('--old-cache', type=Path, default=ROOT / '_workspace/cache_v4/cache_tv')
    q.add_argument('--backbone-cache', type=Path, default=ROOT / '_workspace/backbone_cache')
    q.add_argument('--presence-head', type=Path, default=ROOT / 'results/data_v4_20260923/data_v4_results_20260923T1639Z/train/presence_head.json')
    a = ap.parse_args()
    if (a.mode == 'db' or a.option != 'S4') and not a.ft or a.mode == 'p1' and a.option != 'S4' and a.w is None:
        ap.error('--ft (and --w for p1 R/A) required')
    sys.path[:0] = [str(c) for c in a.code]
    (db if a.mode == 'db' else p1)(a)
