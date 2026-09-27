"""STACKING evaluator: copy of research/top15/work/r2/ads.py (exact ads_compare.py pipeline: shared presence heads, FILE
ensemble refit on val, voice CDF fusion ours 1 : ad w), extended with the add-ons of rule.json and two protocols:
  AC = R2/ads_compare semantics (label-conditioned component scores, missing -> 0)
  RT = runtime semantics (every row scored by every head, voice 0 when silent, tables on non-silent val rows)
  python ads.py val    -> val_report.json + choice_lock.json (rule.json applied; no held-out number is computed)
  python ads.py test   -> test_report.json, test_preds.json, boot.json (needs choice_lock.json)"""
import csv
import hashlib
import itertools
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr
from sklearn.metrics import roc_curve

ROOT = Path('/Users/choiwuseck/Documents/ChatGPT/데이콘')
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / 'data_v4'))
import train_backbone_heads as H  # noqa: E402
import decision_heads as dh  # noqa: E402
from component_fusion import fuse  # noqa: E402
from file_ensemble import fit as fit_file  # noqa: E402
from full_bank_evaluate import score_rows  # noqa: E402
from graph_scoring import official_metrics  # noqa: E402

HERE = Path(__file__).resolve().parent
FROZEN = set(json.loads((HERE / 'frozen_rows.json').read_text())['ids'])
RUNS = {'df': {'max': 'voice_df__mlp_long', 'lme': 'voice_df__mlp_long_lme'},
        'ad': {'max': 'voice_ad__resmlp_long', 'lme': 'voice_ad__resmlp_long_lme'},
        'mu': {'lme': 'music__L6_12', 'mean': 'music__L6_12_mean'}}
ADDONS = ('POOL_v', 'POOL_m', 'SA', 'W1', 'FILE')
FILE_OPTIONS = ('default', 'equal', 'grid', 'max', 'noisy_or', 'max_nopresence')
SA_GRID = [(b, g) for b in (.25, .5, 1, 2) for g in ('soft', 'hard')]


def load_rows():
    return {s: [r for r in csv.DictReader((H.BANK / f).open()) if r['split'] == s and r['ID'] in FROZEN]
            for s, f in (('val', 'bank_trainval.csv'), ('test', 'bank_test.csv'))}


def presence(rows, s):
    """Shared presence heads on the old cache (as ads.py); cached per split in this folder."""
    p = HERE / f'presence_{s}.json'
    if p.exists():
        return json.loads(p.read_text())
    pres, out = json.loads(H.PRESENCE.read_text()), {}
    for r, x in zip(rows, score_rows(rows, H.OLD[s], ROOT / '_keep_best_20260926/train_duration_balanced', 'cpu')):
        out[r['ID']] = list(dh.presence(pres, x['presence_features'], (x['Pv'], x['Pm'])))
    p.write_text(json.dumps(out))
    return out


def scores(name, splits):
    """AC: label-conditioned seed-mean scores (R2 semantics); RT: every-row seed means + silent flags."""
    r = json.loads((HERE / 'runs' / f'{name}.json').read_text())
    ac = {}
    for s in splits:
        ac.update(zip(r[f'{s}_ids'], np.mean(r[f'{s}_preds'], 0).tolist()))
    rt, silent = {}, {}
    for s in splits:
        e = r['every'][s]
        rt.update(zip(e['ids'], np.mean(e['preds'], 0).tolist()))
        silent.update(zip(e['ids'], e['silent']))
    return ac, rt, silent


class Evaluator:
    def __init__(self, splits):
        self.splits, rows = splits, load_rows()
        self.rows = {s: rows[s] for s in splits}
        self.P = {}
        for s in splits:
            self.P.update(presence(self.rows[s], s))
        self.C = {'AC': {}, 'RT': {}}
        self.silent = {}
        for m, v in RUNS.items():
            for pool, name in v.items():
                ac, rt, sil = scores(name, splits)
                self.C['AC'][m, pool], self.C['RT'][m, pool] = ac, rt
                if m == 'df':
                    self.silent = sil
        val = self.rows['val']
        self.file_keep = [i for i, r in enumerate(val) if r['FILE_FAKE_mask'] == '1']
        self.file_y = np.array([int(val[i]['FILE_FAKE']) for i in self.file_keep])
        self.cache = {}

    def comps(self, cfg, prot):
        C = self.C[prot]
        return C['df', cfg['pv']], C['ad', cfg['pv']], C['mu', cfg['pm']]

    def tables(self, cfg, prot):
        df, ad, mu = self.comps(cfg, prot)
        val = self.rows['val']
        if prot == 'AC':   # R2 ads.py / R3: each component's own labelled val rows
            return {n: dh.quantile_table([c[r['ID']] for r in val if r['ID'] in c]) for n, c in (('ours', df), ('ad', ad), ('mu', mu))}
        ns = [r['ID'] for r in val if not self.silent[r['ID']]]   # train_backbone_heads.final: non-silent val rows
        return {n: dh.quantile_table([c[i] for i in ns]) for n, c in (('ours', df), ('ad', ad), ('mu', mu))}

    def fv(self, i, cfg, prot, tabs, sa):
        """sa = None or (b, gate). Returns the fused VOICE score of one file."""
        df, ad, mu = self.comps(cfg, prot)
        if prot == 'RT' and self.silent[i]:
            return 0.
        vals, w = {'ours': df.get(i, 0.), 'ad': ad.get(i, 0.)}, {'ours': 1, 'ad': cfg['w']}
        if sa and (prot == 'RT' or i in mu):
            pm = self.P[i][1]
            vals['mu'], w['mu'] = mu.get(i, 0.), sa[0] * (pm if sa[1] == 'soft' else float(pm >= .5))
        return fuse({'weights': w, 'tables': tabs}, vals)

    def files(self, s, cfg, prot, tabs, sa):
        _, _, mu = self.comps(cfg, prot)
        out = []
        for r in self.rows[s]:
            vp, mp = self.P[r['ID']]
            plain = self.fv(r['ID'], cfg, prot, tabs, None)
            song = self.fv(r['ID'], cfg, prot, tabs, sa) if sa else plain
            out.append({'ID': r['ID'], 'Pv': vp, 'Pm': mp, 'Fv': plain, 'Fv_song': song, 'Fm': mu.get(r['ID'], 0.)})
        return out

    def voice_eer_val(self, fs):
        preds = [{'ID': f['ID'], 'FILE_FAKE_PROB': 0., 'VOICE_FAKE_PROB': f['Fv_song'], 'MUSIC_FAKE_PROB': f['Fm'],
                  'VOICE_PRESENT_PROB': f['Pv'], 'MUSIC_PRESENT_PROB': f['Pm']} for f in fs]
        return official_metrics(self.rows['val'], preds)['metrics']['voice_eer']

    @staticmethod
    def file_inputs(fs, coupled):
        return [{**f, 'Fv': f['Fv_song'] if coupled else f['Fv']} for f in fs]

    def fit(self, fs, option, keep=None, y=None):
        keep = self.file_keep if keep is None else keep
        y = self.file_y if y is None else y
        train = [fs[i] for i in keep]
        if option in ('default', 'grid', 'equal'):
            return fit_file(train, y, min_gain={'default': .002, 'grid': 0., 'equal': float('inf')}[option])[0]
        ens = fit_file(train, y)[0]
        return {**ens, 'weights': {n: int(n == option) for n in ens['candidates']}}

    def preds(self, s, fs, ens, coupled):
        return [{'ID': f['ID'], 'FILE_FAKE_PROB': dh.file_score(ens, self.file_inputs([f], coupled)[0]),
                 'VOICE_FAKE_PROB': f['Fv_song'], 'MUSIC_FAKE_PROB': f['Fm'], 'VOICE_PRESENT_PROB': f['Pv'],
                 'MUSIC_PRESENT_PROB': f['Pm']} for f in fs]

    def metrics(self, s, preds):
        m = official_metrics(self.rows[s], preds)['metrics']
        return {k: round(m[k], 5) for k in ('file_eer', 'voice_eer', 'music_eer')} | {
            'ads': round(.5 * (1 - m['file_eer']) + .2 * (1 - m['voice_eer']) + .3 * (1 - m['music_eer']), 5)}

    def run(self, cfg, prot, want=('val',)):
        """cfg: pv, pm, w, sa (bool), file (option). SA internals (b, gate by val voice EER; coupling by val ADS)."""
        key = (json.dumps(cfg, sort_keys=True), prot, tuple(want))
        if key in self.cache:
            return self.cache[key]
        tabs = self.tables(cfg, prot)
        if cfg['sa']:
            sel = {k: self.voice_eer_val(self.files('val', cfg, prot, tabs, k)) for k in SA_GRID}
            sa = min(SA_GRID, key=lambda k: (round(sel[k], 10), SA_GRID.index(k)))
        else:
            sel, sa = None, None
        fv = self.files('val', cfg, prot, tabs, sa)
        best = None
        for coupled in ((True, False) if sa else (True,)):
            ens = self.fit(self.file_inputs(fv, coupled), cfg['file'])
            m = self.metrics('val', self.preds('val', fv, ens, coupled))
            if best is None or m['ads'] > best[0]['ads']:
                best = (m, coupled, ens)
        out = {'cfg': cfg, 'protocol': prot, 'sa_choice': sa, 'coupled': best[1] if sa else None,
               'file_weights': best[2]['weights'], 'val': best[0],
               'sa_val_voice_eer': {str(k): round(v, 5) for k, v in sel.items()} if sel else None, '_preds': {}}
        out['_preds']['val'] = self.preds('val', fv, best[2], best[1])
        if 'test' in want:
            ft = self.files('test', cfg, prot, tabs, sa)
            out['_preds']['test'] = self.preds('test', ft, best[2], best[1])
            out['test'] = self.metrics('test', out['_preds']['test'])
        self.cache[key] = out
        return out


def folds(rows, k=5):
    return np.array([int(hashlib.sha1(r['group_id'].encode()).hexdigest(), 16) % k for r in rows])


def eer(t, s):
    fpr, tpr, thr = roc_curve(t, s, drop_intermediate=False)
    i = np.argmin(np.abs(fpr - (1 - tpr)))
    return (fpr[i] + 1 - tpr[i]) / 2, thr[i]


def lab(r, c):
    return None if r[c + '_mask'] != '1' or r[c] == '' else int(r[c])


class Boot:
    """Group bootstrap of official ADS (models and fits fixed), boot.py semantics."""
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
        self.n_groups, self.ids, self.memo = len(ug), [r['ID'] for r in rows], {}

    def vec(self, preds):
        assert [p['ID'] for p in preds] == self.ids
        return {k: np.array([p[c] for p in preds]) for k, c in (('file', 'FILE_FAKE_PROB'), ('voice', 'VOICE_FAKE_PROB'), ('music', 'MUSIC_FAKE_PROB'))}

    def ads(self, S, idx=None):
        idx = np.arange(len(self.ids)) if idx is None else idx
        e = {k: eer(self.y[k][idx][self.el[k][idx]], S[k][idx][self.el[k][idx]])[0] for k in S}
        return .5 * (1 - e['file']) + .2 * (1 - e['voice']) + .3 * (1 - e['music'])

    def draws(self, preds):
        if id(preds) not in self.memo:   # preds lists live in Evaluator.cache, so id() is stable
            S = self.vec(preds)
            self.memo[id(preds)] = (self.ads(S), np.array([self.ads(S, i) for i in self.idx]))
        return self.memo[id(preds)]

    def delta(self, preds_a, preds_b):
        """ADS(b) - ADS(a)."""
        (a, da), (b, db) = self.draws(preds_a), self.draws(preds_b)
        d = db - da
        return {'delta': round(b - a, 5), 'ci95': np.percentile(d, [2.5, 97.5]).round(5).tolist(),
                'p_le_0': float((d <= 0).mean()), '_draws': d}


def repro(split):
    """max |seed-mean score here - R2 run file| per S4 member (label-conditioned preds)."""
    out = {}
    for name in (RUNS['df']['max'], RUNS['ad']['max'], RUNS['mu']['lme']):
        a, b = (json.loads(p.read_text()) for p in (HERE / 'runs' / f'{name}.json', ROOT / 'research/top15/work/r2/runs' / f'{name}.json'))
        assert a[f'{split}_ids'] == b[f'{split}_ids'], name
        out[name] = float(np.abs(np.mean(a[f'{split}_preds'], 0) - np.mean(b[f'{split}_preds'], 0)).max())
    return out


def cfg_of(addons, file_opt):
    return {'pv': 'lme' if 'POOL_v' in addons else 'max', 'pm': 'mean' if 'POOL_m' in addons else 'lme',
            'w': 1 if 'W1' in addons else .5, 'sa': 'SA' in addons, 'file': file_opt if 'FILE' in addons else 'default'}


def name_of(addons):
    return 'S4' + ''.join(f'+{a}' for a in ADDONS if a in addons)


def file_cv(E, cfg, prot):
    """5-fold group-CV out-of-fold FILE EER on val for each FILE option (plain S4 FILE inputs)."""
    tabs = E.tables(cfg, prot)
    fv = E.files('val', cfg, prot, tabs, None)
    val = E.rows['val']
    fk = np.array(E.file_keep)
    f = folds([val[i] for i in fk])
    out, oof = {}, {}
    for opt in FILE_OPTIONS:
        sc = np.zeros(len(fk))
        for k in range(5):
            tr, te = np.flatnonzero(f != k), np.flatnonzero(f == k)
            ens = E.fit(fv, opt, keep=fk[tr].tolist(), y=E.file_y[tr])
            sc[te] = [dh.file_score(ens, fv[fk[j]]) for j in te]
        oof[opt] = sc
        out[opt] = {'oof_eer': round(eer(E.file_y, sc)[0], 5),
                    'fold_mean_eer': round(float(np.mean([eer(E.file_y[f == k], sc[f == k])[0] for k in range(5)])), 5)}
    return out, oof, fk


def val_phase():
    E = Evaluator(('val',))
    boot = Boot(E.rows['val'])
    rep = {'n_val_rows': len(E.rows['val']), 'n_val_groups': boot.n_groups}
    # reproduction: S4 under AC must equal R2 ads_report 'R2 all chosen' val (0.93845)
    s4 = {prot: E.run(cfg_of((), None), prot) for prot in ('AC', 'RT')}
    rep['S4_val'] = {p: v['val'] for p, v in s4.items()}
    rep['S4_file_weights'] = {p: v['file_weights'] for p, v in s4.items()}
    r2 = json.loads((ROOT / 'research/top15/work/r2/ads_report.json').read_text())['R2 all chosen']['val']
    rep['S4_AC_val_vs_R2'] = {'r2': r2, 'here': s4['AC']['val']}
    rep['S4_member_val_maxabs_vs_R2_runs'] = repro('val')
    # FILE option by group-CV (RT decides; AC reported)
    rep['file_cv'], file_opt = {}, None
    for prot in ('RT', 'AC'):
        cv, oof, fk = file_cv(E, cfg_of((), None), prot)
        best = min(FILE_OPTIONS, key=lambda o: (cv[o]['oof_eer'], FILE_OPTIONS.index(o)))
        # bootstrap of OOF FILE EER delta (default - best) over val groups, converted to ADS (0.5 * dEER)
        el = boot.el['file']
        pos = {i: j for j, i in enumerate(fk)}
        d = []
        for idx in boot.idx:
            ii = np.array([pos[i] for i in idx if i in pos])
            d.append(.5 * (eer(E.file_y[ii], oof['default'][ii])[0] - eer(E.file_y[ii], oof[best][ii])[0]))
        d = np.array(d)
        gain = .5 * (cv['default']['oof_eer'] - cv[best]['oof_eer'])
        rep['file_cv'][prot] = {'options': cv, 'best': best, 'ads_gain_vs_default': round(gain, 5),
                                'ci95': np.percentile(d, [2.5, 97.5]).round(5).tolist(), 'p_le_0': float((d <= 0).mean())}
        if prot == 'RT':
            file_opt = best
            file_pass = best != 'default' and gain >= .001 and (d <= 0).mean() <= .10
    assert el.sum() == len(fk)
    # single add-ons on S4 (both protocols; RT gates)
    rep['single'] = {}
    passing = []
    for a in ADDONS:
        rep['single'][a] = {}
        for prot in ('RT', 'AC'):
            if a == 'FILE':
                x = E.run(cfg_of(('FILE',), rep['file_cv'][prot]['best']), prot)
            else:
                x = E.run(cfg_of((a,), None), prot)
            b = boot.delta(s4[prot]['_preds']['val'], x['_preds']['val'])
            rep['single'][a][prot] = {'val': x['val'], 'sa_choice': x['sa_choice'], 'coupled': x['coupled'],
                                      'file_weights': x['file_weights'], 'val_boot_vs_S4': {k: v for k, v in b.items() if k != '_draws'}}
            if prot == 'RT' and a != 'FILE' and b['delta'] >= .001 and b['p_le_0'] <= .10:
                passing.append(a)
        if a == 'FILE' and file_pass:
            passing.append('FILE')
    rep['passing'] = passing
    # full factorial of all five add-ons on val (for compatibility); the rule chooses among combos of passing ones
    rep['factorial'] = {}
    for n in range(len(ADDONS) + 1):
        for combo in itertools.combinations(ADDONS, n):
            for prot in ('RT', 'AC'):
                x = E.run(cfg_of(combo, rep['file_cv'][prot]['best']), prot)
                rep['factorial'].setdefault(name_of(combo), {})[prot] = {
                    'val': x['val'], 'sa_choice': x['sa_choice'], 'coupled': x['coupled'], 'file_weights': x['file_weights']}
    cand = [c for n in range(len(passing) + 1) for c in itertools.combinations(passing, n)]
    val_ads = {c: rep['factorial'][name_of(c)]['RT']['val']['ads'] for c in cand}
    top = max(val_ads.values())
    near = [c for c in cand if val_ads[c] >= top - .0005]
    choice = min(near, key=lambda c: (len(c), -val_ads[c]))
    x = E.run(cfg_of(choice, file_opt), 'RT')
    lock = {'written': time.strftime('%Y-%m-%d %H:%M:%S KST'), 'rule': json.loads((HERE / 'rule.json').read_text()),
            'passing_single_addons': passing, 'candidates_val_ads_RT': {name_of(c): v for c, v in val_ads.items()},
            'choice': {'name': name_of(choice), 'addons': list(choice), 'cfg': cfg_of(choice, file_opt),
                       'sa_choice': x['sa_choice'], 'coupled': x['coupled'], 'file_weights': x['file_weights'],
                       'val_RT': x['val']},
            'note': 'written by ads.py val; no held-out (test) prediction or metric had been computed by this study'}
    (HERE / 'val_report.json').write_text(json.dumps(rep, indent=1))
    (HERE / 'choice_lock.json').write_text(json.dumps(lock, indent=1))
    print(json.dumps({k: rep[k] for k in ('S4_val', 'S4_AC_val_vs_R2', 'file_cv', 'passing')}, indent=1))
    for a, v in rep['single'].items():
        print(a, {p: (w['val']['ads'], w['val_boot_vs_S4']['delta'], w['val_boot_vs_S4']['p_le_0'], w['sa_choice'], w['coupled']) for p, w in v.items()})
    for k, v in rep['factorial'].items():
        print(k, v['RT']['val'], v['AC']['val']['ads'])
    print('CHOICE', lock['choice'])


def test_phase():
    lock = json.loads((HERE / 'choice_lock.json').read_text())
    val_rep = json.loads((HERE / 'val_report.json').read_text())
    E = Evaluator(('val', 'test'))
    tb = Boot(E.rows['test'])
    rep = {'choice_lock_written': lock['written'], 'choice': lock['choice']['name'], 'n_test_rows': len(E.rows['test']),
           'n_test_groups': tb.n_groups, 'factorial': {}}
    fopt = {p: val_rep['file_cv'][p]['best'] for p in ('RT', 'AC')}
    res, preds = {}, {}
    for n in range(len(ADDONS) + 1):
        for combo in itertools.combinations(ADDONS, n):
            for prot in ('RT', 'AC'):
                x = E.run(cfg_of(combo, fopt[prot]), prot, ('val', 'test'))
                res[name_of(combo), prot] = x
                rep['factorial'].setdefault(name_of(combo), {})[prot] = {'val': x['val'], 'test': x['test'], 'sa_choice': x['sa_choice'],
                                                                         'coupled': x['coupled']}
    r2 = json.loads((ROOT / 'research/top15/work/r2/ads_report.json').read_text())['R2 all chosen']['test']
    rep['S4_AC_test_vs_R2'] = {'r2': r2, 'here': res['S4', 'AC']['test']}
    rep['S4_member_test_maxabs_vs_R2_runs'] = repro('test')
    s5 = lock['choice']['name']
    rep['boot'] = {}
    draws = {}
    for prot in ('RT', 'AC'):
        base = res['S4', prot]['_preds']['test']
        for combo in [(a,) for a in ADDONS] + list(itertools.combinations(ADDONS, 2)):
            nm = name_of(combo)
            b = tb.delta(base, res[nm, prot]['_preds']['test'])
            draws[nm, prot] = b.pop('_draws')
            rep['boot'].setdefault(prot, {})[nm] = b
        b = tb.delta(base, res[s5, prot]['_preds']['test'])
        b.pop('_draws')
        rep['boot'][prot]['S5=' + s5] = b
        # pairwise interaction on held-out: d(A+B) - d(A) - d(B), bootstrap CI from the same draws
        for A, B in itertools.combinations(ADDONS, 2):
            i = draws[name_of((A, B)), prot] - draws[name_of((A,)), prot] - draws[name_of((B,)), prot]
            pt = rep['boot'][prot][name_of((A, B))]['delta'] - rep['boot'][prot][name_of((A,))]['delta'] - rep['boot'][prot][name_of((B,))]['delta']
            rep.setdefault('interaction', {}).setdefault(prot, {})[f'{A}x{B}'] = {'test': round(pt, 5), 'ci95': np.percentile(i, [2.5, 97.5]).round(5).tolist()}
        for A, B in itertools.combinations(ADDONS, 2):   # same on val (point only)
            f = rep['factorial']
            pt = f[name_of((A, B))][prot]['val']['ads'] - f[name_of((A,))][prot]['val']['ads'] - f[name_of((B,))][prot]['val']['ads'] + f['S4'][prot]['val']['ads']
            rep['interaction'][prot][f'{A}x{B}']['val'] = round(pt, 5)
    rep['diagnostics'] = diagnostics(E, res, s5)
    (HERE / 'test_report.json').write_text(json.dumps(rep, indent=1))
    (HERE / 'test_preds.json').write_text(json.dumps({f'{k[0]}|{k[1]}': v['_preds']['test'] for k, v in res.items()
                                                       if k[0] in ('S4', s5) or k[0].count('+') == 1}))
    print(json.dumps({k: rep[k] for k in ('S4_AC_test_vs_R2', 'boot', 'interaction', 'diagnostics')}, indent=1))
    for k, v in rep['factorial'].items():
        print(k, 'RT', v['RT']['val']['ads'], v['RT']['test'], 'AC', v['AC']['test']['ads'])


def diagnostics(E, res, s5):
    """Held-out: discordant-cell VOICE errors (R3 risk), FILE length dependence (R1), FILE errors by source."""
    test = E.rows['test']
    samples = {}
    for r in test:
        with np.load(H.OLD['test'] / f"{r['ID']}.npz") as d:
            samples[r['ID']] = int(d['samples'])
    out = {}
    for nm in dict.fromkeys(('S4', 'S4+SA', 'S4+POOL_v', 'S4+POOL_m', 'S4+POOL_v+POOL_m', s5)):
        p = {x['ID']: x for x in res[nm, 'RT']['_preds']['test']}
        d = {}
        vi = [r for r in test if lab(r, 'VOICE_PRESENT') == 1 and lab(r, 'VOICE_FAKE') is not None]
        yv, sv = np.array([lab(r, 'VOICE_FAKE') for r in vi]), np.array([p[r['ID']]['VOICE_FAKE_PROB'] for r in vi])
        e, thr = eer(yv, sv)
        cells = defaultdict(lambda: [0, 0])
        for r, y, s in zip(vi, yv, sv):
            mf = lab(r, 'MUSIC_FAKE') if lab(r, 'MUSIC_PRESENT') == 1 else None
            c = cells[f"voice_{'fake' if y else 'real'}|music_{'none' if mf is None else ('fake' if mf else 'real')}"]
            c[0] += 1
            c[1] += int((s >= thr) != bool(y))
        d['voice_cells_at_own_eer_thr'] = {k: [n, round(m / n, 4)] for k, (n, m) in sorted(cells.items())}
        fi = [r for r in test if lab(r, 'FILE_FAKE') is not None]
        yf, sf = np.array([lab(r, 'FILE_FAKE') for r in fi]), np.array([p[r['ID']]['FILE_FAKE_PROB'] for r in fi])
        e, thr = eer(yf, sf)
        src = np.array([r['source'] for r in fi])
        dur = np.array([samples[r['ID']] for r in fi])
        rho, wsum = 0., 0
        for s_ in np.unique(src[yf == 0]):
            m = (src == s_) & (yf == 0)
            if m.sum() >= 20 and len(np.unique(dur[m])) > 1:
                rho += spearmanr(sf[m], dur[m])[0] * m.sum()
                wsum += m.sum()
        d['file_len_rho_within_source_reals'] = round(rho / wsum, 4)
        bins = np.digitize(dur / 16000, [8, 15, 30])
        d['file_fpr_by_duration'] = {lbl: round(float((sf[(yf == 0) & (bins == b)] >= thr).mean()), 4)
                                     for b, lbl in enumerate(('<8s', '8-15s', '15-30s', '>=30s')) if ((yf == 0) & (bins == b)).any()}
        d['file_fnr_by_duration'] = {lbl: round(float((sf[(yf == 1) & (bins == b)] < thr).mean()), 4)
                                     for b, lbl in enumerate(('<8s', '8-15s', '15-30s', '>=30s')) if ((yf == 1) & (bins == b)).any()}
        cells = defaultdict(lambda: [0, 0])
        for s_, y, sc in zip(src, yf, sf):
            c = cells[f"{s_}|{'fake' if y else 'real'}"]
            c[0] += 1
            c[1] += int((sc >= thr) != bool(y))
        d['file_err_by_source_at_own_eer_thr'] = {k: [n, round(m / n, 4)] for k, (n, m) in sorted(cells.items()) if n >= 30}
        out[nm] = d
    return out


if __name__ == '__main__':
    {'val': val_phase, 'test': test_phase}[sys.argv[1]]()
