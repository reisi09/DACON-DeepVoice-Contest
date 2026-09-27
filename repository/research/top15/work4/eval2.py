"""Voice-data variants (choice_lock.json) under both protocols of research/top15/work4/rt_check.py:
  RT = runtime semantics (every frozen val/test row scored by every head, VOICE 0 when the voice stem is silent,
       fusion tables on non-silent val rows)             -- what the package does on the leaderboard
  AC = R2/ads.py semantics (label-conditioned member scores, missing -> 0)   -- the protocol written in the lock
MUSIC = the S5 DB music head (rt_check.py: every-row runs/music_eat__every.json; AC: backbones gate run).
val_completed = val official metrics with rule L applied to the val labels; under AC the completed rows use their own
predictions (runs '*valc*', never used for tables or early stopping) instead of the missing-score default 0 (the
defect of eval.py). base must reproduce S5 (RT 0.95455/0.89673, AC 0.95475/0.89755).
    python eval2.py L [S LS]
"""
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ST = HERE.parent / 'work2' / 'stacking'
sys.path[:0] = [str(HERE.parent / 'work2' / 'backbones'), str(HERE.parent / 'work3' / 'integration'), str(HERE)]
import run as R  # noqa: E402,F401  (puts the project root and data_v4 on sys.path)
import fit_ft_fusion as F  # noqa: E402
from component_fusion import fuse  # noqa: E402
import decision_heads as dh  # noqa: E402
from voice_data import complete, completes  # noqa: E402

variants = ['base'] + [v for v in sys.argv[1:] if v != 'base']
frozen = set(json.loads((ST / 'frozen_rows.json').read_text())['ids'])
BANK = F.ROOT / 'results/data_v4_20260923/data_v4_results_20260923T1639Z/data/bank'
rows = {s: F.bank_rows(BANK, s, frozen) for s in ('val', 'test')}
P = {**json.loads((ST / 'presence_val.json').read_text()), **json.loads((ST / 'presence_test.json').read_text())}
eat_rt = json.loads((HERE / 'runs' / 'music_eat__every.json').read_text())
eat_ac = json.loads((HERE.parent / 'work2' / 'backbones' / 'runs' / 'music_new__mert6_12+eat16.json').read_text())
MU = {'RT': {}, 'AC': {}}
for s in rows:
    MU['RT'].update(zip(eat_rt['every'][s]['ids'], np.mean(eat_rt['every'][s]['preds'], 0).tolist()))
    MU['AC'].update(zip(eat_ac[f'{s}_ids'], np.mean(eat_ac[f'{s}_preds'], 0).tolist()))


def members(v):
    C, silent, valc = {'RT': {}, 'AC': {}}, {}, {}
    for n, m in (('ours', 'df'), ('ad', 'ad')):
        r = json.loads((HERE / 'runs' / f'voice_{m}__{v}.json').read_text())
        for s in rows:
            C['AC'].setdefault(n, {}).update(zip(r[f'{s}_ids'], np.mean(r[f'{s}_preds'], 0).tolist()))
            C['RT'].setdefault(n, {}).update(zip(r['every'][s]['ids'], np.mean(r['every'][s]['preds'], 0).tolist()))
            if n == 'ours':
                silent.update(zip(r['every'][s]['ids'], r['every'][s]['silent']))
        valc[n] = dict(zip(r['valc_ids'], np.mean(r['valc_preds'], 0).tolist()))
    return C, silent, valc


def evaluate(v):
    C, silent, valc = members(v)
    val = rows['val']
    ns = [r['ID'] for r in val if not silent[r['ID']]]
    tables = {'AC': {n: dh.quantile_table([c[r['ID']] for r in val if r['ID'] in c]) for n, c in C['AC'].items()},
              'RT': {n: dh.quantile_table([c[i] for i in ns]) for n, c in C['RT'].items()}}
    tables['AC']['mu'] = dh.quantile_table([MU['AC'][r['ID']] for r in val if r['ID'] in MU['AC']])
    out = {}
    for prot in ('RT', 'AC'):
        c, spec = C[prot], {'weights': {'ours': 1, 'ad': .5}, 'tables': tables[prot]}

        def fv(i, extra=None):
            if prot == 'RT' and silent[i]:
                return 0.
            src = extra if extra and i in extra['ours'] else c
            return fuse(spec, {n: src[n].get(i, 0.) for n in ('ours', 'ad')})
        files = {s: [{'ID': r['ID'], 'Pv': P[r['ID']][0], 'Pm': P[r['ID']][1], 'Fm': MU[prot].get(r['ID'], 0.), 'Fv': fv(r['ID'])}
                     for r in rows[s]] for s in ('val', 'test')}
        ens, _, preds = F.file_preds(files, val)
        # val_completed: rule-L labels; AC fills the completed rows with their own predictions (RT already scores them)
        filled = [dict(p, VOICE_FAKE_PROB=fv(p['ID'], valc if prot == 'AC' else None)) for p in preds['val']]
        comp_rows = [complete(r) for r in val]
        out[prot] = {'val': F.ads_of(val, preds['val']), 'test': F.ads_of(rows['test'], preds['test']),
                     'val_completed': F.ads_of(comp_rows, filled), 'file_weights': ens['weights'], 'preds': preds}
    return out


res = {v: evaluate(v) for v in variants}
expect = {'RT': (.95455, .89673), 'AC': (.95475, .89755)}
for prot, (ev, et) in expect.items():
    got = (res['base'][prot]['val']['ads'], res['base'][prot]['test']['ads'])
    assert got == (ev, et), f'base does not reproduce S5 {prot}: {got}'
boot = F.Boot(rows['test'])
rep = {'rows': {s: len(x) for s, x in rows.items()}, 'val_rows_completed': sum(completes(r) for r in rows['val']), 'variants': {}, 'delta_vs_base_test': {}}
for v in variants:
    rep['variants'][v] = {p: {k: res[v][p][k] for k in ('val', 'val_completed', 'test', 'file_weights')} for p in ('RT', 'AC')}
    if v != 'base':
        rep['delta_vs_base_test'][v] = {p: boot.delta(res['base'][p]['preds']['test'], res[v][p]['preds']['test']) for p in ('RT', 'AC')}
        for p in ('RT', 'AC'):
            d = np.array([boot.ads(res[v][p]['preds']['test'], i) - boot.ads(res['base'][p]['preds']['test'], i) for i in boot.idx])
            rep['delta_vs_base_test'][v][p]['p_lt_-0.003'] = float((d < -.003).mean())
(HERE / f"report2__{'-'.join(variants)}.json").write_text(json.dumps(rep, indent=1))
print(json.dumps(rep, indent=1))
