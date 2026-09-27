"""Generalisation audit: held-out metrics on generators/sources never seen in training, per DB-protocol candidate, with
runtime semantics (RT: every row scored, VOICE 0 when silent, fusion tables on non-silent val rows, FILE refit on val).
Subsets of the held-out (frozen R2 rows): every real row + only the fakes whose generator never occurs among the train
fakes ('unseen-gen') vs + only the fakes with a train generator ('seen-gen'); plus rows whose source never occurs in train.
Candidates: S4 (stacking voice + MERT music), S5 (+ EAT music), work4 voice variants L / S / LS on S5 music, and the
rejected fine-tune ftC1 (option R: ours 1 + ft 2, work3 gate choice) on S4 music.
    python unseen.py
"""
import csv
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ST = HERE.parent / 'work2' / 'stacking'
sys.path[:0] = [str(HERE.parent / 'work2' / 'backbones'), str(HERE.parent / 'work3' / 'integration')]
import run as R  # noqa: E402,F401
import fit_ft_fusion as F  # noqa: E402
from component_fusion import fuse  # noqa: E402
import decision_heads as dh  # noqa: E402

frozen = set(json.loads((ST / 'frozen_rows.json').read_text())['ids'])
BANK = F.ROOT / 'results/data_v4_20260923/data_v4_results_20260923T1639Z/data/bank'
rows = {s: F.bank_rows(BANK, s, frozen) for s in ('val', 'test')}
train = [r for r in csv.DictReader((BANK / 'bank_trainval.csv').open()) if r['split'] == 'train' and r['ID'] in frozen]   # F.bank_rows reads bank_test.csv for any split but 'val'
assert len(train) > 20000, len(train)
P = {**json.loads((ST / 'presence_val.json').read_text()), **json.loads((ST / 'presence_test.json').read_text())}
M, silent = {}, {}


def load_every(name, path, keep_silent=False):
    r = json.loads(path.read_text())
    M[name] = {}
    for s in ('val', 'test'):
        M[name].update(zip(r['every'][s]['ids'], np.mean(r['every'][s]['preds'], 0).tolist()))
        if keep_silent:
            silent.update(zip(r['every'][s]['ids'], r['every'][s]['silent']))


load_every('df_s4', ST / 'runs' / 'voice_df__mlp_long.json', keep_silent=True)
load_every('ad_s4', ST / 'runs' / 'voice_ad__resmlp_long.json')
load_every('mert', ST / 'runs' / 'music__L6_12.json')
load_every('eat', HERE / 'runs' / 'music_eat__every.json')
for v in ('L', 'S', 'LS'):
    load_every(f'df_{v}', HERE / 'runs' / f'voice_df__{v}.json')
    load_every(f'ad_{v}', HERE / 'runs' / f'voice_ad__{v}.json')
M['ft'] = {r['ID']: float(r['ft_max']) for r in csv.DictReader((F.ROOT / '_workspace/ftC1/ft_members.csv').open())}
CONFIGS = {'S4': ({'df_s4': 1, 'ad_s4': .5}, 'mert'), 'ftC1 (rejected)': ({'df_s4': 1, 'ft': 2}, 'mert'),
           'S5': ({'df_s4': 1, 'ad_s4': .5}, 'eat'), 'S5+L (S6)': ({'df_L': 1, 'ad_L': .5}, 'eat'),
           'S5+S': ({'df_S': 1, 'ad_S': .5}, 'eat'), 'S5+LS': ({'df_LS': 1, 'ad_LS': .5}, 'eat')}
val = rows['val']
ns = [r['ID'] for r in val if not silent[r['ID']]]
tables = {n: dh.quantile_table([c[i] for i in ns]) for n, c in M.items()}


def predictions(voice, music):
    first = next(iter(voice))
    spec = {'weights': {'ours' if n == first else n: w for n, w in voice.items()},
            'tables': {('ours' if n == first else n): tables[n] for n in voice}}
    files = {s: [{'ID': r['ID'], 'Pv': P[r['ID']][0], 'Pm': P[r['ID']][1], 'Fm': M[music][r['ID']],
                  'Fv': 0. if silent[r['ID']] else fuse(spec, {('ours' if n == first else n): M[n][r['ID']] for n in voice})}
                 for r in rows[s]] for s in ('val', 'test')}
    return F.file_preds(files, val)[2]['test']


lab = F.lab
gen = lambda r: r['generator'].strip().lower()
train_gens = {gen(r) for r in train if r['FILE_FAKE'] == '1'}
train_sources = {r['source'] for r in train}
test = rows['test']
comp = {'file': ('FILE_FAKE', 'FILE_FAKE_PROB', None), 'voice': ('VOICE_FAKE', 'VOICE_FAKE_PROB', 'VOICE_PRESENT'),
        'music': ('MUSIC_FAKE', 'MUSIC_FAKE_PROB', 'MUSIC_PRESENT')}


def subset_eer(preds, c, keep):
    t, sc, pres = comp[c]
    y, s = [], []
    for r, p in zip(test, preds, strict=True):
        if (pres and lab(r, pres) != 1) or lab(r, t) is None or not keep(r, lab(r, t)):
            continue
        y.append(lab(r, t)); s.append(p[sc])
    return (float(F.eer(y, s)), len(y), int(sum(y))) if len(set(y)) == 2 else (None, len(y), int(sum(y)))


SUBSETS = {'all': lambda r, y: True,
           'unseen-gen fakes + all reals': lambda r, y: y == 0 or gen(r) not in train_gens,
           'seen-gen fakes + all reals': lambda r, y: y == 0 or gen(r) in train_gens,
           'unseen-source rows + seen-source fakes': lambda r, y: r['source'] not in train_sources or y == 1}
report = {'train_fake_generators': len(train_gens), 'unseen_generators_in_test': sorted({gen(r) for r in test if r['FILE_FAKE'] == '1'} - train_gens),
          'configs': {}}
for name, (voice, music) in CONFIGS.items():
    preds = predictions(voice, music)
    report['configs'][name] = {}
    for sub, keep in SUBSETS.items():
        e = {c: subset_eer(preds, c, keep) for c in comp}
        ads = (.5 * (1 - e['file'][0]) + .2 * (1 - e['voice'][0]) + .3 * (1 - e['music'][0])) if all(e[c][0] is not None for c in e) else None
        report['configs'][name][sub] = {'ads': ads and round(ads, 5), **{c: (e[c][0] and round(e[c][0], 5), *e[c][1:]) for c in e}}
    print(name, {sub: report['configs'][name][sub]['ads'] for sub in SUBSETS}, flush=True)
expected = {'S4': .88457, 'ftC1 (rejected)': .88057, 'S5': .89673, 'S5+L (S6)': .89877, 'S5+S': .89736, 'S5+LS': .89933}
for n, v in expected.items():
    assert report['configs'][n]['all']['ads'] == v, (n, report['configs'][n]['all']['ads'], v)
(HERE / 'unseen_report.json').write_text(json.dumps(report, indent=1))
print(json.dumps(report, indent=1))
