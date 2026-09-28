"""Same held-out sets for every candidate (DB protocol, RT semantics as s8_eval / w_eval):
  E1 = DB held-out test (R2 frozen rows, 4,822; natural subset reported too), E2 = new-data held-out ('newtest' pool of
  research/top15/work5/newdata/assign_splits.py: AT-ADD eval generators + 30 % of the open-data lineage groups).
Every config = 4 members trained by the same jobs (voice_df__L, voice_ad__L, voice_new__wavlm8,
music_new__mert6_12+eat16+sslampt5) with EVERY=1; voice fusion 1/0.5/1 on CDF tables of the old val rows; FILE fixed 0/9/1
(CDF tables refit on the old val rows) for every config, so only the heads differ.
  EVAL_SPLITS=newtest (x4) | e3 (Mac: AI-Hub rows scored by score_e3.py); NEW_ROWS = the row CSVs of those splits
  python new_eval.py RUNS REF [CONFIG ...]   RUNS/CONFIG/*.json; REF = the reference config (the S7 recipe, e.g. s7_ref)"""
import csv
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / 's8'))
os.environ['S8_RUNS'] = '/nonexistent'
import s8_eval as E  # noqa: E402  (old val/test rows, presence, silent flags, S7_cpu members)

F, dh, lab = E.F, E.dh, E.F.lab
RUNS, REF = Path(sys.argv[1]), sys.argv[2]
CONFIGS = sys.argv[3:] or sorted(p.name for p in RUNS.iterdir() if p.is_dir())
FILES = {'df': 'voice_df__L.json', 'ad': 'voice_ad__L.json', 'wavlm': 'voice_new__wavlm8.json', 'music': 'music_new__mert6_12+eat16+sslampt5.json'}
S7_FILE = {'max': 0., 'noisy_or': 9., 'max_nopresence': 1.}
NEW = {r['ID']: r for f in os.environ['NEW_ROWS'].split(',') for r in csv.DictReader(open(f))}
val, test = E.rows['val'], E.rows['test']
EVAL = os.environ.get('EVAL_SPLITS', 'newtest').split(',')
M, silent_new = {}, {}
for c in CONFIGS:
    for k, f in FILES.items():
        r = json.loads((RUNS / c / f).read_text())
        name = f'{c}:{k}'
        M[name] = {}
        for s in ('val', 'test', *EVAL):
            M[name].update(zip(r['every'][s]['ids'], np.mean(r['every'][s]['preds'], 0).tolist()))
            if k == 'df' and s in EVAL:
                silent_new.update(zip(r['every'][s]['ids'], r['every'][s]['silent']))
        E.tables[name] = dh.quantile_table([M[name][i] for i in E.ns])
ids_new = sorted(set.intersection(*[set(i for i in M[f'{c}:df'] if i in NEW and NEW[i]['split'] in EVAL) for c in CONFIGS]))
P = dict(E.P)
for i in ids_new:
    with np.load(NEW[i]['npz']) as d:
        P[i] = [float(x) for x in d['presence']] if 'presence' in d.files else None
missing_presence = [i for i in ids_new if P[i] is None]
newtest = [NEW[i] for i in ids_new if P[i] is not None]   # every EVAL split pooled as E2 (x4: newtest, Mac: e3)
silent = {**E.silent, **silent_new}


def predictions(c):
    voice = {f'{c}:df': 1, f'{c}:ad': .5, f'{c}:wavlm': 1}
    first = next(iter(voice))
    key = lambda n: 'ours' if n == first else n
    spec = {'weights': {key(n): w for n, w in voice.items()}, 'tables': {key(n): E.tables[n] for n in voice}}
    files = {s: [{'ID': r['ID'], 'Pv': P[r['ID']][0], 'Pm': P[r['ID']][1], 'Fm': M[f'{c}:music'][r['ID']],
                  'Fv': 0. if silent[r['ID']] else E.fuse(spec, {key(n): M[n][r['ID']] for n in voice})}
                 for r in rows] for s, rows in (('val', val), ('test', test), ('newtest', newtest))}
    ens = F.file_preds({'val': files['val']}, val)[0]
    ens = {**ens, 'weights': dict(S7_FILE)}
    return {s: [{'ID': f['ID'], 'FILE_FAKE_PROB': dh.file_score(ens, f), 'VOICE_FAKE_PROB': f['Fv'], 'MUSIC_FAKE_PROB': f['Fm'],
                 'VOICE_PRESENT_PROB': f['Pv'], 'MUSIC_PRESENT_PROB': f['Pm']} for f in fs] for s, fs in files.items()}


def natural(r):
    s = r['source']
    return not (s.startswith('constructed_') or s == 'existing:constructed_music' or (s == 'haim' and r['generator'] in ('crossfade', 'concat')))


COMP = {'FILE': ('FILE_FAKE', None), 'VOICE': ('VOICE_FAKE', 'VOICE_PRESENT'), 'MUSIC': ('MUSIC_FAKE', 'MUSIC_PRESENT')}


def per_source(p):
    """E2 per dataset: component EER when both classes exist, else the real FPR / fake TPR at the old-val TPR-95 threshold."""
    thr = {}
    for comp, (fake, present) in COMP.items():
        ok = lambda r: lab(r, fake) is not None and (present is None or lab(r, present) == 1)
        thr[comp] = float(np.quantile([q[f'{comp}_FAKE_PROB'] for r, q in zip(val, p['val']) if ok(r) and lab(r, fake) == 1], .05))
    out = defaultdict(dict)
    for comp, (fake, present) in COMP.items():
        by = defaultdict(list)
        for r, q in zip(newtest, p['newtest']):
            if lab(r, fake) is not None and (present is None or lab(r, present) == 1):
                by[r['source']].append((lab(r, fake), q[f'{comp}_FAKE_PROB']))
        for s, v in by.items():
            y = np.array([a for a, _ in v]); x = np.array([b for _, b in v])
            if len(set(y)) == 2:
                out[s][f'{comp}_eer'] = round(float(F.eer(y, x)), 4)
            out[s][f'{comp}_real_fpr' if (y == 0).any() else f'{comp}_fake_tpr'] = round(float(np.mean(x[y == (0 if (y == 0).any() else 1)] >= thr[comp])), 4)
            out[s][f'{comp}_n'] = len(v)
    return dict(out)


def safe(f, *a):
    """A metric needing both classes returns None on a single-class split (E3: all real)."""
    try:
        return f(*a)
    except Exception:
        return None


preds = {c: predictions(c) for c in CONFIGS}
keep = [i for i, r in enumerate(test) if natural(r)]
nat = [test[i] for i in keep]
rep = {'runs': str(RUNS), 'ref': REF, 'n': {'E1': len(test), 'E1_natural': len(nat), 'E2': len(newtest), 'E2_without_presence': len(missing_presence)}}
b1, b2 = F.Boot(test), safe(F.Boot, newtest)
for c in CONFIGS:
    p = preds[c]
    rep[c] = {'val': F.ads_of(val, p['val']), 'E1': F.ads_of(test, p['test']), 'E1_natural': F.ads_of(nat, [p['test'][i] for i in keep]),
              'E2': safe(F.ads_of, newtest, p['newtest']), 'E1+E2': F.ads_of(test + newtest, p['test'] + p['newtest']), 'E2_by_source': per_source(p)}
    if c != REF:
        rep[c]['vs_ref'] = {'E1': b1.delta(preds[REF]['test'], p['test']), 'E2': b2 and safe(b2.delta, preds[REF]['newtest'], p['newtest']),
                            'E1+E2': F.Boot(test + newtest).delta(preds[REF]['test'] + preds[REF]['newtest'], p['test'] + p['newtest'])}
(RUNS / f"new_eval_{'_'.join(EVAL)}.json").write_text(json.dumps(rep, indent=1, default=float))
print(json.dumps(rep['n']))
for c in CONFIGS:
    r = rep[c]
    line = ' '.join(f"{k} {r[k]['ads']:.5f} (F {r[k]['file_eer']:.4f} V {r[k]['voice_eer']:.4f} M {r[k]['music_eer']:.4f})" for k in ('E1', 'E2', 'E1+E2') if r[k])
    print(f'{c:10s} {line}')
    if 'vs_ref' in r:
        print('   vs', REF, {k: (v['delta'], v['ci95'], v['p_gt_0']) for k, v in r['vs_ref'].items() if v})
