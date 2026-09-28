"""DB evaluation of the reweighting candidates (research/top15/work5/FINAL_PLAN_20260928.md gates), RT semantics as s8_eval.
  python w_eval.py [RUNS]   # RUNS/{s7,C,W}/ with voice_df__L, voice_ad__L, voice_new__wavlm8, music_new__mert6_12+eat16+sslampt5
Reference S7_cpu = work4/runs (CPU members of the submitted S7 recipe); s7 = the GPU control of the same recipe.
FILE coefficients fixed at S7's 0/9/1 (primary, CDF tables refit on val); the refit FILE is reported only."""
import json
import os
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / 's8'))
os.environ['S8_RUNS'] = '/nonexistent'   # s8_eval then loads only the S7_cpu members, rows, presence and silent flags
import s8_eval as E  # noqa: E402

F, dh, lab = E.F, E.dh, E.F.lab
RUNS = Path(sys.argv[1]) if len(sys.argv) > 1 else HERE / 'runs'
FILES = {'df': 'voice_df__L.json', 'ad': 'voice_ad__L.json', 'wavlm': 'voice_new__wavlm8.json',
         'music': 'music_new__mert6_12+eat16+sslampt5.json'}
S7_FILE = {'max': 0., 'noisy_or': 9., 'max_nopresence': 1.}
configs = {'S7_cpu': ({'df_L': 1, 'ad_L': .5, 'wavlm8_cpu': 1}, 'music_s7_cpu')}
for scheme in ('s7', 'C', 'W'):
    if all((RUNS / scheme / f).exists() for f in FILES.values()):
        for k, f in FILES.items():
            E.load_every(f'{scheme}:{k}', RUNS / scheme / f)
            E.tables[f'{scheme}:{k}'] = dh.quantile_table([E.M[f'{scheme}:{k}'][i] for i in E.ns])
        configs[scheme] = ({f'{scheme}:df': 1, f'{scheme}:ad': .5, f'{scheme}:wavlm': 1}, f'{scheme}:music')
    else:
        print('skipping incomplete scheme', scheme)
val, test = E.rows['val'], E.rows['test']


def predictions(voice, music, file_weights):
    first = next(iter(voice))
    key = lambda n: 'ours' if n == first else n
    spec = {'weights': {key(n): w for n, w in voice.items()}, 'tables': {key(n): E.tables[n] for n in voice}}
    files = {s: [{'ID': r['ID'], 'Pv': E.P[r['ID']][0], 'Pm': E.P[r['ID']][1], 'Fm': E.M[music][r['ID']],
                  'Fv': 0. if E.silent[r['ID']] else E.fuse(spec, {key(n): E.M[n][r['ID']] for n in voice})}
                 for r in E.rows[s]] for s in ('val', 'test')}
    ens = F.file_preds(files, val)[0]
    if file_weights:
        ens = {**ens, 'weights': dict(file_weights)}
    return {s: [{'ID': f['ID'], 'FILE_FAKE_PROB': dh.file_score(ens, f), 'VOICE_FAKE_PROB': f['Fv'], 'MUSIC_FAKE_PROB': f['Fm'],
                 'VOICE_PRESENT_PROB': f['Pv'], 'MUSIC_PRESENT_PROB': f['Pm']} for f in fs] for s, fs in files.items()}, ens['weights']


def natural(r):
    s = r['source']
    return not (s.startswith('constructed_') or s == 'existing:constructed_music' or (s == 'haim' and r['generator'] in ('crossfade', 'concat')))


keep = [i for i, r in enumerate(test) if natural(r)]
nat = [test[i] for i in keep]
COMP = {'FILE': ('FILE_FAKE', None), 'VOICE': ('VOICE_FAKE', 'VOICE_PRESENT'), 'MUSIC': ('MUSIC_FAKE', 'MUSIC_PRESENT')}


def real_fpr(p):
    """FPR per (component, natural test real source with >= 50 rows) at the val TPR-95% threshold of that component."""
    out = {}
    for comp, (fake, present) in COMP.items():
        ok = lambda r: lab(r, fake) is not None and (present is None or lab(r, present) == 1)
        thr = float(np.quantile([q[f'{comp}_FAKE_PROB'] for r, q in zip(val, p['val']) if ok(r) and lab(r, fake) == 1], .05))
        by = {}
        for r, q in zip(test, p['test']):
            if natural(r) and ok(r) and lab(r, fake) == 0:
                by.setdefault(r['source'], []).append(q[f'{comp}_FAKE_PROB'] >= thr)
        out[comp] = {s: round(float(np.mean(v)), 4) for s, v in sorted(by.items()) if len(v) >= 50}
    return out


rep, preds = {'runs': str(RUNS), 'n_test': len(test), 'n_natural': len(nat)}, {}
for name, (voice, music) in configs.items():
    p, w = predictions(voice, music, S7_FILE)
    q, wr = predictions(voice, music, None)
    preds[name] = p
    rep[name] = {'val': F.ads_of(val, p['val']), 'test': F.ads_of(test, p['test']),
                 'natural': F.ads_of(nat, [p['test'][i] for i in keep]), 'real_fpr': real_fpr(p),
                 'refit_file': {'weights': wr, 'val': F.ads_of(val, q['val']), 'test': F.ads_of(test, q['test'])}}
boot, boot_nat = F.Boot(test), F.Boot(nat)
for name in [n for n in configs if n != 'S7_cpu']:
    r, b = rep[name], rep['S7_cpu']
    r['boot_vs_S7_cpu'] = {'test': boot.delta(preds['S7_cpu']['test'], preds[name]['test']),
                           'natural': boot_nat.delta([preds['S7_cpu']['test'][i] for i in keep], [preds[name]['test'][i] for i in keep])}
    if name != 's7' and 's7' in preds:
        r['boot_vs_s7_gpu'] = {'test': boot.delta(preds['s7']['test'], preds[name]['test']),
                               'natural': boot_nat.delta([preds['s7']['test'][i] for i in keep], [preds[name]['test'][i] for i in keep])}
    fpr_up = {f'{c}:{s}': round(v - b['real_fpr'][c].get(s, v), 4) for c in COMP for s, v in r['real_fpr'][c].items()}
    r['gates'] = {'natural_dads>=-.005': r['natural']['ads'] - b['natural']['ads'] >= -.005,
                  'natural_component_eer_worse<=.02': all(r['natural'][k] - b['natural'][k] <= .02 for k in ('file_eer', 'voice_eer', 'music_eer')),
                  'natural_real_fpr_up<=.05': all(v <= .05 for v in fpr_up.values())}
    r['worst_fpr_up'] = sorted(fpr_up.items(), key=lambda t: -t[1])[:5]
    r['pass'] = all(r['gates'].values())
if 'W' in rep and 'C' in rep:
    rep['order'] = 'C' if rep['C']['natural']['ads'] >= rep['W']['natural']['ads'] + .003 else 'W'
(HERE / 'w_eval.json').write_text(json.dumps(rep, indent=1, default=float))
for name in configs:
    r = rep[name]
    print(f"{name:7s} val {r['val']['ads']:.5f} test {r['test']['ads']:.5f} nat {r['natural']['ads']:.5f} "
          f"(F {r['natural']['file_eer']:.4f} V {r['natural']['voice_eer']:.4f} M {r['natural']['music_eer']:.4f}) "
          f"refitFILE test {r['refit_file']['test']['ads']:.5f} {r['refit_file']['weights']}")
    for k in ('boot_vs_S7_cpu', 'boot_vs_s7_gpu', 'gates', 'worst_fpr_up', 'pass'):
        if k in r:
            print('   ', k, json.dumps(r[k], default=float))
print('order', rep.get('order'))
