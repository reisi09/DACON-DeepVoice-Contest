"""Why did S8 lose 0.0034 on the LB while the DB held-out said +0.0043? Diagnostics on existing DB predictions only
(no training): (1) FILE-ensemble weights: S7 and S8 held-out ADS under val refit / fixed equal / fixed 0-9-1 (the S7
package weights); (2) real-music false positives by source: MUSIC FPR on MUSIC_PRESENT=1, MUSIC_FAKE=0 held-out rows at
each model's val TPR-95 threshold, S7 vs S8; (3) the same for real files (FILE_FAKE=0) with the FILE score.
  S8_RUNS=research/top15/work5/s8/runs_pulled python diag_s8.py"""
import collections
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import s8_eval as E  # noqa: E402
import decision_heads as dh  # noqa: E402
from file_ensemble import fit  # noqa: E402

S8_MUSIC = 'music_new:mert6_12+eat16+sslampt5+mixsslam5'
rows, lab = E.rows, E.F.lab


def files_for(voice, music):
    first = next(iter(voice))
    key = lambda n: 'ours' if n == first else n
    spec = {'weights': {key(n): w for n, w in voice.items()}, 'tables': {key(n): E.tables[n] for n in voice}}
    return {s: [{'ID': r['ID'], 'Pv': E.P[r['ID']][0], 'Pm': E.P[r['ID']][1], 'Fm': E.M[music][r['ID']],
                 'Fv': 0. if E.silent[r['ID']] else E.fuse(spec, {key(n): E.M[n][r['ID']] for n in voice})} for r in rows[s]]
            for s in ('val', 'test')}


def preds_with(files, weights):
    keep = [(f, r) for f, r in zip(files['val'], rows['val']) if r['FILE_FAKE_mask'] == '1']
    ens, rep = fit([f for f, _ in keep], np.array([int(r['FILE_FAKE']) for _, r in keep]))
    if weights is not None:
        ens = {**ens, 'weights': weights}
    return ens, {s: [{'ID': f['ID'], 'FILE_FAKE_PROB': dh.file_score(ens, f), 'VOICE_FAKE_PROB': f['Fv'], 'MUSIC_FAKE_PROB': f['Fm'],
                      'VOICE_PRESENT_PROB': f['Pv'], 'MUSIC_PRESENT_PROB': f['Pm']} for f in fs] for s, fs in files.items()}


report = {'file_weights': {}}
cfg = {'S7_cpu': (E.S7_VOICE, E.S7_MUSIC), 'S8': (E.S7_VOICE, S8_MUSIC)}
preds = {}
for name, (v, m) in cfg.items():
    fl = files_for(v, m)
    for label, w in (('val_refit', None), ('equal', {'max': 1, 'noisy_or': 1, 'max_nopresence': 1}), ('0/9/1', {'max': 0, 'noisy_or': 9, 'max_nopresence': 1})):
        ens, p = preds_with(fl, w)
        report['file_weights'][f'{name} {label}'] = {'weights': ens['weights'], 'val': E.F.ads_of(rows['val'], p['val']), 'test': E.F.ads_of(rows['test'], p['test'])}
        if label == 'val_refit':
            preds[name] = p
for k, v in report['file_weights'].items():
    print(f"{k:22s} weights {v['weights']}  val ADS {v['val']['ads']}  test ADS {v['test']['ads']}  test FILE {v['test']['file_eer']}")


def fpr_by_source(comp, prob, keep_row):
    out = {}
    for n in ('S7_cpu', 'S8'):
        vy = [(lab(r, f'{comp}_FAKE'), p[prob]) for r, p in zip(rows['val'], preds[n]['val'])
              if (comp == 'FILE' or lab(r, f'{comp}_PRESENT') == 1) and lab(r, f'{comp}_FAKE') is not None]
        thr = float(np.quantile([s for y, s in vy if y == 1], .05))
        by = collections.defaultdict(list)
        for r, p in zip(rows['test'], preds[n]['test']):
            if keep_row(r):
                by[r['source']].append(p[prob] >= thr)
        out[n] = {src: (len(v), round(float(np.mean(v)), 3)) for src, v in by.items()}
    return out


for comp, prob, keep in (('MUSIC', 'MUSIC_FAKE_PROB', lambda r: lab(r, 'MUSIC_PRESENT') == 1 and lab(r, 'MUSIC_FAKE') == 0),
                         ('FILE', 'FILE_FAKE_PROB', lambda r: lab(r, 'FILE_FAKE') == 0)):
    res = fpr_by_source(comp, prob, keep)
    report[f'{comp}_real_fpr_by_source'] = res
    print(f'\n{comp} real-row FPR at val TPR-95 threshold (n, S7 -> S8), held-out:')
    for src in sorted(res['S7_cpu'], key=lambda s: -res['S7_cpu'][s][0]):
        n7, f7 = res['S7_cpu'][src]; _, f8 = res['S8'][src]
        print(f'  {src:45s} n={n7:5d}  {f7:.3f} -> {f8:.3f}  ({f8 - f7:+.3f})')
(Path(__file__).resolve().parent / 'diag_s8.json').write_text(json.dumps(report, indent=1))
