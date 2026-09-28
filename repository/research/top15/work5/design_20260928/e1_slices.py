"""E1 (DB held-out, bank test 4,822) weakness map for the DB-protocol configs s7_ref (S7 recipe) and s7_new: official ADS
machinery of new_eval.py / s8_eval.py, per slice; component EER only where the slice has both classes. Local files only
(no AI-Hub rows: the e3 split is never read). Prints aggregates only."""
import os
import sys
from collections import defaultdict

import numpy as np

ROOT = '/Users/choiwuseck/Documents/ChatGPT/데이콘'
sys.argv = ['new_eval.py', f'{ROOT}/_workspace/nd_runs_e3', 's7_ref', 's7_ref', 's7_new', 'W_newB']
os.environ['NEW_ROWS'] = f'{ROOT}/research/top15/work5/newdata/rows/new_rows.csv'
os.environ['EVAL_SPLITS'] = 'test'          # no new rows are 'test' -> E2 empty; E1 = bank test
g = {'__file__': f'{ROOT}/research/top15/work5/newdata/new_eval.py', '__name__': 'x'}
src = open(g['__file__']).read()
exec(compile(src[:src.index("preds = {c: predictions(c) for c in CONFIGS}")] + "preds = {c: predictions(c) for c in CONFIGS}\n", g['__file__'], 'exec'), g)
F, test, preds, LAB = g['F'], g['test'], g['preds'], g['lab']


def eer(y, s):
    y, s = np.asarray(y), np.asarray(s)
    if len(set(y.tolist())) < 2:
        return None
    from sklearn.metrics import roc_curve
    fpr, tpr, _ = roc_curve(y, s, pos_label=1, drop_intermediate=False)
    i = np.argmin(np.abs(fpr - (1 - tpr)))
    return (fpr[i] + 1 - tpr[i]) / 2


def slice_key(r):
    kind = ('mixed' if r['VOICE_PRESENT'] == '1' and r['MUSIC_PRESENT'] == '1' else
            'voice' if r['VOICE_PRESENT'] == '1' else 'music' if r['MUSIC_PRESENT'] == '1' else 'none')
    tel = 'tel' if r.get('codec_chain', '').startswith('tel') else 'nontel'
    lang = 'ko' if (r.get('language') or '').startswith('ko') else 'other'
    return {'source': r['source'], 'type': kind, 'channel': tel, 'lang': lang}


print('E1 rows', len(test))
for c in ('s7_ref', 's7_new', 'W_newB'):
    p = preds[c]['test']
    print(c, 'E1 ADS', round(F.ads_of(test, p)['ads'], 5))
for dim in ('type', 'channel', 'lang', 'source'):
    groups = defaultdict(list)
    for k, r in enumerate(test):
        groups[slice_key(r)[dim]].append(k)
    print(f'\n== by {dim}  (FILE / VOICE / MUSIC EER; "-" = single-class or absent)')
    for name, idx in sorted(groups.items(), key=lambda x: -len(x[1])):
        line = f'  {name:28s} n={len(idx):5d}'
        for c in ('s7_ref', 's7_new', 'W_newB'):
            p = preds[c]['test']
            out = []
            for comp, fake, pres in (('FILE', 'FILE_FAKE', None), ('VOICE', 'VOICE_FAKE', 'VOICE_PRESENT'), ('MUSIC', 'MUSIC_FAKE', 'MUSIC_PRESENT')):
                ii = [k for k in idx if LAB(test[k], fake) is not None and (pres is None or LAB(test[k], pres) == 1)]
                e = eer([LAB(test[k], fake) for k in ii], [p[k][f'{comp}_FAKE_PROB'] for k in ii]) if ii else None
                out.append('-' if e is None else f'{e:.3f}')
            line += f' | {c} {"/".join(out)}'
        print(line)
