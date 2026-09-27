"""Official ADS for the voice-data variants (choice_lock.json) on the ads.py pipeline: shared presence heads, FILE
ensemble refitted on val per configuration, VOICE = val-CDF fusion ours(df) 1 + ad 0.5, MUSIC = S5 baseline run
(music_new__mert6_12+eat16). Rows: R2 pinned val/test rows with an EAT npz (the S5 DB rows).
Reports per variant: val / held-out official metrics on the original labels, val voice EER on rule-L-completed labels,
held-out music-context voice EER (VOICE_PRESENT=1 and MUSIC_PRESENT=1), and a group bootstrap of held-out ADS vs base.
    python eval.py L [S LS ...]          (base is always the reference)
"""
import csv
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_curve

HERE = Path(__file__).resolve().parent
ROOT = Path('/Users/choiwuseck/Documents/ChatGPT/데이콘')
BB = HERE.parent / 'work2' / 'backbones'
sys.path[:0] = [str(ROOT), str(ROOT / 'data_v4'), str(HERE)]
import train_backbone_heads as H  # noqa: E402
import decision_heads as dh  # noqa: E402
from component_fusion import fuse  # noqa: E402
from file_ensemble import fit as fit_file  # noqa: E402
from full_bank_evaluate import score_rows  # noqa: E402
from graph_scoring import official_metrics  # noqa: E402
from voice_data import complete, completes  # noqa: E402  (rule L)

NEW = ROOT / '_workspace/newbackbone_cache'
PINNED = set((BB / 'rows_r2.txt').read_text().split())
variants = ['base'] + [v for v in sys.argv[1:] if v != 'base']
rows = {s: [r for r in csv.DictReader((H.BANK / f).open()) if r['split'] == s and r['ID'] in PINNED and (NEW / f"{r['ID']}.npz").exists()]
        for s, f in (('val', 'bank_trainval.csv'), ('test', 'bank_test.csv'))}


def scores(path):
    r = json.loads(path.read_text())
    out = dict(zip(r['val_ids'], np.mean(r['val_preds'], 0).tolist()))
    out.update(zip(r['test_ids'], np.mean(r['test_preds'], 0).tolist()))
    return out


comp = {'mert': scores(BB / 'runs' / 'music_new__mert6_12+eat16.json')}
for v in variants:
    comp[f'df_{v}'], comp[f'ad_{v}'] = scores(HERE / 'runs' / f'voice_df__{v}.json'), scores(HERE / 'runs' / f'voice_ad__{v}.json')
pres = json.loads(H.PRESENCE.read_text())
P = {}
for s in ('val', 'test'):
    for r, x in zip(rows[s], score_rows(rows[s], H.OLD[s], ROOT / '_keep_best_20260926/train_duration_balanced', 'cpu')):
        P[r['ID']] = dh.presence(pres, x['presence_features'], (x['Pv'], x['Pm']))
val_table = lambda key: dh.quantile_table([comp[key][r['ID']] for r in rows['val'] if r['ID'] in comp[key]])


def component(members, weights):
    spec = {'weights': dict(zip(['ours', *members[1:]], weights)), 'tables': {'ours': val_table(members[0]), **{m: val_table(m) for m in members[1:]}}}
    out = {}
    for s in ('val', 'test'):
        for r in rows[s]:
            i = r['ID']
            values = {'ours': comp[members[0]].get(i, 0.), **{m: comp[m].get(i, 0.) for m in members[1:]}}
            out[i] = values['ours'] if len(members) == 1 else fuse(spec, values)
    return out


def predictions(v):
    voice, music = component([f'df_{v}', f'ad_{v}'], [1, .5]), component(['mert'], [1])
    files = {s: [{'ID': r['ID'], 'Pv': P[r['ID']][0], 'Pm': P[r['ID']][1], 'Fv': voice[r['ID']], 'Fm': music[r['ID']]} for r in rows[s]] for s in rows}
    keep = [i for i, r in enumerate(rows['val']) if r['FILE_FAKE_mask'] == '1']
    ens, _ = fit_file([files['val'][i] for i in keep], np.array([int(rows['val'][i]['FILE_FAKE']) for i in keep]))
    return {s: [{'ID': f['ID'], 'FILE_FAKE_PROB': dh.file_score(ens, f), 'VOICE_FAKE_PROB': f['Fv'], 'MUSIC_FAKE_PROB': f['Fm'],
                 'VOICE_PRESENT_PROB': f['Pv'], 'MUSIC_PRESENT_PROB': f['Pm']} for f in files[s]] for s in rows}


def official(row_list, preds):
    m = official_metrics(row_list, preds)['metrics']
    return {k: round(m[k], 5) for k in ('file_eer', 'voice_eer', 'music_eer')} | {
        'ads': round(.5 * (1 - m['file_eer']) + .2 * (1 - m['voice_eer']) + .3 * (1 - m['music_eer']), 5)}


def eer(t, s):
    fpr, tpr, _ = roc_curve(t, s, drop_intermediate=False)
    i = np.argmin(np.abs(fpr - (1 - tpr)))
    return (fpr[i] + 1 - tpr[i]) / 2


lab = lambda r, c: None if r[c + '_mask'] != '1' or r[c] == '' else int(r[c])
preds = {v: predictions(v) for v in variants}
val_completed = [complete(r) for r in rows['val']]
report = {'rows': {s: len(v) for s, v in rows.items()}, 'val_rows_completed': sum(completes(r) for r in rows['val']), 'variants': {}}
for v in variants:
    ctx = [(r, p) for r, p in zip(rows['test'], preds[v]['test']) if lab(r, 'VOICE_PRESENT') == 1 and lab(r, 'MUSIC_PRESENT') == 1 and lab(r, 'VOICE_FAKE') is not None]
    report['variants'][v] = {'val': official(rows['val'], preds[v]['val']), 'val_completed': official(val_completed, preds[v]['val']),
                             'test': official(rows['test'], preds[v]['test']),
                             'test_music_context_voice_eer': round(float(eer([lab(r, 'VOICE_FAKE') for r, _ in ctx], [p['VOICE_FAKE_PROB'] for _, p in ctx])), 5),
                             'n_test_music_context_voice': len(ctx)}
# group bootstrap of held-out ADS deltas vs base (ads.py semantics)
el = {'file': np.array([lab(r, 'FILE_FAKE') is not None for r in rows['test']]),
      'voice': np.array([lab(r, 'VOICE_PRESENT') == 1 and lab(r, 'VOICE_FAKE') is not None for r in rows['test']]),
      'music': np.array([lab(r, 'MUSIC_PRESENT') == 1 and lab(r, 'MUSIC_FAKE') is not None for r in rows['test']])}
y = {k: np.array([lab(r, c) if e else -1 for r, e in zip(rows['test'], el[k])]) for k, c in (('file', 'FILE_FAKE'), ('voice', 'VOICE_FAKE'), ('music', 'MUSIC_FAKE'))}
S = {v: {k: np.array([p[c] for p in preds[v]['test']]) for k, c in (('file', 'FILE_FAKE_PROB'), ('voice', 'VOICE_FAKE_PROB'), ('music', 'MUSIC_FAKE_PROB'))} for v in variants}


def ads(v, idx):
    e = {k: eer(y[k][idx][el[k][idx]], S[v][k][idx][el[k][idx]]) for k in ('file', 'voice', 'music')}
    return .5 * (1 - e['file']) + .2 * (1 - e['voice']) + .3 * (1 - e['music'])


groups = np.array([r['group_id'] for r in rows['test']])
ug = np.unique(groups)
members = {g: np.flatnonzero(groups == g) for g in ug}
rng, draws = np.random.default_rng(0), {v: [] for v in variants if v != 'base'}
for _ in range(1000):
    idx = np.concatenate([members[g] for g in rng.choice(ug, len(ug))])
    b = ads('base', idx)
    for v in draws:
        draws[v].append(ads(v, idx) - b)
report['delta_vs_base_test'] = {v: {'ads': round(report['variants'][v]['test']['ads'] - report['variants']['base']['test']['ads'], 5),
                                    'ci95': np.percentile(d, [2.5, 97.5]).round(5).tolist(), 'p_gt_0': float((np.array(d) > 0).mean()),
                                    'p_lt_-0.003': float((np.array(d) < -.003).mean())} for v, d in draws.items()}
tag = '-'.join(variants)
(HERE / f'report__{tag}.json').write_text(json.dumps(report, indent=1))
print(json.dumps(report, indent=1))
