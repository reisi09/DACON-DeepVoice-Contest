"""Re-derive the ads_compare.json configurations from the saved per-ID seed-mean scores (ads_compare_scores.json), then
group-bootstrap the held-out test deltas vs baseline. Read-only: writes only lane3/s3_bootstrap.json.
FILE ensemble is refit on val exactly as ads_compare.py does; bootstrap resamples test groups with the fitted models fixed."""
import csv, json, sys
from pathlib import Path
import numpy as np
from sklearn.metrics import roc_curve
ROOT = Path('/Users/choiwuseck/Documents/ChatGPT/데이콘')
sys.path[:0] = [str(ROOT / 'data_v4'), str(ROOT)]
import train_backbone_heads as H
import decision_heads as dh
from component_fusion import fuse
from file_ensemble import fit as fit_file
from graph_scoring import official_metrics

A = ROOT / '_workspace/audit_20260927'
cache = ROOT / '_workspace/backbone_cache'
rows = {s: [r for r in csv.DictReader((H.BANK / f).open()) if r['split'] == s and (cache / f"{r['ID']}.npz").exists()]
        for s, f in (('val', 'bank_trainval.csv'), ('test', 'bank_test.csv'))}
comp = json.loads((A / 'ads_compare_scores.json').read_text())
pres = json.loads(H.PRESENCE.read_text())
P = {}
for s in rows:
    for r in rows[s]:
        with np.load(H.OLD[s] / f"{r['ID']}.npz") as x:
            P[r['ID']] = dh.presence(pres, json.loads(str(x['presence_features'])), (float(x['presence'][0]), float(x['presence'][1])))
vtab = lambda w: {'weights': {'ours': 1, 'ad': w}, 'tables': {
    'ours': dh.quantile_table([comp['voice_df'][r['ID']] for r in rows['val'] if r['ID'] in comp['voice_df']]),
    'ad': dh.quantile_table([comp['voice_ad'][r['ID']] for r in rows['val'] if r['ID'] in comp['voice_ad']])}}
CFG = {'baseline': ('music_12', None), 'm612': ('music_6_12', None), 'm612_ad0.5': ('music_6_12', .5), 'm612_ad1': ('music_6_12', 1.)}


def files(s, mkey, w):
    spec = None if w is None else vtab(w)
    out = []
    for r in rows[s]:
        vp, mp = P[r['ID']]
        fv = comp['voice_df'].get(r['ID'], 0.) if spec is None else fuse(spec, {'ours': comp['voice_df'].get(r['ID'], 0.), 'ad': comp['voice_ad'].get(r['ID'], 0.)})
        out.append({'ID': r['ID'], 'Pv': vp, 'Pm': mp, 'Fv': fv, 'Fm': comp[mkey].get(r['ID'], 0.)})
    return out


def lab(r, c):
    return None if r[c + '_mask'] != '1' or r[c] == '' else int(r[c])


test = rows['test']
groups = np.array([r['group_id'] for r in test])
elig = {'file': np.array([lab(r, 'FILE_FAKE') is not None for r in test]),
        'voice': np.array([lab(r, 'VOICE_PRESENT') == 1 and lab(r, 'VOICE_FAKE') is not None for r in test]),
        'music': np.array([lab(r, 'MUSIC_PRESENT') == 1 and lab(r, 'MUSIC_FAKE') is not None for r in test])}
y = {k: np.array([lab(r, c) if e else -1 for r, e in zip(test, elig[k])]) for k, c in (('file', 'FILE_FAKE'), ('voice', 'VOICE_FAKE'), ('music', 'MUSIC_FAKE'))}


def eer(t, s):
    fpr, tpr, _ = roc_curve(t, s, drop_intermediate=False)
    i = np.argmin(np.abs(fpr - (1 - tpr)))
    return (fpr[i] + 1 - tpr[i]) / 2


def ads(idx, sc):
    e = {k: eer(y[k][idx][elig[k][idx]], sc[k][idx][elig[k][idx]]) for k in ('file', 'voice', 'music')}
    return .5 * (1 - e['file']) + .2 * (1 - e['voice']) + .3 * (1 - e['music']), e


scores, point = {}, {}
for name, (mkey, w) in CFG.items():
    fv = files('val', mkey, w)
    keep = [i for i, r in enumerate(rows['val']) if r['FILE_FAKE_mask'] == '1']
    ens, _ = fit_file([fv[i] for i in keep], np.array([int(rows['val'][i]['FILE_FAKE']) for i in keep]))
    ft = files('test', mkey, w)
    scores[name] = {'file': np.array([dh.file_score(ens, f) for f in ft]), 'voice': np.array([f['Fv'] for f in ft]), 'music': np.array([f['Fm'] for f in ft])}
    preds = [{'ID': f['ID'], 'FILE_FAKE_PROB': s, 'VOICE_FAKE_PROB': f['Fv'], 'MUSIC_FAKE_PROB': f['Fm'], 'VOICE_PRESENT_PROB': f['Pv'], 'MUSIC_PRESENT_PROB': f['Pm']}
             for f, s in zip(ft, scores[name]['file'])]
    m = official_metrics(test, preds)['metrics']
    a, e = ads(np.arange(len(test)), scores[name])
    point[name] = {'official': round(.5 * (1 - m['file_eer']) + .2 * (1 - m['voice_eer']) + .3 * (1 - m['music_eer']), 5), 'fast': round(a, 5),
                   **{k: round(v, 4) for k, v in e.items()}}
print(point, flush=True)
ug, inv = np.unique(groups, return_inverse=True)
members = [np.flatnonzero(inv == g) for g in range(len(ug))]
rng = np.random.default_rng(0)
B = 1000
d = {n: [] for n in CFG if n != 'baseline'}
dc = {n: {k: [] for k in ('file', 'voice', 'music')} for n in d}
for _ in range(B):
    idx = np.concatenate([members[g] for g in rng.integers(0, len(ug), len(ug))])
    base, eb = ads(idx, scores['baseline'])
    for n in d:
        a, e = ads(idx, scores[n])
        d[n].append(a - base)
        for k in e:
            dc[n][k].append(e[k] - eb[k])
res = {'point': point, 'test_rows': len(test), 'test_groups': len(ug), 'B': B, 'delta_ads_vs_baseline': {}}
for n in d:
    v = np.array(d[n])
    res['delta_ads_vs_baseline'][n] = {'mean': round(v.mean(), 5), 'ci95': [round(q, 5) for q in np.percentile(v, [2.5, 97.5])], 'p_le0': float((v <= 0).mean()),
                                       'component_eer_delta_ci95': {k: [round(q, 4) for q in np.percentile(dc[n][k], [2.5, 97.5])] for k in dc[n]}}
v = np.array(d['m612_ad0.5']) - np.array(d['m612_ad1'])
res['ad0.5_minus_ad1'] = {'mean': round(v.mean(), 5), 'ci95': [round(q, 5) for q in np.percentile(v, [2.5, 97.5])]}
(Path(__file__).parent / 's3_bootstrap.json').write_text(json.dumps(res, indent=1))
print(json.dumps(res, indent=1))
