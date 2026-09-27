"""Q1: does the S3 held-out gain (ads_compare 0.86438 -> 0.87267) survive RUNTIME semantics, i.e. Fv/Fm scored on every
file (full_bank_submission.py:120-137) instead of the screen's Fv=0 / Fm=0 on files lacking that component's label?
Q2: does it hold on held-out rows outside the backbones' known training data (fusion_fit.member_seen)?
Q3: how much of the FILE gain is the val refit of the FILE ensemble rather than the components?
Same pipeline as ads_compare.py: shared presence heads, fusion tables = val quantiles of voice-labelled rows, FILE ensemble
fitted on val FILE rows (file_ensemble.fit). Group bootstrap (group_id) on test for deltas.
  python eval_semantics.py [extra component json names...]
"""
import csv, json, sys
from pathlib import Path
import numpy as np
ROOT = Path('/Users/choiwuseck/Documents/ChatGPT/데이콘')
sys.path.insert(0, str(ROOT / 'data_v4')); sys.path.insert(0, str(ROOT))
import train_backbone_heads as H
import decision_heads as dh
from component_fusion import fuse
from file_ensemble import fit as fit_file
from full_bank_evaluate import score_rows
from graph_scoring import official_metrics
from fusion_fit import member_seen

HERE = Path(__file__).resolve().parent
cache = ROOT / '_workspace/backbone_cache'
rows = {s: [r for r in csv.DictReader((H.BANK / f).open()) if r['split'] == s and (cache / f"{r['ID']}.npz").exists()]
        for s, f in (('val', 'bank_trainval.csv'), ('test', 'bank_test.csv'))}
pfile = HERE / 'presence_P.json'
if pfile.exists():
    P = json.loads(pfile.read_text())
else:
    pres, P = json.loads(H.PRESENCE.read_text()), {}
    for s in ('val', 'test'):
        for r, x in zip(rows[s], score_rows(rows[s], H.OLD[s], ROOT / '_keep_best_20260926/train_duration_balanced', 'cpu')):
            P[r['ID']] = list(dh.presence(pres, x['presence_features'], (x['Pv'], x['Pm']))) + [x['Fv'], x['Fm']]
    pfile.write_text(json.dumps(P))
silent = {}
for s in rows:
    for r in rows[s]:
        with np.load(cache / f"{r['ID']}.npz") as d:
            silent[r['ID']] = bool(d['voice_silent'])
screen = json.loads((ROOT / '_workspace/audit_20260927/ads_compare_scores.json').read_text())
names = ['voice_df', 'voice_ad', 'music_12', 'music_6_12'] + sys.argv[1:]
rt = {n: json.loads((HERE / f'comp_{n}.json').read_text())['scores'] for n in names}
# screen semantics for extra components: labelled rows only (as ads_compare / R2 ads.py)
for n in sys.argv[1:]:
    ref = 'voice_df' if 'df' in n else 'voice_ad' if 'ad' in n else 'music_6_12'
    screen[n] = {k: rt[n][k] for k in screen[ref]}


def table(comp, key):
    return dh.quantile_table([comp[key][r['ID']] for r in rows['val'] if r['ID'] in screen[key]])


def build(sem, vkey, mkey, adkey=None, w=.5):
    comp = screen if sem == 'screen' else rt
    spec = adkey and {'weights': {'ours': 1, 'ad': w}, 'tables': {'ours': table(comp, vkey), 'ad': table(comp, adkey)}}

    def one(r):
        i = r['ID']
        vp, mp = P[i][:2]
        if sem == 'screen':   # ads_compare.py:46-47 exactly (fuses every row, missing -> 0)
            fv = comp[vkey].get(i, 0.)
            if spec:
                fv = fuse(spec, {'ours': fv, 'ad': comp[adkey].get(i, 0.)})
            fm = comp[mkey].get(i, 0.)
        else:                 # full_bank_submission.py:127-136
            fv = 0. if silent[i] else comp[vkey][i]
            if spec and not silent[i]:
                fv = fuse(spec, {'ours': fv, 'ad': comp[adkey][i]})
            fm = comp[mkey][i]
        return {'ID': i, 'Pv': vp, 'Pm': mp, 'Fv': min(1., fv), 'Fm': fm}
    files = {s: [one(r) for r in rows[s]] for s in rows}
    keep = [k for k, r in enumerate(rows['val']) if r['FILE_FAKE_mask'] == '1']
    ens, rep = fit_file([files['val'][k] for k in keep], np.array([int(rows['val'][k]['FILE_FAKE']) for k in keep]))
    return files, ens, rep


def preds(files, ens):
    return [{'ID': f['ID'], 'FILE_FAKE_PROB': dh.file_score(ens, f), 'VOICE_FAKE_PROB': f['Fv'], 'MUSIC_FAKE_PROB': f['Fm'],
             'VOICE_PRESENT_PROB': f['Pv'], 'MUSIC_PRESENT_PROB': f['Pm']} for f in files]


def official(rs, pr):
    m = official_metrics(rs, pr)['metrics']
    return {k: round(m[k], 5) for k in ('file_eer', 'voice_eer', 'music_eer')} | {
        'ads': round(.5 * (1 - m['file_eer']) + .2 * (1 - m['voice_eer']) + .3 * (1 - m['music_eer']), 5)}


def eer(y, s, w):
    o = np.argsort(-s, kind='stable'); y, s, w = y[o], s[o], w[o]
    tp, fp = np.cumsum(w * y), np.cumsum(w * (1 - y))
    last = np.r_[s[1:] != s[:-1], True]
    tp, fp = tp[last], fp[last]
    fpr, tpr = np.r_[0., fp / fp[-1]], np.r_[0., tp / tp[-1]]
    k = int(np.argmin(np.abs(fpr - (1 - tpr))))
    return (fpr[k] + 1 - tpr[k]) / 2


def arrays(rs, pr):
    lab = lambda r, c: -1 if r[c] == '' else int(r[c])
    out = {}
    for name, t, c, m in (('file', 'FILE_FAKE', 'FILE_FAKE_PROB', None), ('voice', 'VOICE_FAKE', 'VOICE_FAKE_PROB', 'VOICE_PRESENT'),
                          ('music', 'MUSIC_FAKE', 'MUSIC_FAKE_PROB', 'MUSIC_PRESENT')):
        idx = [k for k, r in enumerate(rs) if lab(r, t) >= 0 and (m is None or lab(r, m) == 1)]
        out[name] = (np.array(idx), np.array([lab(rs[k], t) for k in idx], float), np.array([pr[k][c] for k in idx], float))
    return out


def fast_ads(a, w):
    e = {n: eer(y, s, w[i]) for n, (i, y, s) in a.items()}
    return .5 * (1 - e['file']) + .2 * (1 - e['voice']) + .3 * (1 - e['music'])


CONFIGS = {'baseline': ('voice_df', 'music_12', None), 'music_6_12': ('voice_df', 'music_6_12', None),
           'voicefusion_only': ('voice_df', 'music_12', 'voice_ad'), 'S3': ('voice_df', 'music_6_12', 'voice_ad')}
for n in sys.argv[1:]:
    if 'df' in n:
        CONFIGS[f'S3+{n}'] = (n, 'music_6_12', 'voice_ad')
for n in sys.argv[1:]:
    if 'ad' in n:
        CONFIGS[f'S3+{n}'] = ('voice_df', 'music_6_12', n)
        dfs = [m for m in sys.argv[1:] if 'df' in m]
        if dfs:
            CONFIGS[f'S3+{dfs[0]}+{n}'] = (dfs[0], 'music_6_12', n)
subsets = {'all': rows['test'], 'unseen': [r for r in rows['test'] if not member_seen(r)],
           'seen': [r for r in rows['test'] if member_seen(r)]}
pos = {r['ID']: k for k, r in enumerate(rows['test'])}
report, built = {'n': {k: len(v) for k, v in subsets.items()}}, {}
for sem in ('screen', 'runtime'):
    for cname, (v, m, ad) in CONFIGS.items():
        files, ens, rep = build(sem, v, m, ad)
        pr = {s: preds(files[s], ens) for s in rows}
        built[(sem, cname)] = (files, ens, pr)
        res = {'val': official(rows['val'], pr['val']), 'file_weights': ens['weights']}
        for sub, rs in subsets.items():
            res[f'test_{sub}'] = official(rs, [pr['test'][pos[r['ID']]] for r in rs])
        report[f'{sem}|{cname}'] = res
        print(sem, cname, res, flush=True)
# Q3: S3 components scored with the baseline's FILE ensemble (and the reverse)
for sem in ('screen', 'runtime'):
    fb, eb, _ = built[(sem, 'baseline')]
    fs, es, _ = built[(sem, 'S3')]
    report[f'{sem}|S3 comps + baseline FILE ens'] = official(rows['test'], preds(fs['test'], eb))
    report[f'{sem}|baseline comps + S3 FILE ens'] = official(rows['test'], preds(fb['test'], es))
    print(sem, 'cross-ens', report[f'{sem}|S3 comps + baseline FILE ens'], report[f'{sem}|baseline comps + S3 FILE ens'], flush=True)
# group bootstrap of deltas vs baseline (and vs S3 for extra components)
rng = np.random.default_rng(0)
boot = {}
for sub, rs in subsets.items():
    gid = np.unique([r['group_id'] for r in rs], return_inverse=True)[1]
    ng = gid.max() + 1
    A = {k: arrays(rs, [v[2]['test'][pos[r['ID']]] for r in rs]) for k, v in built.items()}
    pairs = [((sem, c), (sem, 'baseline')) for sem in ('screen', 'runtime') for c in CONFIGS if c != 'baseline']
    pairs += [((sem, c), (sem, 'S3')) for sem in ('screen', 'runtime') for c in CONFIGS if c.startswith('S3+')]
    pairs += [(('runtime', 'S3'), ('screen', 'S3'))]
    D = {p: [] for p in pairs}
    ones = np.ones(len(rs))
    point = {p: fast_ads(A[p[0]], ones) - fast_ads(A[p[1]], ones) for p in pairs}
    for _ in range(1000):
        w = np.bincount(rng.integers(0, ng, ng), minlength=ng)[gid].astype(float)
        cache_ads = {}
        for p in pairs:
            for k in p:
                if k not in cache_ads:
                    cache_ads[k] = fast_ads(A[k], w)
            D[p].append(cache_ads[p[0]] - cache_ads[p[1]])
    for p, d in D.items():
        d = np.array(d)
        key = f'{sub}: {"|".join(p[0])} - {"|".join(p[1])}'
        boot[key] = {'point': round(point[p], 5), 'ci95': [round(float(np.quantile(d, .025)), 5), round(float(np.quantile(d, .975)), 5)],
                     'p_le_0': round(float((d <= 0).mean()), 3), 'groups': int(ng)}
        print(key, boot[key], flush=True)
report['bootstrap'] = boot
out = HERE / ('eval_semantics' + ('_' + '_'.join(sys.argv[1:]) if sys.argv[1:] else '') + '.json')
out.write_text(json.dumps(report, indent=1))
