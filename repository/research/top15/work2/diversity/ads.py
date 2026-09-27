"""Official ADS for the diversity-study heads = R2 ads.py pipeline (shared presence heads, FILE ensemble refitted on
val per configuration, graph_scoring.official_metrics) with VOICE = the DF-Arena MLP head alone (component_fusion.fuse
with only 'ours' passes it through unchanged; the AntiDeepfake member is not retrained on subsets). 2-seed means.

  python ads.py val    val metrics of every run pair (runs/voice_df__X + runs/music__X); writes val_report.json
  python ads.py test   REQUIRES choice_lock.json; held-out metrics, group bootstrap (1000 reps over test group_id),
                       per-family errors; writes report.json
"""
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import subsets as S  # noqa: E402
ROOT = S.ROOT
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / 'data_v4'))
import train_backbone_heads as H  # noqa: E402
import decision_heads as dh  # noqa: E402
from file_ensemble import fit as fit_file  # noqa: E402
from full_bank_evaluate import score_rows  # noqa: E402
from graph_scoring import official_metrics  # noqa: E402

R2 = ROOT / 'research/top15/work/r2/runs'
rows = {s: v for s, v in S.load_rows().items() if s != 'train'}
FAM = {'haim': lambda r: r['source'] == 'haim',
       'real_music': lambda r: r['source'] in ('musan_music', 'constructed_song', 'constructed_mix') and r['MUSIC_FAKE'] in ('0', '0.0'),
       'fleurs_ko': lambda r: r['source'] == 'fleurs_ko', 'constructed_enhanced': lambda r: r['source'] == 'constructed_enhanced',
       'mlaad': lambda r: r['source'] == 'mlaad', 'speechfake': lambda r: r['source'] == 'speechfake'}


def presence():
    p = HERE / 'presence_cache.json'
    if not p.exists():
        pres, P = json.loads(H.PRESENCE.read_text()), {}
        for s in ('val', 'test'):
            for r, x in zip(rows[s], score_rows(rows[s], H.OLD[s], ROOT / '_keep_best_20260926/train_duration_balanced', 'cpu')):
                P[r['ID']] = dh.presence(pres, x['presence_features'], (x['Pv'], x['Pm']))
        p.write_text(json.dumps(P))
    return json.loads(p.read_text())


P = presence()


def run(path):
    r = json.loads(Path(path).read_text())
    return r, {s: dict(zip(r[f'{s}_ids'], np.mean(r[f'{s}_preds'], 0).tolist())) for s in ('val', 'test')}


def official(vd, mu, splits):
    """vd/mu: {'val': {ID: score}, 'test': {...}}; FILE refit on val; returns metrics and test predictions."""
    def files(s):
        return [{'ID': r['ID'], 'Pv': P[r['ID']][0], 'Pm': P[r['ID']][1], 'Fv': vd[s].get(r['ID'], 0.), 'Fm': mu[s].get(r['ID'], 0.)}
                for r in rows[s]]
    fv = files('val')
    keep = [i for i, r in enumerate(rows['val']) if r['FILE_FAKE_mask'] == '1']
    ens, rep = fit_file([fv[i] for i in keep], np.array([int(rows['val'][i]['FILE_FAKE']) for i in keep]))
    res, preds = {'file_weights': ens['weights']}, None
    for s in splits:
        preds = [{'ID': f['ID'], 'FILE_FAKE_PROB': dh.file_score(ens, f), 'VOICE_FAKE_PROB': f['Fv'], 'MUSIC_FAKE_PROB': f['Fm'],
                  'VOICE_PRESENT_PROB': f['Pv'], 'MUSIC_PRESENT_PROB': f['Pm']} for f in (fv if s == 'val' else files(s))]
        m = official_metrics(rows[s], preds)['metrics']
        res[s] = {k: round(m[k], 5) for k in ('file_eer', 'voice_eer', 'music_eer')} | {
            'ads': round(.5 * (1 - m['file_eer']) + .2 * (1 - m['voice_eer']) + .3 * (1 - m['music_eer']), 5)}
    return res, preds


def eer(t, s):
    fpr, tpr, thr = roc_curve(t, s, drop_intermediate=False)
    i = np.argmin(np.abs(fpr - (1 - tpr)))
    return (fpr[i] + 1 - tpr[i]) / 2, thr[i]


def names():
    return sorted(p.name.split('__', 1)[1][:-5] for p in (HERE / 'runs').glob('voice_df__*.json')
                  if (HERE / 'runs' / f'music__{p.name.split("__", 1)[1]}').exists())


def load_pair(name):
    rv, v = run(HERE / 'runs' / f'voice_df__{name}.json')
    rm, m = run(HERE / 'runs' / f'music__{name}.json')
    return rv, v, rm, m


def val_mode():
    out = {}
    for name in names():
        rv, v, rm, m = load_pair(name)
        res, _ = official(v, m, ('val',))
        out[name] = {'n_train': {'voice': rv['n']['train'], 'music': rm['n']['train']},
                     'val_route_eer': {'voice_df': round(rv['val_ens_eer'], 5), 'music': round(rm['val_ens_eer'], 5)},
                     'val_seed_eer': {'voice_df': [round(x['val_eer'], 5) for x in rv['seeds'].values()],
                                      'music': [round(x['val_eer'], 5) for x in rm['seeds'].values()]},
                     'best_epochs': {'voice_df': [x['best_epoch'] for x in rv['seeds'].values()],
                                     'music': [x['best_epoch'] for x in rm['seeds'].values()]},
                     'val_official': res['val'], 'file_weights': res['file_weights']}
        print(name, out[name]['val_route_eer'], out[name]['val_official'], flush=True)
    # reproduction of R2 (same code, same rows): full run seeds 41/42 vs R2 val-chosen runs
    rep = {}
    for g, r2 in (('voice_df', 'voice_df__mlp_long'), ('music', 'music__L6_12')):
        a, b = json.loads((HERE / 'runs' / f'{g}__full.json').read_text()), json.loads((R2 / f'{r2}.json').read_text())
        assert a['val_ids'] == b['val_ids'] and a['test_ids'] == b['test_ids']
        rep[g] = max(float(np.abs(np.array(a['val_preds'][k]) - np.array(b['val_preds'][k])).max()) for k in (0, 1))
    out['_reproduction_max_abs_val_diff_vs_R2'] = rep
    print('reproduction', rep)
    (HERE / 'val_report.json').write_text(json.dumps(out, indent=1))


def test_mode():
    lock = json.loads((HERE / 'choice_lock.json').read_text())
    test = rows['test']
    groups = np.array([r['group_id'] for r in test])
    lab = lambda r, c: None if r[c + '_mask'] != '1' or r[c] == '' else int(float(r[c]))
    el = {'file': np.array([lab(r, 'FILE_FAKE') is not None for r in test]),
          'voice': np.array([lab(r, 'VOICE_PRESENT') == 1 and lab(r, 'VOICE_FAKE') is not None for r in test]),
          'music': np.array([lab(r, 'MUSIC_PRESENT') == 1 and lab(r, 'MUSIC_FAKE') is not None for r in test])}
    y = {k: np.array([lab(r, c) if e else -1 for r, e in zip(test, el[k])])
         for k, c in (('file', 'FILE_FAKE'), ('voice', 'VOICE_FAKE'), ('music', 'MUSIC_FAKE'))}
    rowfam = {r['ID']: [f for f, m in FAM.items() if m(r)] for r in test}
    src = {r['ID']: r['source'] for r in test}
    report, SC, ERR, ITEMS = {'lock_written': lock['written'], 'configs': {}}, {}, defaultdict(dict), {}
    for name in names():
        rv, v, rm, m = load_pair(name)
        res, preds = official(v, m, ('val', 'test'))
        SC[name] = {k: np.array([p[c] for p in preds]) for k, c in (('file', 'FILE_FAKE_PROB'), ('voice', 'VOICE_FAKE_PROB'), ('music', 'MUSIC_FAKE_PROB'))}
        fam = {}
        for route, r in (('voice', rv), ('music', rm)):   # per-family error at the head's own held-out route-EER threshold
            t, s = np.array(r['test_labels']), np.mean(r['test_preds'], 0)
            e, thr = eer(t, s)
            fam[f'{route}_route_eer'] = round(e, 5)
            assert ITEMS.setdefault(route, r['test_ids']) == r['test_ids']
            ERR[name][route] = (s >= thr) != t.astype(bool)
            cells = defaultdict(lambda: [0, 0])
            for i, lbl, sc in zip(r['test_ids'], t, s):
                for key in rowfam[i] + ['ALL']:
                    c = cells[(key, int(lbl))]
                    c[0] += 1
                    c[1] += int((sc >= thr) != bool(lbl))
            fam[route] = {f'{k}|{"fake" if l else "real"}': [n, round(err / n, 4)] for (k, l), (n, err) in sorted(cells.items()) if n >= 20}
            for f in FAM:   # threshold-free: family (one label) vs all held-out rows of the other label on this route
                for l in (0, 1):
                    idx = [j for j, i in enumerate(r['test_ids']) if (f in rowfam[i] and t[j] == l) or t[j] != l]
                    if sum(1 for j in idx if t[j] == l) >= 20:
                        a = roc_auc_score(t[idx], s[idx])
                        fam[route][f'{f}|{"fake" if l else "real"}|auc_vs_rest'] = round(a, 4)
        report['configs'][name] = {'official': res, 'route': fam, 'n_train': {'voice': rv['n']['train'], 'music': rm['n']['train']}}
        print(name, res['test'], fam['voice_route_eer'], fam['music_route_eer'], flush=True)
    # group bootstrap of held-out ADS deltas (models fixed; test groups resampled)
    ug = np.unique(groups)
    members = {g: np.flatnonzero(groups == g) for g in ug}

    def ads(n, idx):
        e = {k: eer(y[k][idx][el[k][idx]], SC[n][k][idx][el[k][idx]])[0] for k in ('file', 'voice', 'music')}
        return np.array([.5 * (1 - e['file']) + .2 * (1 - e['voice']) + .3 * (1 - e['music']), e['file'], e['voice'], e['music']])
    contrasts = {}
    for f in (25, 50, 75):
        ds = [n for n in SC if n.startswith(f'D{f}_')]
        contrasts[f'Q{f}-full'] = ([f'Q{f}_d0'], ['full'])
        contrasts[f'Dmean{f}-full'] = (ds, ['full'])
        contrasts[f'Q{f}-Dmean{f}'] = ([f'Q{f}_d0'], ds)
    for fm in FAM:
        if f'LOFO_{fm}' in SC:
            contrasts[f'LOFO_{fm}-full'] = ([f'LOFO_{fm}'], ['full'])
            contrasts[f'LOFO_{fm}-CTRL_{fm}'] = ([f'LOFO_{fm}'], [f'CTRL_{fm}'])
    contrasts = {k: v for k, v in contrasts.items() if all(n in SC for n in v[0] + v[1])}
    rng = np.random.default_rng(0)
    draws = defaultdict(list)
    point = {n: ads(n, np.arange(len(test))) for n in SC}
    for _ in range(1000):
        idx = np.concatenate([members[g] for g in rng.choice(ug, len(ug))])
        cache = {}
        get = lambda n: cache.setdefault(n, ads(n, idx))
        for k, (a, b) in contrasts.items():
            draws[k].append(np.mean([get(n) for n in a], 0) - np.mean([get(n) for n in b], 0))
    boot = {}
    for k, (a, b) in contrasts.items():
        d = np.array(draws[k])
        pt = np.mean([point[n] for n in a], 0) - np.mean([point[n] for n in b], 0)
        boot[k] = {m: {'delta': round(float(pt[j]), 5), 'ci95': np.percentile(d[:, j], [2.5, 97.5]).round(5).tolist(),
                       'p_le_0': round(float((d[:, j] <= 0).mean()), 3)} for j, m in enumerate(('ads', 'file_eer', 'voice_eer', 'music_eer'))}
        print(k, boot[k]['ads'], flush=True)
    # paired family-error deltas (fixed thresholds), resampling the family's own test groups
    gid = {r['ID']: r['group_id'] for r in test}
    fb = {}
    for k, (a, b) in contrasts.items():
        if not k.startswith('LOFO'):
            continue
        for route, ids in ITEMS.items():
            lbl = dict(zip(ids, json.loads((HERE / 'runs' / f'{"voice_df" if route == "voice" else "music"}__full.json').read_text())['test_labels']))
            for f in FAM:
                for l in (0, 1):
                    idx = np.array([j for j, i in enumerate(ids) if f in rowfam[i] and lbl[i] == l])
                    if len(idx) < 20:
                        continue
                    g = np.array([gid[ids[j]] for j in idx])
                    u = np.unique(g)
                    mem = [idx[g == x] for x in u]
                    diff = ERR[a[0]][route][idx].mean() - ERR[b[0]][route][idx].mean()
                    ds = [ERR[a[0]][route][ii].mean() - ERR[b[0]][route][ii].mean()
                          for ii in (np.concatenate([mem[j] for j in rng.integers(0, len(u), len(u))]) for _ in range(1000))]
                    fb[f'{k}|{route}|{f}|{"fake" if l else "real"}'] = {'n': len(idx), 'groups': len(u), 'delta_err': round(float(diff), 4),
                                                                        'ci95': np.percentile(ds, [2.5, 97.5]).round(4).tolist()}
    report['family_bootstrap'] = fb
    report['bootstrap'] = {'n_groups': len(ug), 'reps': 1000, 'contrasts': boot}
    (HERE / 'report.json').write_text(json.dumps(report, indent=1))


if __name__ == '__main__':
    {'val': val_mode, 'test': test_mode}[sys.argv[1]]()
