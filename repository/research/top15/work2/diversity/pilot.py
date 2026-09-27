"""Smallest local data-diversity pilot (DB protocol, CPU): add the 155 already-featurised Music Arena rows
(_workspace/s2_extra cache_B_dev: sonauto-v2-2 126, sonauto-v3-preview 29; CC BY 4.0, rights review training_allowed
true) to TRAIN only. S2-B caches hold MERT layer 12 + DF-Arena fc5 inputs only (no MERT 6), so:
  music: MERT-12 WindowMIL (the pre-S3 music head design, 50/5) base vs +music_arena, seeds 41/42/43
  voice: DF-Arena MLP (mlp_long) + the 70 Music Arena rows with fake vocals; base = runs/voice_df__full.json (41/42)
Jamendo (training_allowed false) and AI-Hub rows are NOT used.
    python pilot.py music | voice        train, save runs/pilot_*.json (held-out preds saved, not scored)
    python pilot.py voice_s43            post-lock seed-43 replication of voice base and +arena
    python pilot.py test                 REQUIRES choice_lock.json; held-out route EER + family metrics + bootstrap
"""
import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch import nn

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import run as R  # noqa: E402  (R2 run.py copy: train, music_items, stats, Standardize, H, T, Item)

S, T, H, Item = R.S, R.T, R.H, R.Item
S2 = S.ROOT / '_workspace/s2_extra'


def extra(route, source='music_arena'):
    out = []
    for bank, cache in (('bank_B_dev.csv', 'cache_B_dev'), ('bank_B_train.csv', 'cache_B_train')):
        for r in csv.DictReader((S2 / bank).open()):
            p = S2 / cache / f"{r['ID']}.npz"
            if r['source'] != source or not p.exists():
                continue
            lab = T.head_labels(r).get(route)
            with np.load(p) as d:
                silent, samples = bool(d['voice_silent']), int(d['samples'])
                f = d['mert_windows'].astype(np.float16) if route == 'music' else d['voice_embeddings'].astype(np.float32)
                truth = d['window_truth'].astype(np.float32) if route == 'music' else np.full(len(f), np.nan, np.float32)
            if lab is None or (route == 'voice' and r['route'] != 'voice' and silent):
                continue
            out.append((Item({**r, 'split': 'train', 'route': route}, f, truth, lab, silent), samples))
    return out


def fit(group, variants, seeds, tag='', source='music_arena'):
    torch.set_num_threads(4)
    rows = S.load_rows()
    if group == 'music':
        data = {s: R.music_items(rows[s], [12]) for s in rows}
        make, route, fi, cfg = (lambda items: R.WindowMIL(*R.stats(items))), 'music', 0, {}
    else:
        old = torch.load(H.OLD['val'] / 'voice_classifier_initial.pt', map_location='cpu', weights_only=True)
        data = {s: H.items_for(rows[s], R.CACHE, [12], 'voice', dfarena=True) for s in rows}
        make = lambda items: nn.Sequential(R.Standardize(*R.stats(items)), nn.Linear(1280, 256), nn.GELU(), nn.Linear(256, 2))
        route, fi, cfg = 'voice', old['fake_label_index'], R.LONG
    add = extra(route, source)
    print(group, 'extra rows', len(add), 'fake', sum(x.label for x, _ in add), flush=True)
    for v in variants:
        tr = data['train'] + (add if v.startswith('plus') else [])
        res = {'group': group, 'variant': v, 'n': {'train': len(tr)}, 'seeds': {},
               'val_ids': [x.row['ID'] for x, _ in data['val']], 'val_labels': [int(x.label) for x, _ in data['val']],
               'test_ids': [x.row['ID'] for x, _ in data['test']], 'test_labels': [int(x.label) for x, _ in data['test']],
               'val_preds': [], 'test_preds': []}
        for seed in seeds:
            model, info = R.train(route, tr, data['val'], make, fi, seed, **cfg)
            info.pop('history')
            res['seeds'][seed] = info
            for s in ('val', 'test'):
                res[f'{s}_preds'].append(T.predict(model, [x for x, _ in data[s]], route, fi, 256, 10 ** 6, 'cpu').tolist())
            print(group, v, seed, {k: info[k] for k in ('val_eer', 'best_epoch', 'n_train')}, flush=True)
        res['val_ens_eer'] = float(R._roc_metrics(res['val_labels'], np.mean(res['val_preds'], 0).tolist())[0])
        print(group, v, 'val_ens_eer', round(res['val_ens_eer'], 5), flush=True)
        (HERE / 'runs' / f'pilot_{group}__{v}{tag}.json').write_text(json.dumps(res))


def test():
    import ads as A
    assert (HERE / 'choice_lock.json').exists()
    test_rows = {r['ID']: r for r in S.load_rows()['test']}
    out = {}
    for group, base in (('music', 'pilot_music__base'), ('voice', 'voice_df__full')):
        a = json.loads((HERE / 'runs' / f'{base}.json').read_text())
        b = json.loads((HERE / 'runs' / f'pilot_{group}__plus_arena.json').read_text())
        assert a['test_ids'] == b['test_ids']
        t = np.array(a['test_labels'])
        g = np.array([test_rows[i]['group_id'] for i in a['test_ids']])
        haim = np.array([A.FAM['haim'](test_rows[i]) for i in a['test_ids']])
        sa, sb = np.mean(a['test_preds'], 0), np.mean(b['test_preds'], 0)
        res = {}
        for name, s in (('base', sa), ('plus_arena', sb)):
            e, thr = A.eer(t, s)
            err = (s >= thr) != t.astype(bool)
            res[name] = {'val_ens_eer': round(json.loads((HERE / 'runs' / f'{base if name == "base" else "pilot_" + group + "__plus_arena"}.json').read_text())['val_ens_eer'], 5),
                         'test_route_eer': round(e, 5), 'haim_fake_miss': round(float(err[haim & (t == 1)].mean()), 4),
                         'non_haim_fake_miss': round(float(err[~haim & (t == 1)].mean()), 4), 'real_fp': round(float(err[t == 0].mean()), 4),
                         'haim_fake_auc_vs_all_reals': round(float(A.roc_auc_score(t[(haim & (t == 1)) | (t == 0)], s[(haim & (t == 1)) | (t == 0)])), 4)}
        ug = np.unique(g)
        mem = [np.flatnonzero(g == u) for u in ug]
        rng = np.random.default_rng(0)
        d = []
        for _ in range(1000):
            idx = np.concatenate([mem[j] for j in rng.integers(0, len(ug), len(ug))])
            d.append(A.eer(t[idx], sb[idx])[0] - A.eer(t[idx], sa[idx])[0])
        res['delta_test_route_eer'] = {'point': round(res['plus_arena']['test_route_eer'] - res['base']['test_route_eer'], 5),
                                       'ci95': np.percentile(d, [2.5, 97.5]).round(5).tolist(), 'p_le_0': round(float((np.array(d) <= 0).mean()), 3)}
        out[group] = res
        print(group, json.dumps(res, indent=1))
    (HERE / 'pilot_report.json').write_text(json.dumps(out, indent=1))


if __name__ == '__main__':
    if sys.argv[1] == 'test':
        test()
    elif sys.argv[1] == 'voice_s43':   # post-lock replication with a third seed (same configs; not a selection step)
        fit('voice', ('base', 'plus_arena'), (43,), '_s43')
    elif sys.argv[1] == 'music_aihub':   # amendment 1: AI-Hub 71470 (Korean traditional music, REAL; local-only) as real music
        fit('music', ('plus_aihub71470',), (41, 42, 43), source='aihub_71470')
    elif sys.argv[1] == 'music':
        fit('music', ('base', 'plus_arena'), (41, 42, 43))
    elif sys.argv[1] == 'voice':
        fit('voice', ('plus_arena',), (41, 42))


def ads3():
    """post-lock: 3-seed voice pilot (41/42 + 43 replication), per-seed held-out route EER and official ADS with the
    full music head (FILE refit on val), group bootstrap of the ADS delta."""
    import ads as A
    ld = lambda n: json.loads((HERE / 'runs' / f'{n}.json').read_text())
    base, plus = ld('voice_df__full'), ld('pilot_voice__plus_arena')
    b43, p43 = ld('pilot_voice__base_s43'), ld('pilot_voice__plus_arena_s43')
    t = np.array(base['test_labels'])
    seeds = {'base': base['test_preds'] + b43['test_preds'], 'plus_arena': plus['test_preds'] + p43['test_preds']}
    vals = {'base': base['val_preds'] + b43['val_preds'], 'plus_arena': plus['val_preds'] + p43['val_preds']}
    out = {'per_seed_test_route_eer': {k: [round(A.eer(t, np.array(s))[0], 5) for s in v] for k, v in seeds.items()},
           'per_seed_val_eer': {k: [round(float(R._roc_metrics(base['val_labels'], s)[0]), 5) for s in v] for k, v in vals.items()}}
    _, mu = A.run(HERE / 'runs' / 'music__full.json')
    test = A.rows['test']
    SC = {}
    for k in seeds:
        vd = {'val': dict(zip(base['val_ids'], np.mean(vals[k], 0).tolist())), 'test': dict(zip(base['test_ids'], np.mean(seeds[k], 0).tolist()))}
        res, preds = A.official(vd, mu, ('val', 'test'))
        out[f'official_{k}_3seed'] = res
        SC[k] = {c: np.array([p[f'{c.upper()}_FAKE_PROB'] for p in preds]) for c in ('file', 'voice', 'music')}
    lab = lambda r, c: None if r[c + '_mask'] != '1' or r[c] == '' else int(float(r[c]))
    el = {'file': np.array([lab(r, 'FILE_FAKE') is not None for r in test]),
          'voice': np.array([lab(r, 'VOICE_PRESENT') == 1 and lab(r, 'VOICE_FAKE') is not None for r in test]),
          'music': np.array([lab(r, 'MUSIC_PRESENT') == 1 and lab(r, 'MUSIC_FAKE') is not None for r in test])}
    y = {c: np.array([lab(r, c.upper() + '_FAKE') if e else -1 for r, e in zip(test, el[c])]) for c in el}
    ads = lambda k, i: .5 * (1 - A.eer(y['file'][i][el['file'][i]], SC[k]['file'][i][el['file'][i]])[0]) + \
        .2 * (1 - A.eer(y['voice'][i][el['voice'][i]], SC[k]['voice'][i][el['voice'][i]])[0]) + \
        .3 * (1 - A.eer(y['music'][i][el['music'][i]], SC[k]['music'][i][el['music'][i]])[0])
    g = np.array([r['group_id'] for r in test])
    ug = np.unique(g)
    mem = [np.flatnonzero(g == u) for u in ug]
    rng = np.random.default_rng(0)
    d = [ads('plus_arena', i) - ads('base', i) for i in (np.concatenate([mem[j] for j in rng.integers(0, len(ug), len(ug))]) for _ in range(1000))]
    out['ads_delta_3seed'] = {'point': round(out['official_plus_arena_3seed']['test']['ads'] - out['official_base_3seed']['test']['ads'], 5),
                              'ci95': np.percentile(d, [2.5, 97.5]).round(5).tolist(), 'p_le_0': round(float((np.array(d) <= 0).mean()), 3)}
    print(json.dumps(out, indent=1))
    (HERE / 'pilot_voice_3seed.json').write_text(json.dumps(out, indent=1))


if __name__ == '__main__' and sys.argv[1] == 'ads3':
    ads3()


def test_aihub():
    """held-out for amendment 1 (choice_lock_amend1.json): MERT-12 music base vs +AI-Hub 71470 real music."""
    import ads as A
    assert (HERE / 'choice_lock_amend1.json').exists()
    ld = lambda n: json.loads((HERE / 'runs' / f'{n}.json').read_text())
    a, b = ld('pilot_music__base'), ld('pilot_music__plus_aihub71470')
    assert a['test_ids'] == b['test_ids']
    rows = {r['ID']: r for r in S.load_rows()['test']}
    t = np.array(a['test_labels'])
    fam = {f: np.array([m(rows[i]) for i in a['test_ids']]) for f, m in A.FAM.items()}
    g = np.array([rows[i]['group_id'] for i in a['test_ids']])
    out = {}
    for name, r in (('base', a), ('plus_aihub71470', b)):
        s = np.mean(r['test_preds'], 0)
        e, thr = A.eer(t, s)
        err = (s >= thr) != t.astype(bool)
        out[name] = {'val_ens_eer': round(r['val_ens_eer'], 5), 'test_route_eer': round(e, 5), 'real_fp_all': round(float(err[t == 0].mean()), 4),
                     'fake_miss_all': round(float(err[t == 1].mean()), 4),
                     'real_music_family_fp': round(float(err[fam['real_music'] & (t == 0)].mean()), 4),
                     'haim_fake_miss': round(float(err[fam['haim'] & (t == 1)].mean()), 4)}
    sa, sb = np.mean(a['test_preds'], 0), np.mean(b['test_preds'], 0)
    ug = np.unique(g)
    mem = [np.flatnonzero(g == u) for u in ug]
    rng = np.random.default_rng(0)
    d = [A.eer(t[i], sb[i])[0] - A.eer(t[i], sa[i])[0] for i in (np.concatenate([mem[j] for j in rng.integers(0, len(ug), len(ug))]) for _ in range(1000))]
    out['delta_test_route_eer'] = {'point': round(out['plus_aihub71470']['test_route_eer'] - out['base']['test_route_eer'], 5),
                                   'ci95': np.percentile(d, [2.5, 97.5]).round(5).tolist()}
    print(json.dumps(out, indent=1))
    (HERE / 'pilot_aihub_report.json').write_text(json.dumps(out, indent=1))


if __name__ == '__main__' and sys.argv[1] == 'test_aihub':
    test_aihub()


def s4():
    """post-lock context: the voice pilot inside the S4-like fusion (DF MLP 3 seeds, base vs +arena) + R2 AD ResMLP
    (voice_ad__resmlp_long, 3 seeds, NOT retrained with arena rows) at weight 0.5, music = this study's full head."""
    import ads as A
    from component_fusion import fuse
    ld = lambda p: json.loads(Path(p).read_text())
    ad = ld(A.R2 / 'voice_ad__resmlp_long.json')
    adsc = {s: dict(zip(ad[f'{s}_ids'], np.mean(ad[f'{s}_preds'], 0).tolist())) for s in ('val', 'test')}
    _, mu = A.run(HERE / 'runs' / 'music__full.json')
    out = {}
    for k, runs in (('base', ('voice_df__full', 'pilot_voice__base_s43')), ('plus_arena', ('pilot_voice__plus_arena', 'pilot_voice__plus_arena_s43'))):
        rs = [ld(HERE / 'runs' / f'{n}.json') for n in runs]
        df = {s: dict(zip(rs[0][f'{s}_ids'], np.mean(rs[0][f'{s}_preds'] + rs[1][f'{s}_preds'], 0).tolist())) for s in ('val', 'test')}
        spec = {'weights': {'ours': 1, 'ad': .5}, 'tables': {
            'ours': A.dh.quantile_table([df['val'][r['ID']] for r in A.rows['val'] if r['ID'] in df['val']]),
            'ad': A.dh.quantile_table([adsc['val'][r['ID']] for r in A.rows['val'] if r['ID'] in adsc['val']])}}
        vd = {s: {r['ID']: fuse(spec, {'ours': df[s].get(r['ID'], 0.), 'ad': adsc[s].get(r['ID'], 0.)}) for r in A.rows[s]} for s in ('val', 'test')}
        out[k], _ = A.official(vd, mu, ('val', 'test'))
    out['delta_test_ads'] = round(out['plus_arena']['test']['ads'] - out['base']['test']['ads'], 5)
    print(json.dumps(out, indent=1))
    (HERE / 'pilot_voice_s4fusion.json').write_text(json.dumps(out, indent=1))


if __name__ == '__main__' and sys.argv[1] == 's4':
    s4()
