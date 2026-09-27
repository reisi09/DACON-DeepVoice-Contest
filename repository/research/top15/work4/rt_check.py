"""Audit: the S5 music gate (work2/backbones ads.py, AC semantics: missing member score -> 0) re-scored with runtime
semantics (RT: every val/test row scored by every head, VOICE 0 when the voice stem is silent, fusion tables on
non-silent val rows), exactly as research/top15/work3/integration/fit_ft_fusion.py db() does for the S4 baseline.
  1. retrain the DB-protocol S5 music head (music_new mert6_12+eat16, seeds 41/42/43; deterministic) and score every
     frozen val/test row; its labelled-row predictions must equal runs/music_new__mert6_12+eat16.json
  2. S4 = stacking RT/AC members (voice_df mlp_long, voice_ad resmlp_long @0.5, music__L6_12): must reproduce
     RT 0.93773/0.88457 and AC 0.93845/0.88643; S5 = same voice, music replaced; AC must reproduce 0.95475/0.89755
  3. held-out RT and AC deltas S5 - S4 with the group bootstrap (test group_id, 1000 reps, rng 0)
    python rt_check.py
"""
import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
ROOT = Path('/Users/choiwuseck/Documents/ChatGPT/데이콘')
ST = HERE.parent / 'work2' / 'stacking'
sys.path[:0] = [str(HERE.parent / 'work2' / 'backbones'), str(HERE.parent / 'work3' / 'integration')]
import run as R  # noqa: E402  (backbones run.py)
import fit_ft_fusion as F  # noqa: E402  (Boot, bank_rows, ads_of, file_preds, lab, eer)
from component_fusion import fuse  # noqa: E402
import decision_heads as dh  # noqa: E402

OUT = HERE / 'runs'
SPEC = R.parse('mert6_12+eat16')


def every_items(rows):
    """One music Item per row whatever its labels (runtime: every file gets a MUSIC score)."""
    out = []
    for r in rows:
        parts = []
        for name, layers in SPEC:
            base, key, layer_key = R.SOURCES[name]
            with np.load(base / f"{r['ID']}.npz") as d:
                L = list(d[layer_key])
                parts.append(d[key][:, [L.index(k) for k in layers]].reshape(len(d[key]), -1))
        f = np.concatenate(parts, 1)
        out.append(R.Item({**r, 'route': 'music'}, f, np.full(len(f), np.nan, np.float32), 0., False))
    return out


def music_every():
    path = OUT / 'music_eat__every.json'
    if path.exists():
        return json.loads(path.read_text())
    torch.set_num_threads(4)
    rows = {s: [r for r in csv.DictReader((R.H.BANK / f).open()) if r['split'] == s and r['ID'] in R.PINNED]
            for s, f in (('train', 'bank_trainval.csv'), ('val', 'bank_trainval.csv'), ('test', 'bank_test.csv'))}
    data = {s: R.new_items(rows[s], SPEC, 'music') for s in rows}
    frozen = set(json.loads((ST / 'frozen_rows.json').read_text())['ids'])
    every = {s: every_items([r for r in rows[s] if r['ID'] in frozen]) for s in ('val', 'test')}
    ref = json.loads((HERE.parent / 'work2' / 'backbones' / 'runs' / 'music_new__mert6_12+eat16.json').read_text())
    res = {'every': {s: {'ids': [x.row['ID'] for x in v], 'preds': []} for s, v in every.items()}, 'max_abs_vs_ref': []}
    for i, seed in enumerate((41, 42, 43)):
        model, _ = R.train('music', data['train'], data['val'], lambda items: R.WindowMIL(*R.stats(items)), 0, seed)
        vp = R.T.predict(model, [x for x, _ in data['val']], 'music', 0, 256, 10 ** 6, 'cpu')
        res['max_abs_vs_ref'].append(float(np.abs(vp - np.array(ref['val_preds'][i])).max()))
        for s, v in every.items():
            res['every'][s]['preds'].append(R.T.predict(model, v, 'music', 0, 256, 10 ** 6, 'cpu').tolist())
        print('seed', seed, 'max |val pred - ref|', res['max_abs_vs_ref'][-1], flush=True)
    OUT.mkdir(exist_ok=True)
    path.write_text(json.dumps(res))
    return res


def main():
    eat = music_every()
    assert max(eat['max_abs_vs_ref']) == 0.0, f"retrained S5 music head differs from the gate run: {eat['max_abs_vs_ref']}"
    frozen = set(json.loads((ST / 'frozen_rows.json').read_text())['ids'])
    rows = {s: F.bank_rows(F.ROOT / 'results/data_v4_20260923/data_v4_results_20260923T1639Z/data/bank', s, frozen) for s in ('val', 'test')}
    P = {**json.loads((ST / 'presence_val.json').read_text()), **json.loads((ST / 'presence_test.json').read_text())}
    C, silent = {'AC': {}, 'RT': {}}, {}
    for n, run in (('ours', 'voice_df__mlp_long'), ('ad', 'voice_ad__resmlp_long'), ('mert', 'music__L6_12')):
        r = json.loads((ST / 'runs' / f'{run}.json').read_text())
        for s in rows:
            C['AC'].setdefault(n, {}).update(zip(r[f'{s}_ids'], np.mean(r[f'{s}_preds'], 0).tolist()))
            C['RT'].setdefault(n, {}).update(zip(r['every'][s]['ids'], np.mean(r['every'][s]['preds'], 0).tolist()))
            if n == 'ours':
                silent.update(zip(r['every'][s]['ids'], r['every'][s]['silent']))
    ref = json.loads((HERE.parent / 'work2' / 'backbones' / 'runs' / 'music_new__mert6_12+eat16.json').read_text())
    for s in rows:
        C['AC'].setdefault('eat', {}).update(zip(ref[f'{s}_ids'], np.mean(ref[f'{s}_preds'], 0).tolist()))
        C['RT'].setdefault('eat', {}).update(zip(eat['every'][s]['ids'], np.mean(eat['every'][s]['preds'], 0).tolist()))
    for s in rows:
        missing = [r['ID'] for r in rows[s] if r['ID'] not in C['RT']['eat'] or r['ID'] not in C['RT']['mert'] or r['ID'] not in silent]
        assert not missing, f'RT scores missing for {len(missing)} {s} rows'
    val = rows['val']
    ns = [r['ID'] for r in val if not silent[r['ID']]]
    tables = {'AC': {n: dh.quantile_table([c[r['ID']] for r in val if r['ID'] in c]) for n, c in C['AC'].items()},
              'RT': {n: dh.quantile_table([c[i] for i in ns]) for n, c in C['RT'].items()}}

    def run(music, prot):
        c, spec = C[prot], {'weights': {'ours': 1, 'ad': .5}, 'tables': tables[prot]}
        files = {s: [{'ID': r['ID'], 'Pv': P[r['ID']][0], 'Pm': P[r['ID']][1], 'Fm': c[music].get(r['ID'], 0.),
                      'Fv': 0. if prot == 'RT' and silent[r['ID']] else fuse(spec, {n: c[n].get(r['ID'], 0.) for n in ('ours', 'ad')})}
                     for r in rows[s]] for s in ('val', 'test')}
        ens, _, preds = F.file_preds(files, val)
        return {'metrics': {s: F.ads_of(rows[s], preds[s]) for s in ('val', 'test')}, 'file_weights': ens['weights'], 'preds': preds}

    expected = {('mert', 'RT'): (.93773, .88457), ('mert', 'AC'): (.93845, .88643), ('eat', 'AC'): (.95475, .89755)}
    boot = F.Boot(rows['test'])
    rep = {'rows': {s: len(v) for s, v in rows.items()}, 'n_test_groups': boot.n_groups, 'results': {}, 'delta_S5_vs_S4': {}}
    got = {}
    for prot in ('RT', 'AC'):
        for music in ('mert', 'eat'):
            got[(music, prot)] = run(music, prot)
            m = got[(music, prot)]['metrics']
            rep['results'][f"{'S4' if music == 'mert' else 'S5'}|{prot}"] = m | {'file_weights': got[(music, prot)]['file_weights']}
            if (music, prot) in expected:
                assert (m['val']['ads'], m['test']['ads']) == expected[(music, prot)], f'reproduction failed {music} {prot}: {m}'
        rep['delta_S5_vs_S4'][prot] = boot.delta(got[('mert', prot)]['preds']['test'], got[('eat', prot)]['preds']['test'])
    # where the RT/AC difference comes from: rows scored 0 under AC but not under RT
    for s in rows:
        rep.setdefault('ac_zero_rows', {})[s] = {n: sum(r['ID'] not in C['AC'][n] for r in rows[s]) for n in ('ours', 'ad', 'mert', 'eat')}
    (HERE / 'rt_check.json').write_text(json.dumps(rep, indent=1))
    print(json.dumps(rep, indent=1))


if __name__ == '__main__':
    main()
