"""Skeptic check (no training): does seed / subset-draw noise, which the models-fixed group bootstrap ignores, swamp
the study's contrasts?  Recomputes official ADS PER SEED (voice seed k paired with music seed k; FILE refit on val;
shared presence) from the saved run predictions.
  1. exp1 arms: per-seed held-out and val ADS for full, Q*, D* -> seed spread, draw spread, log2-row slope.
  2. Music Arena voice pilot: paired per-seed deltas (base vs +arena, seeds 41/42/43), DF-only and S4-like fusion
     (R2 AD ResMLP 3-seed mean at weight 0.5, as in pilot.s4()).
    python critique_check.py  -> critique_check.json
"""
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import ads as A  # noqa: E402
from component_fusion import fuse  # noqa: E402

ld = lambda n: json.loads((HERE / 'runs' / f'{n}.json').read_text())
per = lambda r, k: {s: dict(zip(r[f'{s}_ids'], r[f'{s}_preds'][k])) for s in ('val', 'test')}
ads = lambda vd, mu: {s: v['ads'] for s, v in A.official(vd, mu, ('val', 'test'))[0].items() if s in ('val', 'test')}
out = {'exp1': {}, 'pilot': {}}

for name in ['full'] + [f'{a}{f}_d{d}' for f in (25, 50, 75) for a, ds in (('Q', (0,)), ('D', (1, 2, 3))) for d in ds]:
    rv, rm = ld(f'voice_df__{name}'), ld(f'music__{name}')
    out['exp1'][name] = [ads(per(rv, k), per(rm, k)) for k in (0, 1)]
    print(name, out['exp1'][name], flush=True)

ad = json.loads((A.R2 / 'voice_ad__resmlp_long.json').read_text())
adsc = {s: dict(zip(ad[f'{s}_ids'], np.mean(ad[f'{s}_preds'], 0).tolist())) for s in ('val', 'test')}
_, mu = A.run(HERE / 'runs' / 'music__full.json')
missing_ad = {s: sum(r['ID'] not in adsc[s] for r in A.rows[s]) for s in ('val', 'test')}
pairs = {'base': [(ld('voice_df__full'), 0), (ld('voice_df__full'), 1), (ld('pilot_voice__base_s43'), 0)],
         'plus_arena': [(ld('pilot_voice__plus_arena'), 0), (ld('pilot_voice__plus_arena'), 1), (ld('pilot_voice__plus_arena_s43'), 0)]}
for k, runs in pairs.items():
    for mode in ('df_only', 's4_fused'):
        res = []
        for r, j in runs:
            df = per(r, j)
            if mode == 's4_fused':
                spec = {'weights': {'ours': 1, 'ad': .5}, 'tables': {
                    'ours': A.dh.quantile_table([df['val'][x['ID']] for x in A.rows['val'] if x['ID'] in df['val']]),
                    'ad': A.dh.quantile_table([adsc['val'][x['ID']] for x in A.rows['val'] if x['ID'] in adsc['val']])}}
                df = {s: {x['ID']: fuse(spec, {'ours': df[s].get(x['ID'], 0.), 'ad': adsc[s].get(x['ID'], 0.)}) for x in A.rows[s]}
                      for s in ('val', 'test')}
            res.append(ads(df, mu))
        out['pilot'][f'{k}|{mode}'] = res
        print(k, mode, res, flush=True)
out['pilot']['ad_missing_rows'] = missing_ad
for mode in ('df_only', 's4_fused'):
    for s in ('val', 'test'):
        d = [p[s] - b[s] for p, b in zip(out['pilot'][f'plus_arena|{mode}'], out['pilot'][f'base|{mode}'])]
        out['pilot'][f'delta|{mode}|{s}'] = {'per_seed': np.round(d, 5).tolist(), 'mean': round(float(np.mean(d)), 5),
                                             'sd': round(float(np.std(d, ddof=1)), 5)}
e = out['exp1']
sm = {n: float(np.mean([x['test'] for x in v])) for n, v in e.items()}
out['exp1_summary'] = {
    'within_config_seed_sd_test_single_seed': round(float(np.sqrt(np.mean([np.var([x['test'] for x in v], ddof=1) for v in e.values()]))), 5),
    'D_draw_sd_of_2seed_mean': {f: round(float(np.std([sm[f'D{f}_d{d}'] for d in (1, 2, 3)], ddof=1)), 5) for f in (25, 50, 75)},
    'two_seed_mean_test': {n: round(v, 5) for n, v in sm.items()},
    'Dmean_curve': {f: round(float(np.mean([sm[f'D{f}_d{d}'] for d in (1, 2, 3)])), 5) for f in (25, 50, 75)} | {100: round(sm['full'], 5)}}
c = out['exp1_summary']['Dmean_curve']
out['exp1_summary']['Dmean_gain_per_doubling'] = {f'{a}->{b}': round((c[b] - c[a]) / np.log2(b / a), 5) for a, b in ((25, 50), (50, 75), (75, 100))}
q = {25: sm['Q25_d0'], 50: sm['Q50_d0'], 75: sm['Q75_d0'], 100: sm['full']}
out['exp1_summary']['Q_gain_per_doubling'] = {f'{a}->{b}': round((q[b] - q[a]) / np.log2(b / a), 5) for a, b in ((25, 50), (50, 75), (75, 100))}
print(json.dumps(out['exp1_summary'], indent=1), json.dumps({k: v for k, v in out['pilot'].items() if k.startswith('delta') or k == 'ad_missing_rows'}, indent=1))
(HERE / 'critique_check.json').write_text(json.dumps(out, indent=1))
