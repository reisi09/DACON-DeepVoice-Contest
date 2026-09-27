"""Q4: how much of the held-out S3-vs-baseline (and R2-MLP-vs-S3-like) ADS delta is training-seed noise?
 (a) single-seed configurations, seeds 41-46 (41-43 = R2 base run files, which reproduce ads_compare_scores.json;
     44-46 and music_12 41-46 = seeds.py retrains with the same H.train calls)
 (b) the disjoint seed triple 44-46 as a second 3-seed ensemble
 (c) 3-seed ensembles drawn at random (per component, without replacement from 6 seeds)
 (d) FILE-rule sensitivity: base with equal weights / s3 with the base's grid weights
Held-out = backbone-cache test rows; everything (CDF tables, FILE grid) refit on val per configuration, as ads_compare.
Read-only on project files; writes seedcheck.json here."""
import sys
sys.dont_write_bytecode = True
import csv, itertools, json, time
from pathlib import Path
import numpy as np
ROOT = Path('/Users/choiwuseck/Documents/ChatGPT/데이콘')
HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(ROOT / 'data_v4'), str(ROOT)]
import core
import train_backbone_heads as H
import decision_heads as dh

t0 = time.time()
cache = ROOT / '_workspace/backbone_cache'
rows = {s: [r for r in csv.DictReader((H.BANK / f).open()) if r['split'] == s and (cache / f"{r['ID']}.npz").exists()]
        for s, f in (('val', 'bank_trainval.csv'), ('test', 'bank_test.csv'))}
ref = json.loads((ROOT / '_workspace/audit_20260927/ads_compare_scores.json').read_text())
comp = {}
R2 = ROOT / 'research/top15/work/r2/runs'
for key, name in (('voice_df', 'voice_df__base'), ('voice_ad', 'voice_ad__base'), ('music_6_12', 'music__L6_12'),
                  ('mlp_df', 'voice_df__mlp_long'), ('mlp_ad', 'voice_ad__resmlp_long')):
    r = json.loads((R2 / f'{name}.json').read_text())
    ids = r['val_ids'] + r['test_ids']
    for j, s in enumerate(r['seeds']):
        comp[f'{key}@{s}'] = dict(zip(ids, np.r_[r['val_preds'][j], r['test_preds'][j]].tolist()))
for f in ('seeds_music.json', 'seeds_voice.json'):
    for key, per in json.loads((HERE / f).read_text()).items():
        for s, v in per.items():
            comp.setdefault(f'{key}@{s}', v)          # R2 files win for 41-43 of voice_df/voice_ad/music_6_12
check = {}
for key in ('voice_df', 'voice_ad', 'music_12', 'music_6_12'):
    ids = list(ref[key])
    m = np.mean([[comp[f'{key}@{s}'][i] for i in ids] for s in (41, 42, 43)], 0)
    check[key] = float(np.abs(m - np.array([ref[key][i] for i in ids])).max())
print('max |seed-mean(41-43) - ads_compare_scores|', check, flush=True)
assert max(check.values()) < 1e-3
pres = json.loads(H.PRESENCE.read_text())
P = {}
for s in rows:
    for r in rows[s]:
        with np.load(H.OLD[s] / f"{r['ID']}.npz") as x:
            P[r['ID']] = dh.presence(pres, json.loads(str(x['presence_features'])), (float(x['presence'][0]), float(x['presence'][1])))


def ens(key, seeds):
    ids = set().union(*(comp[f'{key}@{s}'] for s in seeds))
    return {i: float(np.mean([comp[f'{key}@{s}'][i] for s in seeds])) for i in ids}


SEEDS = (41, 42, 43, 44, 45, 46)
extra = {}
for key in ('voice_df', 'voice_ad', 'music_12', 'music_6_12'):
    extra[f'{key}#A'] = ens(key, (41, 42, 43))
    extra[f'{key}#B'] = ens(key, (44, 45, 46))
for key in ('mlp_df', 'mlp_ad'):
    extra[f'{key}#A'] = ens(key, (41, 42, 43))
D = core.Data(rows['val'], rows['test'], comp | extra, P)
print('loaded', f'{time.time() - t0:.0f}s', flush=True)
base = lambda t: dict(vdf=f'voice_df{t}', m=f'music_12{t}')
s3 = lambda t: dict(vdf=f'voice_df{t}', vad=f'voice_ad{t}', w=.5, m=f'music_6_12{t}')
m612 = lambda t: dict(vdf=f'voice_df{t}', m=f'music_6_12{t}')
r2b = lambda t: dict(vdf=f'voice_df{t}', vad=f'voice_ad{t}', w=.5, m=f'music_6_12{t}', clip=True)
r2 = lambda t, u=None: dict(vdf=f'mlp_df{t}', vad=f'mlp_ad{t}', w=.5, m=f'music_6_12{u or t}', clip=True)
A = lambda cfg: D.ads(D.scores(cfg), D.ti)[0]
out = {'repro_max_abs_seedmean_vs_ads_compare': check}
# (a) single seeds
per = {}
for s in SEEDS:
    t = f'@{s}'
    a = {'base': A(base(t)), 'm612': A(m612(t)), 's3': A(s3(t))}
    per[s] = {'ads': {k: round(v, 5) for k, v in a.items()}, 's3-base': round(a['s3'] - a['base'], 5), 'm612-base': round(a['m612'] - a['base'], 5),
              's3-m612': round(a['s3'] - a['m612'], 5)}
    if s <= 43:
        per[s]['r2_all-r2_base'] = round(A(r2(t)) - A(r2b(t)), 5)
    print('seed', s, per[s], flush=True)
out['a_single_seed'] = per
out['a_single_seed_summary'] = {k: {'mean': round(float(np.mean(v)), 5), 'sd': round(float(np.std(v, ddof=1)), 5), 'min': min(v), 'max': max(v), 'n_gt0': int(sum(x > 0 for x in v)), 'n': len(v)}
                                for k in ('s3-base', 'm612-base', 's3-m612', 'r2_all-r2_base')
                                for v in [[per[s][k] for s in per if k in per[s]]]}
# (b) disjoint triples
tri = {}
for t in ('#A', '#B'):
    a = {'base': A(base(t)), 'm612': A(m612(t)), 's3': A(s3(t))}
    tri[t] = {'ads': {k: round(v, 5) for k, v in a.items()}, 's3-base': round(a['s3'] - a['base'], 5), 'm612-base': round(a['m612'] - a['base'], 5),
              's3-m612': round(a['s3'] - a['m612'], 5)}
tri['#B']['r2_all(A)-r2_base(B)'] = round(A(r2('#A')) - A(r2b('#B')), 5)   # MLP triple 41-43 vs a Linear triple it never shared
tri['#A']['r2_all-r2_base'] = round(A(r2('#A')) - A(r2b('#A')), 5)
out['b_disjoint_triples'] = tri
print('triples', tri, flush=True)
# (c) random 3-of-6 ensembles per component (finite-population: sd of independent triples ~ sd / sqrt(0.6))
rng = np.random.default_rng(7)
combos = list(itertools.combinations(SEEDS, 3))
dc, dm = [], []
for i in range(150):
    pick = {k: combos[j] for k, j in zip(('voice_df', 'voice_ad', 'music_12', 'music_6_12'), rng.integers(0, len(combos), 4))}
    x = {k: ens(k, v) for k, v in pick.items()}
    for k, v in x.items():
        D.comp[f'{k}#R'] = np.array([v.get(r['ID'], 0.) for r in D.rows])
        D.has[f'{k}#R'] = np.array([r['ID'] in v for r in D.rows])
    b = A(base('#R'))
    dc.append(A(s3('#R')) - b)
    dm.append(A(m612('#R')) - b)
out['c_random_triples'] = {k: {'mean': round(float(np.mean(v)), 5), 'sd': round(float(np.std(v, ddof=1)), 5),
                               'sd_indep_approx': round(float(np.std(v, ddof=1) / np.sqrt(.6)), 5), 'p_gt0': round(float(np.mean(np.array(v) > 0)), 3),
                               'q2.5_97.5': np.percentile(v, [2.5, 97.5]).round(5).tolist(), 'n': len(v)} for k, v in (('s3-base', dc), ('m612-base', dm))}
print('random triples', out['c_random_triples'], flush=True)
# (d) FILE-rule sensitivity on the shipped-screen triple (41-43)
EQ = {'max': 1, 'noisy_or': 1, 'max_nopresence': 1}
GW = D.scores(base('#A'))['ens_w']
d = {'base_grid_weights': GW, 'base_grid': A(base('#A')), 'base_equal': A(base('#A') | {'force_w': EQ}),
     's3_equal': A(s3('#A')), 's3_with_base_grid_weights': A(s3('#A') | {'force_w': GW}),
     'm612_equal': A(m612('#A')), 'm612_with_base_grid_weights': A(m612('#A') | {'force_w': GW})}
d = {k: (round(v, 5) if isinstance(v, float) else v) for k, v in d.items()}
d['s3-base, both equal weights'] = round(d['s3_equal'] - d['base_equal'], 5)
d['m612-base, both equal weights'] = round(d['m612_equal'] - d['base_equal'], 5)
d['s3-base, both base grid weights'] = round(d['s3_with_base_grid_weights'] - d['base_grid'], 5)
out['d_file_rule_sensitivity'] = d
print('file rule', d, flush=True)
out['seconds'] = round(time.time() - t0)
(HERE / 'seedcheck.json').write_text(json.dumps(out, indent=1, default=str))
