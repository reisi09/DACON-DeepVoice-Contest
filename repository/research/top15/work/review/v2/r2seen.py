"""Q5: is the R2 MLP held-out ADS gain (+0.0138, seeds 41-43) a favourable seed draw?  Adds MLP seeds 44-46
    (seeds_mlp.json, r2seeds.py) to the Linear seeds 41-46 (R2 runs + seeds.py) and recomputes, in the ads_compare
    pipeline (core.Data, CDF tables + FILE grid refit on val): single-seed paired deltas, disjoint triple B, bootstrap.
Q7: is the S3-style gain on member-SEEN rows (+0.015) really larger than on UNSEEN rows (+0.0078)
    (orchestrator/s3_unseen_check.py, fusion_fit.member_seen)?  Same subsets, fits per subset; test-group bootstrap
    and seed triples/single seeds per subset.
Read-only on project files; writes r2seen.json here."""
import sys
sys.dont_write_bytecode = True
import csv, json, time
from pathlib import Path
import numpy as np
ROOT = Path('/Users/choiwuseck/Documents/ChatGPT/데이콘')
HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(ROOT / 'data_v4'), str(ROOT)]
import core
import train_backbone_heads as H
import decision_heads as dh
from fusion_fit import member_seen

t0 = time.time()
cache = ROOT / '_workspace/backbone_cache'
rows = {s: [r for r in csv.DictReader((H.BANK / f).open()) if r['split'] == s and (cache / f"{r['ID']}.npz").exists()]
        for s, f in (('val', 'bank_trainval.csv'), ('test', 'bank_test.csv'))}
comp = {}
R2 = ROOT / 'research/top15/work/r2/runs'
for key, name in (('voice_df', 'voice_df__base'), ('voice_ad', 'voice_ad__base'), ('music_6_12', 'music__L6_12'),
                  ('mlp_df', 'voice_df__mlp_long'), ('mlp_ad', 'voice_ad__resmlp_long')):
    r = json.loads((R2 / f'{name}.json').read_text())
    for j, s in enumerate(r['seeds']):
        comp[f'{key}@{s}'] = dict(zip(r['val_ids'] + r['test_ids'], np.r_[r['val_preds'][j], r['test_preds'][j]].tolist()))
for f in ('seeds_music.json', 'seeds_voice.json', 'seeds_mlp.json'):
    for key, per in json.loads((HERE / f).read_text()).items():
        for s, v in per.items():
            comp.setdefault(f'{key}@{s}', v)
pres = json.loads(H.PRESENCE.read_text())
P = {}
for s in rows:
    for r in rows[s]:
        with np.load(H.OLD[s] / f"{r['ID']}.npz") as x:
            P[r['ID']] = dh.presence(pres, json.loads(str(x['presence_features'])), (float(x['presence'][0]), float(x['presence'][1])))
ens = lambda key, seeds: {i: float(np.mean([comp[f'{key}@{s}'][i] for s in seeds])) for i in comp[f'{key}@{seeds[0]}']}
for key in ('voice_df', 'voice_ad', 'music_12', 'music_6_12', 'mlp_df', 'mlp_ad'):
    comp[f'{key}#A'], comp[f'{key}#B'] = ens(key, (41, 42, 43)), ens(key, (44, 45, 46))
base = lambda t: dict(vdf=f'voice_df{t}', m=f'music_12{t}')
s3 = lambda t: dict(vdf=f'voice_df{t}', vad=f'voice_ad{t}', w=.5, m=f'music_6_12{t}')
lin = lambda t: dict(vdf=f'voice_df{t}', vad=f'voice_ad{t}', w=.5, m=f'music_6_12{t}', clip=True)
mlp = lambda t, u=None: dict(vdf=f'mlp_df{t}', vad=f'mlp_ad{t}', w=.5, m=f'music_6_12{u or t}', clip=True)
mlp_df = lambda t: dict(vdf=f'mlp_df{t}', vad=f'voice_ad{t}', w=.5, m=f'music_6_12{t}', clip=True)
out = {}


def boot(D, a, b_, B=1000, seed=20260927):
    sa, sb = D.scores(a), D.scores(b_)
    pt = D.ads(sa, D.ti)[0] - D.ads(sb, D.ti)[0]
    tg, rng, v = core.group_index(D.group, D.ti), np.random.default_rng(seed), []
    for _ in range(B):
        idx = core.resample(tg, rng)
        v.append(D.ads(sa, idx)[0] - D.ads(sb, idx)[0])
    return pt, np.array(v)


# Q5
D = core.Data(rows['val'], rows['test'], comp, P)
A = lambda cfg: D.ads(D.scores(cfg), D.ti)[0]
assert abs(A(mlp('#A')) - .88643) < 6e-6 and abs(A(lin('#A')) - .87267) < 6e-6
per = {}
for s in range(41, 47):
    t = f'@{s}'
    a = {'lin': A(lin(t)), 'mlp': A(mlp(t)), 'mlp_df_only': A(mlp_df(t))}
    per[s] = {'ads': {k: round(v, 5) for k, v in a.items()}, 'mlp-lin': round(a['mlp'] - a['lin'], 5), 'mlp_df_only-lin': round(a['mlp_df_only'] - a['lin'], 5)}
    print('Q5 seed', s, per[s], flush=True)
q5 = {'single_seed': per}
q5['single_seed_summary'] = {k: {'mean': round(float(np.mean(v)), 5), 'sd': round(float(np.std(v, ddof=1)), 5), 'min': min(v), 'max': max(v), 'n_gt0': int(sum(x > 0 for x in v))}
                             for k in ('mlp-lin', 'mlp_df_only-lin') for v in [[per[s][k] for s in per]]}
q5['triples'] = {f'mlp{x}-lin{y}': round(A(mlp(f'#{x}')) - A(lin(f'#{y}')), 5) for x in 'AB' for y in 'AB'}
q5['triple_ads'] = {f'{n}#{x}': round(A(f(f'#{x}')), 5) for n, f in (('lin', lin), ('mlp', mlp), ('mlp_df_only', mlp_df)) for x in 'AB'}
pt, v = boot(D, mlp('#B'), lin('#B'))
q5['tripleB_test_group_bootstrap'] = core.summ(v, pt)
out['Q5_r2_mlp'] = q5
print('Q5', q5['single_seed_summary'], q5['triples'], q5['tripleB_test_group_bootstrap'], f'{time.time() - t0:.0f}s', flush=True)

# Q7
q7 = {}
bs = {}
for sub in ('unseen', 'seen'):
    keep = lambda r: member_seen(r) == (sub == 'seen')
    Ds = core.Data([r for r in rows['val'] if keep(r)], [r for r in rows['test'] if keep(r)], comp, P)
    As = lambda cfg: Ds.ads(Ds.scores(cfg), Ds.ti)[0]
    pt, v = boot(Ds, s3('#A'), base('#A'))
    bs[sub] = (pt, v)
    q7[sub] = {'n_val': Ds.nv, 'n_test': len(Ds.ti), 'test_groups': len(core.group_index(Ds.group, Ds.ti)),
               'tripleA_s3-base': round(pt, 5), 'tripleA_boot': core.summ(v, pt),
               'tripleB_s3-base': round(As(s3('#B')) - As(base('#B')), 5),
               'single_seed_s3-base': {s: round(As(s3(f'@{s}')) - As(base(f'@{s}')), 5) for s in range(41, 47)}}
    ss = list(q7[sub]['single_seed_s3-base'].values())
    q7[sub]['single_seed_mean_sd'] = [round(float(np.mean(ss)), 5), round(float(np.std(ss, ddof=1)), 5)]
    print('Q7', sub, {k: q7[sub][k] for k in ('n_val', 'n_test', 'test_groups', 'tripleA_s3-base', 'tripleB_s3-base', 'single_seed_mean_sd')},
          q7[sub]['tripleA_boot'], f'{time.time() - t0:.0f}s', flush=True)
dd = bs['seen'][1] - bs['unseen'][1]
q7['seen_minus_unseen_tripleA'] = core.summ(dd, bs['seen'][0] - bs['unseen'][0])
out['Q7_seen_vs_unseen'] = q7
# Q8: 6-seed (41-46) vs 3-seed ensembles, same pipeline (question: is seed-ensemble size a free lever?)
for key in ('voice_df', 'voice_ad', 'music_12', 'music_6_12', 'mlp_df', 'mlp_ad'):
    e6 = ens(key, tuple(range(41, 47)))
    D.comp[f'{key}#6'] = np.array([e6.get(r['ID'], 0.) for r in D.rows])
    D.has[f'{key}#6'] = D.has[f'{key}#A']
out['Q8_six_seed'] = {n: {t: round(A(f(t)), 5) for t in ('#A', '#B', '#6')} for n, f in (('base', base), ('s3', s3), ('lin', lin), ('mlp', mlp))}
print('Q8', out['Q8_six_seed'], flush=True)
out['seconds'] = round(time.time() - t0)
(HERE / 'r2seen.json').write_text(json.dumps(out, indent=1, default=float))
print('Q7 seen-unseen', q7['seen_minus_unseen_tripleA'], 'seconds', out['seconds'])
