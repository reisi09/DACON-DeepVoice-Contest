"""Q1-Q3 for S3-vs-baseline (ads_compare) and R2-MLP-vs-S3-like, held-out backbone-cache rows:
 Q1 how uncertain is the held-out ADS delta from test sampling alone (test groups resampled, fits fixed)?
 Q2 how much wider when the val-fitted parts (voice CDF tables, FILE ensemble grid) are refit on resampled val groups?
 Q3 does the delta depend on the source mix (leave-one-source-out; Dirichlet reweighting of test sources)?
Read-only on project files; writes boot.json here.   python boot.py [B1 B2 B3]"""
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
from file_ensemble import fit as fit_file_ref
from graph_scoring import official_metrics

B1, B2, B3 = (int(x) for x in sys.argv[1:4]) if len(sys.argv) > 3 else (2000, 500, 2000)
t0 = time.time()
cache = ROOT / '_workspace/backbone_cache'
rows = {s: [r for r in csv.DictReader((H.BANK / f).open()) if r['split'] == s and (cache / f"{r['ID']}.npz").exists()]
        for s, f in (('val', 'bank_trainval.csv'), ('test', 'bank_test.csv'))}
comp = json.loads((ROOT / '_workspace/audit_20260927/ads_compare_scores.json').read_text())
R2 = ROOT / 'research/top15/work/r2/runs'
for key, name in (('voice_df_r2', 'voice_df__mlp_long'), ('voice_ad_r2', 'voice_ad__resmlp_long'), ('music_6_12_r2', 'music__L6_12')):
    r = json.loads((R2 / f'{name}.json').read_text())
    comp[key] = dict(zip(r['val_ids'] + r['test_ids'], np.r_[np.mean(r['val_preds'], 0), np.mean(r['test_preds'], 0)].tolist()))
pres = json.loads(H.PRESENCE.read_text())
P = {}
for s in rows:
    for r in rows[s]:
        with np.load(H.OLD[s] / f"{r['ID']}.npz") as x:
            P[r['ID']] = dh.presence(pres, json.loads(str(x['presence_features'])), (float(x['presence'][0]), float(x['presence'][1])))
D = core.Data(rows['val'], rows['test'], comp, P)
print('loaded', len(rows['val']), len(rows['test']), f'{time.time() - t0:.0f}s', flush=True)

CFG = {'base': dict(vdf='voice_df', m='music_12'),
       'm612': dict(vdf='voice_df', m='music_6_12'),
       's3': dict(vdf='voice_df', vad='voice_ad', w=.5, m='music_6_12'),
       's3_w1': dict(vdf='voice_df', vad='voice_ad', w=1., m='music_6_12'),
       'r2_base': dict(vdf='voice_df', vad='voice_ad', w=.5, m='music_6_12', clip=True),
       'r2_df': dict(vdf='voice_df_r2', vad='voice_ad', w=.5, m='music_6_12', clip=True),
       'r2_all': dict(vdf='voice_df_r2', vad='voice_ad_r2', w=.5, m='music_6_12_r2', clip=True)}
PAIRS = {'s3-base': ('s3', 'base'), 'm612-base': ('m612', 'base'), 's3-m612 (voice AD fusion part)': ('s3', 'm612'),
         's3(w.5)-s3(w1)': ('s3', 's3_w1'), 'r2_df-r2_base': ('r2_df', 'r2_base'), 'r2_all-r2_base': ('r2_all', 'r2_base')}
EXPECT = {'base': .86438, 'm612': .86748, 's3': .87267, 's3_w1': .87209, 'r2_base': .87267, 'r2_df': .88161, 'r2_all': .88643}
SC = {n: D.scores(c) for n, c in CFG.items()}
point = {n: D.ads(SC[n], D.ti) for n in CFG}
out = {'rows': {'val': D.nv, 'test': len(D.ti)}, 'point': {}}
for n in CFG:
    out['point'][n] = {'ads': round(point[n][0], 5), **{k: round(v, 5) for k, v in point[n][1].items()}, 'file_weights': SC[n]['ens_w']}
    assert abs(point[n][0] - EXPECT[n]) < 6e-6, (n, point[n][0])
# cross-check the vectorised pipeline against the project's own fit / official_metrics for S3
fv = [{'Pv': D.pv[i], 'Pm': D.pm[i], 'Fv': SC['s3']['voice'][i], 'Fm': SC['s3']['music'][i]} for i in D.vi if D.fitmask[i]]
ref, _ = fit_file_ref(fv, D.y['file'][D.vi][D.fitmask[D.vi]])
assert ref['weights'] == SC['s3']['ens_w'], (ref['weights'], SC['s3']['ens_w'])
preds = [{'ID': D.rows[i]['ID'], 'FILE_FAKE_PROB': SC['s3']['file'][i], 'VOICE_FAKE_PROB': SC['s3']['voice'][i], 'MUSIC_FAKE_PROB': SC['s3']['music'][i],
          'VOICE_PRESENT_PROB': D.pv[i], 'MUSIC_PRESENT_PROB': D.pm[i]} for i in D.ti]
m = official_metrics(rows['test'], preds)['metrics']
assert abs(.5 * (1 - m['file_eer']) + .2 * (1 - m['voice_eer']) + .3 * (1 - m['music_eer']) - point['s3'][0]) < 1e-9
print('points reproduce ads_compare / R2 ads_report; official_metrics agrees', f'{time.time() - t0:.0f}s', flush=True)
dpoint = {p: point[a][0] - point[b][0] for p, (a, b) in PAIRS.items()}
tg = core.group_index(D.group, D.ti)
vg = core.group_index(D.group, D.vi)
out['test_groups'], out['val_groups'] = len(tg), len(vg)

# Q1: test groups, fits fixed
rng = np.random.default_rng(20260927)
d1 = {p: [] for p in PAIRS}
comp1 = {p: {k: [] for k in ('file', 'voice', 'music')} for p in PAIRS}
for _ in range(B1):
    idx = core.resample(tg, rng)
    a = {n: D.ads(SC[n], idx) for n in CFG}
    for p, (x, y) in PAIRS.items():
        d1[p].append(a[x][0] - a[y][0])
        for k in comp1[p]:
            comp1[p][k].append(a[x][1][k] - a[y][1][k])
out['Q1_test_group_bootstrap'] = {p: core.summ(d1[p], dpoint[p]) | {'eer_delta_pct_ci95': {k: np.percentile(v, [2.5, 97.5]).round(4).tolist() for k, v in comp1[p].items()}} for p in PAIRS}
print('Q1 done', f'{time.time() - t0:.0f}s', flush=True)

# Q2: val groups (refit tables + FILE grid) and test groups, independently, paired across configs
d2 = {p: [] for p in PAIRS}
wchange = {n: 0 for n in CFG}
for _ in range(B2):
    vidx = core.resample(vg, rng)
    idx = core.resample(tg, rng)
    sc = {n: D.scores(c, vidx) for n, c in CFG.items()}
    for n in CFG:
        wchange[n] += sc[n]['ens_w'] != SC[n]['ens_w']
    a = {n: D.ads(sc[n], idx)[0] for n in CFG}
    for p, (x, y) in PAIRS.items():
        d2[p].append(a[x] - a[y])
out['Q2_val_and_test_group_bootstrap'] = {p: core.summ(d2[p], dpoint[p]) for p in PAIRS}
out['Q2_file_weights_differ_from_point_fit_frac'] = {n: round(v / B2, 3) for n, v in wchange.items()}
print('Q2 done', f'{time.time() - t0:.0f}s', flush=True)

# Q3: source mix. LOSO (drop one test source) and Dirichlet(1) weights over test sources (fits fixed)
srcs, counts = np.unique(D.source[D.ti], return_counts=True)
loso = {}
for s, c in zip(srcs, counts):
    idx = D.ti[D.source[D.ti] != s]
    try:
        a = {n: D.ads(SC[n], idx)[0] for n in CFG}
    except (FloatingPointError, ZeroDivisionError, IndexError):
        continue
    loso[s] = {'n': int(c), **{p: round(a[x] - a[y], 5) for p, (x, y) in PAIRS.items()}}
out['Q3_leave_one_source_out'] = loso
out['Q3_loso_min_max'] = {p: [min(v[p] for v in loso.values()), max(v[p] for v in loso.values())] for p in PAIRS}
sidx = {s: np.flatnonzero(D.source[D.ti] == s) for s in srcs}
d3 = {p: [] for p in PAIRS}
with np.errstate(all='raise'):
    for _ in range(B3):
        g = rng.dirichlet(np.ones(len(srcs))) * len(srcs)
        w = np.empty(len(D.ti))
        for s, gi in zip(srcs, g):
            w[sidx[s]] = gi
        a = {n: D.ads(SC[n], D.ti, w)[0] for n in CFG}
        for p, (x, y) in PAIRS.items():
            d3[p].append(a[x] - a[y])
out['Q3_dirichlet_source_reweighting'] = {p: core.summ(d3[p], dpoint[p]) for p in PAIRS}
out['seconds'] = round(time.time() - t0)
(HERE / 'boot.json').write_text(json.dumps(out, indent=1))
for q in ('Q1_test_group_bootstrap', 'Q2_val_and_test_group_bootstrap', 'Q3_dirichlet_source_reweighting'):
    for p, v in out[q].items():
        print(q[:2], p, {k: v[k] for k in ('point', 'boot_mean', 'pct_ci95', 'basic_ci95', 'p_gt0')}, flush=True)
print('LOSO min/max', out['Q3_loso_min_max'], 'seconds', out['seconds'])
