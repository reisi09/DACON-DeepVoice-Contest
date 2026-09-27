"""Q: does type-aware routing (AT-ADD 2026 top systems: route by audio type, type-specific fusion/thresholds) help the
voice member fusion?  VOICE = CDF-mean(ours=DF-Arena heads, ad=AntiDeepfake-embedding heads) with the AD weight
depending on whether the file contains music (singing/music context) or not (speech context). Weights chosen on VAL
only (2-D grid), evaluated once on the held-out test. DB-protocol member scores from
_workspace/audit_20260927/ads_compare_scores.json; routing uses the MUSIC_PRESENT label (oracle; presence heads have
music AUC ~0.999, so this is a near-realistic upper bound). Read-only, CPU, seconds."""
import json, sys, itertools, numpy as np, pandas as pd
from sklearn.metrics import roc_curve
R = '/Users/choiwuseck/Documents/ChatGPT/데이콘/'
sys.path.insert(0, R)
import decision_heads as dh
B = R + 'results/data_v4_20260923/data_v4_results_20260923T1639Z/data/bank/'
bank = pd.concat([pd.read_csv(B + 'bank_trainval.csv'), pd.read_csv(B + 'bank_test.csv')])
S = json.load(open(R + '_workspace/audit_20260927/ads_compare_scores.json'))
d = bank[bank.split.isin(['val', 'test']) & (bank.VOICE_FAKE_mask == 1) & (bank.VOICE_PRESENT == 1)].copy()
d['ours'] = d.ID.map(S['voice_df']); d['ad'] = d.ID.map(S['voice_ad']); d = d.dropna(subset=['ours', 'ad'])
val_all = bank[(bank.split == 'val')].ID
tab = {k: dh.quantile_table([S[m][i] for i in val_all if i in S[m]]) for k, m in (('ours', 'voice_df'), ('ad', 'voice_ad'))}
for k in ('ours', 'ad'):
    d['c_' + k] = [dh.to_cdf(tab[k], x) for x in d[k]]
def eer(y, s):
    fpr, tpr, _ = roc_curve(y, s, drop_intermediate=False); i = np.argmin(abs(fpr - (1 - tpr))); return (fpr[i] + 1 - tpr[i]) / 2
def score(df, w_sp, w_mu):
    w = np.where(df.MUSIC_PRESENT == 1, w_mu, w_sp)
    return (df.c_ours + w * df.c_ad) / (1 + w)
V, T = d[d.split == 'val'], d[d.split == 'test']
print('rows val/test', len(V), len(T), '| test music-context share', round((T.MUSIC_PRESENT == 1).mean(), 3))
grid = [0, .25, .5, 1, 2, 4]
res = {(a, b): eer(V.VOICE_FAKE, score(V, a, b)) for a, b in itertools.product(grid, grid)}
best = min(res, key=res.get)
for name, (a, b) in {'ours only': (0, 0), 'global w=0.5 (S3)': (.5, .5), 'global w=1': (1, 1),
                     f'routed val-best w_speech={best[0]} w_music={best[1]}': best}.items():
    print(f'{name:45s} val voice EER {eer(V.VOICE_FAKE, score(V, a, b)):.4f}  test voice EER {eer(T.VOICE_FAKE, score(T, a, b)):.4f}'
          f'  [test speech-ctx {eer(T[T.MUSIC_PRESENT==0].VOICE_FAKE, score(T[T.MUSIC_PRESENT==0], a, b)):.4f}, music-ctx {eer(T[T.MUSIC_PRESENT==1].VOICE_FAKE, score(T[T.MUSIC_PRESENT==1], a, b)):.4f}]')
# group bootstrap of routed-vs-global(0.5) test delta
rng = np.random.default_rng(0); g = T.group_id.values; ug = np.unique(g); idx = {k: np.where(g == k)[0] for k in ug}
s_r, s_g = score(T, *best).values, score(T, .5, .5).values; y = T.VOICE_FAKE.values
deltas = []
for _ in range(300):
    ii = np.concatenate([idx[k] for k in rng.choice(ug, len(ug))])
    deltas.append(eer(y[ii], s_g[ii]) - eer(y[ii], s_r[ii]))
q = np.quantile(deltas, [.025, .5, .975])
print(f'test voice EER gain routed vs global 0.5 (positive = routed better): median {q[1]:.4f}, 95% CI [{q[0]:.4f}, {q[2]:.4f}], {len(ug)} groups')
print('ADS equivalent = 0.2 x voice EER gain (+ indirect FILE effect not modelled)')
