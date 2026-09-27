"""Q (mirror of cross_component_check.py): in files with BOTH voice and music, does the file's own VOICE member evidence
improve MUSIC_FAKE?  MUSIC = CDF-mean(music MERT-6+12 head, a*DF-Arena voice head, b*AD voice head) applied only when
voice is present (oracle VOICE_PRESENT label; presence AUC ~0.998); weights chosen on VAL; one held-out evaluation over
ALL music rows; discordant-row stress check. DB-protocol scores (ads_compare_scores.json). Read-only, CPU, seconds."""
import json, sys, itertools, numpy as np, pandas as pd
from sklearn.metrics import roc_curve
R = '/Users/choiwuseck/Documents/ChatGPT/데이콘/'
sys.path.insert(0, R)
import decision_heads as dh
B = R + 'results/data_v4_20260923/data_v4_results_20260923T1639Z/data/bank/'
bank = pd.concat([pd.read_csv(B + 'bank_trainval.csv'), pd.read_csv(B + 'bank_test.csv')])
S = json.load(open(R + '_workspace/audit_20260927/ads_compare_scores.json'))
d = bank[bank.split.isin(['val', 'test']) & (bank.MUSIC_FAKE_mask == 1) & (bank.MUSIC_PRESENT == 1)].copy()
val_all = bank[bank.split == 'val'].ID
for k, m in (('mu', 'music_6_12'), ('df', 'voice_df'), ('ad', 'voice_ad')):
    d[k] = d.ID.map(S[m]); t = dh.quantile_table([S[m][i] for i in val_all if i in S[m]])
    d['c_' + k] = [dh.to_cdf(t, x) if x == x else np.nan for x in d[k]]
d = d.dropna(subset=['mu']).reset_index(drop=True)
def eer(y, s):
    fpr, tpr, _ = roc_curve(y, s, drop_intermediate=False); i = np.argmin(abs(fpr - (1 - tpr))); return (fpr[i] + 1 - tpr[i]) / 2
use = (d.VOICE_PRESENT == 1) & d.c_df.notna() & d.c_ad.notna()
def sc(df, a, b):
    u = use[df.index]
    return np.where(u, (df.c_mu + a * df.c_df.fillna(0) + b * df.c_ad.fillna(0)) / (1 + a + b), df.c_mu)
V, T = d[d.split == 'val'], d[d.split == 'test']
print('music rows val/test', len(V), len(T), '| with voice evidence val/test', int(use[V.index].sum()), int(use[T.index].sum()))
grid = [0, .25, .5, 1, 2]
res = {(a, b): eer(V.MUSIC_FAKE, sc(V, a, b)) for a, b in itertools.product(grid, grid)}
best = min(res, key=res.get)
for nm, (a, b) in {'music head only': (0, 0), f'val-best a={best[0]} b={best[1]}': best}.items():
    print(f'{nm:24s} val music EER {eer(V.MUSIC_FAKE, sc(V, a, b)):.4f}  test music EER {eer(T.MUSIC_FAKE, sc(T, a, b)):.4f}')
y = T.MUSIC_FAKE.values; s0, s1 = sc(T, 0, 0), sc(T, *best)
g = T.group_id.values; ug = np.unique(g); idx = {k: np.where(g == k)[0] for k in ug}; rng = np.random.default_rng(0)
dl = [(lambda ii: eer(y[ii], s0[ii]) - eer(y[ii], s1[ii]))(np.concatenate([idx[k] for k in rng.choice(ug, len(ug))])) for _ in range(300)]
print('test music EER gain: median %.4f 95%% CI [%.4f, %.4f], %d groups' % (np.median(dl), *np.quantile(dl, [.025, .975]), len(ug)))
for nm, s in (('head', s0), ('cross', s1)):
    fpr, tpr, th = roc_curve(y, s, drop_intermediate=False); i = np.argmin(abs(fpr - (1 - tpr))); f = s >= th[i]
    tt = T.assign(flag=f)
    print(f'{nm:5s} real-music+fake-voice FPR {tt[(tt.MUSIC_FAKE == 0) & (tt.VOICE_FAKE == 1)].flag.mean():.3f} '
          f'(n {int(((tt.MUSIC_FAKE == 0) & (tt.VOICE_FAKE == 1)).sum())}); fake-music+real-voice miss '
          f'{1 - tt[(tt.MUSIC_FAKE == 1) & (tt.VOICE_FAKE == 0)].flag.mean():.3f} (n {int(((tt.MUSIC_FAKE == 1) & (tt.VOICE_FAKE == 0)).sum())})')
