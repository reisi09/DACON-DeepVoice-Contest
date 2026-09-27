"""Q: in music-context files (voice + music present), does the file's own MUSIC member score carry usable evidence for
VOICE_FAKE (end-to-end generated songs have fake vocals AND fake music; SONICS-style 'full song' view), or is it a
shortcut that breaks on constructed mixes? Val-only weight selection, one held-out evaluation. Also reports the label
cross-tab VOICE_FAKE x MUSIC_FAKE in music-context voice rows per split. DB-protocol scores, read-only, CPU, seconds."""
import json, sys, itertools, numpy as np, pandas as pd
from sklearn.metrics import roc_curve
R = '/Users/choiwuseck/Documents/ChatGPT/데이콘/'
sys.path.insert(0, R)
import decision_heads as dh
B = R + 'results/data_v4_20260923/data_v4_results_20260923T1639Z/data/bank/'
bank = pd.concat([pd.read_csv(B + 'bank_trainval.csv'), pd.read_csv(B + 'bank_test.csv')])
S = json.load(open(R + '_workspace/audit_20260927/ads_compare_scores.json'))
d = bank[bank.split.isin(['val', 'test']) & (bank.VOICE_FAKE_mask == 1) & (bank.VOICE_PRESENT == 1) & (bank.MUSIC_PRESENT == 1)].copy()
for k, m in (('ours', 'voice_df'), ('ad', 'voice_ad'), ('mu', 'music_6_12')):
    d[k] = d.ID.map(S[m])
print('music-context voice rows val/test:', (d.split == 'val').sum(), (d.split == 'test').sum(),
      '| with music score:', d.dropna(subset=['mu']).split.value_counts().to_dict())
print('\nVOICE_FAKE x MUSIC_FAKE (music-context voice rows):')
print(pd.crosstab([d.split, d.VOICE_FAKE], d.MUSIC_FAKE.fillna(-1)).to_string())
d = d.dropna(subset=['ours', 'ad', 'mu'])
val_all = bank[bank.split == 'val'].ID
for k, m in (('ours', 'voice_df'), ('ad', 'voice_ad'), ('mu', 'music_6_12')):
    t = dh.quantile_table([S[m][i] for i in val_all if i in S[m]]); d['c_' + k] = [dh.to_cdf(t, x) for x in d[k]]
def eer(y, s):
    fpr, tpr, _ = roc_curve(y, s, drop_intermediate=False); i = np.argmin(abs(fpr - (1 - tpr))); return (fpr[i] + 1 - tpr[i]) / 2
sc = lambda df, a, b: (df.c_ours + a * df.c_ad + b * df.c_mu) / (1 + a + b)
V, T = d[d.split == 'val'], d[d.split == 'test']
grid = [0, .25, .5, 1, 2, 4]
res = {(a, b): eer(V.VOICE_FAKE, sc(V, a, b)) for a, b in itertools.product(grid, grid)}
best = min(res, key=res.get)
for name, (a, b) in {'ours': (0, 0), 'ours+ad 0.5 (S3)': (.5, 0), 'music member only-ish (b=4,a=0)': (0, 4), f'val-best a={best[0]} b={best[1]}': best}.items():
    print(f'{name:36s} val EER {eer(V.VOICE_FAKE, sc(V, a, b)):.4f}  test EER {eer(T.VOICE_FAKE, sc(T, a, b)):.4f}')

# Overall held-out VOICE EER (all voice rows) when the val-best cross-component rule is applied to music-context rows
# that have a music score, S3 rule elsewhere; group bootstrap; shortcut failure check on discordant rows.
a, b = best
allv = bank[bank.split.isin(['val', 'test']) & (bank.VOICE_FAKE_mask == 1) & (bank.VOICE_PRESENT == 1)].copy()
for k, m in (('ours', 'voice_df'), ('ad', 'voice_ad'), ('mu', 'music_6_12')):
    allv[k] = allv.ID.map(S[m])
allv = allv.dropna(subset=['ours', 'ad'])
for k, m in (('ours', 'voice_df'), ('ad', 'voice_ad'), ('mu', 'music_6_12')):
    t = dh.quantile_table([S[m][i] for i in val_all if i in S[m]])
    allv['c_' + k] = [dh.to_cdf(t, x) if x == x else np.nan for x in allv[k]]
s3 = (allv.c_ours + .5 * allv.c_ad) / 1.5
use = (allv.MUSIC_PRESENT == 1) & allv.c_mu.notna()
new = np.where(use, (allv.c_ours + a * allv.c_ad + b * allv.c_mu.fillna(0)) / (1 + a + b), s3)
TT = allv.split == 'test'
y = allv.VOICE_FAKE[TT].values; s_old = s3[TT].values; s_new = new[TT.values]
print(f'\nALL held-out voice rows n={TT.sum()}: voice EER S3-rule {eer(y, s_old):.4f} -> cross-component {eer(y, s_new):.4f}')
g = allv.group_id[TT].values; ug = np.unique(g); idx = {k: np.where(g == k)[0] for k in ug}; rng = np.random.default_rng(0)
dl = []
for _ in range(300):
    ii = np.concatenate([idx[k] for k in rng.choice(ug, len(ug))]); dl.append(eer(y[ii], s_old[ii]) - eer(y[ii], s_new[ii]))
print('gain (positive = better) median %.4f 95%% CI [%.4f, %.4f], %d groups' % (np.median(dl), *np.quantile(dl, [.025, .975]), len(ug)))
for nm, s in (('S3', s_old), ('cross', s_new)):
    fpr, tpr, th = roc_curve(y, s, drop_intermediate=False); i = np.argmin(abs(fpr - (1 - tpr))); flag = s >= th[i]
    tt = allv[TT].assign(flag=flag)
    disc_r = tt[(tt.VOICE_FAKE == 0) & (tt.MUSIC_FAKE == 1)]; disc_f = tt[(tt.VOICE_FAKE == 1) & (tt.MUSIC_FAKE == 0) & (tt.MUSIC_PRESENT == 1)]
    print(f'{nm:6s} real-voice+fake-music FPR {disc_r.flag.mean():.3f} (n {len(disc_r)}); fake-voice+real-music miss {1 - disc_f.flag.mean():.3f} (n {len(disc_f)}); '
          f'concordant music-ctx FPR {tt[(tt.VOICE_FAKE == 0) & (tt.MUSIC_FAKE == 0)].flag.mean():.3f}')
print('discordant sources (test):', allv[TT & (allv.VOICE_FAKE != allv.MUSIC_FAKE) & (allv.MUSIC_PRESENT == 1)].groupby(['source', 'VOICE_FAKE']).size().to_dict())
