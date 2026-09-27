"""Q: do the DB-protocol TRAINED voice heads (DF-Arena fc5 heads and AntiDeepfake-embedding heads, train split only,
3-seed mean; _workspace/audit_20260927/ads_compare_scores.json) still over-flag narrowband phone (tel_opus8) or
enhanced real voice on the held-out test, as the zero-shot AD model does? This decides whether codec/channel
augmentation (AT-ADD top systems) is worth GPU time. Threshold = held-out voice EER threshold per member. Read-only."""
import json, numpy as np, pandas as pd
from sklearn.metrics import roc_curve
R = '/Users/choiwuseck/Documents/ChatGPT/데이콘/'
bank = pd.read_csv(R + 'results/data_v4_20260923/data_v4_results_20260923T1639Z/data/bank/bank_test.csv')
S = json.load(open(R + '_workspace/audit_20260927/ads_compare_scores.json'))
t = bank[(bank.VOICE_FAKE_mask == 1) & (bank.VOICE_PRESENT == 1)].copy()
for k in ('voice_df', 'voice_ad'):
    t[k] = t.ID.map(S[k])
t = t.dropna(subset=['voice_df', 'voice_ad'])
print('held-out voice rows with both members:', len(t), 'real', int((t.VOICE_FAKE == 0).sum()))
for k in ('voice_df', 'voice_ad'):
    fpr, tpr, th = roc_curve(t.VOICE_FAKE, t[k], drop_intermediate=False); i = np.argmin(abs(fpr - (1 - tpr)))
    t[k + '_flag'] = t[k] >= th[i]
    print(f'{k}: EER {(fpr[i] + 1 - tpr[i]) / 2:.4f} thr {th[i]:.4f}')
real = t[t.VOICE_FAKE == 0]; fake = t[t.VOICE_FAKE == 1]
print('\nREAL voice FPR at EER threshold, by codec_chain:')
print(real.groupby('codec_chain')[['voice_df_flag', 'voice_ad_flag']].agg(['size', 'mean']).round(3).to_string())
print('\nREAL voice FPR by source:')
print(real.groupby('source')[['voice_df_flag', 'voice_ad_flag']].agg(['size', 'mean']).round(3).to_string())
print('\nFAKE voice miss rate by codec_chain:')
print((1 - fake.groupby('codec_chain')[['voice_df_flag', 'voice_ad_flag']].mean()).round(3).to_string())
