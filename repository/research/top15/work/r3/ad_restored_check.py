"""Q: does zero-shot AntiDeepfake-1B over-flag REAL voice that is enhanced/restored or phone-coded,
consistent with its post-training convention (FLEURS-R / LibriTTS-R 'restored' speech = artifact class)?
Reads members_all.csv (zero-shot AD scores, val+test) and the data v4 bank. Read-only."""
import pandas as pd, numpy as np
R = '/Users/choiwuseck/Documents/ChatGPT/데이콘/'
B = R + 'results/data_v4_20260923/data_v4_results_20260923T1639Z/data/bank/'
bank = pd.concat([pd.read_csv(B + 'bank_trainval.csv'), pd.read_csv(B + 'bank_test.csv')])
m = pd.read_csv(R + '_workspace/members_20260927/members_all.csv').merge(bank, on='ID')
m = m[(m.voice_silent == 0) & (m.VOICE_PRESENT == 1) & (m.VOICE_FAKE_mask == 1)]
real = m[m.VOICE_FAKE == 0]
thr = np.quantile(m[m.VOICE_FAKE == 1].xlsr1b_max, 0.5)  # median fake score as a fixed reference
print('n voice rows', len(m), 'real', len(real), 'ref thr (median fake xlsr1b_max)', round(thr, 4))
def tab(g):
    return g.groupby(g.columns[0]).agg(n=('s', 'size'), mean=('s', 'mean'), frac_ge_thr=('s', lambda s: (s >= thr).mean()))
real = real.assign(s=real.xlsr1b_max)
print('\nREAL voice by source (split val+test):')
print(tab(real[['source', 's']]).sort_values('mean').round(3).to_string())
print('\nREAL voice by codec_chain:')
print(tab(real[['codec_chain', 's']]).sort_values('mean').round(3).to_string())
from sklearn.metrics import roc_auc_score
for src in ['constructed_enhanced']:
    x = real[real.source == src].s; y = real[real.source != src].s
    if len(x):
        print(f'\nAUC(real {src} vs other real, higher=more fake-like): {roc_auc_score(np.r_[np.ones(len(x)), np.zeros(len(y))], np.r_[x, y]):.3f} (n={len(x)} vs {len(y)})')

# Refinement: compare against the *parent* speech sources only (constructed_enhanced is built from zeroth/fleurs_ko + MUSAN noise -> HTDemucs vocals).
sp = real[real.source.isin(['zeroth', 'fleurs_ko'])]
en = real[real.source == 'constructed_enhanced']
print(f'\nAUC(enhanced real vs zeroth+fleurs_ko real): {roc_auc_score(np.r_[np.ones(len(en)), np.zeros(len(sp))], np.r_[en.s, sp.s]):.3f}  '
      f'median {en.s.median():.3f} vs {sp.s.median():.3f}; frac>0.5: {(en.s > .5).mean():.3f} vs {(sp.s > .5).mean():.3f}')
print('\nspeech-only REAL (zeroth, fleurs_ko) by codec_chain, frac>0.5:')
print(sp.groupby('codec_chain').s.agg(['size', 'median', lambda s: (s > .5).mean()]).round(3).to_string())
print('\nsinging/music-context REAL (vocalset, constructed_song, musan_music, openmic, haim) frac>0.5:',
      round((real[real.source.isin(['vocalset', 'constructed_song', 'musan_music', 'openmic', 'haim'])].s > .5).mean(), 3))
fk = m[m.VOICE_FAKE == 1]
print('FAKE voice frac>0.5:', round((fk.xlsr1b_max > .5).mean(), 3), 'n', len(fk))
