"""Question: is AntiDeepfake XLS-R 2B zero-shot clearly better than 1B on the held-out VOICE route (both scored on the
voice stem, _workspace/members_20260927/members_all.csv)? If yes, training heads on 2B embeddings is a candidate GPU
lever (not yet extracted); if no, the 1B embedding already captures the family. Also: speech vs music context. CPU, seconds."""
import json, numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score, roc_curve
R = '/Users/choiwuseck/Documents/ChatGPT/데이콘/'
B = R + 'results/data_v4_20260923/data_v4_results_20260923T1639Z/data/bank/'
m = pd.read_csv(R + '_workspace/members_20260927/members_all.csv')
b = pd.concat([pd.read_csv(B + 'bank_trainval.csv'), pd.read_csv(B + 'bank_test.csv')]).merge(m, on='ID')
def eer(y, s):
    fpr, tpr, _ = roc_curve(y, s, drop_intermediate=False); i = np.argmin(np.abs(fpr - 1 + tpr)); return (fpr[i] + 1 - tpr[i]) / 2
out = {}
for sp in ('val', 'test'):
    v = b[(b.split == sp) & (b.VOICE_PRESENT == 1) & (b.VOICE_FAKE_mask == 1) & (b.voice_silent == 0)]
    for ctx, sub in (('all', v), ('speech_ctx', v[v.MUSIC_PRESENT == 0]), ('music_ctx', v[v.MUSIC_PRESENT == 1])):
        if sub.VOICE_FAKE.nunique() < 2: continue
        out[f'{sp}|{ctx}|n={len(sub)}'] = {c: [round(roc_auc_score(sub.VOICE_FAKE, sub[c]), 4), round(eer(sub.VOICE_FAKE, sub[c]), 4)]
                                          for c in ('xlsr1b_max', 'xlsr2b_max', 'xlsr1b_mean', 'xlsr2b_mean')}
print(json.dumps(out, indent=1))
