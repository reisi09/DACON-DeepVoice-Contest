"""Q: is zero-shot AntiDeepfake XLS-R-2B clearly stronger than 1B on data v4 voice rows (val / held-out test, all and
member-unseen)? Decides whether an AD-2B embedding head is worth ~4.3 GB fp16 weights (> 1.71 GB zip headroom) and ~2x
AD runtime on L4. pandas/sklearn only, <1 min, read-only inputs."""
import sys, numpy as np, pandas as pd
from pathlib import Path
from sklearn.metrics import roc_curve, roc_auc_score
R = Path('/Users/choiwuseck/Documents/ChatGPT/데이콘'); sys.path[:0] = [str(R / 'data_v4'), str(R)]
from fusion_fit import member_seen
B = R / 'results/data_v4_20260923/data_v4_results_20260923T1639Z/data/bank'
bank = pd.concat([pd.read_csv(B / f, dtype=str) for f in ('bank_trainval.csv', 'bank_test.csv')])
m = pd.read_csv(R / '_workspace/members_20260927/members_all.csv').merge(bank, on='ID')
m = m[(m.VOICE_FAKE_mask == '1') & (m.voice_silent == 0)]
m['seen'] = [member_seen(r) for r in m.to_dict('records')]
def eer(y, s):
    f, t, _ = roc_curve(y, s, drop_intermediate=False); i = np.argmin(abs(f - (1 - t))); return (f[i] + 1 - t[i]) / 2
for split in ('val', 'test'):
    for sub in ('all', 'unseen'):
        d = m[(m.split == split) & ((sub == 'all') | ~m.seen)]; y = d.VOICE_FAKE.astype(int)
        print(split, sub, 'n', len(d), ' '.join(f'{c} EER {eer(y, d[c]):.4f} AUC {roc_auc_score(y, d[c]):.4f}' for c in ('xlsr1b_max', 'xlsr2b_max')))
dur = m.samples.astype(float) / 16000 if 'samples' in m else None
print('seconds column vs bank duration corr:', None if dur is None else round(np.corrcoef(m.seconds, dur)[0, 1], 4))
