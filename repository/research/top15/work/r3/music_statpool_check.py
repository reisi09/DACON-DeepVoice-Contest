"""Q: does a FILE-level statistics-pooled frozen MERT representation (mean+std over 4.04 s windows) with a plain
logistic-regression head -- Pascu et al. 2024 (LR on frozen SSL beats MLP out of domain), SONICS/Sofia (global
context/structure) -- beat the WindowMIL music head (DB protocol, MERT 6+12) on held-out MUSIC EER?
Train split only; C and layer set chosen on VAL; one held-out evaluation. Cache: _workspace/backbone_cache (fp16).
Read-only, CPU; memory bounded by using layer subsets."""
import json, sys, numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_curve
R = '/Users/choiwuseck/Documents/ChatGPT/데이콘/'
B = R + 'results/data_v4_20260923/data_v4_results_20260923T1639Z/data/bank/'
bank = pd.concat([pd.read_csv(B + 'bank_trainval.csv'), pd.read_csv(B + 'bank_test.csv')])
S = json.load(open(R + '_workspace/audit_20260927/ads_compare_scores.json'))
d = bank[(bank.MUSIC_FAKE_mask == 1) & (bank.MUSIC_PRESENT == 1)].reset_index(drop=True)
LAY = [3, 6, 9, 12, 15, 18, 21, 24]
X, keep = [], []
for i, r in d.iterrows():
    try:
        z = np.load(R + '_workspace/backbone_cache/' + r.ID + '.npz')
    except FileNotFoundError:
        continue
    m = z['mert_layers'].astype(np.float32)          # windows x 8 x 1024
    X.append(m.mean(0).astype(np.float16)); keep.append(i)   # 8 x 1024 (std dropped: v1 run showed no gain and exceeded 4 GB)
d = d.loc[keep].reset_index(drop=True); X = np.stack(X)
print('music rows by split', d.split.value_counts().to_dict(), 'X', X.shape)
def eer(y, s):
    fpr, tpr, _ = roc_curve(y, s, drop_intermediate=False); i = np.argmin(abs(fpr - (1 - tpr))); return (fpr[i] + 1 - tpr[i]) / 2
tr, va, te = (d.split == s for s in ('train', 'val', 'test'))
y = d.MUSIC_FAKE.values.astype(int)
# balanced class x source weights on train (mirrors the class x source balance idea, simplified)
# v1 used 1/(class x source count) weights: tiny cells dominated and EER ~0.40. v2: class-balanced only.
w = np.where(y[tr] == 1, (y[tr] == 0).sum() / (y[tr] == 1).sum(), 1.)
def feats(layers, std):
    cols = [LAY.index(l) for l in layers]
    return X[:, cols, :].reshape(len(X), -1).astype(np.float32)
res = {}
for layers in ([6, 12], [12], [6], [3, 6, 9, 12]):
    for std in (False,):
        F = feats(layers, std); sc = StandardScaler().fit(F[tr]); Ftr, Fva, Fte = sc.transform(F[tr]), sc.transform(F[va]), sc.transform(F[te])
        for C in (1e-4, 1e-3, 1e-2, 1e-1, 1.):
            m = LogisticRegression(C=C, max_iter=1000).fit(Ftr, y[tr], sample_weight=w)
            res[(tuple(layers), std, C)] = (eer(y[va], m.decision_function(Fva)), eer(y[te], m.decision_function(Fte)))
        del F, Ftr, Fva, Fte
for k, (v, t) in sorted(res.items(), key=lambda kv: kv[1][0]):
    print(f'layers {str(k[0]):28s} std {int(k[1])} C {k[2]:g}: val EER {v:.4f} test EER {t:.4f}')
best = min(res, key=lambda k: res[k][0])
ref_ids = d[te].ID; ref = ref_ids.map(S['music_6_12'])
ok = ref.notna().values
print('\nval-selected:', best, 'val %.4f test %.4f' % res[best])
print('WindowMIL MERT 6+12 (DB protocol, ads_compare_scores) test EER on the same test rows with a score: %.4f (n %d of %d)'
      % (eer(y[te][ok], ref.values[ok]), ok.sum(), te.sum()))
