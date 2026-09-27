"""Q: does replacing the voice window MAX with a smoother multi-crop pooling (AT-ADD 2026: median-pooled 4.04 s window
logits, 'conservative multi-crop pooling'; McFee et al. 2018 adaptive pooling) lower held-out VOICE EER and the
length bias, when the pooling is chosen on VAL only?  Uses the DurationBalanced DF-Arena fc5 Linear heads
(_keep_best_20260926/train_duration_balanced/voice_4{1,2,3}.pt; trained on the train split only, with max-MIL) on the
H100 fp32 caches (_workspace/cache_v4). Inference-only change (train/infer pooling mismatch => conservative estimate).
Read-only, CPU."""
import sys, numpy as np, pandas as pd, torch
from scipy.stats import spearmanr
from sklearn.metrics import roc_curve
R = '/Users/choiwuseck/Documents/ChatGPT/데이콘/'
B = R + 'results/data_v4_20260923/data_v4_results_20260923T1639Z/data/bank/'
bank = pd.concat([pd.read_csv(B + 'bank_trainval.csv'), pd.read_csv(B + 'bank_test.csv')])
d = bank[bank.split.isin(['val', 'test']) & (bank.VOICE_FAKE_mask == 1) & (bank.VOICE_PRESENT == 1)].reset_index(drop=True)
heads = []
for s in (41, 42, 43):
    ck = torch.load(R + f'_keep_best_20260926/train_duration_balanced/voice_{s}.pt', map_location='cpu', weights_only=True)
    h = torch.nn.Linear(1280, 2); h.load_state_dict(ck['model']); heads.append((h.eval(), ck['fake_label_index']))
odds, secs, keep = [], [], []
with torch.no_grad():
    for i, r in d.iterrows():
        c = R + '_workspace/cache_v4/' + ('cache_test/' if r.split == 'test' else 'cache_tv/') + r.ID + '.npz'
        try:
            z = np.load(c)
        except FileNotFoundError:
            continue
        if bool(z['voice_silent']):
            continue
        e = torch.from_numpy(z['voice_embeddings']).float()
        o = torch.stack([(lambda v: v[:, fi] - v[:, 1 - fi])(h(e)) for h, fi in heads])  # seeds x windows log-odds
        odds.append(o.numpy()); secs.append(float(z['samples']) / 16000); keep.append(i)
d = d.loc[keep].reset_index(drop=True); d['sec'] = secs
sig = lambda x: 1 / (1 + np.exp(-x))
def pool(o, kind):
    if kind == 'max': return o.max(1)
    if kind == 'mean': return o.mean(1)
    if kind == 'median': return np.median(o, 1)
    if kind.startswith('top'):
        k = int(kind[3:]); return np.sort(o, 1)[:, -min(k, o.shape[1]):].mean(1)
    if kind.startswith('lme'):
        t = float(kind[3:]); return t * (np.logaddexp.reduce(o / t, 1) - np.log(o.shape[1]))
agg = ['max', 'top2', 'top3', 'lme0.5', 'lme1', 'lme2', 'lme4', 'median', 'mean']
for k in agg:   # seed mean of per-seed pooled probability, as in the runtime
    d[k] = [sig(pool(o, k)).mean() for o in odds]
def eer(y, s):
    fpr, tpr, _ = roc_curve(y, s, drop_intermediate=False); i = np.argmin(abs(fpr - (1 - tpr))); return (fpr[i] + 1 - tpr[i]) / 2
def lenbias(df, k):   # n-weighted within-source Spearman(score, duration) among REAL voice files
    num = den = 0
    for _, g in df[df.VOICE_FAKE == 0].groupby('source'):
        if len(g) >= 30 and g.sec.nunique() > 5:
            num += len(g) * spearmanr(g[k], g.sec).correlation; den += len(g)
    return num / den
V, T = d[d.split == 'val'], d[d.split == 'test']
print('rows val/test', len(V), len(T), '| windows median val/test', np.median([o.shape[1] for o, s in zip(odds, d.split) if s == 'val']),
      np.median([o.shape[1] for o, s in zip(odds, d.split) if s == 'test']))
for k in agg:
    print(f'{k:7s} val EER {eer(V.VOICE_FAKE, V[k]):.4f}  test EER {eer(T.VOICE_FAKE, T[k]):.4f}  test len-bias rho(real) {lenbias(T, k):+.3f}'
          f'  test long(>=15s) EER {eer(T[T.sec >= 15].VOICE_FAKE, T[T.sec >= 15][k]):.4f} (n {int((T.sec >= 15).sum())})')
best = min(agg, key=lambda k: eer(V.VOICE_FAKE, V[k]))
print('val-selected pooling:', best)
g = T.group_id.values; ug = np.unique(g); idx = {x: np.where(g == x)[0] for x in ug}; rng = np.random.default_rng(0)
y, a, b = T.VOICE_FAKE.values, T['max'].values, T[best].values
dl = [(lambda ii: eer(y[ii], a[ii]) - eer(y[ii], b[ii]))(np.concatenate([idx[x] for x in rng.choice(ug, len(ug))])) for _ in range(300)]
print('test EER gain val-selected vs max: median %.4f 95%% CI [%.4f, %.4f]' % (np.median(dl), *np.quantile(dl, [.025, .975])))
