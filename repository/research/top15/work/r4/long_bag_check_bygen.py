# Q: with the current DB voice heads (window max, seed mean, as in full_bank_evaluate.score_rows),
# how does held-out VOICE EER change if files are as long as the official test (4-60 s = 1-15 voice windows)?
# Virtual long bags = concatenation of cached 4 s window embeddings of same-source, same-label rows (no audio, no GPU).
import json, sys, numpy as np, pandas as pd, torch
from torch import nn
from pathlib import Path
from sklearn.metrics import roc_curve

ROOT = Path('/Users/choiwuseck/Documents/ChatGPT/데이콘')
BANK = ROOT / 'results/data_v4_20260923/data_v4_results_20260923T1639Z/data/bank'
HEADS = ROOT / '_keep_best_20260926/train_duration_balanced'
rng = np.random.default_rng(20260927)


def eer(y, s):
    fpr, tpr, _ = roc_curve(y, s, drop_intermediate=False)
    i = np.argmin(np.abs(fpr - (1 - tpr)))
    return float((fpr[i] + 1 - tpr[i]) / 2)


heads = []
for seed in (41, 42, 43):
    ck = torch.load(HEADS / f'voice_{seed}.pt', map_location='cpu', weights_only=True)
    h = nn.Linear(1280, 2); h.load_state_dict(ck['model']); heads.append(h.eval()); fi = ck['fake_label_index']

split = sys.argv[1] if len(sys.argv) > 1 else 'test'
bank = pd.read_csv(BANK / ('bank_test.csv' if split == 'test' else 'bank_trainval.csv'))
bank = bank[(bank.split == split) & (bank.VOICE_PRESENT == 1) & bank.VOICE_FAKE.notna()]
cache = ROOT / '_workspace/cache_v4' / ('cache_test' if split == 'test' else 'cache_tv')
rows = []
with torch.no_grad():
    for r in bank.itertuples():
        f = cache / f'{r.ID}.npz'
        if not f.exists():
            continue
        z = np.load(f)
        if bool(z['voice_silent']):
            continue
        e = torch.from_numpy(z['voice_embeddings'].astype(np.float32))
        w = np.stack([h(e).softmax(-1)[:, fi].numpy() for h in heads])  # seeds x windows
        rows.append((f"{r.source}|{r.generator}", int(r.VOICE_FAKE), w))
print(split, 'rows', len(rows))

nat = np.array([w.max(1).mean() for _, _, w in rows]); y = np.array([l for _, l, _ in rows])
nwin = np.array([w.shape[1] for _, _, w in rows])
out = {'split': split, 'n': len(rows), 'natural_windows_pct': np.percentile(nwin, [10, 50, 90]).tolist(),
       'natural_eer': eer(y, nat)}

by = {}
for i, (src, lab, _) in enumerate(rows):
    by.setdefault((src, lab), []).append(i)


def bag(k, src, lab):
    idx = by[(src, lab)]; parts = []; n = 0
    while n < k:
        w = rows[idx[rng.integers(len(idx))]][2]; parts.append(w); n += w.shape[1]
    return np.concatenate(parts, 1)[:, :k].max(1).mean()


keys = [(src, lab) for src, lab, _ in rows]  # one bag per original row keeps the source x label mix
per_k = {}
for k in (1, 2, 4, 8, 15):
    s = np.array([bag(k, *key) for key in keys]); per_k[k] = s
    out[f'eer_k{k}'] = eer(y, s)
    out[f'mean_real_k{k}'] = float(s[y == 0].mean()); out[f'mean_fake_k{k}'] = float(s[y == 1].mean())
ks = rng.integers(1, 16, len(keys))  # official-like: uniform 1..15 windows
mixed = np.array([bag(int(k), *key) for k, key in zip(ks, keys)])
out['eer_uniform_1_15'] = eer(y, mixed)
# length-normalised alternative (top-2 mean instead of max) on the same uniform bags: data-free comparison
print(json.dumps(out, indent=1))
json.dump(out, open(ROOT / f'research/top15/work/r4/long_bag_{split}_bygen.json', 'w'), indent=1)
