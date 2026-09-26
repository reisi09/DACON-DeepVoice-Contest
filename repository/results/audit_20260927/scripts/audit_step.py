"""Audit A6: one real optimizer step through full_bank_train.step on real cached data v4 features (CPU fp32).
Checks: finite loss; nonzero finite grads on every trainable tensor; every tensor changes after the step; the val guard fires."""
import csv, sys, numpy as np, torch
from pathlib import Path
from torch import nn
sys.path.insert(0, '.')
import full_bank_train as T
from localization_models import WindowMIL
B = 'results/data_v4_20260923/data_v4_results_20260923T1639Z/data/bank/'
C = Path('_workspace/cache_v4/cache_tv')
rows = list(csv.DictReader(open(B + 'bank_trainval.csv')))
rng = np.random.default_rng(0)
def items(route, split, n):
    pool = [r for r in rows if r['split'] == split and route in T.head_labels(r)]
    out = []
    for r in rng.permutation(pool)[:n]:
        with np.load(C / f"{r['ID']}.npz") as d:
            f = d['mert_windows' if route == 'music' else 'voice_embeddings'].copy()
            truth = d['window_truth'].copy() if route == 'music' else np.full(len(f), np.nan, np.float32)
            silent = bool(d['voice_silent'])
        out.append(T.Item({**r, 'route': route}, f, truth, T.head_labels(r)[route], silent))
    return out
init = torch.load(C / 'voice_classifier_initial.pt', map_location='cpu', weights_only=True)
for route in ('voice', 'music'):
    tr, va = items(route, 'train', 64), items(route, 'val', 4)
    if route == 'voice':
        model = nn.Linear(1280, 2); model.load_state_dict(init['state_dict'])
    else:
        m = np.stack([x.features.mean(0) for x in tr]); model = WindowMIL(torch.tensor(m.mean(0)), torch.tensor(np.maximum(m.std(0), 1e-6)))
    model.train()
    before = {k: v.detach().clone() for k, v in model.named_parameters()}
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3 if route == 'music' else 1e-4, weight_decay=.01)
    w = {x.row['ID']: 1. for x in tr}
    grads = {}
    orig_step = opt.step
    def spy():
        grads.update({k: float(p.grad.norm()) if p.grad is not None else None for k, p in model.named_parameters()}); orig_step()
    opt.step = spy
    loss = T.step(model, opt, tr, route, init['fake_label_index'], w, 256, 10**9, 'cpu')
    changed = {k: float((p.detach() - before[k]).abs().max()) for k, p in model.named_parameters()}
    try:
        T.step(model, opt, va, route, init['fake_label_index'], {x.row['ID']: 1. for x in va}, 256, 10**9, 'cpu'); guard = 'NOT RAISED'
    except ValueError as e:
        guard = f'raised: {e}'
    print(route, 'labels', sorted({x.label for x in tr}), 'loss', round(loss, 4), '| grad norms', {k: None if v is None else round(v, 5) for k, v in grads.items()},
          '| max param change', {k: f'{v:.2e}' for k, v in changed.items()}, '| val guard', guard)
