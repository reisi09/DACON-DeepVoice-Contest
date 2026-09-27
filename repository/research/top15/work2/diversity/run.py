"""Diversity study runs = R2 run.py (research/top15/work/r2) unchanged except main(): the R2 val-chosen head designs
(voice_df mlp_long: DF-Arena fc5 inputs -> 2-layer MLP, 150/10; music L6_12: MERT 6+12 WindowMIL, 50/5) are trained on
train SUBSETS from subsets.py (val early stop and held-out test untouched, DB protocol). Data are loaded once per process
from the frozen R2 row universe (universe.txt).

  python run.py GROUP PREFIX [PREFIX ...]   GROUP in voice_df, music; runs every subsets.configs() key starting with a PREFIX
"""
from __future__ import annotations

import csv
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import load_file
from torch import nn

ROOT = Path('/Users/choiwuseck/Documents/ChatGPT/데이콘')
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'data_v4'))
import train_backbone_heads as H  # noqa: E402
import full_bank_train as T  # noqa: E402
from eight_heads import FrozenHead  # noqa: E402
from full_bank_batches import Item, duration_balance, epoch_order, loss_weights  # noqa: E402
from graph_scoring import _roc_metrics  # noqa: E402
from localization_models import WindowMIL  # noqa: E402

CACHE = ROOT / '_workspace/backbone_cache'
OUT = Path(__file__).resolve().parent / 'runs'
import subsets as S  # noqa: E402
DESIGN = {'voice_df': 'mlp_long', 'music': 'L6_12'}
SEEDS = (41, 42)
ALL = [3, 6, 9, 12, 15, 18, 21, 24]
LONG = dict(max_epochs=150, patience=10)
VARIANTS = {
    'voice_df': {'base': {}, 'long': LONG, 'long_lr3e-4': {**LONG, 'lr': 3e-4},
                 'resmlp_long': {**LONG, 'arch': 'resmlp'}, 'mlp_long': {**LONG, 'arch': 'mlp'}},
    'voice_ad': {'base': {}, 'long': LONG, 'long_lr3e-4': {**LONG, 'lr': 3e-4},
                 'resmlp_long': {**LONG, 'arch': 'resmlp'}, 'mlp_long': {**LONG, 'arch': 'mlp'},
                 'long_b64': {**LONG, 'batch': 64}, 'long_b1024': {**LONG, 'batch': 1024}},
    'music': {'L6_12': {'layers': [6, 12]}, 'L3_6_9': {'layers': [3, 6, 9]}, 'L3_6_9_12': {'layers': [3, 6, 9, 12]},
              'concat8': {'layers': ALL}, 'sls8': {'layers': ALL, 'arch': 'sls'}},
}


def music_items(rows, layers):
    """H.items_for(route 'music') with fp16 storage (exact: cache is fp16; model promotes to fp32 on subtraction)."""
    out = []
    for r in rows:
        p = CACHE / f"{r['ID']}.npz"
        if not p.exists():
            continue
        with np.load(p) as d:
            silent, L = bool(d['voice_silent']), list(d['layers'])
            f = d['mert_layers'][:, [L.index(k) for k in layers]].reshape(len(d['mert_layers']), -1)
            samples = int(d['samples'])
        truth = np.full(len(f), np.nan, np.float32)
        with np.load(H.OLD[r.get('cache_split', r['split'])] / f"{r['ID']}.npz") as o:
            if len(o['window_truth']) == len(truth):
                truth = o['window_truth'].astype(np.float32)
        label = T.head_labels(r).get('music')
        if label is not None:
            out.append((Item({**r, 'route': 'music'}, f, truth, label, silent), samples))
    return out


class Standardize(nn.Module):
    def __init__(self, mean, scale):
        super().__init__()
        self.register_buffer('mean', mean)
        self.register_buffer('scale', scale)

    def forward(self, x):
        return (x - self.mean) / self.scale


class ResMLP(nn.Module):
    """Pretrained Linear + zero-initialised 2-layer branch on standardised input: equals the Linear head at step 0."""
    def __init__(self, init, mean, scale, hidden=256):
        super().__init__()
        self.lin = nn.Linear(1280, 2)
        self.lin.load_state_dict(init)
        self.branch = nn.Sequential(Standardize(mean, scale), nn.Linear(1280, hidden), nn.GELU(), nn.Linear(hidden, 2))
        nn.init.zeros_(self.branch[-1].weight)
        nn.init.zeros_(self.branch[-1].bias)

    def forward(self, x):
        return self.lin(x) + self.branch(x)


class MixHead(nn.Module):
    """Softmax-weighted sum of per-layer standardised MERT layers (SUPERB/SLS-style), then the FrozenHead MLP."""
    def __init__(self, n):
        super().__init__()
        self.n, self.alpha, self.mlp = n, nn.Parameter(torch.zeros(n)), FrozenHead(1024, 'mlp')

    def forward(self, x):
        x = x.view(len(x), self.n, -1)
        return self.mlp((self.alpha.softmax(0)[:, None] * x).sum(1))


class LayerMixMIL(nn.Module):
    """Same interface T.forward uses for WindowMIL: .mean/.scale buffers and .head returning (logits, hidden)."""
    def __init__(self, mean, scale, n):
        super().__init__()
        self.register_buffer('mean', mean.float())
        self.register_buffer('scale', scale.float())
        self.head = MixHead(n)


def stats(items):
    means = np.stack([x.features.astype(np.float32).mean(0) for x in items])
    return torch.tensor(means.mean(0)), torch.tensor(np.maximum(means.std(0), 1e-6))


def train(route, training, validation, make, fake_index, seed, max_epochs=50, patience=5, lr=None, batch=256):
    items = [x for x, _ in training]
    weights = loss_weights(items)
    ids = [x.row['ID'] for x in items]
    factor = duration_balance([x.label for x in items], [s for _, s in training], [weights[i] for i in ids])
    weights = {i: weights[i] * f for i, f in zip(ids, factor, strict=True)}
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = make(items)
    val = [x for x, _ in validation]
    limit = max(sum(len(x.features) for x in sorted(items, key=lambda x: len(x.features), reverse=True)[:256]), 10 ** 6)
    lr = lr or (1e-3 if route == 'music' else 1e-4)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=.01)
    labels = np.array([x.label for x in val])
    best, state, stale, hist, updates, best_epoch, best_updates = (float('inf'), float('inf')), None, 0, [], 0, 0, 0
    start = time.monotonic()
    for epoch in range(1, max_epochs + 1):
        if stale >= patience:
            break
        model.train()
        order = epoch_order(items, seed, epoch)
        loss = 0.
        for offset in range(0, len(order), batch):
            chunk = [items[i] for i in order[offset:offset + batch]]
            loss += T.step(model, optimizer, chunk, route, fake_index, weights, 256, limit, 'cpu') * len(chunk)
            updates += 1
        pred = T.predict(model, val, route, fake_index, 256, limit, 'cpu')
        eer = float(_roc_metrics([int(v) for v in labels], pred.tolist())[0])
        clipped = np.clip(pred, 1e-7, 1 - 1e-7)
        bce = float(-(labels * np.log(clipped) + (1 - labels) * np.log1p(-clipped)).mean())
        improved = (eer, bce) < best
        if improved:
            best, stale, best_epoch, best_updates = (eer, bce), 0, epoch, updates
            state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            stale += 1
        hist.append([epoch, round(eer, 5), round(bce, 5), round(loss / len(items), 5), improved])
    seconds = time.monotonic() - start
    model.load_state_dict(state)
    return model, {'val_eer': best[0], 'val_bce': best[1], 'best_epoch': best_epoch, 'epochs_run': len(hist),
                   'updates_to_best': best_updates, 'updates_total': updates, 'updates_per_epoch': -(-len(items) // batch),
                   'train_seconds': round(seconds, 1), 'sec_per_epoch': round(seconds / len(hist), 2),
                   'hit_cap': best_epoch == max_epochs, 'n_train': len(items), 'history': hist}


def main(group, prefixes):
    variant = DESIGN[group]
    cfg = dict(VARIANTS[group][variant])
    torch.set_num_threads(4)
    rows = S.load_rows()
    arch, layers = cfg.pop('arch', 'linear' if group != 'music' else 'mil'), cfg.pop('layers', [12])
    t0 = time.monotonic()
    if group == 'music':
        route, fake_index = 'music', 0
        data = {s: music_items(rows[s], layers) for s in rows}
        make = lambda items: WindowMIL(*stats(items))
    else:
        route = 'voice'
        old = torch.load(H.OLD['val'] / 'voice_classifier_initial.pt', map_location='cpu', weights_only=True)
        fake_index = old['fake_label_index']
        data = {s: H.items_for(rows[s], CACHE, [12], 'voice', dfarena=True) for s in rows}
        assert arch == 'mlp'
        make = lambda items: nn.Sequential(Standardize(*stats(items)), nn.Linear(1280, 256), nn.GELU(), nn.Linear(256, 2))
    load_s = time.monotonic() - t0
    todo = {k: v for k, v in S.configs(rows['train']).items() if any(k.startswith(p) for p in prefixes)}
    print(group, 'loaded', {s: len(v) for s, v in data.items()}, round(load_s, 1), 's; configs', list(todo), flush=True)
    for name, keep in todo.items():
        out = OUT / f'{group}__{name}.json'
        if out.exists():
            continue
        train_items = [x for x in data['train'] if x[0].row['ID'] in keep]
        res = {'group': group, 'variant': variant, 'subset': name, 'arch': arch, 'layers': layers,
               'n': {'train': len(train_items), 'val': len(data['val']), 'test': len(data['test'])}, 'seeds': {},
               'val_ids': [x.row['ID'] for x, _ in data['val']], 'val_labels': [int(x.label) for x, _ in data['val']],
               'test_ids': [x.row['ID'] for x, _ in data['test']], 'test_labels': [int(x.label) for x, _ in data['test']]}
        val_preds, test_preds = [], []
        for seed in SEEDS:
            model, info = train(route, train_items, data['val'], make, fake_index, seed, **cfg)
            vp = T.predict(model, [x for x, _ in data['val']], route, fake_index, 256, 10 ** 6, 'cpu')
            tp = T.predict(model, [x for x, _ in data['test']], route, fake_index, 256, 10 ** 6, 'cpu')
            info.pop('history')
            res['seeds'][seed] = info
            val_preds.append(vp.tolist())
            test_preds.append(tp.tolist())
            print(group, name, seed, {k: info[k] for k in ('val_eer', 'best_epoch', 'epochs_run', 'train_seconds', 'n_train')}, flush=True)
        res['val_preds'], res['test_preds'] = val_preds, test_preds
        res['val_ens_eer'] = float(_roc_metrics(res['val_labels'], np.mean(val_preds, 0).tolist())[0])
        print(group, name, 'val_ens_eer', round(res['val_ens_eer'], 5), flush=True)
        out.write_text(json.dumps(res))


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2:])
