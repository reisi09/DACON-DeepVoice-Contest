"""Backbones study head runs: copy of research/top15/work/r2/run.py (DB protocol: train split -> val early stop; test
predictions saved, not scored here), unchanged for the R2 groups, plus two groups on the new frozen-encoder cache
(extract_new_backbones.py; directory $NEWCACHE, default ./cache):
  music_new VARIANT   WindowMIL music head (R2 music base config) on concatenated window features, e.g.
                      muq6, muq4_8, mert6_12+muq6, eat12, mert6_12+muq6+eat12, mert6_12 (control = R2 music__L6_12)
  voice_new VARIANT   voice-route R2 'mlp' head (Standardize -> 256 GELU -> 2, window max, 150/10) on e.g. muqv6;
                      'ad' = AntiDeepfake embeddings (control = R2 voice_ad__mlp_long)
Sources: mert/ad = _workspace/backbone_cache, muq/muqv/eat = $NEWCACHE (muq_backing / muq_voice / eat_backing).

  python run.py GROUP VARIANT
"""
from __future__ import annotations

import csv
import json
import sys
import time
import os
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
NEW = Path(os.environ.get('NEWCACHE', Path(__file__).resolve().parent / 'cache'))
S6 = Path(os.environ.get('S6CACHE', Path(__file__).resolve().parents[1] / 's6' / 'cache'))
# rows pinned to R2's 31,270 backbone rows (the live backbone_cache keeps growing; R2 components exist only for these)
PINNED = set(Path(os.environ.get('ROWS', Path(__file__).resolve().parent / 'rows_r2.txt')).read_text().split())
SOURCES = {'mert': (CACHE, 'mert_layers', 'layers'), 'ad': (CACHE, 'ad_embeddings', None),
           'muq': (NEW, 'muq_backing', 'muq_layers'), 'muqv': (NEW, 'muq_voice', 'muq_layers'), 'eat': (NEW, 'eat_backing', 'eat_layers'),
           # S6 candidates (research/top15/work2/s6/extract_s6.py, directory $S6CACHE)
           'sslampt': (S6, 'sslam_pt_backing', 'sslam_layers'), 'sslamft': (S6, 'sslam_ft_backing', 'sslam_layers'),
           'wavlm': (S6, 'wavlm_voice', 'wavlm_layers'), 'hubert': (S6, 'hubert_voice', 'hubert_layers')}
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


def parse(variant):
    """'mert6_12+muq4' -> [('mert', [6, 12]), ('muq', [4])]; 'ad' -> [('ad', None)]."""
    out = []
    for token in variant.split('+'):
        name = token.rstrip('0123456789_')
        if name not in SOURCES:
            raise ValueError(f'unknown feature source in {variant!r}: {name!r}')
        out.append((name, [int(k) for k in token[len(name):].split('_')] if token[len(name):] else None))
    return out


def new_items(rows, spec, route):
    """H.items_for / music_items semantics on features concatenated over sources (all on the runtime windows):
    labels from T.head_labels, music window truth from the old cache, a silent voice stem drops non-voice-route rows."""
    out = []
    for r in rows:
        if not all((SOURCES[n][0] / f"{r['ID']}.npz").exists() for n, _ in spec):
            continue
        parts = []
        for index, (name, layers) in enumerate(spec):
            base, key, layer_key = SOURCES[name]
            with np.load(base / f"{r['ID']}.npz") as d:
                x = d[key]
                if layers is not None:
                    L = list(d[layer_key])
                    x = x[:, [L.index(k) for k in layers]].reshape(len(x), -1)
                if index == 0:
                    silent, samples = bool(d['voice_silent']), int(d['samples'])
            parts.append(x)
        if len({len(x) for x in parts}) != 1:
            raise ValueError(f"window counts differ across sources for {r['ID']}: {[len(x) for x in parts]}")
        f = np.concatenate(parts, 1)
        label = T.head_labels(r).get(route)
        if label is None or (route == 'voice' and r['route'] != 'voice' and silent):
            continue
        truth = np.full(len(f), np.nan, np.float32)
        if route == 'music':
            with np.load(H.OLD[r.get('cache_split', r['split'])] / f"{r['ID']}.npz") as o:
                if len(o['window_truth']) == len(truth):
                    truth = o['window_truth'].astype(np.float32)
        else:
            f = f.astype(np.float32)   # as H.items_for (voice); music stays fp16 like music_items (model promotes)
        out.append((Item({**r, 'route': route}, f, truth, label, silent), samples))
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


def main(group, variant):
    config = {'music_new': {}, 'voice_new': {**LONG, 'arch': 'mlp'}}[group] if group.endswith('_new') else VARIANTS[group][variant]
    cfg = dict(config)
    torch.set_num_threads(4)
    rows = {s: [r for r in csv.DictReader((H.BANK / f).open()) if r['split'] == s and r['ID'] in PINNED]
            for s, f in (('train', 'bank_trainval.csv'), ('val', 'bank_trainval.csv'), ('test', 'bank_test.csv'))}
    arch, layers = cfg.pop('arch', 'mil' if group in ('music', 'music_new') else 'linear'), cfg.pop('layers', [12])
    t0 = time.monotonic()
    if group in ('music_new', 'voice_new'):
        route, fake_index, spec = group.split('_')[0], 0, parse(variant)
        data = {s: new_items(rows[s], spec, route) for s in rows}
        if route == 'music':
            make = lambda items: WindowMIL(*stats(items))
        else:
            def make(items):
                mean, scale = stats(items)
                return nn.Sequential(Standardize(mean, scale), nn.Linear(len(mean), 256), nn.GELU(), nn.Linear(256, 2))
    elif group == 'music':
        route, fake_index = 'music', 0
        data = {s: music_items(rows[s], layers) for s in rows}
        if arch == 'sls':
            make = lambda items: LayerMixMIL(*stats(items), len(layers))
        else:
            make = lambda items: WindowMIL(*stats(items))
    else:
        route = 'voice'
        if group == 'voice_df':
            old = torch.load(H.OLD['val'] / 'voice_classifier_initial.pt', map_location='cpu', weights_only=True)
            init, fake_index = old['state_dict'], old['fake_label_index']
        else:
            native = load_file(str(H.AD / 'model.safetensors'))
            init, fake_index = {'weight': native['proj_fc.weight'].float(), 'bias': native['proj_fc.bias'].float()}, 0
        data = {s: H.items_for(rows[s], CACHE, [12], 'voice', dfarena=group == 'voice_df') for s in rows}

        def make(items):
            if arch == 'linear':
                m = nn.Linear(1280, 2)
                m.load_state_dict(init)
                return m
            mean, scale = stats(items)
            if arch == 'resmlp':
                return ResMLP(init, mean, scale)
            return nn.Sequential(Standardize(mean, scale), nn.Linear(1280, 256), nn.GELU(), nn.Linear(256, 2))
    load_s = time.monotonic() - t0
    res = {'group': group, 'variant': variant, 'arch': arch, 'layers': layers, 'config': config,
           'load_seconds': round(load_s, 1), 'n': {s: len(v) for s, v in data.items()}, 'seeds': {},
           'val_ids': [x.row['ID'] for x, _ in data['val']], 'val_labels': [int(x.label) for x, _ in data['val']],
           'test_ids': [x.row['ID'] for x, _ in data['test']], 'test_labels': [int(x.label) for x, _ in data['test']]}
    val_preds, test_preds = [], []
    for seed in (41, 42, 43):
        model, info = train(route, data['train'], data['val'], make, fake_index, seed, **cfg)
        vp = T.predict(model, [x for x, _ in data['val']], route, fake_index, 256, 10 ** 6, 'cpu')
        tp = T.predict(model, [x for x, _ in data['test']], route, fake_index, 256, 10 ** 6, 'cpu')
        if arch == 'sls':
            info['layer_weights'] = [round(float(w), 4) for w in model.head.alpha.detach().softmax(0)]
        info['n_params'] = sum(p.numel() for p in model.parameters())
        res['seeds'][seed] = info
        val_preds.append(vp.tolist())
        test_preds.append(tp.tolist())
        print(group, variant, seed, {k: info[k] for k in ('val_eer', 'best_epoch', 'epochs_run', 'updates_to_best',
                                                           'train_seconds', 'hit_cap')}, flush=True)
    res['val_preds'], res['test_preds'] = val_preds, test_preds
    ens = np.mean(val_preds, 0)
    res['val_ens_eer'] = float(_roc_metrics(res['val_labels'], ens.tolist())[0])
    print(group, variant, 'val_ens_eer', round(res['val_ens_eer'], 5), flush=True)
    (OUT / f'{group}__{variant}.json').write_text(json.dumps(res))


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2])
