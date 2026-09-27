"""Depth study (work2/depth): R2 run.py (research/top15/work/r2/run.py) copy; train() unchanged.
DB protocol: train split -> val early stop; test predictions saved, not scored here.

  python run.py GROUP VARIANT [VARIANT ...]   (GROUP in fullvoice_db, voice_ad, music; data loaded once per process)
"""
from __future__ import annotations

import csv
import json
import resource
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
from full_bank_batches import Item, duration_balance, epoch_order, loss_weights  # noqa: E402
from graph_scoring import _roc_metrics  # noqa: E402
from localization_models import WindowMIL  # noqa: E402

CACHE = ROOT / '_workspace/backbone_cache'
OUT = Path(__file__).resolve().parent / 'runs'

LONG = dict(max_epochs=150, patience=10)
VOICE = {'mlp256': ('mlp', [256], 0.), 'mlp512': ('mlp', [512], 0.), 'mlp3': ('mlp', [512, 256], 0.),
         'mlp4': ('mlp', [512, 512, 256], 0.), 'resln': ('resln', None, 0.), 'resln_d1': ('resln', None, .1),
         'resln_d3': ('resln', None, .3)}
VARIANTS = {
    'fullvoice_db': {k: {**LONG, 'arch': v} for k, v in VOICE.items()},
    'voice_ad': {**{k: {**LONG, 'arch': v} for k, v in VOICE.items()}, 'resmlp': {**LONG, 'arch': ('r2resmlp', None, 0.)},
                 'res_mlp3': {**LONG, 'arch': ('res', ('mlp', [512, 256], 0.), 0.)},
                 'res_resln_d1': {**LONG, 'arch': ('res', ('resln', None, .1), 0.)}},
    'music': {'mil128_r2': {'arch': ('r2mil', None, 0.)}, 'mil128': {**LONG, 'arch': ('r2mil', None, 0.)},
              'wide512': {**LONG, 'arch': ('mlp', [512], 0.)}, 'deep': {**LONG, 'arch': ('mlp', [512, 128], 0.)},
              'deep_d1': {**LONG, 'arch': ('mlp', [512, 128], .1)}, 'deep_d3': {**LONG, 'arch': ('mlp', [512, 128], .3)},
              'resln': {**LONG, 'arch': ('resln', None, 0.)}, 'resln_d1': {**LONG, 'arch': ('resln', None, .1)}},
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


class ResLN(nn.Module):
    """Linear(din, w) -> blocks x [x + W2(drop(GELU(W1(LN x))))] -> LN -> Linear(w, dout)."""
    def __init__(self, din, dout, p=0., width=512, blocks=2):
        super().__init__()
        self.inp = nn.Linear(din, width)
        self.blocks = nn.ModuleList(nn.Sequential(nn.LayerNorm(width), nn.Linear(width, width), nn.GELU(), nn.Dropout(p),
                                                  nn.Linear(width, width)) for _ in range(blocks))
        self.norm, self.out = nn.LayerNorm(width), nn.Linear(width, dout)

    def forward(self, x):
        x = self.inp(x)
        for b in self.blocks:
            x = x + b(x)
        return self.out(self.norm(x))


def net(spec, din, dout):
    kind, hidden, p = spec
    if kind == 'resln':
        return ResLN(din, dout, p)
    layers, d = [], din
    for h in hidden:
        layers += [nn.Linear(d, h), nn.GELU()] + ([nn.Dropout(p)] if p else [])
        d = h
    return nn.Sequential(*layers, nn.Linear(d, dout))


def last_linear(m):
    return m.out if isinstance(m, ResLN) else m[-1]


class ResBranch(nn.Module):
    """Pretrained AD Linear + zero-initialised deeper branch on standardised input (generalises R2 ResMLP)."""
    def __init__(self, init, mean, scale, spec):
        super().__init__()
        self.lin = nn.Linear(1280, 2)
        self.lin.load_state_dict(init)
        self.branch = nn.Sequential(Standardize(mean, scale), net(spec, 1280, 2))
        nn.init.zeros_(last_linear(self.branch[1]).weight)
        nn.init.zeros_(last_linear(self.branch[1]).bias)

    def forward(self, x):
        return self.lin(x) + self.branch(x)


class MILHead(nn.Module):
    """T.forward interface for music: .mean/.scale buffers, .head(x) -> (window logits, hidden)."""
    def __init__(self, mean, scale, spec):
        super().__init__()
        self.register_buffer('mean', mean.float())
        self.register_buffer('scale', scale.float())
        self.head = _Squeeze(net(spec, 2048, 1))


class _Squeeze(nn.Module):
    def __init__(self, m):
        super().__init__()
        self.m = m

    def forward(self, x):
        return self.m(x).squeeze(-1), x


def voice_items(rows, cache, force_split=None):
    """r2/full_voice.py voice_items verbatim: full_bank_train.read_items semantics for the voice head."""
    out = []
    for r in rows:
        with np.load(cache / f"{r['ID']}.npz") as d:
            f, silent, samples = d['voice_embeddings'].copy(), bool(d['voice_silent']), int(d['samples'])
        for head, label in T.head_labels(r).items():
            if head != 'voice' or (head != r['route'] and silent):
                continue
            out.append((Item({**r, 'route': 'voice', **({'split': force_split} if force_split else {})}, f,
                             np.full(len(f), np.nan, np.float32), label, silent), samples))
    return out


def load(group):
    if group == 'fullvoice_db':   # r2/full_voice.py main('db', ...)
        tv = list(csv.DictReader((H.BANK / 'bank_trainval.csv').open()))
        items = voice_items(tv, H.OLD['train'])
        data = {'train': [x for x in items if x[0].row['split'] == 'train'], 'val': [x for x in items if x[0].row['split'] == 'val'],
                'test': voice_items(list(csv.DictReader((H.BANK / 'bank_test_as_val.csv').open())), H.OLD['test'], 'test')}
        old = torch.load(H.OLD['val'] / 'voice_classifier_initial.pt', map_location='cpu', weights_only=True)
        return data, 'voice', old['fake_label_index'], None
    # backbone_cache grew 31,270 -> 35,609 files at 14:00-15:01 KST (another lane); keep R2's frozen 31,270 rows
    frozen = set(json.loads((Path(__file__).resolve().parent / 'rows_r2.json').read_text())['ids'])
    rows = {s: [r for r in csv.DictReader((H.BANK / f).open()) if r['split'] == s and r['ID'] in frozen]
            for s, f in (('train', 'bank_trainval.csv'), ('val', 'bank_trainval.csv'), ('test', 'bank_test.csv'))}
    if group == 'music':
        return {s: music_items(rows[s], [6, 12]) for s in rows}, 'music', 0, None
    native = load_file(str(H.AD / 'model.safetensors'))
    init = {'weight': native['proj_fc.weight'].float(), 'bias': native['proj_fc.bias'].float()}
    return {s: H.items_for(rows[s], CACHE, [12], 'voice') for s in rows}, 'voice', 0, init


def maker(group, spec, init):
    kind = spec[0]

    def make(items):
        mean, scale = stats(items)
        if group == 'music':
            return WindowMIL(mean, scale) if kind == 'r2mil' else MILHead(mean, scale, spec)
        if kind == 'r2resmlp':
            return ResMLP(init, mean, scale)
        if kind == 'res':
            return ResBranch(init, mean, scale, spec[1])
        return nn.Sequential(Standardize(mean, scale), net(spec, 1280, 2))   # same init order as R2 mlp
    return make


def main(group, variants):
    torch.set_num_threads(4)
    t0 = time.monotonic()
    data, route, fake_index, init = load(group)
    load_s = time.monotonic() - t0
    print(group, 'loaded', {s: len(v) for s, v in data.items()}, round(load_s, 1), 's', flush=True)
    for variant in variants:
        path = OUT / f'{group}__{variant}.json'
        if path.exists():
            continue
        cfg = dict(VARIANTS[group][variant])
        make = maker(group, cfg.pop('arch'), init)
        res = {'group': group, 'variant': variant, 'config': repr(VARIANTS[group][variant]), 'load_seconds': round(load_s, 1),
               'n': {s: len(v) for s, v in data.items()}, 'seeds': {},
               'val_ids': [x.row['ID'] for x, _ in data['val']], 'val_labels': [int(x.label) for x, _ in data['val']],
               'test_ids': [x.row['ID'] for x, _ in data['test']], 'test_labels': [int(x.label) for x, _ in data['test']]}
        val_preds, test_preds = [], []
        for seed in (41, 42, 43):
            model, info = train(route, data['train'], data['val'], make, fake_index, seed, **cfg)
            vp = T.predict(model, [x for x, _ in data['val']], route, fake_index, 256, 10 ** 6, 'cpu')
            tp = T.predict(model, [x for x, _ in data['test']], route, fake_index, 256, 10 ** 6, 'cpu')
            info['n_params'] = sum(p.numel() for p in model.parameters())
            info['peak_rss_gb'] = round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2 ** 30, 2)
            res['seeds'][seed] = info
            val_preds.append(vp.tolist())
            test_preds.append(tp.tolist())
            print(group, variant, seed, {k: info[k] for k in ('val_eer', 'best_epoch', 'epochs_run', 'train_seconds',
                                                               'n_params', 'peak_rss_gb')}, flush=True)
        res['val_preds'], res['test_preds'] = val_preds, test_preds
        res['val_ens_eer'] = float(_roc_metrics(res['val_labels'], np.mean(val_preds, 0).tolist())[0])
        print(group, variant, 'val_ens_eer', round(res['val_ens_eer'], 5), flush=True)
        path.write_text(json.dumps(res))


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2:])
