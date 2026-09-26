"""Heads on new frozen backbones (data_v4/extract_backbones.py cache), same objective/weights/duration balance/optimizer/
lr/patience/epochs as full_bank_train.py:
  voice: Linear(1280, 2) on AntiDeepfake-1B voice-stem window embeddings, initialized from the AntiDeepfake classifier,
         file = max over windows (the DF-Arena voice head, other backbone)
  music: WindowMIL on MERT hidden layers of the backing stem (layer 12 = the current music feature), log-mean-exp
Screen (DB protocol): train split -> val early stopping -> one-shot held-out test, one seed, then official ADS for
voice/music swaps and CDF fusions with the current heads (FILE ensemble refitted on val for each configuration).

  python data_v4/train_backbone_heads.py --cache BACKBONE_CACHE --out DIR [--layers 12 6 ...] [--final]
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import load_file
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'data_v4'))
import decision_heads as dh  # noqa: E402
import full_bank_train as T  # noqa: E402
from component_fusion import fuse  # noqa: E402
from file_ensemble import fit as fit_file  # noqa: E402
from full_bank_batches import Item, duration_balance, epoch_order, loss_weights  # noqa: E402
from full_bank_evaluate import score_rows  # noqa: E402
from graph_scoring import _roc_metrics, official_metrics  # noqa: E402
from localization_models import WindowMIL  # noqa: E402

BANK = ROOT / 'results/data_v4_20260923/data_v4_results_20260923T1639Z/data/bank'
OLD = {'val': ROOT / '_workspace/cache_v4/cache_tv', 'train': ROOT / '_workspace/cache_v4/cache_tv', 'test': ROOT / '_workspace/cache_v4/cache_test'}
PRESENCE = ROOT / 'results/data_v4_20260923/data_v4_results_20260923T1639Z/train/presence_head.json'
AD = ROOT / '_workspace/models/xls-r-1b-anti-deepfake'


def items_for(rows, cache: Path, layers, route_filter):
    out = []
    for r in rows:
        p = cache / f"{r['ID']}.npz"
        if not p.exists():
            continue
        with np.load(p) as d:
            silent, L = bool(d['voice_silent']), list(d['layers'])
            feats = {'voice': d['ad_embeddings'].astype(np.float32),
                     'music': d['mert_layers'][:, [L.index(k) for k in layers]].reshape(len(d['mert_layers']), -1).astype(np.float32)}
            samples = int(d['samples'])
        truth = np.full(len(feats['music']), np.nan, np.float32)
        with np.load(OLD[r['split']] / f"{r['ID']}.npz") as o:   # same runtime windows (checked on parity files)
            if len(o['window_truth']) == len(truth):
                truth = o['window_truth'].astype(np.float32)
        for head, label in T.head_labels(r).items():
            if head != route_filter or (head != r['route'] and head == 'voice' and silent):
                continue
            f = feats[head]
            out.append((Item({**r, 'route': head}, f, truth if head == 'music' else np.full(len(f), np.nan, np.float32), label, silent), samples))
    return out


def train(route, training, validation, init=None, seed=41, max_epochs=50, patience=5):
    items = [x for x, _ in training]
    weights = loss_weights(items)
    ids = [x.row['ID'] for x in items]
    factor = duration_balance([x.label for x in items], [s for _, s in training], [weights[i] for i in ids])
    weights = {i: weights[i] * f for i, f in zip(ids, factor, strict=True)}
    torch.manual_seed(seed)
    np.random.seed(seed)
    if route == 'music':
        means = np.stack([x.features.mean(0) for x in items])
        model = WindowMIL(torch.tensor(means.mean(0)), torch.tensor(np.maximum(means.std(0), 1e-6)))
    else:
        model = nn.Linear(1280, 2)
        model.load_state_dict(init)
    val = [x for x, _ in validation]
    limit = max(sum(len(x.features) for x in sorted(items, key=lambda x: len(x.features), reverse=True)[:256]), 10 ** 6)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3 if route == 'music' else 1e-4, weight_decay=.01)
    labels = np.array([x.label for x in val])
    best, state, stale, epochs = (float('inf'), float('inf')), None, 0, 0
    for epoch in range(1, max_epochs + 1):
        if stale >= patience:
            break
        model.train()
        order = epoch_order(items, seed, epoch)
        for offset in range(0, len(order), 256):
            T.step(model, optimizer, [items[i] for i in order[offset:offset + 256]], route, 0, weights, 256, limit, 'cpu')
        pred = T.predict(model, val, route, 0, 256, limit, 'cpu')
        eer = float(_roc_metrics([int(v) for v in labels], pred.tolist())[0])
        clipped = np.clip(pred, 1e-7, 1 - 1e-7)
        bce = float(-(labels * np.log(clipped) + (1 - labels) * np.log1p(-clipped)).mean())
        epochs = epoch
        if (eer, bce) < best:
            best, stale, state = (eer, bce), 0, {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            stale += 1
    model.load_state_dict(state)
    return model, best[0], epochs


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--cache', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--layers', type=int, nargs='+', default=[3, 6, 9, 12, 15, 18, 21, 24])
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    rows = {s: [r for r in csv.DictReader((BANK / f).open()) if r['split'] == s]
            for s, f in (('train', 'bank_trainval.csv'), ('val', 'bank_trainval.csv'), ('test', 'bank_test.csv'))}
    native = load_file(str(AD / 'model.safetensors'))
    init = {'weight': native['proj_fc.weight'].float(), 'bias': native['proj_fc.bias'].float()}
    report, preds = {'rows_with_cache': {}}, {}
    # new voice head (AntiDeepfake embeddings)
    data = {s: items_for(rows[s], a.cache, [12], 'voice') for s in rows}
    report['rows_with_cache']['voice'] = {s: len(v) for s, v in data.items()}
    model, val_eer, epochs = train('voice', data['train'], data['val'], init)
    for s in ('val', 'test'):
        p = T.predict(model, [x for x, _ in data[s]], 'voice', 0, 256, 10 ** 6, 'cpu')
        preds.setdefault('voice_ad', {}).update({x.row['ID']: float(v) for (x, _), v in zip(data[s], p)})
    report['voice_ad'] = {'val_eer': round(val_eer, 5), 'epochs': epochs}
    print('voice_ad', report['voice_ad'], report['rows_with_cache']['voice'], flush=True)
    torch.save(model.state_dict(), a.out / 'voice_ad_41.pt')
    # music heads per layer
    for k in a.layers:
        data = {s: items_for(rows[s], a.cache, [k], 'music') for s in rows}
        model, val_eer, epochs = train('music', data['train'], data['val'])
        for s in ('val', 'test'):
            p = T.predict(model, [x for x, _ in data[s]], 'music', 0, 256, 10 ** 6, 'cpu')
            preds.setdefault(f'music_L{k}', {}).update({x.row['ID']: float(v) for (x, _), v in zip(data[s], p)})
        test_items = [x for x, _ in data['test']]
        test_eer = float(_roc_metrics([int(x.label) for x in test_items], [preds[f'music_L{k}'][x.row['ID']] for x in test_items])[0])
        report[f'music_L{k}'] = {'val_eer': round(val_eer, 5), 'test_eer_route': round(test_eer, 5), 'epochs': epochs, 'train_rows': len(data['train'])}
        print(f'music_L{k}', report[f'music_L{k}'], flush=True)
        torch.save(model.state_dict(), a.out / f'music_L{k}_41.pt')
    with (a.out / 'predictions.json').open('w') as f:
        json.dump(preds, f)
    (a.out / 'screen_report.json').write_text(json.dumps(report, indent=1) + '\n')


if __name__ == '__main__':
    main()
