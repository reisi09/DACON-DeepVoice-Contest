"""Audit experiment: does head quality still grow with more independent training groups?

Same objective, weights, duration balance, optimizer, lr, patience and max epochs as full_bank_train.py (DataV4
duration-balanced setup: train split only, val for early stopping), one seed, nested subsets of train *groups*
(group_id) at 12.5/25/50/100 %. Evaluation: val (selection split) and the held-out test split (never trained on).
Records rows, groups, optimizer updates and best epoch, so more-data effects are not confused with longer training.

  .venv/bin/python _workspace/audit_20260927/learning_curve.py --out _workspace/audit_20260927/learning_curve.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import full_bank_train as T  # noqa: E402
from full_bank_batches import duration_balance, epoch_order, loss_weights  # noqa: E402
from graph_scoring import _roc_metrics  # noqa: E402
from localization_models import WindowMIL  # noqa: E402

BANK = ROOT / 'results/data_v4_20260923/data_v4_results_20260923T1639Z/data/bank'
CACHE = ROOT / '_workspace/cache_v4'
FRACTIONS = (.125, .25, .5, 1.)


def subset(training, fraction, seed=20260927):
    groups = sorted({x.row['group_id'] for x in training}, key=lambda g: hashlib.sha256(f'{seed}:{g}'.encode()).hexdigest())
    keep = set(groups[:max(1, round(len(groups) * fraction))])   # nested: a prefix of one fixed group order
    return [x for x in training if x.row['group_id'] in keep], len(keep)


def run(route, training, validation, test, samples, init, fake_index, max_epochs=50, patience=5, seed=41):
    weights = loss_weights(training)
    ids = [x.row['ID'] for x in training]
    factor = duration_balance([x.label for x in training], [samples[i] for i in ids], [weights[i] for i in ids])
    weights = {i: weights[i] * f for i, f in zip(ids, factor, strict=True)}
    torch.manual_seed(seed)
    np.random.seed(seed)
    if route == 'music':
        means = np.stack([x.features.mean(0) for x in training])
        model = WindowMIL(torch.tensor(means.mean(0)), torch.tensor(np.maximum(means.std(0), 1e-6)))
    else:
        model = nn.Linear(1280, 2)
        model.load_state_dict(init)
    limit = max(sum(len(x.features) for x in sorted(training, key=lambda x: len(x.features), reverse=True)[:256]),
                max(len(x.features) for x in validation + test))
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3 if route == 'music' else 1e-4, weight_decay=.01)
    labels = [x.label for x in validation]
    best, best_state, best_epoch, stale, updates, history = (float('inf'), float('inf')), None, 0, 0, 0, []
    for epoch in range(1, max_epochs + 1):
        if stale >= patience:
            break
        model.train()
        order = epoch_order(training, seed, epoch)
        for offset in range(0, len(order), 256):
            T.step(model, optimizer, [training[i] for i in order[offset:offset + 256]], route, fake_index, weights, 256, limit, 'cpu')
            updates += 1
        pred = T.predict(model, validation, route, fake_index, 256, limit, 'cpu')
        eer = float(_roc_metrics([int(v) for v in labels], pred.tolist())[0])
        clipped = np.clip(pred, 1e-7, 1 - 1e-7)
        bce = float(-(np.array(labels) * np.log(clipped) + (1 - np.array(labels)) * np.log1p(-clipped)).mean())
        history.append(round(eer, 5))
        if (eer, bce) < best:
            best, best_epoch, stale = (eer, bce), epoch, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            stale += 1
    model.load_state_dict(best_state)
    test_pred = T.predict(model, test, route, fake_index, 256, limit, 'cpu')
    test_eer = float(_roc_metrics([int(x.label) for x in test], test_pred.tolist())[0])
    return {'val_eer': round(best[0], 5), 'test_eer': round(test_eer, 5), 'best_epoch': best_epoch,
            'epochs_run': len(history), 'updates': updates, 'val_history': history}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', type=Path, required=True)
    a = ap.parse_args()
    started = time.time()
    items = T.read_items(BANK / 'bank_trainval.csv', CACHE / 'cache_tv')
    test_items = T.read_items(BANK / 'bank_test_as_val.csv', CACHE / 'cache_test')
    samples = {k: v['samples'] for c in ('cache_tv', 'cache_test')
               for k, v in json.loads((CACHE / c / 'index.json').read_text())['items'].items()}
    init = torch.load(CACHE / 'cache_tv' / 'voice_classifier_initial.pt', map_location='cpu', weights_only=True)
    report = {'setup': 'DataV4 duration-balanced objective, seed 41, CPU fp32, nested group subsets', 'results': []}
    for route in ('music', 'voice'):
        training = [x for x in items if x.row['route'] == route and x.row['split'] == 'train']
        validation = [x for x in items if x.row['route'] == route and x.row['split'] == 'val']
        test = [x for x in test_items if x.row['route'] == route]
        for fraction in FRACTIONS:
            part, n_groups = subset(training, fraction)
            r = run(route, part, validation, test, samples, init['state_dict'], init['fake_label_index'])
            r.update(route=route, fraction=fraction, rows=len(part), groups=n_groups,
                     positives=int(sum(x.label for x in part)), minutes=round((time.time() - started) / 60, 1))
            report['results'].append(r)
            print(json.dumps({k: v for k, v in r.items() if k != 'val_history'}), flush=True)
            a.out.write_text(json.dumps(report, indent=1) + '\n')


if __name__ == '__main__':
    main()
