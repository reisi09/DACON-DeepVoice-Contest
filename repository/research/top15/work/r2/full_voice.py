"""Shipped-configuration voice head (DF-Arena fc5 inputs, full old cache_v4, all rows) under longer schedules.
  python full_voice.py db  CONFIG   train split -> val early stop; held-out test preds saved (DurBal protocol, CPU)
  python full_voice.py p1  CONFIG   train + held-out test as train -> val early stop (P1 protocol; exact CPU rerun of
                                    _workspace/p1_refit_cpu for CONFIG=base); no held-out estimate exists
CONFIG in base (50/5, lr 1e-4), long (150/10), long_lr3e-4."""
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run import H, T, Item, OUT, LONG, train, _roc_metrics, Standardize, stats  # noqa: E402

CONFIGS = {'base': {}, 'long': LONG, 'long_lr3e-4': {**LONG, 'lr': 3e-4}, 'mlp_long': {**LONG, 'arch': 'mlp'}}
BANK = H.BANK


def voice_items(rows, cache, force_split=None):
    """full_bank_train.read_items semantics for the voice head (no hash re-validation; same arrays)."""
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


def main(protocol, config):
    torch.set_num_threads(4)
    tv = list(csv.DictReader((BANK / 'bank_trainval.csv').open()))
    test = list(csv.DictReader((BANK / 'bank_test_as_val.csv').open()))
    items = voice_items(tv, H.OLD['train'])
    data = {'train': [x for x in items if x[0].row['split'] == 'train'], 'val': [x for x in items if x[0].row['split'] == 'val']}
    held = voice_items(test, H.OLD['test'], 'train' if protocol == 'p1' else 'test')
    if protocol == 'p1':
        data['train'] += held
    else:
        data['test'] = held
    old = torch.load(H.OLD['val'] / 'voice_classifier_initial.pt', map_location='cpu', weights_only=True)

    cfg = dict(CONFIGS[config])
    arch = cfg.pop('arch', 'linear')

    def make(items):
        if arch == 'mlp':
            return torch.nn.Sequential(Standardize(*stats(items)), torch.nn.Linear(1280, 256), torch.nn.GELU(), torch.nn.Linear(256, 2))
        m = torch.nn.Linear(1280, 2)
        m.load_state_dict(old['state_dict'])
        return m
    res = {'group': f'fullvoice_{protocol}', 'variant': config, 'config': CONFIGS[config], 'seeds': {},
           'n': {s: len(v) for s, v in data.items()}, 'val_ids': [x.row['ID'] for x, _ in data['val']],
           'val_labels': [int(x.label) for x, _ in data['val']]}
    if 'test' in data:
        res |= {'test_ids': [x.row['ID'] for x, _ in data['test']], 'test_labels': [int(x.label) for x, _ in data['test']]}
    vps, tps = [], []
    for seed in (41, 42, 43):
        model, info = train('voice', data['train'], data['val'], make, old['fake_label_index'], seed, **cfg)
        vps.append(T.predict(model, [x for x, _ in data['val']], 'voice', old['fake_label_index'], 256, 10 ** 6, 'cpu').tolist())
        if 'test' in data:
            tps.append(T.predict(model, [x for x, _ in data['test']], 'voice', old['fake_label_index'], 256, 10 ** 6, 'cpu').tolist())
        res['seeds'][seed] = info
        print(protocol, config, seed, {k: info[k] for k in ('val_eer', 'best_epoch', 'epochs_run', 'updates_to_best',
                                                            'train_seconds', 'hit_cap', 'n_train')}, flush=True)
    res['val_preds'], res['val_ens_eer'] = vps, float(_roc_metrics(res['val_labels'], np.mean(vps, 0).tolist())[0])
    if tps:
        res['test_preds'] = tps
    print(protocol, config, 'val_ens_eer', round(res['val_ens_eer'], 5), flush=True)
    (OUT / f'fullvoice_{protocol}__{config}.json').write_text(json.dumps(res))


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2])
