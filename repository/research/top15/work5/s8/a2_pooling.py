"""Alternative A2 (s8_lock.json, exploratory, VAL ONLY): window pooling of a voice member, max (runtime) vs mean, on the SAME
heads. Trains the run.py voice_new recipe for VARIANT (seeds 41/42/43, GPU fp32 via HEAD_DEVICE) and writes every-row
predictions of both poolings (per-window softmax fake prob -> max or mean, seed mean) in the run.py 'every' format:
  $RUNS_OUT/voice_new__{VARIANT}.max.json and .mean.json   (silent voice stems -> 0 as T.predict)
  python a2_pooling.py wavlm8"""
import csv
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
from torch import nn

sys.path.insert(0, str(Path('/Users/choiwuseck/Documents/ChatGPT/데이콘/research/top15/work2/backbones')))
import run as R  # noqa: E402

variant = sys.argv[1]
torch.set_num_threads(4)
rows = {s: [r for r in csv.DictReader((R.H.BANK / f).open()) if r['split'] == s and r['ID'] in R.PINNED]
        for s, f in (('train', 'bank_trainval.csv'), ('val', 'bank_trainval.csv'), ('test', 'bank_test.csv'))}
spec = R.parse(variant)
data = {s: R.new_items(rows[s], spec, 'voice') for s in rows}
every = {s: R.every_items(rows[s], spec, 'voice') for s in ('val', 'test')}


def make(items):
    mean, scale = R.stats(items)
    return nn.Sequential(R.Standardize(mean, scale), nn.Linear(len(mean), 256), nn.GELU(), nn.Linear(256, 2))


out = {p: {'every': {s: {'ids': [x.row['ID'] for x in v], 'silent': [bool(x.silent) for x in v], 'preds': []} for s, v in every.items()}}
       for p in ('max', 'mean')}
for seed in (41, 42, 43):
    model, info = R.train('voice', data['train'], data['val'], make, 0, seed, **R.LONG)
    model.eval()
    with torch.no_grad():
        for s, v in every.items():
            pooled = {'max': [], 'mean': []}
            for x in v:
                p = model(torch.from_numpy(x.features).to(R.DEVICE)).float().softmax(-1)[:, 0]
                pooled['max'].append(0. if x.silent else float(p.max()))
                pooled['mean'].append(0. if x.silent else float(p.mean()))
            for k in pooled:
                out[k]['every'][s]['preds'].append(pooled[k])
    print(variant, seed, {k: info[k] for k in ('val_eer', 'best_epoch')}, flush=True)
dest = Path(os.environ['RUNS_OUT'])
for k, res in out.items():
    (dest / f'voice_new__{variant}.{k}.json').write_text(json.dumps(res))
print('written', [f'voice_new__{variant}.{k}.json' for k in out])
