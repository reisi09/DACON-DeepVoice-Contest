"""Every-row (runtime semantics) predictions of the DB-protocol WavLM layer-8 voice member (backbones run.py voice_new
wavlm8, seeds 41/42/43; deterministic): labelled-row predictions must equal runs/voice_new__wavlm8.json."""
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch import nn

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE.parent / 'work2' / 'backbones')]
import run as R  # noqa: E402

ST = HERE.parent / 'work2' / 'stacking'
SPEC = R.parse('wavlm8')


def every_items(rows):
    out = []
    for r in rows:
        with np.load(R.H.OLD[r.get('cache_split', r['split'])] / f"{r['ID']}.npz") as o:
            silent = bool(o['voice_silent'])
        with np.load(R.S6 / f"{r['ID']}.npz") as d:
            L = list(d['wavlm_layers'])
            f = d['wavlm_voice'][:, [L.index(8)]].reshape(len(d['wavlm_voice']), -1).astype(np.float32)
        out.append(R.Item({**r, 'route': 'voice'}, f, np.full(len(f), np.nan, np.float32), 0., silent))
    return out


torch.set_num_threads(4)
import csv  # noqa: E402
rows = {s: [r for r in csv.DictReader((R.H.BANK / f).open()) if r['split'] == s and r['ID'] in R.PINNED]
        for s, f in (('train', 'bank_trainval.csv'), ('val', 'bank_trainval.csv'), ('test', 'bank_test.csv'))}
data = {s: R.new_items(rows[s], SPEC, 'voice') for s in rows}
frozen = set(json.loads((ST / 'frozen_rows.json').read_text())['ids'])
every = {s: every_items([r for r in rows[s] if r['ID'] in frozen]) for s in ('val', 'test')}
ref = json.loads((HERE.parent / 'work2' / 'backbones' / 'runs' / 'voice_new__wavlm8.json').read_text())


def make(items):
    mean, scale = R.stats(items)
    return nn.Sequential(R.Standardize(mean, scale), nn.Linear(len(mean), 256), nn.GELU(), nn.Linear(256, 2))


res = {'every': {s: {'ids': [x.row['ID'] for x in v], 'silent': [bool(x.silent) for x in v], 'preds': []} for s, v in every.items()}, 'max_abs_vs_ref': []}
for i, seed in enumerate((41, 42, 43)):
    model, _ = R.train('voice', data['train'], data['val'], make, 0, seed, **R.LONG)
    vp = R.T.predict(model, [x for x, _ in data['val']], 'voice', 0, 256, 10 ** 6, 'cpu')
    res['max_abs_vs_ref'].append(float(np.abs(vp - np.array(ref['val_preds'][i])).max()))
    for s, v in every.items():
        res['every'][s]['preds'].append(R.T.predict(model, v, 'voice', 0, 256, 10 ** 6, 'cpu').tolist())
    print('seed', seed, 'max |val pred - ref|', res['max_abs_vs_ref'][-1], flush=True)
(HERE / 'runs' / 'voice_wavlm8__every.json').write_text(json.dumps(res))
