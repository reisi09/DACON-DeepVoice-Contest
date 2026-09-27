"""S6 encoder study (work2/s6/choice_lock.json) under runtime semantics; derived from the S5 audit below.
Audit: the S5 music gate (work2/backbones ads.py, AC semantics: missing member score -> 0) re-scored with runtime
semantics (RT: every val/test row scored by every head, VOICE 0 when the voice stem is silent, fusion tables on
non-silent val rows), exactly as research/top15/work3/integration/fit_ft_fusion.py db() does for the S4 baseline.
  1. retrain the DB-protocol S5 music head (music_new mert6_12+eat16, seeds 41/42/43; deterministic) and score every
     frozen val/test row; its labelled-row predictions must equal runs/music_new__mert6_12+eat16.json
  2. S4 = stacking RT/AC members (voice_df mlp_long, voice_ad resmlp_long @0.5, music__L6_12): must reproduce
     RT 0.93773/0.88457 and AC 0.93845/0.88643; S5 = same voice, music replaced; AC must reproduce 0.95475/0.89755
  3. held-out RT and AC deltas S5 - S4 with the group bootstrap (test group_id, 1000 reps, rng 0)
    python rt_check.py
"""
import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
ROOT = Path('/Users/choiwuseck/Documents/ChatGPT/데이콘')
ST = HERE.parent / 'work2' / 'stacking'
sys.path[:0] = [str(HERE.parent / 'work2' / 'backbones'), str(HERE.parent / 'work3' / 'integration')]
import run as R  # noqa: E402  (backbones run.py)
import fit_ft_fusion as F  # noqa: E402  (Boot, bank_rows, ads_of, file_preds, lab, eer)
from component_fusion import fuse  # noqa: E402
import decision_heads as dh  # noqa: E402

OUT = HERE / 'runs'
SPEC = R.parse('mert6_12+eat16+sslampt5')


def every_items(rows):
    """One music Item per row whatever its labels (runtime: every file gets a MUSIC score)."""
    out = []
    for r in rows:
        parts = []
        for name, layers in SPEC:
            base, key, layer_key = R.SOURCES[name]
            with np.load(base / f"{r['ID']}.npz") as d:
                L = list(d[layer_key])
                parts.append(d[key][:, [L.index(k) for k in layers]].reshape(len(d[key]), -1))
        f = np.concatenate(parts, 1)
        out.append(R.Item({**r, 'route': 'music'}, f, np.full(len(f), np.nan, np.float32), 0., False))
    return out


def music_every():
    path = OUT / 'music_sslam__every.json'
    if path.exists():
        return json.loads(path.read_text())
    torch.set_num_threads(4)
    rows = {s: [r for r in csv.DictReader((R.H.BANK / f).open()) if r['split'] == s and r['ID'] in R.PINNED]
            for s, f in (('train', 'bank_trainval.csv'), ('val', 'bank_trainval.csv'), ('test', 'bank_test.csv'))}
    data = {s: R.new_items(rows[s], SPEC, 'music') for s in rows}
    frozen = set(json.loads((ST / 'frozen_rows.json').read_text())['ids'])
    every = {s: every_items([r for r in rows[s] if r['ID'] in frozen]) for s in ('val', 'test')}
    ref = json.loads((HERE.parent / 'work2' / 'backbones' / 'runs' / 'music_new__mert6_12+eat16+sslampt5.json').read_text())
    res = {'every': {s: {'ids': [x.row['ID'] for x in v], 'preds': []} for s, v in every.items()}, 'max_abs_vs_ref': []}
    for i, seed in enumerate((41, 42, 43)):
        model, _ = R.train('music', data['train'], data['val'], lambda items: R.WindowMIL(*R.stats(items)), 0, seed)
        vp = R.T.predict(model, [x for x, _ in data['val']], 'music', 0, 256, 10 ** 6, 'cpu')
        res['max_abs_vs_ref'].append(float(np.abs(vp - np.array(ref['val_preds'][i])).max()))
        for s, v in every.items():
            res['every'][s]['preds'].append(R.T.predict(model, v, 'music', 0, 256, 10 ** 6, 'cpu').tolist())
        print('seed', seed, 'max |val pred - ref|', res['max_abs_vs_ref'][-1], flush=True)
    OUT.mkdir(exist_ok=True)
    path.write_text(json.dumps(res))
    return res



if __name__ == '__main__':
    r = music_every()
    print('max |val pred - gate run|', r['max_abs_vs_ref'])
