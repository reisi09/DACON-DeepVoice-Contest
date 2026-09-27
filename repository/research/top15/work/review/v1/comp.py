"""Q: under RUNTIME semantics (full_bank_submission.py scores Fv on every file's voice stem and Fm on every file's backing
stem), what do the ads_compare DB-protocol heads output for val/test files WITHOUT that component's label?
Retrains exactly as _workspace/audit_20260927/ads_compare.py::seed_mean (H.train, seeds 41-43, train split only), then
predicts on every val/test backbone-cache row. Checks reproduction against ads_compare_scores.json on labelled rows.

  python comp.py voice_df|voice_ad|music_12|music_6_12
"""
import csv, json, sys, time
from pathlib import Path
import numpy as np, torch
ROOT = Path('/Users/choiwuseck/Documents/ChatGPT/데이콘')
sys.path.insert(0, str(ROOT / 'data_v4')); sys.path.insert(0, str(ROOT))
import train_backbone_heads as H
import full_bank_train as T
from full_bank_batches import Item
from safetensors.torch import load_file

HERE = Path(__file__).resolve().parent
name = sys.argv[1]
torch.set_num_threads(4)
cache = ROOT / '_workspace/backbone_cache'
rows = {s: [r for r in csv.DictReader((H.BANK / f).open()) if r['split'] == s and (cache / f"{r['ID']}.npz").exists()]
        for s, f in (('train', 'bank_trainval.csv'), ('val', 'bank_trainval.csv'), ('test', 'bank_test.csv'))}
route = 'music' if name.startswith('music') else 'voice'
layers = {'music_12': [12], 'music_6_12': [6, 12]}.get(name, [12])
dfarena = name == 'voice_df'
if name == 'voice_df':
    o = torch.load(H.OLD['val'] / 'voice_classifier_initial.pt', map_location='cpu', weights_only=True)
    init, fi = o['state_dict'], o['fake_label_index']
elif name == 'voice_ad':
    n = load_file(str(H.AD / 'model.safetensors'))
    init, fi = {'weight': n['proj_fc.weight'].float(), 'bias': n['proj_fc.bias'].float()}, 0
else:
    init, fi = None, 0
t0 = time.monotonic()
train = H.items_for(rows['train'], cache, layers, route, dfarena=dfarena)
val_l = H.items_for(rows['val'], cache, layers, route, dfarena=dfarena)


def all_items(rs):
    """Every row, runtime features: DF-Arena fc5 inputs (old cache) / AD embeddings / MERT layers; runtime silence flag."""
    out = []
    for r in rs:
        with np.load(cache / f"{r['ID']}.npz") as d:
            L = list(d['layers'])
            if route == 'music':
                f = d['mert_layers'][:, [L.index(k) for k in layers]].reshape(len(d['mert_layers']), -1)  # fp16 kept
            else:
                f = d['ad_embeddings'].astype(np.float32)   # as items_for
            silent = bool(d['voice_silent'])
        if dfarena:
            with np.load(H.OLD[r['split']] / f"{r['ID']}.npz") as o:
                f, silent = o['voice_embeddings'], bool(o['voice_silent'])
        out.append(Item({**r, 'route': route}, f, np.full(len(f), np.nan, np.float32), 0., silent))
    return out


ev = {s: all_items(rows[s]) for s in ('val', 'test')}
print(name, 'loaded', round(time.monotonic() - t0, 1), 's', {s: len(v) for s, v in ev.items()}, 'train', len(train), flush=True)
acc = {}
for seed in (41, 42, 43):
    model, _, _ = H.train(route, train, val_l, init, fi, seed=seed)
    for s in ('val', 'test'):
        p = T.predict(model, ev[s], route, fi, 256, 10 ** 6, 'cpu')
        for x, v in zip(ev[s], p):
            acc.setdefault(x.row['ID'], []).append(float(v))
    print(name, 'seed', seed, round(time.monotonic() - t0, 1), 's', flush=True)
out = {k: float(np.mean(v)) for k, v in acc.items()}
ref = json.loads((ROOT / '_workspace/audit_20260927/ads_compare_scores.json').read_text())[name]
diff = max(abs(out[k] - v) for k, v in ref.items())
print(name, 'labelled rows', len(ref), 'max |diff| vs ads_compare_scores', diff, flush=True)
(HERE / f'comp_{name}.json').write_text(json.dumps({'scores': out, 'repro_max_abs_diff': diff, 'n_labelled': len(ref)}))
