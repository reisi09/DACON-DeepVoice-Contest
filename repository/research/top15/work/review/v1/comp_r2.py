"""Q: under RUNTIME semantics, what do R2's val-chosen MLP voice heads (voice_df mlp_long, voice_ad resmlp_long) output on
val/test files WITHOUT a voice label (music-only files: the runtime still scores their voice stem)? Retrains exactly as
r2/run.py (same train(), same make(), seeds 41-43, LONG schedule, backbone-cache rows, train split only) and predicts
on every val/test backbone-cache row. Checks reproduction against r2/runs/<run>.json on labelled rows.
  python comp_r2.py voice_df_mlp|voice_ad_resmlp"""
import csv, importlib.util, json, sys, time
from pathlib import Path
import numpy as np, torch
from torch import nn
ROOT = Path('/Users/choiwuseck/Documents/ChatGPT/데이콘')
sys.path.insert(0, str(ROOT / 'data_v4')); sys.path.insert(0, str(ROOT))
import train_backbone_heads as H
import full_bank_train as T
from full_bank_batches import Item
from safetensors.torch import load_file
spec = importlib.util.spec_from_file_location('r2run', ROOT / 'research/top15/work/r2/run.py')
R = importlib.util.module_from_spec(spec); spec.loader.exec_module(R)

HERE = Path(__file__).resolve().parent
name = sys.argv[1]
group, variant, arch = {'voice_df_mlp': ('voice_df', 'mlp_long', 'mlp'), 'voice_ad_resmlp': ('voice_ad', 'resmlp_long', 'resmlp')}[name]
torch.set_num_threads(4)
cache = R.CACHE
rows = {s: [r for r in csv.DictReader((H.BANK / f).open()) if r['split'] == s]
        for s, f in (('train', 'bank_trainval.csv'), ('val', 'bank_trainval.csv'), ('test', 'bank_test.csv'))}
dfarena = group == 'voice_df'
if dfarena:
    o = torch.load(H.OLD['val'] / 'voice_classifier_initial.pt', map_location='cpu', weights_only=True)
    init, fi = o['state_dict'], o['fake_label_index']
else:
    n = load_file(str(H.AD / 'model.safetensors'))
    init, fi = {'weight': n['proj_fc.weight'].float(), 'bias': n['proj_fc.bias'].float()}, 0


def make(items):   # r2/run.py main() make(), voice branch
    mean, scale = R.stats(items)
    if arch == 'resmlp':
        return R.ResMLP(init, mean, scale)
    return nn.Sequential(R.Standardize(mean, scale), nn.Linear(1280, 256), nn.GELU(), nn.Linear(256, 2))


t0 = time.monotonic()
data = {s: H.items_for(rows[s], cache, [12], 'voice', dfarena=dfarena) for s in ('train', 'val')}


def all_items(rs):   # every backbone-cache row, runtime features and runtime silence flag (as comp.py)
    out = []
    for r in rs:
        p = cache / f"{r['ID']}.npz"
        if not p.exists():
            continue
        with np.load(p) as d:
            f, silent = d['ad_embeddings'].astype(np.float32), bool(d['voice_silent'])
        if dfarena:
            with np.load(H.OLD[r['split']] / f"{r['ID']}.npz") as o:
                f, silent = o['voice_embeddings'], bool(o['voice_silent'])
        out.append(Item({**r, 'route': 'voice'}, f, np.full(len(f), np.nan, np.float32), 0., silent))
    return out


ev = {s: all_items(rows[s]) for s in ('val', 'test')}
print(name, 'loaded', round(time.monotonic() - t0, 1), 's', {s: len(v) for s, v in ev.items()}, flush=True)
acc = {}
for seed in (41, 42, 43):
    model, info = R.train('voice', data['train'], data['val'], make, fi, seed, **R.LONG)
    for s in ('val', 'test'):
        for x, v in zip(ev[s], T.predict(model, ev[s], 'voice', fi, 256, 10 ** 6, 'cpu')):
            acc.setdefault(x.row['ID'], []).append(float(v))
    print(name, 'seed', seed, round(info['val_eer'], 5), info['best_epoch'], round(time.monotonic() - t0, 1), 's', flush=True)
out = {k: float(np.mean(v)) for k, v in acc.items()}
run = json.loads((ROOT / f'research/top15/work/r2/runs/{group}__{variant}.json').read_text())
ref = dict(zip(run['val_ids'], np.mean(run['val_preds'], 0))) | dict(zip(run['test_ids'], np.mean(run['test_preds'], 0)))
diff = max(abs(out[k] - v) for k, v in ref.items())
print(name, 'labelled rows', len(ref), 'max |diff| vs r2 run', diff, flush=True)
(HERE / f'comp_{name}.json').write_text(json.dumps({'scores': out, 'repro_max_abs_diff': diff, 'n_labelled': len(ref)}))
