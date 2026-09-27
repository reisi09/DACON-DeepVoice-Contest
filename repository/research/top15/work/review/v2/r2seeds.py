"""Q5: is the R2 MLP held-out gain a favourable seed draw?  Trains R2's val-chosen voice heads (voice_df mlp_long,
voice_ad resmlp_long) for extra seeds with r2/run.py's own train/make code, backbone-cache rows, DB protocol; saves
per-seed val+test probabilities here (seeds_mlp.json).  Read-only on project files and on r2/.
  python -B r2seeds.py 44,45,46"""
import sys
sys.dont_write_bytecode = True
import csv, json, time
from pathlib import Path
import torch
from safetensors.torch import load_file
from torch import nn
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / 'r2'))
import run as R                       # r2/run.py (inserts project root + data_v4 on sys.path)
H, T = R.H, R.T

torch.set_num_threads(4)
seeds = [int(s) for s in sys.argv[1].split(',')]
rows = {s: [r for r in csv.DictReader((H.BANK / f).open()) if r['split'] == s]
        for s, f in (('train', 'bank_trainval.csv'), ('val', 'bank_trainval.csv'), ('test', 'bank_test.csv'))}
old = torch.load(H.OLD['val'] / 'voice_classifier_initial.pt', map_location='cpu', weights_only=True)
native = load_file(str(H.AD / 'model.safetensors'))
out_path = HERE / 'seeds_mlp.json'
out = json.loads(out_path.read_text()) if out_path.exists() else {}
for key, group, arch, init, fi in (('mlp_df', 'voice_df', 'mlp', old['state_dict'], old['fake_label_index']),
                                   ('mlp_ad', 'voice_ad', 'resmlp', {'weight': native['proj_fc.weight'].float(), 'bias': native['proj_fc.bias'].float()}, 0)):
    t0 = time.time()
    data = {s: H.items_for(rows[s], R.CACHE, [12], 'voice', dfarena=group == 'voice_df') for s in rows}

    def make(items, arch=arch, init=init):   # == run.main make() for these two archs
        mean, scale = R.stats(items)
        if arch == 'resmlp':
            return R.ResMLP(init, mean, scale)
        return nn.Sequential(R.Standardize(mean, scale), nn.Linear(1280, 256), nn.GELU(), nn.Linear(256, 2))
    print(key, 'loaded', {s: len(v) for s, v in data.items()}, f'{time.time() - t0:.0f}s', flush=True)
    for seed in seeds:
        model, info = R.train('voice', data['train'], data['val'], make, fi, seed, **R.LONG)
        rec = {}
        for s in ('val', 'test'):
            p = T.predict(model, [x for x, _ in data[s]], 'voice', fi, 256, 10 ** 6, 'cpu')
            rec.update({x.row['ID']: float(v) for (x, _), v in zip(data[s], p)})
        out.setdefault(key, {})[str(seed)] = rec
        print(key, seed, {k: info[k] for k in ('val_eer', 'best_epoch', 'epochs_run', 'hit_cap', 'train_seconds')}, flush=True)
        out_path.write_text(json.dumps(out))
