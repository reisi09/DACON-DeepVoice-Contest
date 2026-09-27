"""Q4: does the held-out S3-vs-baseline ADS delta survive other training seeds?  Retrains the ads_compare.py heads
(same H.items_for / H.train calls, DB protocol, backbone-cache rows) for the requested components and seeds and saves
per-seed val+test probabilities to this folder only.  Read-only on project files.
  python seeds.py OUT.json COMPONENT:SEED[,SEED..] ...     COMPONENT in voice_df, voice_ad, music_12, music_6_12"""
import sys
sys.dont_write_bytecode = True
import csv, json, time
from pathlib import Path
import numpy as np, torch
ROOT = Path('/Users/choiwuseck/Documents/ChatGPT/데이콘')
sys.path[:0] = [str(ROOT / 'data_v4'), str(ROOT)]
import train_backbone_heads as H
import full_bank_train as T
from safetensors.torch import load_file

torch.set_num_threads(4)
cache = ROOT / '_workspace/backbone_cache'
rows = {s: [r for r in csv.DictReader((H.BANK / f).open()) if r['split'] == s and (cache / f"{r['ID']}.npz").exists()]
        for s, f in (('train', 'bank_trainval.csv'), ('val', 'bank_trainval.csv'), ('test', 'bank_test.csv'))}
old = torch.load(H.OLD['val'] / 'voice_classifier_initial.pt', map_location='cpu', weights_only=True)
native = load_file(str(H.AD / 'model.safetensors'))
SPEC = {'voice_df': ('voice', [12], True, old['state_dict'], old['fake_label_index']),
        'voice_ad': ('voice', [12], False, {'weight': native['proj_fc.weight'].float(), 'bias': native['proj_fc.bias'].float()}, 0),
        'music_12': ('music', [12], False, None, 0), 'music_6_12': ('music', [6, 12], False, None, 0)}
out_path = Path(__file__).parent / sys.argv[1]
out = json.loads(out_path.read_text()) if out_path.exists() else {}
for arg in sys.argv[2:]:
    name, seeds = arg.split(':')
    route, layers, dfarena, init, fi = SPEC[name]
    t0 = time.time()
    data = {s: H.items_for(rows[s], cache, layers, route, dfarena=dfarena) for s in rows}
    print(name, 'loaded', {s: len(v) for s, v in data.items()}, f'{time.time() - t0:.0f}s', flush=True)
    for seed in map(int, seeds.split(',')):
        t1 = time.time()
        model, val_eer, epochs = H.train(route, data['train'], data['val'], init, fi, seed=seed)
        rec = {}
        for s in ('val', 'test'):
            p = T.predict(model, [x for x, _ in data[s]], route, fi, 256, 10 ** 6, 'cpu')
            rec.update({x.row['ID']: float(v) for (x, _), v in zip(data[s], p)})
        out.setdefault(name, {})[str(seed)] = rec
        print(name, seed, 'val_eer', round(val_eer, 5), 'epochs', epochs, f'{time.time() - t1:.0f}s', flush=True)
        out_path.write_text(json.dumps(out))
