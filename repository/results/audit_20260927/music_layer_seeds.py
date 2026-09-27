"""Is MERT layer 6 (or a concat) better than layer 12 across seeds? Same rows (backbone cache), same objective."""
import csv, json, sys, numpy as np
from pathlib import Path
sys.path.insert(0, 'data_v4'); sys.path.insert(0, '.')
import train_backbone_heads as H
import full_bank_train as T
from graph_scoring import _roc_metrics
cache = Path('_workspace/backbone_cache')
rows = {s: [r for r in csv.DictReader((H.BANK / f).open()) if r['split'] == s]
        for s, f in (('train', 'bank_trainval.csv'), ('val', 'bank_trainval.csv'), ('test', 'bank_test.csv'))}
out = {}
for layers in ([12], [6], [6, 12], [3, 6, 9]):
    data = {s: H.items_for(rows[s], cache, layers, 'music') for s in rows}
    res = []
    for seed in (41, 42, 43):
        model, val_eer, epochs = H.train('music', data['train'], data['val'], seed=seed)
        test = [x for x, _ in data['test']]
        p = T.predict(model, test, 'music', 0, 256, 10 ** 6, 'cpu')
        res.append((val_eer, float(_roc_metrics([int(x.label) for x in test], p.tolist())[0])))
    v, t = np.array(res).T
    out[str(layers)] = {'val': [round(x, 4) for x in v], 'test': [round(x, 4) for x in t], 'val_mean': round(v.mean(), 4), 'test_mean': round(t.mean(), 4)}
    print(layers, out[str(layers)], flush=True)
Path('_workspace/audit_20260927/music_layer_seeds.json').write_text(json.dumps(out, indent=1))
