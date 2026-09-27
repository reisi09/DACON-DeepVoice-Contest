"""Lane 4 check W: how concentrated are the S3 final training loss weights (class x source x generator balance, then
duration balance)? Reports per-route N, #cells, max/min weight, Kish effective sample size (sum w)^2/sum w^2, and the share
of total weight carried by the 1% heaviest rows. Rows = train + held-out test with a backbone npz (train_backbone_heads.final);
head assignment copied from full_bank_train.head_labels (not imported, to avoid importing full_bank_features)."""
import csv, json
from collections import Counter
from pathlib import Path
import sys
import numpy as np

R = Path('/Users/choiwuseck/Documents/ChatGPT/데이콘')
sys.path.insert(0, str(R))
from full_bank_batches import Item, loss_weights, duration_balance  # noqa: E402

BANK = R / 'results/data_v4_20260923/data_v4_results_20260923T1639Z/data/bank'
BB = R / '_workspace/backbone_cache'
rows = [r for r in csv.DictReader((BANK / 'bank_trainval.csv').open()) if r['split'] == 'train']
rows += [{**r, 'split': 'train'} for r in csv.DictReader((BANK / 'bank_test.csv').open())]
rows = [r for r in rows if (BB / f"{r['ID']}.npz").exists()]

def known(r, n):
    return r.get(n, '') in ('0', '1', '0.0', '1.0') and r.get(n + '_mask', '1') not in ('0', '0.0', 'False')

def head_labels(r):
    heads = {r['route']: float(r['MUSIC_FAKE' if r['route'] == 'music' else 'VOICE_FAKE'])}
    for head, fake, present in (('music', 'MUSIC_FAKE', 'MUSIC_PRESENT'), ('voice', 'VOICE_FAKE', 'VOICE_PRESENT')):
        if head not in heads and known(r, fake) and r.get(present, '') in ('1', '1.0'):
            heads[head] = float(r[fake])
    return heads

out = {}
items = {'voice': [], 'music': []}
for r in rows:
    with np.load(BB / f"{r['ID']}.npz") as d:
        silent, samples = bool(d['voice_silent']), int(d['samples'])
    for head, label in head_labels(r).items():
        if head != r['route'] and head == 'voice' and silent:
            continue
        items[head].append((Item({**r, 'route': head}, np.zeros((1, 1), np.float32), np.zeros(1, np.float32), label, silent), samples))
for route, data in items.items():
    its = [x for x, _ in data]
    w = loss_weights(its)
    base = np.array([w[x.row['ID']] for x in its])
    f = duration_balance([x.label for x in its], [s for _, s in data], base)
    fin = base * f
    cells = Counter((x.label, x.row['source'], x.row['generator']) for x in its)
    def stats(v):
        s = np.sort(v)[::-1]
        return {'max': round(float(v.max()), 2), 'min': round(float(v.min()), 4), 'mean': round(float(v.mean()), 4),
                'kish_ess': int(v.sum() ** 2 / (v ** 2).sum()), 'top1pct_weight_share': round(float(s[:max(1, len(s) // 100)].sum() / s.sum()), 3)}
    small = sorted(((n, c) for c, n in cells.items()), key=lambda t: t[0])[:5]
    out[route] = {'N': len(its), 'cells': len(cells), 'cells_per_class': dict(Counter(c[0] for c in cells)),
                  'smallest_cells': [[n, list(map(str, c))] for n, c in small],
                  'class_source_generator_weights': stats(base), 'after_duration_balance': stats(fin)}
json.dump(out, sys.stdout, indent=1); print()
(R / 'research/top15/work/lane4/weights_check.out.json').write_text(json.dumps(out, indent=1) + '\n')
