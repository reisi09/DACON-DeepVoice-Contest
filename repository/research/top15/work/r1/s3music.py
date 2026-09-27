"""R1 (c5): S3 music heads (MERT layers 6+12, backbone_cache rows) - LME (S3 as built) vs mean pooling.
DB protocol (train -> val early stop -> held-out test), seeds 41/42/43, data_v4/train_backbone_heads.train unchanged
except the patched file pooling. -> s3music.json"""
import json, sys, time
from pathlib import Path
import numpy as np, torch
ROOT = Path('/Users/choiwuseck/Documents/ChatGPT/데이콘')
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / 'data_v4'))
import csv
import full_bank_train as T  # noqa: E402
import train_backbone_heads as B  # noqa: E402
sys.path.insert(0, str(Path(__file__).parent))
import agg  # noqa: E402

ORIG = T.forward


def mean_forward(model, items, route, fake_index, device):
    sizes = [len(i.features) for i in items]
    f = torch.from_numpy(np.concatenate([i.features for i in items])).to(device)
    logits, _ = model.head((f - model.mean) / model.scale)
    truth = torch.from_numpy(np.concatenate([i.truth for i in items])).to(device)
    known = torch.isfinite(truth)
    return torch.stack([p.float().mean() for p in logits.split(sizes)]), logits[known], truth[known]


def main():
    t0 = time.time(); torch.set_num_threads(4)
    rows = {s: [r for r in csv.DictReader((B.BANK / f).open()) if r['split'] == s and 'music' in T.head_labels(r)]
            for s, f in (('train', 'bank_trainval.csv'), ('val', 'bank_trainval.csv'), ('test', 'bank_test.csv'))}
    data = {s: B.items_for(rows[s], ROOT / '_workspace/backbone_cache', [6, 12], 'music') for s in rows}
    print({s: len(v) for s, v in data.items()}, f'load {time.time() - t0:.0f}s', flush=True)
    lab = {s: np.array([x.label for x, _ in data[s]]).astype(int) for s in ('val', 'test')}
    grp = np.array([x.row['group_id'] for x, _ in data['test']])
    res, probs, win = {}, {}, {}
    for pool in ('lme', 'mean'):
        T.forward = ORIG if pool == 'lme' else mean_forward
        per = {'val': [], 'test': []}
        win[pool] = []
        for seed in (41, 42, 43):
            model, val_eer, ep = B.train('music', data['train'], data['val'], seed=seed)
            for s in ('val', 'test'):
                per[s].append(T.predict(model, [x for x, _ in data[s]], 'music', 0, 256, 10 ** 6, 'cpu'))
            with torch.no_grad():
                win[pool].append({x.row['ID']: model.head((torch.from_numpy(x.features) - model.mean) / model.scale)[0].reshape(-1).numpy() for x, _ in data['test']})
            print(pool, seed, 'epochs', ep, 'val', round(val_eer, 5), 'test', round(agg.eer(lab['test'], per['test'][-1]), 5), f'{time.time() - t0:.0f}s', flush=True)
        probs[pool] = {s: np.mean(v, 0) for s, v in per.items()}
        res[pool] = {'per_seed_val': [round(agg.eer(lab['val'], p), 5) for p in per['val']],
                     'per_seed_test': [round(agg.eer(lab['test'], p), 5) for p in per['test']],
                     'ens_val': round(agg.eer(lab['val'], probs[pool]['val']), 5), 'ens_test': round(agg.eer(lab['test'], probs[pool]['test']), 5)}
    T.forward = ORIG
    res['boot_delta_test_mean_minus_lme'] = agg.boot_delta(lab['test'], probs['lme']['test'], probs['mean']['test'], grp)
    res['n'] = {s: [int((lab[s] == 0).sum()), int((lab[s] == 1).sum())] for s in lab}
    json.dump(res, open(Path(__file__).parent / 's3music.json', 'w'), indent=1)
    # synthetic length protocol (music route) on the backbone-cache test rows, same partner machinery as agg.part_b
    ids = np.load(agg.OUT / 'windows.npz')['ids']
    keep = np.array([i in win['lme'][0] for i in ids])
    for pool in ('lme', 'mean'):
        mz = [np.stack([w[i] for w in win[pool]], 1) if k else np.zeros((1, 3), np.float32) for i, k in zip(ids, keep)]
        agg._SETS.clear()
        agg.part_b(mz=mz, magg=pool, tasks=[('music', 'MUSIC_FAKE', pool)], keep=keep, out_name=f'b_s3music_{pool}.json')
    print(json.dumps(res, indent=1))


if __name__ == '__main__':
    main()
