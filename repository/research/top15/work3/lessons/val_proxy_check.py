"""Lessons diagnostic (CPU): does in-source val EER track held-out EER epoch by epoch on AntiDeepfake-1B features,
and does a train-carved source-held-out set (val_ood) track it better?

Heads on the frozen AD-1B voice-stem window embeddings (_workspace/backbone_cache), the S3 voice_ad objective
(full_bank_train.step: weighted BCE, loss_weights x duration_balance, AdamW 1e-4 wd .01, 256 files/update).
No early stopping: every epoch is scored on val, val_ood and held-out test (diagnostic only, never used to pick a model).
Epoch 0 = the unmodified AntiDeepfake classifier (the frozen reference).

  python research/top15/work3/lessons/val_proxy_check.py --cache _workspace/backbone_cache --out OUT.json [--epochs 30]
"""
import argparse, csv, json, sys, time
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import load_file

ROOT = Path(__file__).resolve().parents[4]
sys.path[:0] = [str(ROOT), str(ROOT / 'data_v4')]
import full_bank_train as T  # noqa: E402
import voice_heads  # noqa: E402
from full_bank_batches import duration_balance, epoch_order, loss_weights  # noqa: E402
from graph_scoring import _roc_metrics  # noqa: E402
from train_backbone_heads import BANK, AD, items_for  # noqa: E402

# Train-carved "unseen family" set: whole components of these fake generators / real sources leave training.
OOD_FAKE = {('mlaad', 'Chatterbox Multilingual'), ('mlaad', 'MOSS-TTS-8B'), ('mlaad', 'MOSS-TTS-1.7B'), ('speechfake', 'SeedVC')}
OOD_REAL = {'musan_speech', 'existing:librispeech_test-clean'}


def eer(items, pred):
    return float(_roc_metrics([int(x.label) for x in items], pred.tolist())[0])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cache', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--epochs', type=int, default=30)
    ap.add_argument('--arch', nargs='+', default=['linear', 'mlp'])
    a = ap.parse_args()
    torch.set_num_threads(4)
    rows = {s: [r for r in csv.DictReader((BANK / f).open()) if r['split'] == s]
            for s, f in (('train', 'bank_trainval.csv'), ('val', 'bank_trainval.csv'), ('test', 'bank_test.csv'))}
    ood = lambda r: (r['source'], r['generator']) in OOD_FAKE if r['VOICE_FAKE'] in ('1', '1.0') else r['source'] in OOD_REAL
    comps = {r['component_id'] for r in rows['train'] if ood(r)}
    carved = {'train': [r for r in rows['train'] if r['component_id'] not in comps],
              'val_ood': [r for r in rows['train'] if r['component_id'] in comps]}
    t0 = time.time()
    data = {s: items_for(v, a.cache, [12], 'voice') for s, v in {**carved, 'val': rows['val'], 'test': rows['test']}.items()}
    print({s: (len(v), round(float(np.mean([x.label for x, _ in v])), 3)) for s, v in data.items()}, f'{time.time()-t0:.0f}s', flush=True)
    native = load_file(str(AD / 'model.safetensors'))
    init = {'weight': native['proj_fc.weight'].float(), 'bias': native['proj_fc.bias'].float()}
    items = [x for x, _ in data['train']]
    w = loss_weights(items)
    ids = [x.row['ID'] for x in items]
    w = {i: w[i] * f for i, f in zip(ids, duration_balance([x.label for x in items], [s for _, s in data['train']], [w[i] for i in ids]))}
    ev = {s: [x for x, _ in data[s]] for s in ('val', 'val_ood', 'test')}
    means = np.stack([x.features.mean(0) for x in items])
    report = {'n': {s: len(v) for s, v in data.items()}, 'ood_fake': sorted(map(list, OOD_FAKE)), 'ood_real': sorted(OOD_REAL)}
    for arch in a.arch:
        torch.manual_seed(41)
        model = voice_heads.build(arch, init, torch.tensor(means.mean(0)), torch.tensor(np.maximum(means.std(0), 1e-6)))
        opt = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=.01)
        hist = []
        for epoch in range(a.epochs + 1):
            if epoch:
                model.train()
                order = epoch_order(items, 41, epoch)
                for o in range(0, len(order), 256):
                    T.step(model, opt, [items[i] for i in order[o:o + 256]], 'voice', 0, w, 256, 10 ** 6, 'cpu')
            h = {'epoch': epoch}
            for s, v in ev.items():
                p = T.predict(model, v, 'voice', 0, 256, 10 ** 6, 'cpu')
                h[s] = round(eer(v, p), 5)
                h[s + '_sat'] = round(float(np.mean((p > 1 - 1e-6) | (p < 1e-6))), 4)   # saturated-probability share
            hist.append(h)
            print(arch, h, flush=True)
        pick = lambda k: min(hist[1:], key=lambda h: h[k])['epoch']
        report[arch] = {'history': hist, 'epoch_by_val': pick('val'), 'epoch_by_val_ood': pick('val_ood'), 'epoch_by_test': pick('test'),
                        'test_at_val_pick': hist[pick('val')]['test'], 'test_at_val_ood_pick': hist[pick('val_ood')]['test'],
                        'test_frozen': hist[0]['test'], 'test_best': min(h['test'] for h in hist)}
        print(arch, {k: v for k, v in report[arch].items() if k != 'history'}, flush=True)
    a.out.write_text(json.dumps(report, indent=1) + '\n')


if __name__ == '__main__':
    main()
