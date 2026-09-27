"""R1 (c): retrain the DF-Arena fc5 voice head with a different file pooling. DB protocol, seed 41, CPU fp32.
  python retrain.py max|lme|mean          -> retrain_<pool>/{voice_41.pt, history.json, windows_voice.npz}
  python retrain.py lme|mean music        -> retrain_music_<pool>/{music_41.pt, history.json, windows_music.npz}
Music: WindowMIL, lr 1e-3, 0.25 window-truth loss unchanged; 'lme' is the unpatched control.
Same items (full_bank_train.read_items), loss weights x duration_balance, fc5 init, AdamW 1e-4/.01, 256-file
updates, clip 1, (val EER, val BCE) selection, patience 5, <=50 epochs. Only the window->file pooling differs;
'max' runs the unpatched full_bank_train code path and is the matched control.
"""
import json, sys, time, hashlib
from pathlib import Path
import numpy as np, torch
from torch import nn

ROOT = Path('/Users/choiwuseck/Documents/ChatGPT/데이콘')
sys.path.insert(0, str(ROOT))
import full_bank_train as T  # noqa: E402
from full_bank_batches import loss_weights, duration_balance, epoch_order  # noqa: E402

BANK = ROOT / 'results/data_v4_20260923/data_v4_results_20260923T1639Z/data/bank'
CACHE = ROOT / '_workspace/cache_v4/cache_tv'
POOL = sys.argv[1]
ROUTE = sys.argv[2] if len(sys.argv) > 2 else 'voice'
SEED = int(sys.argv[3]) if len(sys.argv) > 3 else 41
OUT = Path(__file__).parent / ((f'retrain_{POOL}' if ROUTE == 'voice' else f'retrain_music_{POOL}') + ('' if SEED == 41 else f'_s{SEED}'))
POOLS = {'lme': lambda p: p.logsumexp(0) - np.log(len(p)), 'mean': lambda p: p.mean()}

if ROUTE == 'music' and POOL != 'lme':
    _orig = T.forward

    def forward(model, items, route, fake_index, device):
        assert route == 'music' and POOL == 'mean'
        sizes = [len(item.features) for item in items]
        features = torch.from_numpy(np.concatenate([item.features for item in items])).to(device)
        logits, _ = model.head((features - model.mean) / model.scale)
        truth = torch.from_numpy(np.concatenate([item.truth for item in items])).to(device)
        known = torch.isfinite(truth)
        return torch.stack([part.float().mean() for part in logits.split(sizes)]), logits[known], truth[known]
    T.forward = forward
elif ROUTE == 'voice' and POOL != 'max':
    def forward(model, items, route, fake_index, device):
        assert route == 'voice'
        sizes = [len(item.features) for item in items]
        values = model(torch.from_numpy(np.concatenate([item.features for item in items])).to(device)).float()
        odds = values[:, fake_index] - values[:, 1 - fake_index]
        return torch.stack([POOLS[POOL](part) for part in odds.split(sizes)]), odds[:0], odds[:0]
    T.forward = forward   # step() and predict() resolve forward from the module at call time


def main():
    t0 = time.time()
    torch.set_num_threads(4)
    OUT.mkdir(exist_ok=True)
    items = T.read_items(BANK / 'bank_trainval.csv', CACHE)
    training = [x for x in items if x.row['route'] == ROUTE and x.row['split'] == 'train']
    validation = [x for x in items if x.row['route'] == ROUTE and x.row['split'] == 'val']
    del items
    weights = loss_weights(training)
    samples = {k: v['samples'] for k, v in json.loads((CACHE / 'index.json').read_text())['items'].items()}
    ids = [x.row['ID'] for x in training]
    factor = duration_balance([x.label for x in training], [samples[i] for i in ids], [weights[i] for i in ids])
    weights = {i: weights[i] * f for i, f in zip(ids, factor, strict=True)}
    voice = torch.load(CACHE / 'voice_classifier_initial.pt', map_location='cpu', weights_only=True)
    fi = voice['fake_label_index']
    torch.manual_seed(SEED); np.random.seed(SEED)
    if ROUTE == 'music':
        means = np.stack([x.features.mean(0) for x in training])
        model = T.WindowMIL(torch.tensor(means.mean(0)), torch.tensor(np.maximum(means.std(0), 1e-6)))
    else:
        model = nn.Linear(1280, 2); model.load_state_dict(voice['state_dict'])
    longest = sorted(training, key=lambda x: len(x.features), reverse=True)[:256]
    microbatch, window_limit = 256, sum(len(x.features) for x in longest)
    window_limit = max(window_limit, max(len(x.features) for x in validation))
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3 if ROUTE == 'music' else 1e-4, weight_decay=.01)
    labels = np.asarray([x.label for x in validation])
    best, stale, history, state = (float('inf'), float('inf')), 0, [], None
    print(f'train {len(training)} val {len(validation)} load {time.time() - t0:.0f}s', flush=True)
    for epoch in range(1, 51):
        if stale >= 5:
            break
        model.train()
        order = epoch_order(training, SEED, epoch)
        for off in range(0, len(order), 256):
            T.step(model, opt, [training[i] for i in order[off:off + 256]], ROUTE, fi, weights, microbatch, window_limit, 'cpu')
        p = T.predict(model, validation, ROUTE, fi, microbatch, window_limit, 'cpu')
        eer = float(T._roc_metrics(labels.tolist(), p.tolist())[0])
        c = np.clip(p, 1e-7, 1 - 1e-7)
        bce = float(-(labels * np.log(c) + (1 - labels) * np.log1p(-c)).mean())
        improved = (eer, bce) < best
        if improved:
            best, stale, state = (eer, bce), 0, {'model': {k: v.clone() for k, v in model.state_dict().items()}, 'epoch': epoch,
                                                 'validation_eer': eer, 'fake_label_index': fi, 'pool': POOL}
        else:
            stale += 1
        history.append({'epoch': epoch, 'val_eer': eer, 'val_bce': bce, 'improved': improved})
        print(f'{POOL} epoch {epoch} val_eer {eer:.5f} bce {bce:.4f} {time.time() - t0:.0f}s', flush=True)
    torch.save(state, OUT / f'{ROUTE}_{SEED}.pt')
    (OUT / 'history.json').write_text(json.dumps({'history': history, 'best_epoch': state['epoch'], 'best_val_eer': state['validation_eer'],
                                                   'n_train': len(training), 'n_val': len(validation), 'seconds': time.time() - t0}, indent=1))
    # per-window z for val + held-out test in windows.npz order (voice only)
    model.load_state_dict(state['model']); model.eval()
    w = np.load(Path(__file__).parent / 'windows.npz')
    ids_, split = w['ids'], w['split']
    zs, offs = [], [0]
    with torch.no_grad():
        for i, s in zip(ids_, split):
            cache = ROOT / '_workspace/cache_v4' / ('cache_test' if s == 'test' else 'cache_tv')
            with np.load(cache / f'{i}.npz', allow_pickle=False) as z:
                if ROUTE == 'music':
                    f = torch.from_numpy(z['mert_windows'])
                    v = model.head((f - model.mean) / model.scale)[0].reshape(-1, 1)
                    zs.append(v.numpy())
                else:
                    v = model(torch.from_numpy(z['voice_embeddings']))
                    zs.append((v[:, fi] - v[:, 1 - fi]).numpy()[:, None])
            offs.append(offs[-1] + len(v))
    key = 'm' if ROUTE == 'music' else 'v'
    np.savez(OUT / f'windows_{ROUTE}.npz', **{key + 'z': np.concatenate(zs).astype(np.float32), key + 'o': np.array(offs)})
    assert np.array_equal(np.array(offs), w[key + 'o']), 'window layout differs from windows.npz'
    print('done', POOL, state['epoch'], state['validation_eer'], f'{time.time() - t0:.0f}s')


if __name__ == '__main__':
    main()
