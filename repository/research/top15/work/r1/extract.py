"""R1 step 0: per-window DB-head logits for val + held-out test from cache_v4 (CPU fp32, read-only).
Writes windows.npz: ids, split, voice z (fake-real logit diff, 3 seeds), music logits (3 seeds), ragged via offsets."""
import json, sys, time
from pathlib import Path
import numpy as np, pandas as pd, torch

ROOT = Path('/Users/choiwuseck/Documents/ChatGPT/데이콘')
sys.path.insert(0, str(ROOT))
from full_bank_evaluate import load_selected_heads  # noqa: E402

OUT = Path(__file__).parent
BANK = ROOT / 'results/data_v4_20260923/data_v4_results_20260923T1639Z/data/bank'
DB = ROOT / '_keep_best_20260926/train_duration_balanced'
torch.set_num_threads(4)


def main():
    t = time.time()
    tv = pd.read_csv(BANK / 'bank_trainval.csv', dtype=str, keep_default_na=False)
    te = pd.read_csv(BANK / 'bank_test.csv', dtype=str, keep_default_na=False)
    rows = pd.concat([tv[tv.split == 'val'], te], ignore_index=True)
    music, voice, fi, _ = load_selected_heads(DB, 'cpu')
    vz, mz, vo, mo, silent = [], [], [0], [0], []
    with torch.no_grad():
        for i, s in zip(rows.ID, rows.split):
            cache = ROOT / '_workspace/cache_v4' / ('cache_test' if s == 'test' else 'cache_tv')
            with np.load(cache / f'{i}.npz', allow_pickle=False) as z:
                e = torch.from_numpy(z['voice_embeddings']); f = torch.from_numpy(z['mert_windows'])
                silent.append(bool(z['voice_silent']))
            v = torch.stack([h(e) for h in voice])                        # [3, n, 2]
            vz.append((v[..., fi] - v[..., 1 - fi]).numpy().T)            # [n, 3]
            mz.append(torch.stack([h.head((f - h.mean) / h.scale)[0].reshape(-1) for h in music]).numpy().T)
            vo.append(vo[-1] + len(e)); mo.append(mo[-1] + len(f))
    np.savez(OUT / 'windows.npz', ids=rows.ID.values.astype(str), split=rows.split.values.astype(str),
             vz=np.concatenate(vz).astype(np.float32), mz=np.concatenate(mz).astype(np.float32),
             vo=np.array(vo), mo=np.array(mo), silent=np.array(silent), fake_index=fi)
    print(len(rows), 'rows', vo[-1], 'voice windows', mo[-1], 'music windows', f'{time.time() - t:.0f}s')


if __name__ == '__main__':
    main()
