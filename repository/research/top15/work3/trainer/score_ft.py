"""Runtime AntiDeepfake member on HTDemucs voice stems for any bank rows, with a fine-tuned (or zero-shot) checkpoint.

  torchrun --standalone --nproc_per_node 4 score_ft.py --ckpt RUN/best --manifest /root/ft/bank_trainval.csv \
      /root/ft/bank_test.csv --splits val test --stems /root/stems --code-root /root/ft/code --out RUN/ft_members.csv
Output CSV: ID,ft_max,ft_top30,ft_mean,voice_silent, one row per manifest row (manifest order), raw member values for
every row including silent stems (as members_all.csv; consumers zero silent voice as the runtime does). The checkpoint
is loaded by antideepfake_runtime.AntiDeepfake itself; fp32 by default = L4 runtime numerics (--bf16 for speed).
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist

from finetune_ad import load_voice, members, window_probs

FIELDS = ['ID', 'ft_max', 'ft_top30', 'ft_mean', 'voice_silent']


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--ckpt', type=Path, required=True, help='dir with model.safetensors + antideepfake_config.json')
    ap.add_argument('--manifest', type=Path, nargs='+', required=True)
    ap.add_argument('--splits', nargs='*', default=[], help='keep rows whose split is listed (default all)')
    ap.add_argument('--stems', type=Path, required=True)
    ap.add_argument('--code-root', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--device', default='cuda', choices=('cuda', 'cpu'))
    ap.add_argument('--batch', type=int, default=64, help='windows per forward')
    ap.add_argument('--files', type=int, default=128, help='files loaded per chunk')
    ap.add_argument('--bf16', action='store_true')
    ap.add_argument('--threads', type=int, default=4)
    a = ap.parse_args(argv)
    distributed = 'RANK' in os.environ
    if distributed:
        dist.init_process_group('gloo')   # barrier only
    rank, world = (dist.get_rank(), dist.get_world_size()) if distributed else (0, 1)
    device = torch.device(f"cuda:{os.environ.get('LOCAL_RANK', 0)}" if a.device == 'cuda' else 'cpu')
    if device.type == 'cuda':
        torch.cuda.set_device(device)
    torch.set_num_threads(a.threads)
    sys.path.insert(0, str(a.code_root))
    from antideepfake_runtime import AntiDeepfake
    ids = [r['ID'] for m in a.manifest for r in csv.DictReader(m.open(newline='')) if not a.splits or r['split'] in a.splits]
    if len(set(ids)) != len(ids):
        raise ValueError('Duplicate IDs across manifests')
    model = AntiDeepfake(a.ckpt, str(device))
    mine = [i for i in ids[rank::world] if (a.stems / f'{i}.npz').exists()]
    part = a.out.with_name(f'{a.out.name}.part{rank}')
    a.out.parent.mkdir(parents=True, exist_ok=True)
    started = time.time()
    with part.open('w', newline='') as stream:
        writer = csv.writer(stream)
        for o in range(0, len(mine), a.files):
            chunk = mine[o:o + a.files]
            silent, voices = [], []
            for i in chunk:
                with np.load(a.stems / f'{i}.npz') as z:
                    silent.append(int(z['voice_silent']))
                voices.append(load_voice(a.stems / f'{i}.npz'))
            for i, p, s in zip(chunk, window_probs(model, voices, a.batch, a.bf16), silent, strict=True):
                writer.writerow([i, *members(p), s])
            stream.flush()
            print(f'rank {rank}: {o + len(chunk)}/{len(mine)} files {time.time() - started:.0f}s', flush=True)
    if distributed:
        dist.barrier()
    if rank == 0:
        parts = [a.out.with_name(f'{a.out.name}.part{r}') for r in range(world)]
        scored = {row[0]: row for p in parts for row in csv.reader(p.open(newline=''))}
        with a.out.open('w', newline='') as stream:
            writer = csv.writer(stream)
            writer.writerow(FIELDS)
            writer.writerows(scored[i] for i in ids if i in scored)
        for p in parts:
            p.unlink()
        print(f'wrote {a.out}: {len(scored)}/{len(ids)} rows scored, {len(ids) - len(scored)} without stems, '
              f'{time.time() - started:.0f}s', flush=True)
    if distributed:
        dist.barrier()
        dist.destroy_process_group()


if __name__ == '__main__':
    main()
