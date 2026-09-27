"""Partial fine-tuning of AntiDeepfake XLS-R-1B on HTDemucs voice stems (research/top15/work3/DESIGN.md, Track C).

  torchrun --standalone --nproc_per_node 4 finetune_ad.py --manifest /root/ft/bank_trainval.csv --stems /root/stems \
      --model /root/models/xlsr1b --code-root /root/ft/code --out RUN
Runs single-process without torchrun (CPU smoke: --device cpu). Train split only; val selects; test is never read.
Writes RUN/metrics.jsonl (train + eval records), RUN/evals.json, RUN/val_best.csv, RUN/result.json and RUN/best/
(fp16 model.safetensors in the native AntiDeepfake key layout + antideepfake_config.json + meta.json), which
antideepfake_runtime.AntiDeepfake(RUN/best) loads unchanged; the round trip is re-checked at the end of the run.
"""
from __future__ import annotations

import argparse
import csv
import gc
import json
import math
import os
import re
import shutil
import struct
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
import torchaudio
from safetensors import safe_open

from full_bank_batches import Item, duration_balance, loss_weights  # verbatim copy of the project module

WINDOW, SR = 64600, 16000


# ---------------------------------------------------------------- rows (full_bank_train.head_labels semantics)
def _known(row, name):
    return row.get(name, '') in ('0', '1', '0.0', '1.0') and row.get(name + '_mask', '1') not in ('0', '0.0', 'False')


def voice_label(row) -> float | None:
    """The voice head of full_bank_train.head_labels: route label, or a known+present VOICE_FAKE on a mixed row."""
    if row['route'] == 'voice':
        return float(row['VOICE_FAKE'])
    if _known(row, 'VOICE_FAKE') and row.get('VOICE_PRESENT', '') in ('1', '1.0'):
        return float(row['VOICE_FAKE'])
    return None


def _stem_meta(path: Path):
    if not path.exists():
        return None
    with np.load(path) as z:
        return int(z['samples']), bool(z['voice_silent'])


def voice_items(manifest: Path, stems: Path, split: str):
    """(items, samples by ID, missing-stem count). As full_bank_train.read_items: a mixed row whose separated voice
    stem is silent carries no voice evidence and is dropped; a voice-route row is kept."""
    rows = [(r, y) for r in csv.DictReader(manifest.open(newline='')) if r['split'] == split
            for y in [voice_label(r)] if y is not None]
    with ThreadPoolExecutor(16) as pool:
        metas = list(pool.map(lambda r: _stem_meta(stems / f"{r[0]['ID']}.npz"), rows))
    items, samples, missing = [], {}, 0
    for (row, label), meta in zip(rows, metas, strict=True):
        if meta is None:
            missing += 1
            continue
        if meta[0] < 1 or (row['route'] != 'voice' and meta[1]):
            continue
        items.append(Item(row, np.empty(0), np.empty(0), label, meta[1]))
        samples[row['ID']] = meta[0]
    return items, samples, missing


def load_voice(path: Path) -> np.ndarray:
    """float16 voice stem, padded to 400 samples like score_members.py / AntiDeepfake.windows."""
    with np.load(path) as z:
        v = z['voice']
    return v if len(v) >= 400 else np.pad(v, (0, 400 - len(v)))


class Crops(torch.utils.data.Dataset):
    """One random WINDOW crop of the voice stem; shorter clips repeat like baseline extract_segment."""

    def __init__(self, items, stems: Path):
        self.paths = [stems / f"{x.row['ID']}.npz" for x in items]
        self.targets = [0 if x.label == 1. else 1 for x in items]   # AntiDeepfake class 0 = fake

    def __len__(self):
        return len(self.paths)

    tries = 1   # orchestrator: best-of-k random crops by RMS so song windows without vocals rarely carry the file label

    def __getitem__(self, i):
        with np.load(self.paths[i]) as z:
            v = z['voice']
        if len(v) < WINDOW:
            v = np.tile(v, WINDOW // len(v) + 1)[:WINDOW]
        else:
            starts = [int(torch.randint(len(v) - WINDOW + 1, ())) for _ in range(self.tries)]
            s = max(starts, key=lambda s: float(np.square(v[s:s + WINDOW], dtype=np.float32).mean()))
            v = v[s:s + WINDOW].copy()
        return torch.from_numpy(v), self.targets[i]


# ---------------------------------------------------------------- augmentation (train only, on the batch device)
def fir_lowpass(cutoff: torch.Tensor, sr: int, taps: int = 101) -> torch.Tensor:
    """Hann-windowed sinc low-pass kernels [B, taps], unit DC gain; cutoff (Hz) per row."""
    n = torch.arange(taps, device=cutoff.device) - taps // 2
    fc = (cutoff / (sr / 2))[:, None]
    h = fc * torch.sinc(fc * n) * torch.hann_window(taps, periodic=False, device=cutoff.device)
    return h / h.sum(1, keepdim=True)


def fir(x: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
    """Row-wise 'same' FIR filtering of x [B, N] with symmetric kernels h [B, K]."""
    return F.conv1d(x[None], h[:, None], padding=h.shape[1] // 2, groups=len(x))[0]


def telephone(x: torch.Tensor) -> torch.Tensor:
    """8 kHz resample + 300-3400 Hz band-pass, back to 16 kHz."""
    y = torchaudio.functional.resample(x, SR, 8000)
    edges = torch.tensor([3400., 300.], device=x.device)
    band = fir_lowpass(edges, 8000, 129)
    y = fir(y, (band[0] - band[1]).expand(len(y), -1))
    return torchaudio.functional.resample(y, 8000, SR)[:, :x.shape[1]]


def add_noise(x: torch.Tensor, snr_db: torch.Tensor, pink: torch.Tensor) -> torch.Tensor:
    white = torch.randn_like(x)
    shaped = torch.fft.irfft(torch.fft.rfft(white) / torch.arange(x.shape[1] // 2 + 1, device=x.device).clamp(min=1).sqrt(),
                             n=x.shape[1])
    noise = torch.where(pink[:, None], shaped, white)
    noise = noise / noise.pow(2).mean(1, keepdim=True).clamp(min=1e-12).sqrt()
    return x + noise * (x.pow(2).mean(1, keepdim=True) / 10 ** (snr_db[:, None] / 10)).sqrt()


def augment(x: torch.Tensor) -> torch.Tensor:
    """DESIGN.md: telephone (p .15), FIR low-pass 3.4-8 kHz (p .3), white/pink noise SNR 10-40 dB (p .5), gain +-6 dB,
    random polarity. (Gain is a no-op for AntiDeepfake, whose forward layer-normalises the waveform.)"""
    u = lambda lo=0., hi=1.: torch.empty(len(x), device=x.device).uniform_(lo, hi)  # noqa: E731
    m = u() < .15
    if m.any():
        x[m] = telephone(x[m])
    m, cutoff = u() < .3, u(3400., 8000.)
    if m.any():
        x[m] = fir(x[m], fir_lowpass(cutoff[m], SR))
    m, snr, pink = u() < .5, u(10., 40.), u() < .5
    if m.any():
        x[m] = add_noise(x[m], snr[m], pink[m])
    return x * (10 ** (u(-6., 6.) / 20) * torch.where(u() < .5, -1., 1.))[:, None]


# ---------------------------------------------------------------- runtime member (AntiDeepfake.predict windows)
def window_starts(n: int) -> list[int]:
    starts = list(range(0, max(1, n - WINDOW + 1), WINDOW // 2))
    if n > WINDOW and starts[-1] != n - WINDOW:
        starts.append(n - WINDOW)
    return starts


@torch.no_grad()
def window_probs(model, voices, batch: int = 64, amp: bool = False) -> list[list[float]]:
    """Per-file window fake probabilities (softmax index 0) on AntiDeepfake.predict's layout. Full windows are batched
    across files; a sub-window file is one unpadded clip at its own length (AD pooling uses every frame)."""
    device = next(model.parameters()).device
    out = [[] for _ in voices]

    def run(clips):
        x = torch.from_numpy(np.stack(clips)).to(device).float()
        with torch.autocast(device.type, torch.bfloat16, enabled=amp):
            return model(x).float().softmax(-1)[:, 0].tolist()

    full = [(i, s) for i, v in enumerate(voices) if len(v) >= WINDOW for s in window_starts(len(v))]
    for o in range(0, len(full), batch):
        chunk = full[o:o + batch]
        for (i, _), p in zip(chunk, run([voices[i][s:s + WINDOW] for i, s in chunk]), strict=True):
            out[i].append(p)
    for i, v in enumerate(voices):
        if len(v) < WINDOW:
            out[i] = run([v])
    return out


def members(p: list[float]) -> tuple[float, float, float]:
    """(max, top-30% mean, mean) exactly as AntiDeepfake.predict + component_fusion.detector_members."""
    top = sorted(p, reverse=True)[:max(1, math.ceil(.3 * len(p)))]
    return max(p), float(np.mean(top)), float(np.mean(p))


# ---------------------------------------------------------------- model, freeze, native checkpoint layout
def trainable(name: str, first: int) -> bool:
    m = re.match(r'encoder\.encoder\.layers\.(\d+)\.', name)
    return name.startswith(('classifier.', 'encoder.encoder.layer_norm.')) or (m is not None and int(m[1]) >= first)


def build(model_dir: Path, device: torch.device, first: int):
    from antideepfake_runtime import AntiDeepfake
    model = AntiDeepfake(model_dir, str(device))
    model.encoder.freeze_feature_encoder()
    for name, p in model.named_parameters():
        p.requires_grad_(trainable(name, first))
    return model


def train_mode(model, first: int) -> None:
    """Frozen modules stay in eval (no dropout, no layerdrop: the encoder's own flag stays False); trainable layers
    train with the config's hidden/attention dropout."""
    model.eval()
    for layer in model.encoder.encoder.layers[first:]:
        layer.train()


def hf_key(name: str) -> str | None:
    """antideepfake_runtime.AntiDeepfake's native -> converted tensor name (None = training-only tensor)."""
    if name.startswith('proj_fc.'):
        return name.replace('proj_fc.', 'classifier.')
    key = name.removeprefix('m_ssl.model.')
    if key.startswith(('quantizer.', 'project_q.', 'final_proj.')):
        return None
    match = re.fullmatch(r'feature_extractor.conv_layers.(\d+).(0|2.1).(weight|bias)', key)
    if match:
        layer, block, parameter = match.groups()
        key = f'feature_extractor.conv_layers.{layer}.{"conv" if block == "0" else "layer_norm"}.{parameter}'
    elif key.startswith('post_extract_proj.'):
        key = key.replace('post_extract_proj.', 'feature_projection.projection.')
    elif key.startswith('layer_norm.'):
        key = 'feature_projection.' + key
    elif key == 'mask_emb':
        key = 'masked_spec_embed'
    elif key.startswith('encoder.pos_conv.0.'):
        key = key.replace('encoder.pos_conv.0.', 'encoder.pos_conv_embed.conv.')
        key = key.replace('weight_g', 'parametrizations.weight.original0').replace('weight_v', 'parametrizations.weight.original1')
    elif key.startswith('encoder.layers.'):
        key = key.replace('.self_attn_layer_norm.', '.layer_norm.').replace('.self_attn.', '.attention.')
        key = key.replace('.fc1.', '.feed_forward.intermediate_dense.').replace('.fc2.', '.feed_forward.output_dense.')
    return 'encoder.' + key


def save_native(model, base: Path, target: Path, meta: dict) -> None:
    """Every native key of base/model.safetensors as fp16 (training-only tensors copied from base), streamed one
    tensor at a time (no second full copy in RAM), then an atomic directory swap."""
    state = model.state_dict()
    with safe_open(str(base / 'model.safetensors'), 'pt') as f:
        names = list(f.keys())
        shapes = {n: f.get_slice(n).get_shape() for n in names}
        missing = [n for n in names if hf_key(n) is not None and hf_key(n) not in state]
        extra = set(state) - {hf_key(n) for n in names}
        if missing or extra:
            raise ValueError(f'Native layout mismatch: missing {missing[:3]} extra {sorted(extra)[:3]}')
        header, offset = {}, 0
        for n in names:
            size = 2 * math.prod(shapes[n])
            header[n] = {'dtype': 'F16', 'shape': shapes[n], 'data_offsets': [offset, offset + size]}
            offset += size
        blob = json.dumps(header, separators=(',', ':')).encode()
        blob += b' ' * (-len(blob) % 8)
        tmp = target.with_name(target.name + '.tmp')
        shutil.rmtree(tmp, ignore_errors=True)
        tmp.mkdir(parents=True)
        with (tmp / 'model.safetensors').open('wb') as out:
            out.write(struct.pack('<Q', len(blob)) + blob)
            for n in names:
                value = f.get_tensor(n) if hf_key(n) is None else state[hf_key(n)]
                if list(value.shape) != list(shapes[n]):
                    raise ValueError(f'Shape changed: {n}')
                out.write(value.detach().to('cpu', torch.float16).contiguous().numpy().tobytes())
    shutil.copy2(base / 'antideepfake_config.json', tmp)
    (tmp / 'meta.json').write_text(json.dumps(meta))
    shutil.rmtree(target, ignore_errors=True)
    tmp.rename(target)


def fingerprint(model, half: bool = False) -> dict[str, list[float]]:
    """Per-tensor (sum, sum of squares) in fp64; half=True fingerprints the fp16-rounded weights a checkpoint stores."""
    cast = (lambda t: t.half().double()) if half else (lambda t: t.double())  # noqa: E731
    return {n: [float(cast(p.detach()).sum()), float(cast(p.detach()).square().sum())] for n, p in model.named_parameters()}


# ---------------------------------------------------------------- training
def parse(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--manifest', type=Path, required=True, help='bank_trainval.csv (train + val rows)')
    ap.add_argument('--stems', type=Path, required=True, help='dir of <ID>.npz (voice, backing, samples, voice_silent)')
    ap.add_argument('--model', type=Path, required=True, help='AntiDeepfake dir (model.safetensors + antideepfake_config.json)')
    ap.add_argument('--code-root', type=Path, required=True, help='dir with antideepfake_runtime.py, graph_scoring.py')
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--device', default='cuda', choices=('cuda', 'cpu'))
    ap.add_argument('--first-trainable-layer', type=int, default=32)
    ap.add_argument('--batch', type=int, default=32, help='windows per GPU')
    ap.add_argument('--lr-backbone', type=float, default=1e-5)
    ap.add_argument('--lr-head', type=float, default=1e-4)
    ap.add_argument('--wd', type=float, default=.01)
    ap.add_argument('--warmup', type=float, default=.05)
    ap.add_argument('--clip', type=float, default=1.)
    ap.add_argument('--max-epochs', type=float, default=4)
    ap.add_argument('--max-steps', type=int, default=0, help='override max-epochs')
    ap.add_argument('--eval-every', type=int, default=0, help='steps; default a quarter epoch')
    ap.add_argument('--patience', type=int, default=3, help='evals without improvement')
    ap.add_argument('--eval-batch', type=int, default=64)
    ap.add_argument('--eval-precision', default='bf16', choices=('bf16', 'fp32'), help='cuda only; cpu is fp32')
    ap.add_argument('--augment', action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument('--workers', type=int, default=8)
    ap.add_argument('--threads', type=int, default=4)
    ap.add_argument('--seed', type=int, default=41)
    ap.add_argument('--log-every', type=int, default=10)
    ap.add_argument('--ref-files', type=int, default=16, help='val files whose windows check the checkpoint round trip')
    ap.add_argument('--weight-cap', type=float, default=5., help='cap sampler weight at cap x mean (0 = off)')
    ap.add_argument('--crop-tries', type=int, default=4, help='random crops per item, keep the highest-RMS one')
    return ap.parse_args(argv)


def main(argv=None) -> None:
    a = parse(argv)
    distributed = 'RANK' in os.environ
    if distributed:
        dist.init_process_group('nccl' if a.device == 'cuda' else 'gloo')
    rank, world = (dist.get_rank(), dist.get_world_size()) if distributed else (0, 1)
    device = torch.device(f"cuda:{os.environ.get('LOCAL_RANK', 0)}" if a.device == 'cuda' else 'cpu')
    if device.type == 'cuda':
        torch.cuda.set_device(device)
    torch.set_num_threads(a.threads)
    torch.manual_seed(a.seed + rank)
    sys.path.insert(0, str(a.code_root))
    from graph_scoring import _roc_metrics
    lead = rank == 0
    a.out.mkdir(parents=True, exist_ok=True)
    log = (a.out / 'metrics.jsonl').open('a') if lead else None

    def emit(record):
        if lead:
            log.write(json.dumps(record) + '\n')
            log.flush()
            print(json.dumps(record), flush=True)

    started = time.time()
    train, samples, train_missing = voice_items(a.manifest, a.stems, 'train')
    val, _, val_missing = voice_items(a.manifest, a.stems, 'val')
    if {x.row['component_id'] for x in train} & {x.row['component_id'] for x in val}:
        raise ValueError('Train and val share components (DB protocol)')
    if {x.label for x in val} != {0., 1.}:
        raise ValueError('Validation needs both classes')
    base = loss_weights(train)
    ids = [x.row['ID'] for x in train]
    factor = duration_balance([x.label for x in train], [samples[i] for i in ids], [base[i] for i in ids])
    weights = torch.tensor([base[i] * f for i, f in zip(ids, factor, strict=True)], dtype=torch.double)
    if a.weight_cap > 0:   # orchestrator: cap per-row draw weight (uncapped: ESS 4,211 of 24,623, top row ~297 draws)
        for _ in range(5):
            weights = torch.minimum(weights, a.weight_cap * weights.mean())
    ess = float(weights.sum() ** 2 / (weights ** 2).sum())
    Crops.tries = a.crop_tries

    steps_per_epoch = math.ceil(len(train) / (a.batch * world))
    total = a.max_steps or max(1, round(a.max_epochs * steps_per_epoch))
    eval_every = a.eval_every or max(1, steps_per_epoch // 4)
    warm = max(1, round(a.warmup * total))
    amp = device.type == 'cuda'
    eval_amp = amp and a.eval_precision == 'bf16'

    model = build(a.model, device, a.first_trainable_layer)
    names = [n for n, p in model.named_parameters() if p.requires_grad]
    head = [p for n, p in model.named_parameters() if p.requires_grad and n.startswith('classifier.')]
    body = [p for n, p in model.named_parameters() if p.requires_grad and not n.startswith('classifier.')]
    before = fingerprint(model) if lead else None
    net = torch.nn.parallel.DistributedDataParallel(model, device_ids=[device] if amp else None,
                                                    gradient_as_bucket_view=True) if distributed else model
    optimizer = torch.optim.AdamW([{'params': body, 'lr': a.lr_backbone}, {'params': head, 'lr': a.lr_head}],
                                  weight_decay=a.wd)
    schedule = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda s: (s + 1) / warm if s < warm else
                                                 .5 * (1 + math.cos(math.pi * (s - warm) / max(1, total - warm))))
    generator = torch.Generator().manual_seed(a.seed * 1000 + rank)
    loader = torch.utils.data.DataLoader(
        Crops(train, a.stems), batch_size=a.batch, drop_last=True, num_workers=a.workers, pin_memory=amp,
        sampler=torch.utils.data.WeightedRandomSampler(weights, total * a.batch, replacement=True, generator=generator),
        persistent_workers=a.workers > 0, prefetch_factor=4 if a.workers else None)

    shard = val[rank::world]
    voices = [load_voice(a.stems / f"{x.row['ID']}.npz") for x in shard]
    ref = [i for i, x in enumerate(shard) if not x.silent][:a.ref_files]   # silent stems are scored 0 anyway
    ref_ids = [shard[i].row['ID'] for i in ref]
    emit({'kind': 'setup', 'world': world, 'sampler_ess': round(ess, 1), 'weight_cap': a.weight_cap, 'crop_tries': a.crop_tries, 'train_items': len(train), 'train_fake': sum(x.label for x in train),
          'val_items': len(val), 'val_fake': sum(x.label for x in val), 'missing_stems': [train_missing, val_missing],
          'trainable_params': sum(p.numel() for p in head + body), 'trainable_tensors': len(names),
          'steps_per_epoch': steps_per_epoch, 'total_steps': total, 'eval_every': eval_every, 'warmup_steps': warm,
          'global_batch': a.batch * world, 'args': {k: str(v) for k, v in vars(a).items()},
          'setup_seconds': round(time.time() - started, 1)})

    history, best, stale = [], None, 0

    def evaluate(step):
        nonlocal best, stale
        t0 = time.time()
        model.eval()
        probs = window_probs(model, voices, a.eval_batch, eval_amp)
        local = [(x.row['ID'], x.label, x.silent, p) for x, p in zip(shard, probs, strict=True)]
        gathered = [None] * world
        if distributed:
            dist.all_gather_object(gathered, local)
        else:
            gathered = [local]
        rows = sorted((r for part in gathered for r in part), key=lambda r: r[0])
        scores = [0. if silent else max(p) for _, _, silent, p in rows]   # runtime: silent voice stem -> 0
        labels = np.array([y for _, y, _, _ in rows])
        eer, auc = _roc_metrics(labels.astype(int).tolist(), scores)
        clipped = np.clip(scores, 1e-7, 1 - 1e-7)
        bce = float(-(labels * np.log(clipped) + (1 - labels) * np.log1p(-clipped)).mean())
        improved = best is None or (eer, bce) < best['key']
        record = {'kind': 'eval', 'step': step, 'epoch': round(step / steps_per_epoch, 3), 'val_eer': eer, 'val_auc': auc,
                  'val_bce': bce, 'improved': improved, 'windows': sum(len(p) for *_, p in rows), 'seconds': round(time.time() - t0, 1)}
        if improved:
            best, stale = {'key': (eer, bce), **record}, 0
            if lead:
                with open(a.out / 'val_best.csv', 'w', newline='') as stream:
                    w = csv.writer(stream)
                    w.writerow(['ID', 'label', 'ft_max', 'ft_top30', 'ft_mean', 'voice_silent'])
                    w.writerows([i, int(y), *members(p), int(s)] for i, y, s, p in rows)
                reference = window_probs(model, [voices[i] for i in ref], a.eval_batch, False)   # fp32, pre-fp16-cast
                save_native(model, a.model, a.out / 'best', {**record, 'reference': dict(zip(ref_ids, reference, strict=True)),
                                                           'fp16_fingerprint': fingerprint(model, half=True),
                                                           'first_trainable_layer': a.first_trainable_layer})
        else:
            stale += 1
        history.append(record)
        emit(record)
        if lead:
            (a.out / 'evals.json').write_text(json.dumps(history, indent=1))
        train_mode(model, a.first_trainable_layer)

    evaluate(0)   # zero-shot baseline in the same code path
    params = head + body
    step, window_loss, t_train, t_log, windows = 0, torch.zeros((), device=device), 0., time.time(), 0
    for x, y in loader:
        t0 = time.time()
        step += 1
        x, y = x.to(device, non_blocking=True).float(), y.to(device, non_blocking=True)
        if a.augment:
            x = augment(x)
        with torch.autocast(device.type, torch.bfloat16, enabled=amp):
            logits = net(x)
        loss = F.cross_entropy(logits.float(), y)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(params, a.clip, error_if_nonfinite=True)
        optimizer.step()
        schedule.step()
        window_loss += loss.detach()
        windows += len(x) * world
        if step % a.log_every == 0 or step == total:
            n = step % a.log_every or a.log_every
            if amp:
                torch.cuda.synchronize()
            now = time.time()
            emit({'kind': 'train', 'step': step, 'epoch': round(step / steps_per_epoch, 3), 'loss': float(window_loss) / n,
                  'grad_norm': float(norm), 'lr_backbone': optimizer.param_groups[0]['lr'],
                  'lr_head': optimizer.param_groups[1]['lr'], 'windows_per_s': round(n * len(x) * world / (now - t_log), 1),
                  'peak_gb': round(torch.cuda.max_memory_allocated() / 1e9, 2) if amp else None})
            window_loss.zero_()
            t_log = now
        t_train += time.time() - t0
        if step % eval_every == 0 or step == total:
            evaluate(step)
            t_log = time.time()
            if stale >= a.patience:
                break

    result = {'best': {k: v for k, v in best.items() if k != 'key'}, 'zero_shot': history[0], 'steps': step,
              'total_steps': total, 'stopped_early': step < total, 'train_windows': windows,
              'train_loop_seconds': round(t_train, 1), 'wall_seconds': round(time.time() - started, 1),
              'eval_seconds': round(sum(h['seconds'] for h in history), 1),
              'peak_gpu_gb': round(torch.cuda.max_memory_allocated() / 1e9, 2) if amp else None}
    if lead:   # only intended tensors moved; then the saved checkpoint must reproduce the pre-cast reference
        after = fingerprint(model)
        changed = {n for n in after if after[n] != before[n]}
        result['params'] = {'trainable_tensors': len(names), 'changed_tensors': len(changed),
                            'changed_outside_trainable': sorted(changed - set(names)),
                            'trainable_unchanged': sorted(set(names) - changed)}
        del net, model, optimizer, schedule, params, head, body, loss, logits   # AccumulateGrad nodes pin params
        gc.collect()
        from antideepfake_runtime import AntiDeepfake
        meta = json.loads((a.out / 'best' / 'meta.json').read_text())
        reloaded = AntiDeepfake(a.out / 'best', str(device))
        again = window_probs(reloaded, [voices[i] for i in ref], a.eval_batch, False)
        diff = max(abs(p - q) for i, r in zip(ref_ids, again, strict=True) for p, q in zip(meta['reference'][i], r, strict=True))
        exact = fingerprint(reloaded) == meta['fp16_fingerprint']   # layout + serialization lossless beyond the fp16 cast
        result['roundtrip'] = {'weights_exact': exact, 'files': len(ref_ids), 'windows': sum(map(len, again)),
                               'max_abs_prob_diff_vs_fp32': diff, 'ok': exact and diff <= 1e-3,
                               'unused_training_only': len(reloaded.unused_training_only)}
        (a.out / 'result.json').write_text(json.dumps(result, indent=1))
        emit({'kind': 'result', **result})
    if distributed:
        dist.barrier()
        dist.destroy_process_group()


if __name__ == '__main__':
    main()
