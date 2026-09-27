"""S6 candidate frozen encoders on the stored HTDemucs stems (data_v4/dump_stems.py: runtime preprocessing, fp16):
  backing stem -> SSLAM pretrain / AS2M fine-tuned (EAT code, 16 kHz Kaldi fbank 128): patch-mean of blocks SSLAM_LAYERS
  voice stem   -> WavLM-Large / HuBERT-Large ll60k: frame-mean of hidden_states WAVLM_LAYERS / HUBERT_LAYERS
Windows = runtime window_layout (64600 @16 kHz, hop 32300), fp32 compute, TF32 off, fp16 cache (choice_lock.json).
  python extract_s6.py --ids IDS.txt --stems /root/stems --models DIR --code-root CODE --eat-core VENDOR --out OUT [--shard k --shards n]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

SSLAM_LAYERS = (4, 5, 6, 7, 8, 9)
WAVLM_LAYERS = (4, 6, 8, 10, 12, 16)
HUBERT_LAYERS = (2, 4, 6, 8, 10, 12)
EAT_MEAN, EAT_STD = -4.268, 4.569
BATCH = 8


def load_eat_family(directory: Path, device):
    from eat_core import EAT, EATConfig
    from safetensors.torch import load_file
    cfg = json.loads((directory / 'config.json').read_text())
    model = EAT(EATConfig(**{k: v for k, v in cfg.items() if k not in ('architectures', 'auto_map', 'model_type', 'torch_dtype', 'transformers_version')}))
    state = {k.removeprefix('model.'): v.float() for k, v in load_file(str(directory / 'model.safetensors')).items()}
    model.load_state_dict(state, strict=True)
    return model.to(device).eval().requires_grad_(False)


@torch.inference_mode()
def eat_family_features(model, chunks: np.ndarray, layers, device) -> np.ndarray:
    import torchaudio
    out = []
    for offset in range(0, len(chunks), BATCH):
        mels = []
        for x in torch.from_numpy(chunks[offset:offset + BATCH]).to(device):
            mel = torchaudio.compliance.kaldi.fbank((x - x.mean())[None], htk_compat=True, sample_frequency=16000, use_energy=False,
                                                    window_type='hanning', num_mel_bins=128, dither=0.0, frame_shift=10)
            mels.append((mel[:len(mel) // 16 * 16] - EAT_MEAN) / (EAT_STD * 2))
        x = model.local_encoder(torch.stack(mels)[:, None])
        x = x + model.fixed_positional_encoder(x, None)[:, :x.size(1), :]
        x = model.pos_drop(model.pre_norm(torch.cat((model.extra_tokens.expand(len(x), -1, -1), x), 1)))
        got = []
        for index, block in enumerate(model.blocks[:max(layers)], 1):
            x, _ = block(x)
            if index in layers:
                got.append(x[:, 1:].mean(1))
        out.append(torch.stack(got, 1).half().cpu().numpy())
    return np.concatenate(out)


@torch.inference_mode()
def ssl_features(model, chunks: np.ndarray, layers, device) -> np.ndarray:
    """Wav2Vec2FeatureExtractor do_normalize (per-window zero mean / unit variance), then frame-mean hidden states."""
    out = []
    for offset in range(0, len(chunks), BATCH):
        x = torch.from_numpy(chunks[offset:offset + BATCH]).to(device)
        x = (x - x.mean(1, keepdim=True)) / torch.sqrt(x.var(1, unbiased=False, keepdim=True) + 1e-7)
        hidden = model(x, output_hidden_states=True).hidden_states
        out.append(torch.stack([hidden[k].mean(1) for k in layers], 1).half().cpu().numpy())
    return np.concatenate(out)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--ids', type=Path, required=True)
    ap.add_argument('--stems', type=Path, required=True)
    ap.add_argument('--models', type=Path, required=True, help='dir with sslam_pt/, sslam_ft/, wavlm/, hubert/')
    ap.add_argument('--code-root', type=Path, required=True, help='dir with mert12_submission.py (window_layout, windows)')
    ap.add_argument('--eat-core', type=Path, required=True, help='dir containing the vendored eat_core package')
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--shard', type=int, default=0)
    ap.add_argument('--shards', type=int, default=1)
    ap.add_argument('--device', default='cuda')
    ap.add_argument('--deadline', type=float, default=float('inf'))
    a = ap.parse_args()
    sys.path[:0] = [str(a.code_root), str(a.eat_core)]
    from mert12_submission import window_layout, windows
    from transformers import AutoModel
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    device = torch.device(a.device)
    a.out.mkdir(parents=True, exist_ok=True)
    ids = a.ids.read_text().split()[a.shard::a.shards]
    sslam = {k: load_eat_family(a.models / k, device) for k in ('sslam_pt', 'sslam_ft')}
    ssl = {k: AutoModel.from_pretrained(a.models / k, local_files_only=True).to(device).eval().requires_grad_(False) for k in ('wavlm', 'hubert')}
    seconds, done, started = {k: 0. for k in (*sslam, *ssl)}, 0, time.time()
    for i, row_id in enumerate(ids):
        target = a.out / f'{row_id}.npz'
        if target.exists():
            continue
        if time.time() > a.deadline:
            print(f'{a.shard}: deadline reached at {i}/{len(ids)}', flush=True)
            break
        src = a.stems / f'{row_id}.npz'
        if not src.exists():
            print(f'{a.shard}: missing stems {row_id}', flush=True)
            continue
        with np.load(src) as d:
            voice, backing = d['voice'].astype(np.float32), d['backing'].astype(np.float32)
            samples, silent = int(d['samples']), bool(d['voice_silent'])
        starts = window_layout(samples)
        wb, wv = np.asarray(windows(backing, starts), np.float32), np.asarray(windows(voice, starts), np.float32)
        extra = {}
        for k, m in sslam.items():
            t = time.time(); extra[f'{k}_backing'] = eat_family_features(m, wb, SSLAM_LAYERS, device); seconds[k] += time.time() - t
        for k, m in ssl.items():
            t = time.time(); extra[f'{k}_voice'] = ssl_features(m, wv, WAVLM_LAYERS if k == 'wavlm' else HUBERT_LAYERS, device); seconds[k] += time.time() - t
        partial = target.with_suffix('.tmp.npz')
        np.savez(partial, **extra, sslam_layers=np.array(SSLAM_LAYERS), wavlm_layers=np.array(WAVLM_LAYERS),
                 hubert_layers=np.array(HUBERT_LAYERS), voice_silent=np.array(silent), samples=np.array(samples))
        partial.replace(target)
        done += 1
        if done % 500 == 0 or i == len(ids) - 1:
            print(f'{a.shard}/{a.shards} {i + 1}/{len(ids)} {done / (time.time() - started):.2f} files/s',
                  {k: round(v, 1) for k, v in seconds.items()}, flush=True)


if __name__ == '__main__':
    main()
