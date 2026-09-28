"""S8 candidate frozen features for all 42,039 data v4 backbone rows (H100), on the exact runtime inputs:
  raw mix      = baseline load_audio (16 kHz mono) -> peak cap 0.6, fp32 (what full_bank_submission separates)
                 -> WavLM-Large hidden_states SSL_LAYERS (mixwavlm), SSLAM-pretrain blocks SSLAM_LAYERS (mixsslam)
  voice stem   = /root/stems (data_v4/dump_stems.py, fp16 storage) -> data2vec-audio-large (d2v), UniSpeech-SAT-large (usat)
  backing stem = /root/stems -> EnCodec 24 kHz encoder latent (pre-quantisation, 128 x 75 Hz), time mean ++ std (encodec)
Windows = runtime window_layout (64600 @16 kHz, hop 32300); SSL input per-window zero-mean/unit-var (do_normalize);
fp32 compute, TF32 off, fp16 cache. SSL encoders run only the first max(SSL_LAYERS)+1 layers: hidden_states[k] for
k <= max is appended before layer k runs, so it is identical to the full model (checked by --check-truncation).
  python extract_s8.py --ids IDS --manifest backbone_manifest.csv --audio /root/data_v4/out --stems /root/stems \
      --models /root/models --assets /root/bb_assets --code-root /root/ft/code --eat-core VENDOR --out OUT [--shard k --shards n]
"""
from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

import numpy as np
import torch

SSL_LAYERS = (4, 6, 8, 10, 12, 16)
SSLAM_LAYERS = (4, 5, 6, 7, 8, 9)
BATCH = 8


def truncated(model, keep):
    model.encoder.layers = model.encoder.layers[:keep]
    return model


@torch.inference_mode()
def encodec_features(model, chunks: np.ndarray, device) -> np.ndarray:
    import torchaudio
    out = []
    for offset in range(0, len(chunks), BATCH):
        x = torchaudio.functional.resample(torch.from_numpy(chunks[offset:offset + BATCH]).to(device), 16000, 24000)
        z = model.encoder(x[:, None])                      # (B, 128, T') continuous latent before the RVQ
        out.append(torch.cat([z.mean(-1), z.std(-1)], -1).half().cpu().numpy())
    return np.concatenate(out)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--ids', type=Path, required=True)
    ap.add_argument('--manifest', type=Path, required=True)
    ap.add_argument('--audio', type=Path, required=True)
    ap.add_argument('--stems', type=Path, required=True)
    ap.add_argument('--models', type=Path, required=True, help='dir with s6/{wavlm,sslam_pt}, s8/{d2v,usat,encodec}')
    ap.add_argument('--assets', type=Path, required=True)
    ap.add_argument('--code-root', type=Path, required=True)
    ap.add_argument('--eat-core', type=Path, required=True)
    ap.add_argument('--s6-code', type=Path, required=True, help='dir with extract_s6.py (eat_family_features, ssl_features)')
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--shard', type=int, default=0)
    ap.add_argument('--shards', type=int, default=1)
    ap.add_argument('--device', default='cuda')
    ap.add_argument('--deadline', type=float, default=float('inf'))
    ap.add_argument('--check-truncation', action='store_true')
    a = ap.parse_args()
    sys.path[:0] = [str(a.code_root), str(a.eat_core), str(a.s6_code)]
    from mert12_submission import baseline_module, pin_torchaudio_loader, window_layout, windows
    from transformers import AutoModel, EncodecModel
    import extract_s6 as X
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    device = torch.device(a.device)
    a.out.mkdir(parents=True, exist_ok=True)
    paths = {r['ID']: r['path'] for r in csv.DictReader(a.manifest.open())}
    ids = a.ids.read_text().split()[a.shard::a.shards]
    base = baseline_module(a.assets)
    pin_torchaudio_loader(base)
    keep = max(SSL_LAYERS) + 1
    ssl = {k: AutoModel.from_pretrained(a.models / p, local_files_only=True).to(device).eval().requires_grad_(False)
           for k, p in (('mixwavlm', 's6/wavlm'), ('d2v', 's8/d2v'), ('usat', 's8/usat'))}
    if a.check_truncation:
        x = torch.randn(2, 64600, device=device)
        full = {k: [h.clone() for h in m(x, output_hidden_states=True).hidden_states] for k, m in ssl.items()}
    for m in ssl.values():
        truncated(m, keep)
    if a.check_truncation:
        for k, m in ssl.items():
            cut = m(x, output_hidden_states=True).hidden_states
            worst = max(float((cut[j] - full[k][j]).abs().max()) for j in SSL_LAYERS)
            print(f'truncation {k}: max |hidden_states[k] full - truncated| over {SSL_LAYERS} = {worst}', flush=True)
            assert worst == 0.0, k
        del full
    sslam = X.load_eat_family(a.models / 's6' / 'sslam_pt', device)
    encodec = EncodecModel.from_pretrained(a.models / 's8' / 'encodec', local_files_only=True).to(device).eval().requires_grad_(False)
    seconds, done, started = {k: 0. for k in ('decode', *ssl, 'mixsslam', 'encodec')}, 0, time.time()
    for i, row_id in enumerate(ids):
        target = a.out / f'{row_id}.npz'
        if target.exists():
            continue
        if time.time() > a.deadline:
            print(f'{a.shard}: deadline reached at {i}/{len(ids)}', flush=True)
            break
        src = a.stems / f'{row_id}.npz'
        if not src.exists() or row_id not in paths:
            print(f'{a.shard}: missing stems/path {row_id}', flush=True)
            continue
        t = time.time()
        raw = base.load_audio(a.audio / paths[row_id])
        peak = float(np.abs(raw).max())
        if peak > .6:
            raw = (raw * (.6 / peak)).astype(np.float32)
        with np.load(src) as d:
            voice, backing = d['voice'].astype(np.float32), d['backing'].astype(np.float32)
            samples, silent = int(d['samples']), bool(d['voice_silent'])
        if len(raw) != samples:
            print(f'{a.shard}: SKIP {row_id} decoded {len(raw)} samples != stems {samples}', flush=True)
            continue
        starts = window_layout(samples)
        wm = np.asarray(windows(np.asarray(raw, np.float32), starts), np.float32)
        wv = np.asarray(windows(voice, starts), np.float32)
        wb = np.asarray(windows(backing, starts), np.float32)
        seconds['decode'] += time.time() - t
        extra = {}
        for k, m in ssl.items():
            t = time.time(); extra[f'{k}'] = X.ssl_features(m, wm if k == 'mixwavlm' else wv, SSL_LAYERS, device); seconds[k] += time.time() - t
        t = time.time(); extra['mixsslam'] = X.eat_family_features(sslam, wm, SSLAM_LAYERS, device); seconds['mixsslam'] += time.time() - t
        t = time.time(); extra['encodec'] = encodec_features(encodec, wb, device); seconds['encodec'] += time.time() - t
        partial = target.with_suffix('.tmp.npz')
        np.savez(partial, **extra, ssl_layers=np.array(SSL_LAYERS), sslam_layers=np.array(SSLAM_LAYERS),
                 voice_silent=np.array(silent), samples=np.array(samples))
        partial.replace(target)
        done += 1
        if done % 500 == 0 or i == len(ids) - 1:
            print(f'{a.shard}/{a.shards} {i + 1}/{len(ids)} {done / (time.time() - started):.2f} files/s',
                  {k: round(v, 1) for k, v in seconds.items()}, flush=True)


if __name__ == '__main__':
    main()
