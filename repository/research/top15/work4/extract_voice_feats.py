"""Voice-route features for external rows, with the runtime semantics of full_bank_submission.Engine.predict:
peak cap 0.6 -> HTDemucs voice stem (padded/cropped to the input) -> DF-Arena fc5-input embedding per
base.get_segment_starts segment (voice_embeddings, float32 as the old cache) + AntiDeepfake.windows (ad_embeddings,
float16 as backbone_cache) + voice_silent + samples. fp32 compute, TF32 off.
  python extract_voice_feats.py --manifest M.csv --input ROOT --assets ASSETS --antideepfake DIR --code-root CODE --out OUT [--shard k --shards n]
M.csv needs columns ID, path (relative to ROOT).
"""
from __future__ import annotations

import argparse
import csv
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import torch


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--manifest', type=Path, required=True)
    ap.add_argument('--input', type=Path, required=True)
    ap.add_argument('--assets', type=Path, required=True, help='needs baseline/ (HTDemucs, DF-Arena)')
    ap.add_argument('--antideepfake', type=Path, required=True)
    ap.add_argument('--code-root', type=Path, required=True, help='dir with mert12_submission.py and antideepfake_runtime.py')
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--shard', type=int, default=0)
    ap.add_argument('--shards', type=int, default=1)
    ap.add_argument('--device', default='cuda')
    a = ap.parse_args()
    sys.path.insert(0, str(a.code_root))
    from mert12_submission import baseline_module, pin_torchaudio_loader
    from antideepfake_runtime import AntiDeepfake
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    device = torch.device(a.device)
    a.out.mkdir(parents=True, exist_ok=True)
    rows = list(csv.DictReader(a.manifest.open()))[a.shard::a.shards]
    base = baseline_module(a.assets)
    pin_torchaudio_loader(base)
    demucs = base.load_htdemucs_model().to(device).eval().requires_grad_(False)
    df, _ = base.load_df_arena_model(device)
    df.requires_grad_(False)
    captured = []
    df.backbone.conformer.fc5.register_forward_pre_hook(lambda _m, inputs: captured.append(inputs[0].detach().float().cpu()))
    detector = AntiDeepfake(a.antideepfake, a.device)
    done, started = 0, time.time()
    for i, row in enumerate(rows):
        target = a.out / f"{row['ID']}.npz"
        if target.exists():
            continue
        raw = base.load_audio(a.input / row['path'])
        peak = float(np.abs(raw).max())
        if peak > .6:
            raw = (raw * (.6 / peak)).astype(np.float32)
        with tempfile.TemporaryDirectory(prefix='voicefeats_') as temporary:
            normalized = Path(temporary) / 'normalized.wav'
            sf.write(normalized, raw, 16000, subtype='FLOAT')
            with torch.inference_mode():
                voice, _ = base.separate_voice_and_music(normalized, demucs, device)
        voice = np.pad(voice[:len(raw)], (0, max(0, len(raw) - len(voice)))).astype(np.float32)
        captured.clear()
        with torch.inference_mode():
            for start in base.get_segment_starts(len(raw)):
                df(input_values=torch.from_numpy(base.extract_segment(voice, start)).to(device))
        emb, prob = detector.windows(voice)
        partial = target.with_suffix('.tmp.npz')
        np.savez(partial, voice_embeddings=torch.cat(captured).numpy().astype(np.float32), ad_embeddings=emb.astype(np.float16),
                 ad_probs=prob, voice_silent=np.array(base.calculate_rms(voice) < base.SILENCE_RMS), samples=np.array(len(raw)),
                 source_peak=np.array(peak))
        partial.replace(target)
        done += 1
        if done % 250 == 0 or i == len(rows) - 1:
            print(f'{a.shard}/{a.shards} {i + 1}/{len(rows)} {done / (time.time() - started):.2f} files/s', flush=True)


if __name__ == '__main__':
    main()
