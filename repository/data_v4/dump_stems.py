"""HTDemucs voice/backing stems for fine-tuning, on the exact runtime preprocessing (full_bank_submission.Engine.predict):
librosa 16 kHz mono load -> peak cap 0.6 -> baseline separate_voice_and_music -> pad/crop to len(raw).
One npz per row (voice, backing: float16 [samples]; samples), resumable, shardable. Non-AI-Hub data v4 rows only.

  python data_v4/dump_stems.py --manifest M.csv --input OUT --assets ASSETS --out STEMS --shard 0 --shards 20
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
    ap.add_argument('--assets', type=Path, required=True)
    ap.add_argument('--code-root', type=Path, required=True, help='folder with mert12_submission.py / v6_runtime.py')
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--shard', type=int, default=0)
    ap.add_argument('--shards', type=int, default=1)
    ap.add_argument('--device', default='cuda')
    a = ap.parse_args()
    sys.path.insert(0, str(a.code_root))
    from mert12_submission import baseline_module, pin_torchaudio_loader
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    device = torch.device(a.device)
    a.out.mkdir(parents=True, exist_ok=True)
    rows = list(csv.DictReader(a.manifest.open()))[a.shard::a.shards]
    base = baseline_module(a.assets)
    pin_torchaudio_loader(base)
    demucs = base.load_htdemucs_model().to(device).eval().requires_grad_(False)
    started, done = time.time(), 0
    for row in rows:
        target = a.out / f"{row['ID']}.npz"
        if target.exists():
            continue
        raw = base.load_audio(a.input / row['path'])
        peak = float(np.abs(raw).max())
        if peak > .6:
            raw = (raw * (.6 / peak)).astype(np.float32)
        with tempfile.TemporaryDirectory(prefix='stems_') as temporary:
            normalized = Path(temporary) / 'normalized.wav'
            sf.write(normalized, raw, 16000, subtype='FLOAT')
            voice, backing = base.separate_voice_and_music(normalized, demucs, device)
        voice = np.pad(voice[:len(raw)], (0, max(0, len(raw) - len(voice))))
        backing = np.pad(backing[:len(raw)], (0, max(0, len(raw) - len(backing))))
        partial = target.with_suffix('.tmp.npz')
        np.savez(partial, voice=voice.astype(np.float16), backing=backing.astype(np.float16), samples=np.array(len(raw)),
                 voice_silent=np.array(base.calculate_rms(voice) < base.SILENCE_RMS))
        partial.replace(target)
        done += 1
        if done % 200 == 0:
            print(f'{a.shard}/{a.shards} {done} rows {done / (time.time() - started):.2f}/s', flush=True)


if __name__ == '__main__':
    main()
