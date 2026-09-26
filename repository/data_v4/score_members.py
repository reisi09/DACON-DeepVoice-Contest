"""Zero-shot member scores for component fusion, on the exact runtime preprocessing (full_bank_submission.Engine.predict):
load -> peak cap 0.6 -> HTDemucs voice stem. AntiDeepfake models score the voice stem; SONICS scores the capped mix.

  python data_v4/score_members.py --manifest BANK.csv --split val --input OUT --assets ASSETS \
      --antideepfake xlsr1b=DIR [xlsr2b=DIR ...] --sonics DIR --sonics-code DIR --out scores.csv --shard 0 --shards 4
Resumable: IDs already in --out are skipped. One CSV row per manifest row.
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

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from mert12_submission import baseline_module, pin_torchaudio_loader  # noqa: E402
from antideepfake_runtime import AntiDeepfake  # noqa: E402
from component_fusion import detector_members, sonics_prob  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--manifest', type=Path, required=True)
    ap.add_argument('--split', default=None, help='keep only rows with this split value')
    ap.add_argument('--input', type=Path, required=True)
    ap.add_argument('--assets', type=Path, required=True)
    ap.add_argument('--antideepfake', nargs='+', default=[], help='name=model_dir')
    ap.add_argument('--sonics', type=Path)
    ap.add_argument('--sonics-code', type=Path)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--shard', type=int, default=0)
    ap.add_argument('--shards', type=int, default=1)
    ap.add_argument('--device', default='cuda')
    a = ap.parse_args()
    device = torch.device(a.device)
    rows = [r for r in csv.DictReader(a.manifest.open()) if a.split is None or r['split'] == a.split]
    rows = rows[a.shard::a.shards]
    done = {r['ID'] for r in csv.DictReader(a.out.open())} if a.out.exists() else set()
    base = baseline_module(a.assets)
    pin_torchaudio_loader(base)
    demucs = base.load_htdemucs_model().to(device).eval().requires_grad_(False)
    detectors = {name: AntiDeepfake(Path(d), a.device) for name, d in (s.split('=', 1) for s in a.antideepfake)}
    sonics = None
    if a.sonics:
        sys.path.insert(0, str(a.sonics_code))
        from sonics.models.hf_model import HFAudioClassifier
        sonics = HFAudioClassifier.from_pretrained(str(a.sonics)).to(device).eval()
    fields = ['ID', 'voice_silent', 'seconds'] + [f'{n}_{k}' for n in detectors for k in ('max', 'top30', 'mean')] + (['sonics'] if sonics else [])
    new = not a.out.exists()
    with a.out.open('a', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        if new:
            writer.writeheader()
        for i, row in enumerate(rows):
            if row['ID'] in done:
                continue
            started = time.time()
            raw = base.load_audio(a.input/row['path'])
            peak = float(np.abs(raw).max())
            if peak > .6:
                raw = (raw*(.6/peak)).astype(np.float32)
            with tempfile.TemporaryDirectory(prefix='members_') as temporary:
                normalized = Path(temporary)/'normalized.wav'
                sf.write(normalized, raw, 16000, subtype='FLOAT')
                voice, _ = base.separate_voice_and_music(normalized, demucs, device)
            voice = np.pad(voice[:len(raw)], (0, max(0, len(raw)-len(voice)))).astype(np.float32)
            out = {'ID': row['ID'], 'voice_silent': int(base.calculate_rms(voice) < base.SILENCE_RMS)}
            for name, model in detectors.items():
                out.update(detector_members(name, model.predict(voice if len(voice) >= 400 else np.pad(voice, (0, 400-len(voice))), float('inf'))))
            if sonics:
                out['sonics'] = sonics_prob(sonics, raw, device)
            out['seconds'] = round(time.time()-started, 3)
            writer.writerow(out)
            stream.flush()
            if i % 100 == 0:
                print(f'{a.shard}/{a.shards} {i}/{len(rows)} {out["seconds"]}s', flush=True)


if __name__ == '__main__':
    main()
