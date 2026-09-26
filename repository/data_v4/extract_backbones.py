"""New-backbone features for trained heads, on the exact runtime preprocessing (full_bank_submission.Engine.predict):
load -> peak cap 0.6 -> HTDemucs voice/backing stems.
  voice stem   -> AntiDeepfake XLS-R-1B, 64600-sample windows (hop 32300): pooled last hidden state (1280, fp16) + fake prob
  backing stem -> MERT-v1-330M on the runtime windows (24 kHz): time-mean of hidden layers LAYERS (fp16)
One npz per row, resumable (existing npz skipped), shardable.

  python data_v4/extract_backbones.py --manifest M.csv --input OUT --assets ASSETS --antideepfake DIR --out CACHE --shard 0 --shards 5
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
from mert12_submission import AutoFeatureExtractor, AutoModel, baseline_module, pin_torchaudio_loader, window_layout, windows  # noqa: E402
from antideepfake_runtime import AntiDeepfake  # noqa: E402

LAYERS = (3, 6, 9, 12, 15, 18, 21, 24)   # hidden_states index; 12 = the current music feature
AD_WINDOW, AD_HOP = 64600, 32300


@torch.inference_mode()
def ad_windows(model: AntiDeepfake, audio: np.ndarray, device, batch: int = 16) -> tuple[np.ndarray, np.ndarray]:
    """Same windowing and pooling as AntiDeepfake.predict, returning the pooled embedding the classifier sees."""
    audio = audio if len(audio) >= 400 else np.pad(audio, (0, 400 - len(audio)))
    starts = list(range(0, max(1, len(audio) - AD_WINDOW + 1), AD_HOP))
    if len(audio) > AD_WINDOW and starts[-1] != len(audio) - AD_WINDOW:
        starts.append(len(audio) - AD_WINDOW)
    emb, prob = [], []
    for offset in range(0, len(starts), batch):
        clips = torch.from_numpy(np.stack([audio[s:s + AD_WINDOW] for s in starts[offset:offset + batch]])).to(device)
        hidden = model.encoder(torch.nn.functional.layer_norm(clips, clips.shape[-1:])).last_hidden_state.mean(1)
        emb.append(hidden.half().cpu().numpy())
        prob.append(model.classifier(hidden).softmax(-1)[:, 0].float().cpu().numpy())
    return np.concatenate(emb), np.concatenate(prob)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--manifest', type=Path, required=True)
    ap.add_argument('--input', type=Path, required=True)
    ap.add_argument('--assets', type=Path, required=True, help='needs baseline/ (HTDemucs) and mert/')
    ap.add_argument('--antideepfake', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--shard', type=int, default=0)
    ap.add_argument('--shards', type=int, default=1)
    ap.add_argument('--device', default='cuda')
    ap.add_argument('--deadline', type=float, default=float('inf'), help='unix time; stop cleanly after it')
    a = ap.parse_args()
    device = torch.device(a.device)
    a.out.mkdir(parents=True, exist_ok=True)
    rows = list(csv.DictReader(a.manifest.open()))[a.shard::a.shards]
    base = baseline_module(a.assets)
    pin_torchaudio_loader(base)
    demucs = base.load_htdemucs_model().to(device).eval().requires_grad_(False)
    detector = AntiDeepfake(a.antideepfake, a.device)
    processor = AutoFeatureExtractor.from_pretrained(a.assets / 'mert', local_files_only=True)
    mert = AutoModel.from_pretrained(a.assets / 'mert', trust_remote_code=True, local_files_only=True).to(device).eval()
    import librosa
    done = 0
    for i, row in enumerate(rows):
        target = a.out / f"{row['ID']}.npz"
        if target.exists():
            continue
        if time.time() > a.deadline:
            print(f'{a.shard}: deadline reached at {i}/{len(rows)}', flush=True)
            break
        raw = base.load_audio(a.input / row['path'])
        peak = float(np.abs(raw).max())
        if peak > .6:
            raw = (raw * (.6 / peak)).astype(np.float32)
        with tempfile.TemporaryDirectory(prefix='backbones_') as temporary:
            normalized = Path(temporary) / 'normalized.wav'
            sf.write(normalized, raw, 16000, subtype='FLOAT')
            voice, backing = base.separate_voice_and_music(normalized, demucs, device)
        voice = np.pad(voice[:len(raw)], (0, max(0, len(raw) - len(voice)))).astype(np.float32)
        backing = np.pad(backing[:len(raw)], (0, max(0, len(raw) - len(backing)))).astype(np.float32)
        emb, prob = ad_windows(detector, voice, device)
        starts = window_layout(len(raw))
        layers = []
        with torch.inference_mode():
            for offset in range(0, len(starts), 8):   # runtime MERT batch
                audio = [librosa.resample(x, orig_sr=16000, target_sr=24000, res_type='soxr_hq') for x in windows(backing, starts[offset:offset + 8])]
                values = processor(audio, sampling_rate=24000, return_tensors='pt', padding=True).input_values.to(device)
                hidden = mert(values, output_hidden_states=True).hidden_states
                layers.append(torch.stack([hidden[k].mean(1) for k in LAYERS], 1).half().cpu().numpy())
        partial = target.with_suffix('.tmp.npz')
        np.savez(partial, ad_embeddings=emb, ad_probs=prob, mert_layers=np.concatenate(layers), layers=np.array(LAYERS),
                 voice_silent=np.array(base.calculate_rms(voice) < base.SILENCE_RMS), samples=np.array(len(raw)))
        partial.replace(target)
        done += 1
        if done % 200 == 0:
            print(f'{a.shard}/{a.shards} {i}/{len(rows)}', flush=True)


if __name__ == '__main__':
    main()
