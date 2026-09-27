"""Frozen MuQ (+ optional EAT-large) window features for trained heads, on the exact runtime preprocessing
(full_bank_submission.Engine.predict): load -> peak cap 0.6 -> HTDemucs voice/backing stems -> runtime window layout
(mert12_submission.window_layout: 64600-sample windows, hop 32300, exact tail; right-zero-padded).
  backing stem -> MuQ-large-msd-iter (24 kHz, soxr_hq like the MERT path): time-mean of hidden states MUQ_LAYERS (fp16)
  voice stem   -> the same MuQ on the same windows (singing-voice feature)                                   (fp16)
  backing stem -> EAT-large AS2M fine-tuned (16 kHz Kaldi fbank 128): mean over patch tokens of blocks EAT_LAYERS (fp16)
Weights are loaded from fp16 safetensors (written once from the HF fp32 file; the same file ships in the package) and
computed in fp32 with TF32 off, like the L4 runtime. One npz per row, resumable, shardable, same CLI as
data_v4/extract_backbones.py.

  python extract_new_backbones.py --manifest M.csv --input OUT --assets ASSETS --muq MUQ_DIR [--eat EAT_DIR] \
      --out CACHE --shard 0 --shards 4
MUQ_DIR = HF snapshot of OpenMuQ/MuQ-large-msd-iter (config.json, model.safetensors);
EAT_DIR = HF snapshot of worstchan/EAT-large_epoch20_finetune_AS2M (config.json, model.safetensors).
Needs no packages beyond the evaluation server's (vendor/ replaces muq's easydict and EAT's timm helpers).
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from safetensors.torch import load_file, save_file

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / 'vendor'))
MUQ_LAYERS = (0, 2, 4, 6, 8, 10, 12)      # hidden_states index: 0 = conv front end, 12 = last Conformer layer (layer-normed)
EAT_LAYERS = (4, 8, 12, 16, 20, 24)       # block outputs of the 24-block EAT-large
EAT_MEAN, EAT_STD = -4.268, 4.569          # EAT model card normalisation
BATCH = 8                                  # runtime MERT batch


def fp16_weights(directory: Path) -> Path:
    """The packaged weight file: fp16 copy of the HF fp32 safetensors (deterministic, written once)."""
    target = directory / 'model_fp16.safetensors'
    if not target.exists():
        state = {k: v.half().contiguous() if v.is_floating_point() else v for k, v in load_file(str(directory / 'model.safetensors')).items()}
        partial = target.with_suffix(f'.{os.getpid()}.tmp')   # shards may race here: identical content, atomic rename
        save_file(state, str(partial))
        partial.replace(target)
    return target


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1 << 24), b''):
            h.update(block)
    return h.hexdigest()


def load_muq(directory: Path, device):
    from muq_core import MuQ, MuQConfig
    model = MuQ(MuQConfig(**json.loads((directory / 'config.json').read_text())))
    state = {k: v.float() for k, v in load_file(str(fp16_weights(directory))).items()}
    model.load_state_dict(state, strict=True)
    return model.to(device).eval().requires_grad_(False)


def load_eat(directory: Path, device):
    from eat_core import EAT, EATConfig
    cfg = json.loads((directory / 'config.json').read_text())
    model = EAT(EATConfig(**{k: v for k, v in cfg.items() if k not in ('architectures', 'auto_map', 'model_type', 'torch_dtype', 'transformers_version')}))
    state = {k.removeprefix('model.'): v.float() for k, v in load_file(str(fp16_weights(directory))).items()}
    model.load_state_dict(state, strict=True)
    return model.to(device).eval().requires_grad_(False)


@torch.inference_mode()
def muq_features(model, chunks: np.ndarray, device) -> np.ndarray:
    """chunks: (n, 64600) 16 kHz windows -> (n, len(MUQ_LAYERS), 1024) fp16, time-mean per hidden state."""
    import librosa
    out = []
    for offset in range(0, len(chunks), BATCH):
        audio = np.stack([librosa.resample(x, orig_sr=16000, target_sr=24000, res_type='soxr_hq') for x in chunks[offset:offset + BATCH]]).astype(np.float32)
        hidden = model(torch.from_numpy(audio).to(device), output_hidden_states=True).hidden_states
        out.append(torch.stack([hidden[k].mean(1) for k in MUQ_LAYERS], 1).half().cpu().numpy())
    return np.concatenate(out)


@torch.inference_mode()
def eat_features(model, chunks: np.ndarray, device) -> np.ndarray:
    """chunks: (n, 64600) 16 kHz windows -> (n, len(EAT_LAYERS), 1024) fp16, mean over patch tokens (CLS excluded)."""
    import torchaudio
    out = []
    for offset in range(0, len(chunks), BATCH):
        mels = []
        for x in torch.from_numpy(chunks[offset:offset + BATCH]).to(device):
            mel = torchaudio.compliance.kaldi.fbank((x - x.mean())[None], htk_compat=True, sample_frequency=16000, use_energy=False,
                                                    window_type='hanning', num_mel_bins=128, dither=0.0, frame_shift=10)
            mels.append((mel[:len(mel) // 16 * 16] - EAT_MEAN) / (EAT_STD * 2))   # whole 16-frame patches (402 -> 400 frames)
        x = model.local_encoder(torch.stack(mels)[:, None])
        x = x + model.fixed_positional_encoder(x, None)[:, :x.size(1), :]
        x = model.pos_drop(model.pre_norm(torch.cat((model.extra_tokens.expand(len(x), -1, -1), x), 1)))
        layers = []
        for index, block in enumerate(model.blocks, 1):
            x, _ = block(x)
            if index in EAT_LAYERS:
                layers.append(x[:, 1:].mean(1))
        out.append(torch.stack(layers, 1).half().cpu().numpy())
    return np.concatenate(out)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--manifest', type=Path, required=True)
    ap.add_argument('--input', type=Path, required=True)
    ap.add_argument('--assets', type=Path, required=True, help='needs baseline/ (HTDemucs)')
    ap.add_argument('--muq', type=Path, required=True)
    ap.add_argument('--eat', type=Path, default=None, help='omit to skip EAT-large')
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--code-root', type=Path, default=HERE.parents[3], help='project root (mert12_submission.py)')
    ap.add_argument('--shard', type=int, default=0)
    ap.add_argument('--shards', type=int, default=1)
    ap.add_argument('--device', default='cuda')
    ap.add_argument('--deadline', type=float, default=float('inf'), help='unix time; stop cleanly after it')
    a = ap.parse_args()
    sys.path.insert(0, str(a.code_root))
    from mert12_submission import baseline_module, pin_torchaudio_loader, window_layout, windows
    torch.backends.cuda.matmul.allow_tf32 = False   # runtime (full_bank_submission.run) computes without TF32
    torch.backends.cudnn.allow_tf32 = False
    device = torch.device(a.device)
    a.out.mkdir(parents=True, exist_ok=True)
    rows = list(csv.DictReader(a.manifest.open()))[a.shard::a.shards]
    base = baseline_module(a.assets)
    pin_torchaudio_loader(base)
    demucs = base.load_htdemucs_model().to(device).eval().requires_grad_(False)
    muq = load_muq(a.muq, device)
    eat = load_eat(a.eat, device) if a.eat else None
    if a.shard == 0:
        (a.out / 'weights.json').write_text(json.dumps({'muq': sha256(fp16_weights(a.muq)), 'eat': a.eat and sha256(fp16_weights(a.eat)),
                                                        'muq_layers': MUQ_LAYERS, 'eat_layers': EAT_LAYERS}, indent=1))
    seconds, done, started = {'load_demucs': 0., 'muq': 0., 'eat': 0.}, 0, time.time()
    for i, row in enumerate(rows):
        target = a.out / f"{row['ID']}.npz"
        if target.exists():
            continue
        if time.time() > a.deadline:
            print(f'{a.shard}: deadline reached at {i}/{len(rows)}', flush=True)
            break
        t = time.time()
        raw = base.load_audio(a.input / row['path'])
        peak = float(np.abs(raw).max())
        if peak > .6:
            raw = (raw * (.6 / peak)).astype(np.float32)
        with tempfile.TemporaryDirectory(prefix='newbackbones_') as temporary:
            normalized = Path(temporary) / 'normalized.wav'
            sf.write(normalized, raw, 16000, subtype='FLOAT')
            voice, backing = base.separate_voice_and_music(normalized, demucs, device)
        voice = np.pad(voice[:len(raw)], (0, max(0, len(raw) - len(voice)))).astype(np.float32)
        backing = np.pad(backing[:len(raw)], (0, max(0, len(raw) - len(backing)))).astype(np.float32)
        starts = window_layout(len(raw))
        seconds['load_demucs'] += time.time() - t
        t = time.time()
        extra = {'muq_backing': muq_features(muq, windows(backing, starts), device),
                 'muq_voice': muq_features(muq, windows(voice, starts), device)}
        seconds['muq'] += time.time() - t
        if eat is not None:
            t = time.time()
            extra['eat_backing'] = eat_features(eat, windows(backing, starts), device)
            seconds['eat'] += time.time() - t
        partial = target.with_suffix('.tmp.npz')
        np.savez(partial, **extra, muq_layers=np.array(MUQ_LAYERS), eat_layers=np.array(EAT_LAYERS),
                 voice_silent=np.array(base.calculate_rms(voice) < base.SILENCE_RMS), samples=np.array(len(raw)))
        partial.replace(target)
        done += 1
        if done % 200 == 0 or i == len(rows) - 1:
            rate = done / (time.time() - started)
            print(f'{a.shard}/{a.shards} {i + 1}/{len(rows)} {rate:.2f} files/s', {k: round(v, 1) for k, v in seconds.items()}, flush=True)


if __name__ == '__main__':
    main()
