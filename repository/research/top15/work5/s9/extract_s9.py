"""S9 candidate features for all 42,039 backbone rows: the S7 music encoders on the RAW MIX (post peak cap, fp32; the input
of S8's mixsslam member), on the exact training paths of their backing-stem features:
  mixeat  = EAT-large blocks EAT_LAYERS, mean over patch tokens (research/top15/work2/backbones/extract_new_backbones.py)
  mixmert = MERT-v1-330M hidden_states MERT_LAYERS, time mean after per-window soxr_hq 16->24 kHz (data_v4/extract_backbones.py)
Weights = the packaged assets (S8 tree model/assets/{eat,mert}), loaded like full_bank_submission.py. fp32 compute, TF32 off,
fp16 cache, per-row atomic write under a per-process temporary name.
  python extract_s9.py --ids IDS --manifest backbone_manifest.csv --audio /root/data_v4/out --assets TREE/model/assets \
      --runtime TREE/model/runtime --bb-assets /root/bb_assets --code-root /root/ft/code --out OUT [--shard k --shards n]
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

EAT_LAYERS = (4, 8, 12, 16, 20, 24)
MERT_LAYERS = (3, 6, 9, 12, 15, 18, 21, 24)
EAT_MEAN, EAT_STD = -4.268, 4.569
BATCH = 8


@torch.inference_mode()
def eat_features(model, chunks: np.ndarray, device) -> np.ndarray:
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
        layers = []
        for index, block in enumerate(model.blocks, 1):
            x, _ = block(x)
            if index in EAT_LAYERS:
                layers.append(x[:, 1:].mean(1))
        out.append(torch.stack(layers, 1).half().cpu().numpy())
    return np.concatenate(out)


@torch.inference_mode()
def mert_features(mert, processor, chunks: np.ndarray, device) -> np.ndarray:
    import librosa
    out = []
    for offset in range(0, len(chunks), BATCH):
        audio = [librosa.resample(x, orig_sr=16000, target_sr=24000, res_type='soxr_hq') for x in chunks[offset:offset + BATCH]]
        values = processor(audio, sampling_rate=24000, return_tensors='pt', padding=True).input_values.to(device)
        hidden = mert(values, output_hidden_states=True).hidden_states
        out.append(torch.stack([hidden[k].mean(1) for k in MERT_LAYERS], 1).half().cpu().numpy())
    return np.concatenate(out)


def main() -> None:
    ap = argparse.ArgumentParser()
    for name in ('--ids', '--manifest', '--audio', '--assets', '--runtime', '--bb-assets', '--code-root', '--out'):
        ap.add_argument(name, type=Path, required=True)
    ap.add_argument('--shard', type=int, default=0)
    ap.add_argument('--shards', type=int, default=1)
    ap.add_argument('--device', default='cuda')
    ap.add_argument('--deadline', type=float, default=float('inf'))
    a = ap.parse_args()
    sys.path[:0] = [str(a.runtime), str(a.code_root)]   # the packaged runtime modules first
    from mert12_submission import AutoFeatureExtractor, AutoModel, baseline_module, pin_torchaudio_loader, window_layout, windows
    from full_bank_submission import load_eat_family
    import mert12_submission, full_bank_submission, hashlib
    for mod in (mert12_submission, full_bank_submission):   # provenance: which copies were imported
        print(a.shard, mod.__name__, mod.__file__, hashlib.sha256(Path(mod.__file__).read_bytes()).hexdigest()[:16], flush=True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    device = torch.device(a.device)
    a.out.mkdir(parents=True, exist_ok=True)
    paths = {r['ID']: r['path'] for r in csv.DictReader(a.manifest.open())}
    ids = a.ids.read_text().split()
    assert len(ids) == len(set(ids)), 'duplicate ids'
    ids = ids[a.shard::a.shards]
    base = baseline_module(a.bb_assets)
    pin_torchaudio_loader(base)
    eat = load_eat_family(a.assets / 'eat', device)
    processor = AutoFeatureExtractor.from_pretrained(a.assets / 'mert', local_files_only=True)
    mert, info = AutoModel.from_pretrained(a.assets / 'mert', trust_remote_code=True, local_files_only=True, output_loading_info=True)
    assert not any(info.values()), info
    mert.to(device).eval().requires_grad_(False)
    done, failed, started = 0, [], time.time()
    for i, row_id in enumerate(ids):
        target = a.out / f'{row_id}.npz'
        if target.exists():
            continue
        if time.time() > a.deadline:
            print(f'{a.shard}: deadline reached at {i}/{len(ids)}', flush=True)
            break
        raw = base.load_audio(a.audio / paths[row_id])
        peak = float(np.abs(raw).max())
        if peak > .6:
            raw = (raw * (.6 / peak)).astype(np.float32)
        starts = window_layout(len(raw))
        wm = np.asarray(windows(np.asarray(raw, np.float32), starts), np.float32)
        partial = target.with_name(f'{row_id}.{os.getpid()}.tmp.npz')
        np.savez(partial, mixeat=eat_features(eat, wm, device), mixmert=mert_features(mert, processor, wm, device),
                 eat_layers=np.array(EAT_LAYERS), mert_layers=np.array(MERT_LAYERS), samples=np.array(len(raw)))
        partial.replace(target)
        done += 1
        if done % 500 == 0 or i == len(ids) - 1:
            print(f'{a.shard}/{a.shards} {i + 1}/{len(ids)} {done / (time.time() - started):.2f} files/s', flush=True)
    print(f'{a.shard}: finished {done} new files', flush=True)


if __name__ == '__main__':
    main()
