"""Local (mps/cpu) copy of full_bank_features.prepare for data that must not leave this machine.

Generated mechanically from full_bank_features.prepare; only the device gate, device, MERT batch benchmark and
CUDA cache calls differ. Its own hash is part of the cache identity.
ponytail: duplicated extractor; fold a device parameter into full_bank_features after the deadline (changing that
file now would invalidate every existing cache identity).
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import time
import tempfile

import numpy as np
import torch

from eight_fixed_io import atomic_json
from full_bank_features import load_rows, row_identity, validate_cache, window_truth
from mert12_submission import WINDOW, HOP, pin_torchaudio_loader, window_layout, windows
from v6_data import sha256
from v6_runtime import baseline_module
import decision_heads


def prepare_local(args: argparse.Namespace) -> None:
    """Bound device batches while visiting every window of every original file."""
    os.environ.update(HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1')
    import librosa
    import soundfile as sf
    from transformers import AutoFeatureExtractor, AutoModel
    if args.device == 'cuda':
        raise RuntimeError('Use full_bank_features.py on the authorized H100; this copy is for local mps/cpu only')
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    args.cache.mkdir(parents=True, exist_ok=True)
    args.output.mkdir(parents=True, exist_ok=True)
    rows = load_rows(args.manifest)
    asset_hashes = json.loads(args.asset_hashes.read_text())
    for name, expected in asset_hashes.items():
        if sha256(args.assets / name) != expected:
            raise ValueError(f'Asset identity mismatch: {name}')
    identity = {'manifest_sha256': sha256(args.manifest), 'assets': asset_hashes,
                'implementation': {name: sha256(Path(__file__).parent / name) for name in
                                   ('full_bank_features.py', 'mert12_submission.py', 'v6_runtime.py', 'decision_heads.py', 'local_features.py')},
                'window': WINDOW, 'hop': HOP, 'mert_layer': 12, 'dtype': 'float32',
                'duration': 'all samples, no crop', 'torch': torch.__version__, 'device': args.device}
    identity_key = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    index_path = args.cache / 'index.json'
    index = json.loads(index_path.read_text()) if index_path.exists() else {'identity': identity, 'items': {}}
    if index['identity'] != identity:
        raise ValueError('Refuse to resume a cache with different data/models/implementation')
    base = baseline_module(args.assets)
    pin_torchaudio_loader(base)
    device = torch.device(args.device)
    separator = base.load_htdemucs_model().to(device).eval().requires_grad_(False)
    voice_model, fake_index = base.load_df_arena_model(device)
    voice_model.requires_grad_(False)
    classifier = voice_model.backbone.conformer.fc5
    initial_path = args.cache / 'voice_classifier_initial.pt'
    if not initial_path.exists():
        torch.save({'state_dict': {k: v.detach().cpu() for k, v in classifier.state_dict().items()},
                    'module': 'backbone.conformer.fc5', 'fake_label_index': fake_index}, initial_path)
    initial = torch.load(initial_path, map_location='cpu', weights_only=True)
    if initial['fake_label_index'] != fake_index or any(not torch.equal(v.cpu(), initial['state_dict'][k]) for k, v in classifier.state_dict().items()):
        raise ValueError('Cached voice classifier initialization changed')
    panns, voice_indices, music_indices = base.load_panns_model(device)
    from panns_inference import labels as panns_labels
    label_groups = json.loads((base.PANNS_DIR / 'component_labels.json').read_text())
    processor = AutoFeatureExtractor.from_pretrained(args.assets / 'mert', local_files_only=True)
    mert, loading = AutoModel.from_pretrained(args.assets / 'mert', trust_remote_code=True,
                                             local_files_only=True, output_loading_info=True)
    if any(loading.values()):
        raise RuntimeError(f'MERT loading mismatch: {loading}')
    mert.to(device).eval().requires_grad_(False)
    captured = []
    def capture(_module, inputs):
        captured.append(inputs[0].detach().float().cpu().numpy().copy())
    handle = classifier.register_forward_pre_hook(capture)
    def mert_batch(audio):
        resampled = [librosa.resample(x, orig_sr=16000, target_sr=24000, res_type='soxr_hq') for x in audio]
        values = processor(resampled, sampling_rate=24000, return_tensors='pt', padding=True).input_values.to(device)
        with torch.inference_mode():
            return mert(values, output_hidden_states=True).hidden_states[12].mean(1).float().cpu().numpy()
    batch_receipt = args.output / 'mert_batch_benchmark.json'
    batch = json.loads(batch_receipt.read_text())['chosen'] if batch_receipt.exists() else None
    began = time.time()
    for number, row in enumerate(rows, 1):
        key = row_identity(row, identity_key)
        target = args.cache / f"{row['ID']}.npz"
        entry = index['items'].get(row['ID'])
        if entry:
            validate_cache(target, key, entry['sha256'])
            continue
        path = (args.input / row['path']).resolve()
        if not path.is_relative_to(args.input.resolve()) or sha256(path) != row['audio_sha256']:
            raise ValueError(f"Input bytes changed or escaped input directory: {row['ID']}")
        raw = base.load_audio(path)
        peak = float(np.abs(raw).max())
        preserve_gain = row.get('input_kind', '').startswith(('legacy', 'constructed'))
        gain = 1. if preserve_gain or peak <= .6 else .6 / peak
        raw = (raw * gain).astype(np.float32)
        with tempfile.TemporaryDirectory(prefix='full_bank_audio_', dir=args.cache) as temporary_audio:
            normalized = Path(temporary_audio) / 'normalized.wav'
            sf.write(normalized, raw, 16000, subtype='FLOAT')
            with torch.inference_mode():
                voice, backing = base.separate_voice_and_music(normalized, separator, device)
        voice = np.pad(voice[:len(raw)], (0, max(0, len(raw)-len(voice))))
        backing = np.pad(backing[:len(raw)], (0, max(0, len(raw)-len(backing))))
        starts = window_layout(len(raw))
        valid = np.minimum(WINDOW, len(raw)-starts)
        vectors = np.empty((0, 1024), np.float32)
        if batch is None:  # no VRAM to benchmark off-GPU: fixed batch (windows are fixed-length, so batching does not change values)
            batch = args.mert_batch
        pieces = []
        for offset in range(0, len(starts), batch):
            pieces.append(mert_batch(windows(backing, starts[offset:offset+batch])))
        vectors = np.concatenate(pieces)
        captured.clear()
        voice_scores, presence = [], np.zeros(2, np.float32)
        # Original DF-Arena forward accepts one waveform, not a native batched input.
        for start in base.get_segment_starts(len(raw)):
            segment = base.extract_segment(voice, start)
            with torch.inference_mode():
                logits = voice_model(input_values=torch.from_numpy(segment).to(device))['logits']
            voice_scores.append(float(logits.float().softmax(-1)[0, fake_index]))
        # one batched PANNs pass over the same segments as the legacy per-segment loop; legacy presence = its max
        probs = panns.inference(base.make_panns_segments(raw))[0]
        presence = np.asarray(decision_heads.legacy_presence(probs, panns_labels, label_groups), np.float32)
        presence_feats = decision_heads.presence_features(probs, panns_labels, label_groups, raw, voice)
        voice_silent = base.calculate_rms(voice) < base.SILENCE_RMS
        temporary = target.with_suffix('.tmp.npz')
        np.savez(temporary, identity=np.array(key), mert_windows=vectors, starts=starts, valid_samples=valid,
                 window_truth=window_truth(row, starts, valid), voice_embeddings=np.concatenate(captured),
                 voice_original=np.array(0. if voice_silent else max(voice_scores)),
                 voice_silent=np.array(voice_silent), presence=np.asarray(presence, np.float32),
                 presence_features=np.array(json.dumps(presence_feats)),
                 samples=np.array(len(raw)), source_peak=np.array(peak), normalization_gain=np.array(gain),
                 normalized_rms=np.array(base.calculate_rms(raw)))
        temporary.replace(target)
        index['items'][row['ID']] = {'path': target.name, 'sha256': sha256(target), 'identity': key,
                                    'windows': len(starts), 'samples': len(raw)}
        atomic_json(index_path, index)
        atomic_json(args.output / 'progress.json', {'stage': 'features', 'done': len(index['items']),
                    'total': len(rows), 'last_id': row['ID'], 'elapsed_seconds': time.time()-began})
        print(f"features {number}/{len(rows)} {row['ID']} samples={len(raw)}", flush=True)
    handle.remove()
    if len(index['items']) != len(rows):
        raise RuntimeError('Feature coverage is incomplete')
    atomic_json(args.output / 'features_complete.json', {'rows': len(rows), 'index_sha256': sha256(index_path),
                'voice_classifier_sha256': sha256(args.cache / 'voice_classifier_initial.pt')})
    del mert, separator, voice_model, panns
    gc.collect()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('manifest', 'input', 'cache', 'output', 'assets', 'asset-hashes'):
        parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--device', default='mps', choices=('mps', 'cpu'))
    parser.add_argument('--mert-batch', type=int, default=16)
    prepare_local(parser.parse_args())
