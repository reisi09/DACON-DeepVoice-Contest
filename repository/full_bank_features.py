"""Resume-safe, full-duration MERT and original DF-Arena classifier features.

Run with the pinned project environment; no network or wall-clock cutoff.
"""
from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
import os
from pathlib import Path
import time
import tempfile
from collections.abc import Mapping

import numpy as np
import torch

from eight_fixed_io import atomic_json
from mert12_submission import WINDOW, HOP, pin_torchaudio_loader, window_layout, windows
from localization_features import select_batch
from v6_data import sha256
from v6_runtime import baseline_module
import decision_heads
from full_bank_contract import role


def row_identity(row: Mapping[str, str], identity: str) -> str:
    """Bind audio bytes, truth, split, and feature implementation together."""
    return hashlib.sha256((identity + json.dumps(dict(row), sort_keys=True)).encode()).hexdigest()


def window_truth(row: Mapping[str, str], starts: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """Use explicit known intervals; weak file labels never invent local labels."""
    events = [x for x in json.loads(row.get('event_intervals') or '[]') if x.get('component') == 'music']
    result = np.full(len(starts), np.nan, np.float32)
    for i, (start, count) in enumerate(zip(starts, valid, strict=True)):
        real_bed = row.get('variant', '').startswith(('partial_', 'sham'))
        truth = np.zeros(int(count), np.float32) if real_bed else np.full(int(count), np.nan, np.float32)
        for event in events:
            if event.get('label') not in (0, 1, '0', '1'):
                continue
            left = max(0, round(float(event['start']) * 16000) - int(start))
            right = min(int(count), round(float(event['end']) * 16000) - int(start))
            if right > left:
                truth[left:right] = float(event['label'])
        if np.isfinite(truth).all():
            result[i] = float(truth.mean())
    return result


def voice_window_truth(row, starts, samples, segment_samples):
    """Match DF-Arena's repeated short segments; uncovered samples stay unknown."""
    labels = np.full(samples, np.nan, np.float32)
    for event in json.loads(row.get('event_intervals') or '[]'):
        if event.get('component') != 'voice':
            continue
        left, right = round(float(event['start']) * 16000), round(float(event['end']) * 16000)
        if not 0 <= left < right <= samples or event['label'] not in (0, 1):
            raise ValueError('Invalid voice event interval')
        if np.isfinite(labels[left:right]).any():
            raise ValueError('Overlapping voice event intervals')
        labels[left:right] = float(event['label'])
    output = np.full(len(starts), np.nan, np.float32)
    for i, start in enumerate(starts):
        segment = labels[int(start):int(start) + segment_samples]
        if len(segment) and np.isfinite(segment).all():
            output[i] = np.resize(segment, segment_samples).mean()
    return output


def load_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline='') as stream:
        rows = list(csv.DictReader(stream))
    if not rows or len({r['ID'] for r in rows}) != len(rows):
        raise ValueError('Manifest must contain unique nonempty IDs')
    for row in rows:
        if row['split'] not in ('train', 'val', 'test') or row['route'] not in ('music', 'voice'):
            raise ValueError(f"Invalid split/route: {row['ID']}")
        label = row['MUSIC_FAKE' if row['route'] == 'music' else 'VOICE_FAKE']
        if label not in ('0', '1', '0.0', '1.0'):
            raise ValueError(f"Missing supervised route label: {row['ID']}")
        if Path(row['ID']).name != row['ID']:
            raise ValueError('Unsafe cache identifier')
    assignments = {}
    for row in rows:
        assigned = role(row)
        if assigned not in ('train', 'selection', 'calibration', 'development', 'sealed'):
            raise ValueError('Unknown data role')
        expected_split = 'train' if assigned == 'train' else 'val' if assigned in ('selection', 'calibration') else 'test'
        if row['split'] != expected_split:
            raise ValueError('Data role conflicts with legacy split')
        assignments.setdefault(row['component_id'], set()).add(assigned)
    if any(len(v) > 1 for v in assignments.values()):
        raise ValueError('Source component crosses roles')
    train = {r['component_id'] for r in rows if r['split'] == 'train'}
    val = {r['component_id'] for r in rows if r['split'] == 'val'}
    if train & val:
        raise ValueError('Source component leakage between train and validation')
    return rows


def validate_cache(path: Path, expected: str, digest: str) -> None:
    """A completed cache must retain its exact bytes and numerical invariants."""
    if sha256(path) != digest:
        raise ValueError(f'Cache hash changed: {path}')
    with np.load(path, allow_pickle=False) as data:
        if str(data['identity'].item()) != expected:
            raise ValueError(f'Cache identity changed: {path}')
        for key in ('voice_embeddings', 'presence', 'starts', 'valid_samples'):
            if not np.isfinite(data[key]).all():
                raise ValueError(f'Nonfinite cache {path}: {key}')
        mert = data['mert_windows']
        if len(mert) and (len(mert) != len(data['starts']) or not np.isfinite(mert).all()):
            raise ValueError(f'Invalid MERT windows: {path}')
        samples = int(data['samples'])
        if not np.array_equal(data['starts'], window_layout(samples)) or not np.array_equal(data['valid_samples'], np.minimum(WINDOW, samples-data['starts'])):
            raise ValueError(f'Full-duration window coverage changed: {path}')
        if data['voice_embeddings'].shape[1:] != (1280,) or mert.shape[1:] != (1024,):
            raise ValueError(f'Encoder features incompatible with trained architecture: {path}')
        truth = data['window_truth']
        if np.any(np.isfinite(truth) & ((truth < 0) | (truth > 1))):
            raise ValueError(f'Invalid local occupancy labels: {path}')
        if 'voice_window_truth' in data.files:
            segment = int(data['voice_segment_samples'])
            if segment != 64600:
                raise ValueError('DF-Arena segment length differs from the baseline contract')
            last = max(0, samples - segment)
            expected_starts = list(range(0, last + 1, segment))
            if expected_starts[-1] != last:
                expected_starts.append(last)
            if not np.array_equal(data['voice_starts'], expected_starts) or len(data['voice_embeddings']) != len(expected_starts):
                raise ValueError('Voice cache timing no longer matches encoder windows')
            local = data['voice_window_truth']
            if len(local) != len(expected_starts) or np.isinf(local).any() or np.any(np.isfinite(local) & ((local < 0) | (local > 1))):
                raise ValueError('Invalid voice local supervision')
        if not len(data['voice_embeddings']):
            raise ValueError(f'Empty voice features: {path}')


def prepare(args: argparse.Namespace) -> None:
    """Bound device batches while visiting every window of every original file."""
    os.environ.update(HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1')
    import librosa
    import soundfile as sf
    from transformers import AutoFeatureExtractor, AutoModel
    if not torch.cuda.is_available() or 'H100' not in torch.cuda.get_device_name(0):
        raise RuntimeError('This runner requires the authorized H100')
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
                                   ('full_bank_features.py', 'mert12_submission.py', 'v6_runtime.py', 'decision_heads.py', 'full_bank_contract.py')},
                'window': WINDOW, 'hop': HOP, 'mert_layer': 12, 'dtype': 'float32',
                'duration': 'all samples, no crop', 'torch': torch.__version__}
    identity_key = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    index_path = args.cache / 'index.json'
    index = json.loads(index_path.read_text()) if index_path.exists() else {'identity': identity, 'items': {}}
    if index['identity'] != identity:
        raise ValueError('Refuse to resume a cache with different data/models/implementation')
    base = baseline_module(args.assets)
    pin_torchaudio_loader(base)
    device = torch.device('cuda')
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
        if batch is None:
            sample = windows(backing, np.resize(starts, 256))
            batch = select_batch(lambda n: mert_batch(sample[:n]), (16, 32, 64, 128, 256),
                                 args.output / 'mert_batch_benchmark.json', float('inf'))
        pieces = []
        for offset in range(0, len(starts), batch):
            pieces.append(mert_batch(windows(backing, starts[offset:offset+batch])))
        vectors = np.concatenate(pieces)
        captured.clear()
        voice_scores, presence = [], np.zeros(2, np.float32)
        # Original DF-Arena forward accepts one waveform, not a native batched input.
        voice_starts = np.asarray(base.get_segment_starts(len(raw)), dtype=np.int64)
        for start in voice_starts:
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
                 voice_starts=voice_starts, voice_segment_samples=np.array(base.SEGMENT_SAMPLES),
                 voice_window_truth=voice_window_truth(row, voice_starts, len(raw), base.SEGMENT_SAMPLES),
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
    torch.cuda.empty_cache()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('manifest', 'input', 'cache', 'output', 'assets', 'asset-hashes'):
        parser.add_argument('--'+name, type=Path, required=True)
    prepare(parser.parse_args())
