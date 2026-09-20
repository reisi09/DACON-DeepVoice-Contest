"""Shared, file-independent model evidence; baseline preprocessing is retained."""

import gc
import hashlib
import importlib.util
import importlib.metadata
import json
import time
from pathlib import Path

import numpy as np
import torch

from graph_fusion import Feature


def atomic_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, allow_nan=False), encoding='utf-8')
    temporary.replace(path)


def digest(path: Path) -> str:
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


class Extractor:
    """Own the three frozen models and accumulate timing for one experiment."""

    def __init__(self, baseline_script: Path, include_raw: bool = True):
        spec = importlib.util.spec_from_file_location('official_baseline', baseline_script)
        if spec is None or spec.loader is None:
            raise ImportError(f'Cannot import baseline at {baseline_script}')
        self.base = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.base)
        self.device = self.base.select_device('cuda')
        self.include_raw = include_raw
        torch.manual_seed(20260913)
        np.random.seed(20260913)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        self.panns, self.vi, self.mi = self.base.load_panns_model(self.device)
        self.df, self.fake_index = self.base.load_df_arena_model(self.device)
        self.demucs = self.base.load_htdemucs_model().to(self.device)
        self.signature = hashlib.sha256((
            digest(Path(__file__)) + digest(baseline_script)
            + (self.base.MODEL_DIR / 'SHA256SUMS.txt').read_text()
            + '|'.join(importlib.metadata.version(k) for k in
                ('torch', 'torchaudio', 'transformers', 'librosa', 'demucs', 'panns-inference'))
            + f'fp32-serial-shifts0-overlap0.25-raw{include_raw}'
        ).encode()).hexdigest()

    def fake_windows(self, audio: np.ndarray) -> list[float]:
        starts = self.base.get_segment_starts(audio.size)
        if self.base.calculate_rms(audio) < self.base.SILENCE_RMS:
            return [0.0] * len(starts)
        result = []
        for start in starts:
            segment = torch.from_numpy(self.base.extract_segment(audio, start)).to(self.device)
            with torch.inference_mode():
                logits = self.df(input_values=segment)['logits']
                probability = torch.softmax(logits.float(), dim=-1)[0, self.fake_index]
            result.append(float(probability))
        return result

    def extract(self, audio_path: Path) -> Feature:
        """No other file or accumulated prediction enters this computation."""
        audio = self.base.load_audio(audio_path)
        predictions, _ = self.panns.inference(self.base.make_panns_segments(audio))
        presence = np.stack((predictions[:, self.vi].max(1), predictions[:, self.mi].max(1)), 1)
        voice, music = self.base.separate_voice_and_music(audio_path, self.demucs, self.device)
        feature = Feature(
            ID=audio_path.stem, starts=self.base.get_segment_starts(audio.size),
            presence=presence.astype(float).tolist(),
            voice_fake=self.fake_windows(voice), music_fake=self.fake_windows(music),
            raw_fake=self.fake_windows(audio) if self.include_raw else [0.0] * len(presence),
            stem_rms=[self.base.calculate_rms(voice), self.base.calculate_rms(music)],
            duration=float(audio.size / self.base.AUDIO_SAMPLE_RATE),
        )
        lengths = {len(feature[name]) for name in ('starts', 'presence', 'voice_fake', 'music_fake', 'raw_fake')}
        if len(lengths) != 1:
            raise ValueError(f'Window alignment mismatch: {audio_path}, {lengths}')
        return feature

    def close(self) -> None:
        del self.panns, self.df, self.demucs
        gc.collect()
        torch.cuda.empty_cache()


def extract_cached(extractor: Extractor, paths: list[Path], cache_dir: Path, deadline: float) -> list[Feature]:
    """Only complete atomic records can resume; content and code changes invalidate keys."""
    features = []
    for path in paths:
        key = hashlib.sha256((digest(path) + extractor.signature).encode()).hexdigest()
        cache_path = cache_dir / f'{key}.json'
        if cache_path.is_file():
            record = json.loads(cache_path.read_text())
            if record['signature'] != extractor.signature or record['input_sha256'] != digest(path):
                raise ValueError(f'Cache provenance mismatch: {cache_path}')
            feature = record['feature']
        else:
            if time.time() >= deadline:
                break
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
            start = time.perf_counter()
            feature = extractor.extract(path)
            torch.cuda.synchronize()
            atomic_json(cache_path, {
                'feature': feature, 'input_sha256': digest(path), 'signature': extractor.signature,
                'seconds': time.perf_counter() - start,
                'max_allocated_gib': torch.cuda.max_memory_allocated() / 1024**3,
                'max_reserved_gib': torch.cuda.max_memory_reserved() / 1024**3,
            })
        features.append({**feature, 'ID': path.stem})
        print(f'FEATURE {len(features)}/{len(paths)} {path.stem}', flush=True)
    return features
