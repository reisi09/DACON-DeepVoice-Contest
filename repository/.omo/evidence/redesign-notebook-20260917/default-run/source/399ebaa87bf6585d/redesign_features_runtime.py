"""Frozen GPU encoder runtime for aligned redesign feature windows."""

from __future__ import annotations

import importlib.util
import os
import sys
import time
from pathlib import Path
from typing import Final

import numpy as np

from redesign_features_cache import FeatureRecord

WINDOW: Final = 64_600
SILENCE_LOG_FLOOR: Final = 1e-12


class FeatureRuntimeError(ValueError):
    """The local encoder runtime cannot preserve the extraction contract."""


def _load_baseline(path: Path):
    spec = importlib.util.spec_from_file_location("redesign_official_baseline", path)
    if spec is None or spec.loader is None:
        raise FeatureRuntimeError("Cannot load pinned baseline source")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class FrozenEncoders:
    """Own the one shared set of frozen separator, MERT, DF-Arena, and PANNs models."""

    def __init__(self, assets: Path, device: str, deadline: float | None = None) -> None:
        import torch
        from transformers import AutoFeatureExtractor, AutoModel

        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        if device == "cuda" and not torch.cuda.is_available():
            raise FeatureRuntimeError("CUDA was requested but is unavailable; CPU fallback is disabled")
        self.torch = torch
        self.device = torch.device(device)
        self.deadline = deadline
        self.base = _load_baseline(assets / "baseline/script.py")
        self.demucs = self.base.load_htdemucs_model().eval().requires_grad_(False)
        self.df, self.fake_index = self.base.load_df_arena_model(self.device)
        self.df.eval().requires_grad_(False)
        self.panns, self.voice_indices, self.music_indices = self.base.load_panns_model(self.device)
        self.processor = AutoFeatureExtractor.from_pretrained(assets / "mert", local_files_only=True)
        self.mert, loading = AutoModel.from_pretrained(
            assets / "mert", trust_remote_code=True, local_files_only=True, output_loading_info=True
        )
        if any(loading.values()) or int(self.mert.config.hidden_size) != 1024:
            raise FeatureRuntimeError(f"MERT checkpoint did not load exactly: {loading}")
        self.mert.to(self.device).eval().requires_grad_(False)
        self.parity_checked = False

    def _check_deadline(self) -> None:
        if self.deadline is not None and time.time() >= self.deadline:
            raise TimeoutError("Feature extraction deadline reached inside encoder work")

    def _mert_windows(self, waveform: np.ndarray, starts: np.ndarray) -> np.ndarray:
        import librosa

        vectors: list[np.ndarray] = []
        for offset in range(0, len(starts), 16):
            self._check_deadline()
            pieces = [self.base.extract_segment(waveform, int(start)) for start in starts[offset:offset + 16]]
            resampled = [
                librosa.resample(piece, orig_sr=16_000, target_sr=24_000, res_type="soxr_hq")
                for piece in pieces
            ]
            inputs = self.processor(resampled, sampling_rate=24_000, return_tensors="pt")
            hidden = self.mert(inputs.input_values.to(self.device)).last_hidden_state.mean(1)
            vectors.extend(hidden.float().cpu().numpy())
        return np.asarray(vectors, dtype=np.float32)

    def _voice_windows(self, vocal: np.ndarray, starts: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        torch = self.torch
        embeddings: list[np.ndarray] = []
        logits: list[float] = []
        probabilities: list[float] = []
        if self.base.calculate_rms(vocal) < self.base.SILENCE_RMS:
            return (
                np.zeros((len(starts), 1280), dtype=np.float32),
                np.full(len(starts), np.log(1e-12 / (1 - 1e-12)), dtype=np.float32),
                np.zeros(len(starts), dtype=np.float32),
            )
        classifier = self.df.get_submodule("backbone.conformer.fc5")
        captured = []

        def capture(_module, arguments) -> None:
            captured.append(arguments[0].detach())

        handle = classifier.register_forward_pre_hook(capture)
        try:
            for start in starts:
                self._check_deadline()
                segment = self.base.extract_segment(vocal, int(start))
                native = self.df(input_values=torch.from_numpy(segment).to(self.device))["logits"].float()[0]
                if len(captured) != 1 or captured[0].shape != (1, 1280):
                    raise FeatureRuntimeError("DF-Arena classifier hook did not expose one 1280-d input")
                fake = native[self.fake_index]
                real = native[1 - self.fake_index]
                delta = fake - real
                probability = torch.softmax(native, dim=-1)[self.fake_index]
                embeddings.append(captured.pop().cpu().numpy()[0].astype(np.float32))
                logits.append(float(delta.cpu()))
                probabilities.append(float(probability.cpu()))
        finally:
            handle.remove()
        return np.asarray(embeddings), np.asarray(logits, dtype=np.float32), np.asarray(probabilities, dtype=np.float32)

    def extract(self, path: Path) -> FeatureRecord:
        torch = self.torch
        base = self.base
        self._check_deadline()
        audio = base.load_audio(path)
        with torch.inference_mode():
            vocal, stem = base.separate_voice_and_music(path, self.demucs, self.device)
            vocal = np.pad(vocal[: len(audio)], (0, max(0, len(audio) - len(vocal))))
            stem = np.pad(stem[: len(audio)], (0, max(0, len(audio) - len(stem))))
            starts = np.asarray(base.get_segment_starts(len(audio)), dtype=np.int64)
            raw_segments = [base.extract_segment(audio, int(start)) for start in starts]
            stem_segments = [base.extract_segment(stem, int(start)) for start in starts]
            vocal_segments = [base.extract_segment(vocal, int(start)) for start in starts]
            panns_input = base.make_panns_segments(audio)
            panns_scores, _ = self.panns.inference(panns_input)
            presence = np.stack(
                (panns_scores[:, self.voice_indices].max(1), panns_scores[:, self.music_indices].max(1)), axis=1
            ).astype(np.float32)
            voice_embedding, voice_logits, voice_probability = self._voice_windows(vocal, starts)
            stem_features = self._mert_windows(stem, starts)
            raw_features = self._mert_windows(audio, starts)
        duration = np.clip((len(audio) - starts) / WINDOW, 0, 1).astype(np.float32)
        raw_rms = np.asarray([base.calculate_rms(piece) for piece in raw_segments])
        stem_rms = np.asarray([base.calculate_rms(piece) for piece in stem_segments])
        vocal_rms = np.asarray([base.calculate_rms(piece) for piece in vocal_segments])
        raw_audible = raw_rms >= base.SILENCE_RMS
        stem_audible = stem_rms >= base.SILENCE_RMS
        aux = np.column_stack((
            np.log(np.maximum(raw_rms, SILENCE_LOG_FLOOR)),
            np.log(np.maximum(stem_rms, SILENCE_LOG_FLOOR)),
            np.log(np.maximum(vocal_rms, SILENCE_LOG_FLOOR)),
            raw_audible.astype(np.float32), stem_audible.astype(np.float32), duration,
        )).astype(np.float32)
        evidence = np.asarray(
            [voice_probability.max(), presence[:, 0].max(), presence[:, 1].max()], dtype=np.float32
        )
        if not self.parity_checked:
            baseline_presence = base.predict_presence(self.panns, self.voice_indices, self.music_indices, audio)
            baseline_voice = base.predict_fake(self.df, self.fake_index, vocal, self.device)
            if not np.allclose(evidence[1:], baseline_presence, atol=1e-6, rtol=0):
                raise FeatureRuntimeError("PANNs window evidence diverges from baseline bag prediction")
            if not np.isclose(evidence[0], baseline_voice, atol=1e-6, rtol=0):
                raise FeatureRuntimeError("DF-Arena window evidence diverges from baseline bag prediction")
            self.parity_checked = True
        return FeatureRecord(
            stem=stem_features, raw=raw_features, aux=aux, starts=starts,
            valid=duration > 0, stem_audible=stem_audible,
            voice_embedding=voice_embedding, voice_logits=voice_logits,
            voice_probability=voice_probability, presence=presence, evidence=evidence,
        )
