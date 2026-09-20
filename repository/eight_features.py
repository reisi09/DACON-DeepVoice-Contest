"""Shared H100 feature extraction for the eight controlled experiments."""

from __future__ import annotations

import csv
import gc
import hashlib
import json
import os
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Final

import numpy as np

from eight_feature_schema import ALL_FEATURE_KEYS, feature_profiles, required_keys, validate_feature_arrays
from eight_feature_identity import (
    feature_identity as _identity,
    file_sha256 as _sha256,
    resolve_parent_hashes as _resolve_parent_hashes,
)
from eight_spectral import spectrum_features

WINDOW: Final = 64_600
HOP: Final = 32_300
SILENCE_RMS: Final = 1e-5
LAYERS: Final = (6, 12, 24)
ANTI_LAYERS: Final = (12, 24, 48)


class FeatureError(RuntimeError):
    """Feature extraction cannot preserve its declared identity or alignment."""


def window_layout(samples: int) -> tuple[np.ndarray, np.ndarray]:
    """Cover audio with a half-window hop and one deduplicated exact tail."""
    if samples <= 0:
        raise FeatureError("Decoded audio is empty")
    last = max(0, samples - WINDOW)
    starts = list(range(0, last + 1, HOP))
    if starts[-1] != last:
        starts.append(last)
    offsets = np.asarray(starts, dtype=np.int64)
    valid = np.minimum(WINDOW, samples - offsets).astype(np.int64)
    return offsets, valid


def _windows(audio: np.ndarray, starts: np.ndarray) -> np.ndarray:
    result = np.zeros((len(starts), WINDOW), dtype=np.float32)
    for index, start in enumerate(starts):
        piece = audio[int(start) : int(start) + WINDOW]
        result[index, : len(piece)] = piece
    return result


def _atomic_json(path: Path, value: dict[str, str | int | float]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


class _Engine:
    """Own frozen models and expose aligned, inference-only window batches."""

    def __init__(self, assets: Path, anti_path: Path, device: str, deadline: float, batch_size: int) -> None:
        import torch
        from transformers import AutoFeatureExtractor, AutoModel
        from antideepfake_runtime import AntiDeepfake, verify_native
        from redesign_features_runtime import _load_baseline

        self.torch = torch
        self.device = torch.device(device)
        self.deadline = deadline
        self.batch_size = batch_size
        self.base = _load_baseline(assets / "baseline/script.py")
        parity = verify_native(anti_path, device)
        if parity["real_audio_cases"] < 1:
            raise FeatureError("AntiDeepfake native parity has no real-audio case")
        self.separator = self.base.load_htdemucs_model().eval().requires_grad_(False)
        self.df, self.df_fake = self.base.load_df_arena_model(self.device)
        self.df.eval().requires_grad_(False)
        self.panns, self.voice_indices, self.music_indices = self.base.load_panns_model(self.device)
        if hasattr(self.panns, "model"):
            self.panns.model.eval().requires_grad_(False)
        self.processor = AutoFeatureExtractor.from_pretrained(assets / "mert", local_files_only=True)
        self.mert = AutoModel.from_pretrained(assets / "mert", trust_remote_code=True, local_files_only=True)
        self.mert.to(self.device).eval().requires_grad_(False)
        self.anti = AntiDeepfake(anti_path, device)
        self.parity = parity

    def _batches(self, windows: np.ndarray):
        for offset in range(0, len(windows), self.batch_size):
            if time.time() >= self.deadline:
                raise TimeoutError("Feature extraction deadline reached before a model batch")
            yield windows[offset : offset + self.batch_size]

    def _mert(self, windows: np.ndarray, layers: tuple[int, ...]) -> dict[int, np.ndarray]:
        import librosa

        collected = {layer: [] for layer in layers}
        with self.torch.inference_mode():
            for batch in self._batches(windows):
                audio = [librosa.resample(item, orig_sr=16_000, target_sr=24_000, res_type="soxr_hq") for item in batch]
                inputs = self.processor(audio, sampling_rate=24_000, return_tensors="pt", padding=True)
                output = self.mert(inputs.input_values.to(self.device), output_hidden_states=True)
                for layer in layers:
                    collected[layer].append(output.hidden_states[layer].mean(1).float().cpu().numpy())
        return {layer: np.concatenate(collected[layer]).astype(np.float32) for layer in layers}

    def _anti(self, windows: np.ndarray, keep_layers: bool) -> tuple[np.ndarray, dict[int, np.ndarray]]:
        probabilities: list[np.ndarray] = []
        collected = {layer: [] for layer in ANTI_LAYERS}
        with self.torch.inference_mode():
            for batch in self._batches(windows):
                audio = self.torch.from_numpy(batch).to(self.device)
                normalized = self.torch.nn.functional.layer_norm(audio, audio.shape[-1:])
                output = self.anti.encoder(normalized, output_hidden_states=keep_layers)
                logits = self.anti.classifier(output.last_hidden_state.mean(1))
                probabilities.append(logits.float().softmax(-1)[:, 0].cpu().numpy())
                if keep_layers:
                    for layer in ANTI_LAYERS:
                        collected[layer].append(output.hidden_states[layer].mean(1).float().cpu().numpy())
        layers = {layer: np.concatenate(parts).astype(np.float32) for layer, parts in collected.items() if parts}
        return np.concatenate(probabilities).astype(np.float32), layers

    def _df(self, windows: np.ndarray) -> np.ndarray:
        probabilities: list[float] = []
        with self.torch.inference_mode():
            for batch in self._batches(windows):
                for window in batch:
                    logits = self.df(input_values=self.torch.from_numpy(window).to(self.device))["logits"].float()[0]
                    probabilities.append(float(logits.softmax(-1)[self.df_fake].cpu()))
        return np.asarray(probabilities, dtype=np.float32)

    def _presence(self, windows: np.ndarray) -> np.ndarray:
        import librosa

        audio = np.stack([librosa.resample(item, orig_sr=16_000, target_sr=32_000, res_type="soxr_hq") for item in windows])
        with self.torch.inference_mode():
            scores, _ = self.panns.inference(audio.astype(np.float32))
        return np.column_stack((scores[:, self.voice_indices].max(1), scores[:, self.music_indices].max(1))).astype(np.float32)

    def extract(self, path: Path, profile: str) -> dict[str, np.ndarray]:
        if time.time() >= self.deadline:
            raise TimeoutError("Feature extraction deadline reached before an audio record")
        raw = self.base.load_audio(path)
        required = required_keys(profile)
        need_vocals = any(key.endswith("_vocals") for key in required)
        need_stem = any(key.startswith("mert_") or key in {"stem_mask", "spectrum_stem"} for key in required)
        vocals = stem = None
        with self.torch.inference_mode():
            if need_vocals or need_stem:
                vocals, stem = self.base.separate_voice_and_music(path, self.separator, self.device)
        if vocals is not None:
            vocals = np.pad(vocals[: len(raw)], (0, max(0, len(raw) - len(vocals))))
        if stem is not None:
            stem = np.pad(stem[: len(raw)], (0, max(0, len(raw) - len(stem))))
        starts, valid = window_layout(len(raw))
        raw_windows = _windows(raw, starts)
        vocal_windows = _windows(vocals, starts) if need_vocals and vocals is not None else None
        stem_windows = _windows(stem, starts) if need_stem and stem is not None else None
        raw_rms = np.sqrt(np.mean(raw_windows.astype(np.float64) ** 2, axis=1))
        result = {
            "starts": starts, "valid_samples": valid, "padding_samples": WINDOW - valid,
            "raw_mask": (raw_rms >= SILENCE_RMS).astype(np.uint8),
        }
        if "stem_mask" in required and stem_windows is not None:
            stem_rms = np.sqrt(np.mean(stem_windows.astype(np.float64) ** 2, axis=1))
            result["stem_mask"] = (stem_rms >= SILENCE_RMS).astype(np.uint8)
        anti_layers = tuple(layer for layer in ANTI_LAYERS if f"anti_l{layer}" in required)
        if "anti_raw" in required or anti_layers:
            anti_raw, layers = self._anti(raw_windows, bool(anti_layers))
            if "anti_raw" in required:
                result["anti_raw"] = anti_raw
            result.update({f"anti_l{layer}": layers[layer] for layer in anti_layers})
        if "anti_vocals" in required and vocal_windows is not None:
            result["anti_vocals"], _ = self._anti(vocal_windows, False)
        if "df_raw" in required:
            result["df_raw"] = self._df(raw_windows)
        if "df_vocals" in required and vocal_windows is not None:
            result["df_vocals"] = self._df(vocal_windows)
        if "presence" in required:
            result["presence"] = self._presence(raw_windows)
        mert_layers = tuple(layer for layer in LAYERS if f"mert_l{layer}" in required)
        if mert_layers and stem_windows is not None:
            result.update({f"mert_l{layer}": value for layer, value in self._mert(stem_windows, mert_layers).items()})
        if "spectrum_stem" in required and stem_windows is not None:
            result["spectrum_stem"] = spectrum_features(stem_windows)
        validate_feature_arrays(result, profile)
        return result


def _load_rows(root: Path) -> list[dict[str, str]]:
    manifest = root / "source/dataset/manifest.csv"
    try:
        from eight_contract import read_manifest
        return [dict(row) for row in read_manifest(manifest)]
    except ImportError:
        with manifest.open(encoding="utf-8", newline="") as stream:
            return list(csv.DictReader(stream))


def _validate_cache(path: Path, identity: Mapping, profile: str) -> int:
    """Reject stale profiles and validate exactly the arrays declared by this row role."""
    with np.load(path, allow_pickle=False) as saved:
        if json.loads(str(saved["identity_json"])) != identity:
            raise FeatureError(f"Stale feature cache identity: {identity['ID']}")
        arrays = {name: saved[name] for name in saved.files if name != "identity_json"}
        if set(arrays).difference(ALL_FEATURE_KEYS):
            raise FeatureError(f"Unknown arrays in feature cache: {identity['ID']}")
        return validate_feature_arrays(arrays, profile)


def extract(
    config_path: str | Path, row_ids: Sequence[str] | None = None,
) -> dict[str, str | int | float | None]:
    """Extract or identity-validate every manifest record into output/features."""
    from redesign_assets import validate_assets

    config = json.loads(Path(config_path).read_text(encoding="utf-8"))
    root, assets, anti_path = Path(config["root"]), Path(config["assets"]), Path(config["antideepfake"])
    deadline = float(config["pause_deadline_epoch"]) - float(config.get("export_reserve_seconds", 600))
    threads, batch_size = int(config.get("cpu_threads", 4)), int(config.get("feature_batch_size", 8))
    os.environ.update(OMP_NUM_THREADS=str(threads), MKL_NUM_THREADS=str(threads), TOKENIZERS_PARALLELISM="false")
    asset_hashes = validate_assets(assets)
    anti_hashes = {name: _sha256(anti_path / name) for name in ("model.safetensors", "antideepfake_config.json", "native_reference.json", "parity_audio.npz")}
    all_rows, output = _load_rows(root), root / "output/features"
    parent_hashes = _resolve_parent_hashes(root, all_rows)
    profiles = feature_profiles(all_rows)
    requested = set(row_ids) if row_ids is not None else None
    rows = [row for row in all_rows if requested is None or row["ID"] in requested]
    if requested is not None and {row["ID"] for row in rows} != requested:
        raise FeatureError("Requested feature preflight IDs are absent from the manifest")
    output.mkdir(parents=True, exist_ok=True)
    requested_windows = sum(len(window_layout(max(1, round(float(row["duration_seconds"]) * 16_000)))[0]) for row in rows)
    device = str(config.get("device", "cuda"))
    import torch
    if device == "cuda" and not torch.cuda.is_available():
        raise FeatureError("CUDA extraction requested but unavailable")
    torch.set_num_threads(threads)
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    engine, extracted, cached = None, 0, 0
    processed_windows = extracted_windows = 0
    engine_ready_at: float | None = None
    profile_inference = {profile: {"records": 0, "windows": 0, "seconds": 0.0}
                         for profile in sorted(set(profiles[row["ID"]] for row in rows))}
    began = time.time()
    for index, row in enumerate(rows):
        source = root / "source/dataset" / row["path"]
        if _sha256(source) != row["audio_sha256"]:
            raise FeatureError(f"Manifest audio identity changed: {row['ID']}")
        profile = profiles[row["ID"]]
        identity = _identity(row, asset_hashes, anti_hashes, parent_hashes[row["ID"]], profile, required_keys(profile))
        target = output / f"{row['ID']}.npz"
        if target.exists():
            count = _validate_cache(target, identity, profile)
            cached += 1
        else:
            if engine is None:
                engine = _Engine(assets, anti_path, device, deadline, batch_size)
                engine_ready_at = time.time()
            inference_began = time.time()
            values = engine.extract(source, profile)
            inference_seconds = time.time() - inference_began
            count = len(values["starts"])
            temporary = target.with_suffix(".partial")
            with temporary.open("wb") as stream:
                np.savez_compressed(stream, **values, identity_json=np.asarray(json.dumps(identity, sort_keys=True)))
            temporary.replace(target)
            extracted += 1
            extracted_windows += count
            profile_inference[profile]["records"] += 1
            profile_inference[profile]["windows"] += count
            profile_inference[profile]["seconds"] += inference_seconds
        processed_windows += count
        progress = {"stage": "features", "completed": index + 1, "total": len(rows),
                    "processed_windows": processed_windows, "requested_windows": requested_windows,
                    "extracted_windows": extracted_windows,
                    "profile_inference": profile_inference,
                    "elapsed_after_engine_init_seconds": None if engine_ready_at is None else time.time() - engine_ready_at,
                    "current_id": row["ID"], "time": time.time()}
        _atomic_json(root / "output/progress.json", progress)
        _atomic_json(Path(config["shared"]) / "progress.json", progress)
    peak_gpu = int(torch.cuda.max_memory_allocated()) if device == "cuda" else 0
    parity_error = None if engine is None else float(engine.parity["max_logit_error"])
    report = {"status": "complete", "records": len(rows), "extracted": extracted, "cached": cached,
              "processed_windows": processed_windows, "extracted_windows": extracted_windows,
              "requested_windows": requested_windows,
              "profile_inference": profile_inference,
              "elapsed_after_engine_init_seconds": None if engine_ready_at is None else time.time() - engine_ready_at,
              "elapsed_seconds": time.time() - began, "peak_gpu_bytes": peak_gpu, "anti_native_max_logit_error": parity_error}
    _atomic_json(root / "output/extraction.json", report)
    del engine
    gc.collect()
    if device == "cuda":
        torch.cuda.empty_cache()
    return report


def preflight(config_path: str | Path) -> Mapping:
    """Extract and verify one real record for every role-specific cache contract."""
    config = json.loads(Path(config_path).read_text(encoding="utf-8"))
    root, rows = Path(config["root"]), _load_rows(Path(config["root"]))
    profiles = feature_profiles(rows)
    selected = []
    for profile in (
        "core_music_base", "core_music_processed", "core_speech_base",
        "core_speech_processed", "swap", "partial",
    ):
        candidates = [row for row in rows if profiles[row["ID"]] == profile]
        if profile == "partial":
            candidates = [row for row in candidates if float(row["duration_seconds"]) >= 60]
        if not candidates:
            raise FeatureError(f"Feature preflight lacks profile: {profile}")
        selected.append(candidates[0])
    extraction = extract(config_path, [row["ID"] for row in selected])
    checked: list[str] = []
    receipts: dict[str, str] = {}
    for row in selected:
        path = root / "output/features" / f"{row['ID']}.npz"
        with np.load(path, allow_pickle=False) as saved:
            identity = json.loads(str(saved["identity_json"]))
            if identity["audio_sha256"] != row["audio_sha256"]:
                raise FeatureError(f"Preflight identity mismatch: {row['ID']}")
            profile = profiles[row["ID"]]
            arrays = {name: saved[name] for name in saved.files if name != "identity_json"}
            validate_feature_arrays(arrays, profile)
            receipts[row["ID"]] = hashlib.sha256(str(saved["identity_json"]).encode()).hexdigest()
        checked.append(row["ID"])
    report = {"status": "complete", "records_checked": len(checked), "IDs": checked,
              "feature_profiles": {row["ID"]: profiles[row["ID"]] for row in selected},
              "required_modalities": {row["ID"]: sorted(required_keys(profiles[row["ID"]])) for row in selected},
              "identity_receipts": receipts, "extraction": extraction}
    _atomic_json(root / "output/feature_preflight.json", report)
    return report
