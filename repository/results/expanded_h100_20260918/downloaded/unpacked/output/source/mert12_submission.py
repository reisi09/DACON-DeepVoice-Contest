"""Offline MERT12 full-refit inference with baseline voice and presence models."""
from __future__ import annotations

import argparse
import csv
import json
import os
import resource
import sys
import time
from collections.abc import Sequence
from pathlib import Path
from types import ModuleType
from typing import Final, TypedDict

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"

import librosa
import numpy as np
import torch
from transformers import AutoFeatureExtractor, AutoModel

from eight_heads import FrozenHead
from graph_scoring import PREDICTION_COLUMNS, SUBMISSION_COLUMNS, validate_predictions
from v6_data import sha256
from v6_runtime import baseline_module

WINDOW: Final = 64_600
HOP: Final = 32_300
SEEDS: Final = (41, 42, 43)
DEMUCS_INPUT_DECODER: Final = "torchaudio.load+demucs.audio.convert_audio"


class SubmissionError(ValueError):
    """The packaged runtime, checkpoint, or prediction violates its fixed contract."""


class HeadBundle(TypedDict):
    model: FrozenHead
    mean: np.ndarray
    scale: np.ndarray
    seed: int


def window_layout(samples: int) -> np.ndarray:
    """Return half-hop starts plus one deduplicated exact tail."""
    if samples <= 0:
        raise SubmissionError("Decoded audio is empty")
    last = max(0, samples - WINDOW)
    starts = list(range(0, last + 1, HOP))
    if starts[-1] != last:
        starts.append(last)
    return np.asarray(starts, dtype=np.int64)


def windows(audio: np.ndarray, starts: np.ndarray) -> np.ndarray:
    """Right-zero-pad each exact 16 kHz window."""
    result = np.zeros((len(starts), WINDOW), dtype=np.float32)
    for index, start in enumerate(starts):
        piece = audio[int(start): int(start) + WINDOW]
        result[index, :len(piece)] = piece
    return result


def load_heads(paths: Sequence[Path], device: str) -> list[HeadBundle]:
    """Reload exactly the three serialized full-refit heads and normalizers."""
    if len(paths) != len(SEEDS):
        raise SubmissionError("Exactly three MERT12 checkpoints are required")
    result: list[HeadBundle] = []
    for path, seed in zip(paths, SEEDS, strict=True):
        payload = torch.load(path, map_location="cpu", weights_only=True)
        mean = np.asarray(payload["mean"], dtype=np.float32)
        scale = np.asarray(payload["scale"], dtype=np.float32)
        if (payload["seed"] != seed or payload["steps"] != 500 or payload["batch_size"] != 32
                or payload["mode"] != "source_generator_parent+coverage_prefix"
                or payload["architecture"] != "mlp" or mean.shape != (1024,) or scale.shape != (1024,)
                or not np.isfinite(mean).all() or not np.isfinite(scale).all() or not (scale > 0).all()):
            raise SubmissionError(f"Invalid full-refit checkpoint: seed{seed}")
        model = FrozenHead(1024, "mlp").to(device)
        model.load_state_dict(payload["model"], strict=True)
        result.append({"model": model.eval(), "mean": mean, "scale": scale, "seed": seed})
    return result


def predict_music(heads: Sequence[HeadBundle], feature_windows: np.ndarray, device: str) -> float:
    """Mean all valid windows first, then average three serialized head probabilities."""
    values = np.asarray(feature_windows, dtype=np.float32)
    if values.ndim != 2 or values.shape[1] != 1024 or not len(values) or not np.isfinite(values).all():
        raise SubmissionError("MERT12 inference requires finite non-empty 1024-wide windows")
    pooled = values.mean(0)
    probabilities: list[float] = []
    with torch.inference_mode():
        for head in heads:
            standardized = (pooled - head["mean"]) / head["scale"]
            logits, _ = head["model"](torch.as_tensor(standardized, device=device)[None])
            probabilities.append(float(logits.sigmoid()[0].cpu()))
    return float(np.mean(probabilities))


def combine(voice: float, music: float, presence: tuple[float, float]) -> list[float]:
    """Keep the current-best baseline file-max fusion unchanged."""
    voice_present, music_present = presence
    return [max(voice_present * voice, music_present * music), voice, music,
            voice_present, music_present]


def load_track_torchaudio(path: Path, channels: int, sample_rate: int) -> torch.Tensor:
    """Decode and resample with the fixed training-feature fallback path."""
    import torchaudio
    from demucs.audio import convert_audio

    waveform, source_rate = torchaudio.load(str(path))
    return convert_audio(waveform, source_rate, sample_rate, channels)


def pin_torchaudio_loader(base: ModuleType) -> None:
    """Remove system-FFmpeg-dependent branching from baseline stem extraction."""
    base.load_track = load_track_torchaudio


def verify_package(root: Path) -> dict:  # noqa: DICT_OK
    manifest = json.loads((root / "model/manifest.json").read_text(encoding="utf-8"))
    choice = manifest["choice"]
    if (choice["seeds"] != list(SEEDS) or choice["candidate"] != "E6_MERT_layer12_full_data"
            or choice["file_fusion"] != "max" or len(choice["checkpoints"]) != 3
            or choice.get("demucs_input_decoder") != DEMUCS_INPUT_DECODER):
        raise SubmissionError("Package choice is not the locked MERT12 full-data ensemble")
    for name, expected in manifest["sha256"].items():
        path = (root / name).resolve()
        if not path.is_relative_to(root.resolve()) or sha256(path) != expected:
            raise SubmissionError(f"Packaged member hash mismatch: {name}")
    return manifest


def extract_mert12(backing: np.ndarray, processor, mert, device: str) -> tuple[np.ndarray, np.ndarray]:
    starts = window_layout(len(backing))
    vectors: list[np.ndarray] = []
    with torch.inference_mode():
        for segment in windows(backing, starts):
            resampled = librosa.resample(segment, orig_sr=16_000, target_sr=24_000, res_type="soxr_hq")
            inputs = processor(resampled, sampling_rate=24_000, return_tensors="pt")
            output = mert(inputs.input_values.to(device), output_hidden_states=True)
            vectors.append(output.hidden_states[12].mean(1)[0].float().cpu().numpy())
    return np.asarray(vectors, dtype=np.float32), starts


def run(root: Path) -> None:
    began = time.monotonic()
    root = root.resolve()
    manifest = verify_package(root)
    assets = root / "model/assets"
    base = baseline_module(assets)
    pin_torchaudio_loader(base)
    args = base.parse_arguments()
    device = base.select_device(args.device)
    if str(device) == "cuda" and not torch.cuda.is_available():
        raise SubmissionError("CUDA requested but unavailable")
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    if str(device) == "cuda":
        torch.cuda.reset_peak_memory_stats()
    columns, rows = base.read_sample_submission(args.sample_submission)
    if columns != list(SUBMISSION_COLUMNS):
        raise SubmissionError("Unexpected sample-submission columns")
    files = base.order_audio_files(base.find_audio_files(args.test_dir), rows)
    presence = base.predict_presence_for_all_files(files, device)
    df, fake_index = base.load_df_arena_model(device)
    demucs = base.load_htdemucs_model()
    processor = AutoFeatureExtractor.from_pretrained(assets / "mert", local_files_only=True)
    mert, info = AutoModel.from_pretrained(assets / "mert", trust_remote_code=True,
                                           local_files_only=True, output_loading_info=True)
    if any(info.values()):
        raise SubmissionError(f"MERT checkpoint did not load exactly: {info}")
    mert.to(device).eval().requires_grad_(False)
    heads = load_heads([root / name for name in manifest["choice"]["checkpoints"]], str(device))
    total_windows = 0
    for index, (row, path) in enumerate(zip(rows, files, strict=True), 1):
        audio = base.load_audio(path)
        with torch.inference_mode():
            vocal, backing = base.separate_voice_and_music(path, demucs, device)
            voice = base.predict_fake(df, fake_index, vocal, device)
        backing = np.pad(backing[:len(audio)], (0, max(0, len(audio) - len(backing))))
        features, starts = extract_mert12(backing, processor, mert, str(device))
        music = predict_music(heads, features, str(device))
        values = combine(voice, music, presence[path.stem])
        total_windows += len(starts)
        row.update({key: round(value, 10) for key, value in zip(PREDICTION_COLUMNS, values, strict=True)})
        print(f"MERT12 {index}/{len(files)} {path.stem}", flush=True)
    validate_predictions(rows, [row["ID"] for row in rows])
    temporary = args.output.with_suffix(".csv.partial")
    base.save_submission(temporary, columns, rows)
    temporary.replace(args.output)
    peak_rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    receipt = {"files": len(rows), "windows": total_windows, "device": args.device,
               "elapsed_seconds": time.monotonic() - began,
               "peak_gpu_allocated_bytes": torch.cuda.max_memory_allocated() if str(device) == "cuda" else 0,
               "peak_gpu_reserved_bytes": torch.cuda.max_memory_reserved() if str(device) == "cuda" else 0,
               "peak_rss_bytes": int(peak_rss if sys.platform == "darwin" else peak_rss * 1024),
               "submission_sha256": sha256(args.output), "manifest_sha256": sha256(root / "model/manifest.json"),
               "official_l4_validated": False, "window_pooling": "all_valid_mean_no_audible_mask"}
    receipt["demucs_input_decoder"] = DEMUCS_INPUT_DECODER
    (args.output.parent / "runtime_receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")


def compare(reference: Path, permuted: Path, subset: Path, cached_reference_path: Path,
            output: Path) -> None:
    def read(path: Path) -> dict[str, dict[str, str]]:
        with path.open(encoding="utf-8", newline="") as stream:
            return {row["ID"]: row for row in csv.DictReader(stream)}
    expected, reordered, reduced = read(reference), read(permuted), read(subset)
    if set(expected) != set(reordered) or not reduced or not set(reduced) < set(expected):
        raise SubmissionError("QA requires a full permutation and a strict non-empty subset")
    columns = list(PREDICTION_COLUMNS)
    maximum = max(abs(float(expected[key][column]) - float(candidate[key][column]))
                  for candidate in (reordered, reduced) for key in candidate for column in columns)
    if maximum > 1e-7:
        raise SubmissionError(f"Cross-file order/subset independence failed: {maximum}")
    cached = json.loads(cached_reference_path.read_text(encoding="utf-8"))["music_probability_by_id"]
    if set(cached) != set(expected):
        raise SubmissionError("Cached MERT12 reference IDs differ from the raw-audio smoke set")
    cached_error = max(abs(float(expected[key]["MUSIC_FAKE_PROB"]) - float(value))
                       for key, value in cached.items())
    if cached_error > 1e-5:
        raise SubmissionError(f"Cached/raw-audio MERT12 parity failed: {cached_error}")
    output.write_text(json.dumps({"status": "passed", "reference_files": len(expected),
        "subset_files": len(reduced), "max_probability_error": maximum,
        "cached_raw_music_tolerance": 1e-5,
        "cached_raw_music_max_abs_error": cached_error}, indent=2) + "\n")


def cached_reference(features: Path, checkpoints: Sequence[Path], sample_submission: Path,
                     output: Path) -> None:
    """Score the hash-verified H100 MERT12 cache with the serialized final heads."""
    with sample_submission.open(encoding="utf-8", newline="") as stream:
        identifiers = [row["ID"] for row in csv.DictReader(stream)]
    heads = load_heads(checkpoints, "cpu")
    scores: dict[str, float] = {}
    feature_hashes: dict[str, str] = {}
    for identifier in identifiers:
        path = features / f"{identifier}.npz"
        with np.load(path, allow_pickle=False) as saved:
            identity = json.loads(str(saved["identity_json"]))
            valid = np.asarray(saved["valid_samples"]) > 0
            values = np.asarray(saved["mert_l12"], dtype=np.float32)
        if identity.get("ID") != identifier or valid.shape != (len(values),) or not valid.any():
            raise SubmissionError(f"Invalid cached MERT12 smoke reference: {identifier}")
        scores[identifier] = predict_music(heads, values[valid], "cpu")
        feature_hashes[identifier] = sha256(path)
    output.write_text(json.dumps({"music_probability_by_id": scores,
                                  "feature_sha256_by_id": feature_hashes}, indent=2) + "\n")


def cli() -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    qa = subparsers.add_parser("compare")
    for name in ("reference", "permuted", "subset", "output"):
        qa.add_argument(f"--{name}", type=Path, required=True)
    qa.add_argument("--cached-reference", type=Path, required=True)
    cached = subparsers.add_parser("cached-reference")
    cached.add_argument("--features", type=Path, required=True)
    cached.add_argument("--checkpoints", type=Path, nargs=3, required=True)
    cached.add_argument("--sample-submission", type=Path, required=True)
    cached.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "compare":
        compare(args.reference, args.permuted, args.subset, args.cached_reference, args.output)
    else:
        cached_reference(args.features, args.checkpoints, args.sample_submission, args.output)


if __name__ == "__main__":
    cli()
