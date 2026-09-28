"""Isolate MERT batch geometry on one retained backing stem per audio file."""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import json
import os
import subprocess
import time
from pathlib import Path
from typing import Final

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

import librosa
import numpy as np
import torch
from transformers import AutoFeatureExtractor, AutoModel

from mert12_submission import load_heads, predict_music, verify_package, window_layout, windows
from v6_data import sha256
from v6_runtime import baseline_module

EXPECTED_FILES: Final = 4


def difference(left: np.ndarray, right: np.ndarray) -> dict[str, float]:
    """Return decision-useful absolute error statistics for equal arrays."""
    if left.shape != right.shape:
        raise ValueError(f"shape mismatch: {left.shape} != {right.shape}")
    delta = left.astype(np.float64) - right.astype(np.float64)
    return {"max_abs": float(np.max(np.abs(delta))), "rms": float(np.sqrt(np.mean(delta**2)))}


def seed_probabilities(heads: list, feature_windows: np.ndarray) -> list[float]:  # noqa: ANN401
    """Score one pooled feature with every serialized head independently."""
    pooled = np.asarray(feature_windows, dtype=np.float32).mean(0)
    result: list[float] = []
    with torch.inference_mode():
        for head in heads:
            value = torch.as_tensor((pooled - head["mean"]) / head["scale"])[None]
            logits, _ = head["model"](value)
            result.append(float(logits.sigmoid()[0]))
    return result


def file_sha256_from_array(value: np.ndarray) -> str:
    """Bind retained intermediate values without exporting large arrays."""
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def run(args: argparse.Namespace) -> None:
    """Run the same-stem batch-versus-singleton MERT probe."""
    if args.device != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("This diagnostic requires the actual CUDA runtime")
    if time.time() >= args.deadline_epoch:
        raise TimeoutError("Diagnostic deadline already expired")
    package = args.package_root.resolve()
    manifest = verify_package(package)
    assets = package / "model/assets"
    base = baseline_module(assets)
    with args.sample_submission.open(encoding="utf-8", newline="") as stream:
        identifiers = [row["ID"] for row in csv.DictReader(stream)]
    if len(identifiers) != EXPECTED_FILES or len(set(identifiers)) != EXPECTED_FILES:
        raise ValueError("Probe requires the exact four-file smoke set")
    files = base.order_audio_files(base.find_audio_files(args.test_dir), [{"ID": value} for value in identifiers])

    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.set_float32_matmul_precision("highest")
    torch.cuda.reset_peak_memory_stats()
    separator = base.load_htdemucs_model().eval().requires_grad_(False)
    processor = AutoFeatureExtractor.from_pretrained(assets / "mert", local_files_only=True)
    mert, loading = AutoModel.from_pretrained(
        assets / "mert", trust_remote_code=True, local_files_only=True, output_loading_info=True,
    )
    if any(loading.values()):
        raise RuntimeError(f"MERT checkpoint did not load exactly: {loading}")
    mert.to(args.device).eval().requires_grad_(False)
    heads = load_heads([package / name for name in manifest["choice"]["checkpoints"]], "cpu")

    records: list[dict] = []
    arrays_dir = args.output.parent / f"{args.output.stem}_arrays"
    arrays_dir.mkdir(parents=True, exist_ok=True)
    began = time.monotonic()
    for path in files:
        if time.time() >= args.deadline_epoch:
            raise TimeoutError("Diagnostic deadline reached")
        audio = base.load_audio(path)
        with torch.inference_mode():
            _, backing = base.separate_voice_and_music(path, separator, args.device)
        backing = np.pad(backing[: len(audio)], (0, max(0, len(audio) - len(backing)))).astype(np.float32)
        starts = window_layout(len(backing))
        resampled = [
            librosa.resample(item, orig_sr=16_000, target_sr=24_000, res_type="soxr_hq")
            for item in windows(backing, starts)
        ]
        batch_input = processor(resampled, sampling_rate=24_000, return_tensors="pt", padding=True).input_values
        single_input = torch.cat([
            processor(item, sampling_rate=24_000, return_tensors="pt").input_values for item in resampled
        ])
        processor_error = difference(batch_input.numpy(), single_input.numpy())
        if processor_error["max_abs"] != 0.0:
            raise RuntimeError(f"Processor changed values across batch geometry: {processor_error}")
        with torch.inference_mode():
            batch_features = mert(batch_input.to(args.device), output_hidden_states=True).hidden_states[12].mean(1).float().cpu().numpy()
            single_features = np.concatenate([
                mert(batch_input[index:index + 1].to(args.device), output_hidden_states=True)
                .hidden_states[12].mean(1).float().cpu().numpy()
                for index in range(len(batch_input))
            ])
        cached_path = args.cached_features / f"{path.stem}.npz"
        with np.load(cached_path, allow_pickle=False) as saved:
            identity = json.loads(str(saved["identity_json"]))
            cached = np.asarray(saved["mert_l12"], dtype=np.float32)
            cached_starts = np.asarray(saved["starts"], dtype=np.int64)
        if identity["audio_sha256"] != sha256(path) or not np.array_equal(cached_starts, starts):
            raise RuntimeError(f"Cached identity/layout mismatch: {path.stem}")
        arrays_path = arrays_dir / f"{path.stem}.npz"
        temporary = arrays_path.with_suffix(".npz.partial")
        with temporary.open("wb") as stream:
            np.savez_compressed(
                stream,
                backing=backing,
                batch_features=batch_features,
                single_features=single_features,
                cached_starts=cached_starts,
            )
        temporary.replace(arrays_path)
        records.append({
            "ID": path.stem,
            "audio_sha256": sha256(path),
            "backing_sha256_float32": file_sha256_from_array(backing),
            "arrays_path": arrays_path.relative_to(args.output.parent).as_posix(),
            "arrays_sha256": sha256(arrays_path),
            "starts": starts.tolist(),
            "processor_batch_vs_single": processor_error,
            "mert_batch_vs_single": difference(batch_features, single_features),
            "mert_batch_vs_h100_cache": difference(batch_features, cached),
            "mert_single_vs_h100_cache": difference(single_features, cached),
            "probability": {
                "batch": predict_music(heads, batch_features, "cpu"),
                "single": predict_music(heads, single_features, "cpu"),
                "h100_cache": predict_music(heads, cached, "cpu"),
                "batch_per_seed": seed_probabilities(heads, batch_features),
                "single_per_seed": seed_probabilities(heads, single_features),
                "h100_cache_per_seed": seed_probabilities(heads, cached),
            },
        })
        del batch_features, single_features, backing
        torch.cuda.empty_cache()

    ffmpeg = subprocess.run(["ffmpeg", "-version"], check=True, capture_output=True, text=True).stdout.splitlines()[0]
    receipt = {
        "schema_version": 1,
        "status": "complete",
        "scope": "same retained stem and exact processor tensor; MERT batch-of-file-windows versus singleton forwards",
        "files": len(records),
        "records": records,
        "environment": {
            "python": os.sys.version.split()[0],
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "transformers": importlib.metadata.version("transformers"),
            "librosa": importlib.metadata.version("librosa"),
            "numpy": np.__version__,
            "demucs": importlib.metadata.version("demucs"),
            "ffmpeg": ffmpeg,
            "gpu": torch.cuda.get_device_name(0),
            "matmul_tf32": torch.backends.cuda.matmul.allow_tf32,
            "cudnn_tf32": torch.backends.cudnn.allow_tf32,
        },
        "elapsed_seconds": time.monotonic() - began,
        "peak_gpu_allocated_bytes": torch.cuda.max_memory_allocated(),
        "peak_gpu_reserved_bytes": torch.cuda.max_memory_reserved(),
        "package_manifest_sha256": sha256(package / "model/manifest.json"),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")


def cli() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--package-root", type=Path, required=True)
    parser.add_argument("--test-dir", type=Path, required=True)
    parser.add_argument("--sample-submission", type=Path, required=True)
    parser.add_argument("--cached-features", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--deadline-epoch", type=float, required=True)
    parser.add_argument("--device", choices=("cuda",), default="cuda")
    run(parser.parse_args())


if __name__ == "__main__":
    cli()
