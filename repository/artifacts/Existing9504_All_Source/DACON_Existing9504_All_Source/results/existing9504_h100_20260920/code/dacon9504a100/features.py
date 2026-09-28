"""H100 full-duration frozen-encoder features for all 9504 rows (isolated venv).

Derived from source/full_bank_features.py (``prepare``).  The per-file
computation is the original one, reusing unchanged functions by import:
  * v6_runtime.baseline_module + mert12_submission.pin_torchaudio_loader
    (HTDemucs input decoded by torchaudio.load + demucs.audio.convert_audio),
  * input gain: legacy/constructed input_kind preserved; originals peak-capped at 0.6,
  * HTDemucs vocals / summed-other backing, trimmed/padded to the raw timeline,
  * MERT-v1-330M hidden_states[12] mean over every 64,600-sample half-hop window
    (mert12_submission.window_layout/windows; batch chosen by localization_features.select_batch),
  * DF-Arena 1B segments of the vocal stem with the original fc5 input captured by a
    forward pre-hook (1280-d) and the original fake probability kept as voice_original,
  * PANNs presence max over baseline segments of the gain-normalized audio,
  * full_bank_features.window_truth / row_identity / validate_cache unchanged.
Changes versus the original: H100 (not H100) identity check, all three splits
allowed (features carry no training decision), a separate pilot cache and a
recorded runtime projection, per-component timing, and exact 9504 coverage.
"""
from __future__ import annotations

import argparse
import json
import os
import tempfile
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch

from full_bank_features import row_identity, validate_cache, window_truth
from localization_features import select_batch
from mert12_submission import HOP, WINDOW, pin_torchaudio_loader, window_layout, windows
from v6_runtime import baseline_module

from .common import PipelineError, StageLock, atomic_json, now_iso, read_json, sha256_file
from .dataset import is_singing, read_manifest
from .fingerprint import differences, feature_implementation, stage_fingerprint
from .gpu import configure_numerics, memory, release, require_device, synchronize

MERT_BATCH_CANDIDATES = (16, 32, 64, 128, 256)


def load_config(path: Path) -> dict[str, Any]:
    return read_json(path)


def implementation_hashes(config: Mapping[str, Any]) -> dict[str, str]:
    return feature_implementation(config)


def identity_for(config: Mapping[str, Any], batch: int, purpose: str, device_info: Mapping[str, Any]) -> dict[str, Any]:
    env = read_json(Path(config["env_receipt"])) if config.get("env_receipt") else {}
    return {"manifest_sha256": sha256_file(Path(config["manifest"])),
            "assets": read_json(Path(config["assets_dir"]) / "asset_hashes.json"),
            "implementation": implementation_hashes(config), "window": WINDOW, "hop": HOP, "mert_layer": 12,
            "dtype": "float32", "duration": "all samples, no crop", "torch": str(torch.__version__),
            "torch_cuda": torch.version.cuda, "gpu": device_info.get("name"), "env_lock_sha256": env.get("lock_sha256"),
            "mert_batch": int(batch), "purpose": purpose,
            "gain_policy": "input_kind legacy*/constructed* preserved; otherwise peak capped at 0.6",
            "demucs_input_decoder": "torchaudio.load+demucs.audio.convert_audio"}


def identity_key(identity: Mapping[str, Any]) -> str:
    import hashlib
    return hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()


class Encoders:
    """The four frozen models, loaded exactly as the original extraction did."""

    def __init__(self, assets: Path, device: str) -> None:
        os.environ.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
        import librosa
        import soundfile
        from transformers import AutoFeatureExtractor, AutoModel
        self.librosa, self.soundfile = librosa, soundfile
        self.device = torch.device(device)
        began = time.monotonic()
        self.base = baseline_module(assets)
        pin_torchaudio_loader(self.base)
        self.separator = self.base.load_htdemucs_model().to(self.device).eval().requires_grad_(False)
        self.voice_model, self.fake_index = self.base.load_df_arena_model(self.device)
        self.voice_model.requires_grad_(False)
        self.classifier = self.voice_model.backbone.conformer.fc5
        self.panns, self.voice_indices, self.music_indices = self.base.load_panns_model(self.device)
        self.processor = AutoFeatureExtractor.from_pretrained(assets / "mert", local_files_only=True)
        self.mert, loading = AutoModel.from_pretrained(assets / "mert", trust_remote_code=True,
                                                       local_files_only=True, output_loading_info=True)
        if any(loading.values()):
            raise PipelineError(f"MERT loading mismatch: {loading}")
        self.mert.to(self.device).eval().requires_grad_(False)
        self.captured: list[np.ndarray] = []
        self.handle = self.classifier.register_forward_pre_hook(self._capture)
        self.load_seconds = time.monotonic() - began

    def _capture(self, _module, inputs) -> None:
        self.captured.append(inputs[0].detach().float().cpu().numpy().copy())

    def classifier_state(self) -> dict[str, Any]:
        return {"state_dict": {k: v.detach().cpu() for k, v in self.classifier.state_dict().items()},
                "module": "backbone.conformer.fc5", "fake_label_index": self.fake_index}

    def mert_batch(self, audio: np.ndarray) -> np.ndarray:
        resampled = [self.librosa.resample(x, orig_sr=16000, target_sr=24000, res_type="soxr_hq") for x in audio]
        values = self.processor(resampled, sampling_rate=24000, return_tensors="pt", padding=True).input_values
        with torch.inference_mode():
            return self.mert(values.to(self.device), output_hidden_states=True).hidden_states[12].mean(1).float().cpu().numpy()

    def separate(self, raw: np.ndarray, scratch: Path) -> tuple[np.ndarray, np.ndarray]:
        with tempfile.TemporaryDirectory(prefix="full_bank_audio_", dir=scratch) as temporary:
            normalized = Path(temporary) / "normalized.wav"
            self.soundfile.write(normalized, raw, 16000, subtype="FLOAT")
            with torch.inference_mode():
                voice, backing = self.base.separate_voice_and_music(normalized, self.separator, self.device)
        voice = np.pad(voice[:len(raw)], (0, max(0, len(raw) - len(voice))))
        backing = np.pad(backing[:len(raw)], (0, max(0, len(raw) - len(backing))))
        return voice, backing

    def load_normalized(self, row: Mapping[str, str], data_dir: Path) -> tuple[np.ndarray, float, float]:
        path = (data_dir / row["path"]).resolve()
        if not path.is_relative_to(data_dir.resolve()) or sha256_file(path) != row["audio_sha256"]:
            raise PipelineError(f"Input bytes changed or escaped input directory: {row['ID']}")
        raw = self.base.load_audio(path)
        peak = float(np.abs(raw).max())
        preserve_gain = row.get("input_kind", "").startswith(("legacy", "constructed"))
        gain = 1. if preserve_gain or peak <= .6 else .6 / peak
        return (raw * gain).astype(np.float32), peak, gain

    def process(self, row: Mapping[str, str], data_dir: Path, scratch: Path, batch: int,
                key: str) -> tuple[dict[str, np.ndarray], dict[str, float]]:
        timing: dict[str, float] = {}
        mark = time.perf_counter()

        def lap(name: str) -> None:
            nonlocal mark
            synchronize(self.device.type)
            now = time.perf_counter()
            timing[name] = now - mark
            mark = now

        raw, peak, gain = self.load_normalized(row, data_dir)
        lap("decode")
        voice, backing = self.separate(raw, scratch)
        lap("htdemucs")
        starts = window_layout(len(raw))
        valid = np.minimum(WINDOW, len(raw) - starts)
        pieces = [self.mert_batch(windows(backing, starts[offset:offset + batch]))
                  for offset in range(0, len(starts), batch)]
        vectors = np.concatenate(pieces)
        lap("mert")
        self.captured.clear()
        voice_scores, presence = [], np.zeros(2, np.float32)
        for start in self.base.get_segment_starts(len(raw)):
            segment = self.base.extract_segment(voice, start)
            with torch.inference_mode():
                logits = self.voice_model(input_values=torch.from_numpy(segment).to(self.device))["logits"]
            voice_scores.append(float(logits.float().softmax(-1)[0, self.fake_index]))
            a, b = self.base.predict_presence(self.panns, self.voice_indices, self.music_indices,
                                              self.base.extract_segment(raw, start))
            presence = np.maximum(presence, [a, b])
        voice_silent = self.base.calculate_rms(voice) < self.base.SILENCE_RMS
        lap("dfarena_panns")
        arrays = {"identity": np.array(key), "mert_windows": vectors, "starts": starts, "valid_samples": valid,
                  "window_truth": window_truth(row, starts, valid), "voice_embeddings": np.concatenate(self.captured),
                  "voice_original": np.array(0. if voice_silent else max(voice_scores)),
                  "voice_silent": np.array(voice_silent), "presence": np.asarray(presence, np.float32),
                  "samples": np.array(len(raw)), "source_peak": np.array(peak), "normalization_gain": np.array(gain),
                  "normalized_rms": np.array(self.base.calculate_rms(raw))}
        timing["total"] = sum(timing.values())
        return arrays, timing

    def close(self) -> None:
        self.handle.remove()
        for name in ("separator", "voice_model", "panns", "mert", "classifier"):
            if hasattr(self, name):
                delattr(self, name)
        release(self.device.type)


def save_npz(target: Path, arrays: Mapping[str, np.ndarray]) -> str:
    temporary = target.with_suffix(".tmp.npz")
    np.savez(temporary, **arrays)
    temporary.replace(target)
    return sha256_file(target)


class FeatureCache:
    """index.json + <ID>.npz, atomically updated per row (original resume format)."""

    def __init__(self, directory: Path, identity: Mapping[str, Any]) -> None:
        self.directory = directory
        self.directory.mkdir(parents=True, exist_ok=True)
        self.index_path = directory / "index.json"
        self.identity = dict(identity)
        self.key = identity_key(identity)
        self.index = read_json(self.index_path) if self.index_path.is_file() else {"identity": self.identity, "items": {}}
        if self.index["identity"] != self.identity:
            raise PipelineError(f"Refuse to resume {directory}: cache identity differs (data/models/implementation)")

    def verify_existing(self, rows_by_id: Mapping[str, Mapping[str, str]]) -> int:
        for identifier, entry in self.index["items"].items():
            if identifier not in rows_by_id:
                raise PipelineError(f"Cache holds an ID outside the manifest: {identifier}")
            validate_cache(self.directory / entry["path"], row_identity(rows_by_id[identifier], self.key), entry["sha256"])
        return len(self.index["items"])

    def ensure_initial_classifier(self, state: Mapping[str, Any]) -> str:
        path = self.directory / "voice_classifier_initial.pt"
        if not path.exists():
            torch.save(dict(state), path)
        initial = torch.load(path, map_location="cpu", weights_only=True)
        if initial["fake_label_index"] != state["fake_label_index"] or any(
                not torch.equal(value.cpu(), initial["state_dict"][name]) for name, value in state["state_dict"].items()):
            raise PipelineError("Cached voice classifier initialization changed")
        return sha256_file(path)

    def add(self, row: Mapping[str, str], arrays: Mapping[str, np.ndarray]) -> dict[str, Any]:
        target = self.directory / f"{row['ID']}.npz"
        digest = save_npz(target, arrays)
        entry = {"path": target.name, "sha256": digest, "identity": str(arrays["identity"]),
                 "windows": int(len(arrays["starts"])), "samples": int(arrays["samples"])}
        self.index["items"][row["ID"]] = entry
        atomic_json(self.index_path, self.index)
        return entry


def duration_of(row: Mapping[str, str]) -> float | None:
    try:
        value = float(row.get("duration_seconds") or "nan")
    except ValueError:
        return None
    return value if np.isfinite(value) and value > 0 else None


def stratum(row: Mapping[str, str]) -> str:
    if row["route"] == "voice":
        return "singing" if is_singing(row) else "speech"
    return "music_legacy_gain_preserved" if row.get("input_kind", "").startswith(("legacy", "constructed")) \
        else "music_original_peak_capped"


def select_pilot_rows(rows: Sequence[Mapping[str, str]], quantiles=(0., .25, .5, .75, .95, 1.)) -> list[Mapping[str, str]]:
    """Deterministic, train-only, duration-quantile rows per stratum plus one per file extension."""
    train = [row for row in rows if row["split"] == "train" and duration_of(row) is not None]
    chosen: dict[str, Mapping[str, str]] = {}
    for name in sorted({stratum(row) for row in train}):
        members = sorted((row for row in train if stratum(row) == name), key=lambda r: (duration_of(r), r["ID"]))
        for q in quantiles:
            row = members[min(len(members) - 1, int(round(q * (len(members) - 1))))]
            chosen[row["ID"]] = row
    for extension in sorted({Path(row["path"]).suffix.lower() for row in train}):
        if not any(Path(row["path"]).suffix.lower() == extension for row in chosen.values()):
            candidates = sorted((row for row in train if Path(row["path"]).suffix.lower() == extension),
                                key=lambda r: (duration_of(r), r["ID"]))
            row = candidates[len(candidates) // 2]
            chosen[row["ID"]] = row
    return sorted(chosen.values(), key=lambda r: (duration_of(r), r["ID"]))


def fit_runtime(records: Sequence[Mapping[str, Any]]) -> dict[str, float]:
    duration = np.asarray([record["duration_seconds"] for record in records], np.float64)
    seconds = np.asarray([record["timing"]["total"] for record in records], np.float64)
    design = np.stack([np.ones_like(duration), duration], 1)
    (intercept, slope), *_ = np.linalg.lstsq(design, seconds, rcond=None)
    if slope <= 0:
        intercept, slope = 0.0, float(seconds.sum() / max(duration.sum(), 1e-9))
    intercept = max(0.0, float(intercept))
    residual = seconds - (intercept + slope * duration)
    return {"intercept_seconds": intercept, "seconds_per_audio_second": float(slope),
            "rmse_seconds": float(np.sqrt(np.mean(residual ** 2))), "rows": len(records)}


def project(rows: Sequence[Mapping[str, str]], fit: Mapping[str, float]) -> dict[str, Any]:
    durations = [duration_of(row) for row in rows]
    known = [value for value in durations if value is not None]
    fallback = float(np.median(known)) if known else 0.0
    seconds = [fit["intercept_seconds"] + fit["seconds_per_audio_second"] * (value if value is not None else fallback)
               for value in durations]
    by_split = {split: float(sum(s for s, row in zip(seconds, rows) if row["split"] == split))
                for split in ("train", "val", "test")}
    return {"projected_feature_seconds": float(sum(seconds)), "projected_feature_hours": float(sum(seconds) / 3600),
            "by_split_seconds": by_split, "rows": len(rows), "rows_without_duration": durations.count(None),
            "audio_hours": float(sum(known) / 3600)}


def run_pilot(config: Mapping[str, Any]) -> dict[str, Any]:
    device = config["device"]
    info = require_device(device, config["expected_gpu"])
    configure_numerics()
    rows = read_manifest(Path(config["manifest"]))
    pilot_dir = Path(config["pilot_dir"])
    if (pilot_dir / "cache").exists():  # an interrupted pilot restarts clean (its batch may be re-measured)
        import shutil
        shutil.rmtree(pilot_dir / "cache")
    pilot_dir.mkdir(parents=True, exist_ok=True)
    data_dir = Path(config["data_dir"])
    selected = select_pilot_rows(rows)
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    encoders = Encoders(Path(config["assets_dir"]), device)
    print(f"pilot: models loaded in {encoders.load_seconds:.1f}s; {len(selected)} train rows selected", flush=True)
    warm = selected[0]
    raw, _, _ = encoders.load_normalized(warm, data_dir)
    warm_began = time.monotonic()
    encoders.separate(raw, pilot_dir)
    encoders.mert_batch(windows(raw, window_layout(len(raw))[:1]))
    warmup_seconds = time.monotonic() - warm_began
    receipt_path = pilot_dir / "mert_batch_benchmark.json"
    if config.get("mert_batch_override"):
        batch = int(config["mert_batch_override"])
        atomic_json(receipt_path, {"chosen": batch, "measurements": [], "policy": "test override (CPU only)"})
    else:
        music = [row for row in selected if row["route"] == "music"]
        reference = max(music or selected, key=lambda r: (duration_of(r) or 0, r["ID"]))
        raw, _, _ = encoders.load_normalized(reference, data_dir)
        _, backing = encoders.separate(raw, pilot_dir)
        sample = windows(backing, np.resize(window_layout(len(raw)), max(MERT_BATCH_CANDIDATES)))
        batch = select_batch(lambda n: encoders.mert_batch(sample[:n]), MERT_BATCH_CANDIDATES, receipt_path,
                             float("inf"))
        benchmark = read_json(receipt_path)
        benchmark["reference_id"] = reference["ID"]
        atomic_json(receipt_path, benchmark)
    print(f"pilot: MERT batch={batch}", flush=True)
    identity = identity_for(config, batch, "pilot", info)
    cache = FeatureCache(pilot_dir / "cache", identity)
    cache.ensure_initial_classifier(encoders.classifier_state())
    records = []
    for number, row in enumerate(selected, 1):
        key = row_identity(row, cache.key)
        arrays, timing = encoders.process(row, data_dir, pilot_dir, batch, key)
        cache.add(row, arrays)
        record = {"ID": row["ID"], "stratum": stratum(row), "split": row["split"], "route": row["route"],
                  "extension": Path(row["path"]).suffix.lower(), "duration_seconds": duration_of(row),
                  "samples": int(arrays["samples"]), "windows": int(len(arrays["starts"])),
                  "voice_segments": int(len(arrays["voice_embeddings"])), "timing": timing, "memory": memory(device)}
        records.append(record)
        print(f"pilot {number}/{len(selected)} {row['ID']} dur={record['duration_seconds']:.1f}s "
              f"total={timing['total']:.2f}s", flush=True)
    fit = fit_runtime(records)
    projection = project(rows, fit)
    peak = memory(device)
    receipt = {"device": info, "rows": records, "fit": fit, "projection": projection, "mert_batch": batch,
               "mert_batch_benchmark": read_json(receipt_path), "model_load_seconds": encoders.load_seconds,
               "warmup_seconds": warmup_seconds, "memory": peak, "identity": identity,
               "fingerprint": stage_fingerprint(config, "pilot"),
               "selection": "train split only; duration quantiles per stratum + one row per extension",
               "longest_duration_seconds_all_rows": max((duration_of(r) or 0) for r in rows),
               "longest_duration_seconds_pilot": max((duration_of(r) or 0) for r in selected),
               "finished_at": now_iso()}
    encoders.close()
    atomic_json(pilot_dir / "pilot_receipt.json", receipt)
    print(f"pilot: projected feature extraction {projection['projected_feature_hours']:.2f} h for "
          f"{projection['rows']} rows ({projection['audio_hours']:.1f} h audio)", flush=True)
    return receipt


def run_full(config: Mapping[str, Any]) -> dict[str, Any]:
    device = config["device"]
    info = require_device(device, config["expected_gpu"])
    configure_numerics()
    rows = read_manifest(Path(config["manifest"]))
    by_id = {row["ID"]: row for row in rows}
    pilot = read_json(Path(config["pilot_dir"]) / "pilot_receipt.json")
    stale = differences(pilot.get("fingerprint"), stage_fingerprint(config, "pilot"))
    if stale:
        raise PipelineError(f"Pilot receipt does not match the current inputs/code/environment/config: {stale[:10]}")
    batch = int(pilot["mert_batch"])
    identity = identity_for(config, batch, "full", info)
    cache_dir = Path(config["cache_dir"])
    cache = FeatureCache(cache_dir, identity)
    done = cache.verify_existing(by_id)
    fit = pilot["fit"]
    predicted = {row["ID"]: fit["intercept_seconds"] + fit["seconds_per_audio_second"] * (duration_of(row) or 0.)
                 for row in rows}
    progress_path = Path(config["features_dir"]) / "progress.json"
    stats_path = Path(config["features_dir"]) / "row_timings.jsonl"
    Path(config["features_dir"]).mkdir(parents=True, exist_ok=True)
    print(f"features: {done}/{len(rows)} rows already cached and verified", flush=True)
    began = time.monotonic()
    session_predicted = session_actual = 0.0
    encoders = None
    if done < len(rows):
        if device == "cuda":
            torch.cuda.reset_peak_memory_stats()
        encoders = Encoders(Path(config["assets_dir"]), device)
        cache.ensure_initial_classifier(encoders.classifier_state())
    remaining = sum(predicted[r["ID"]] for r in rows if r["ID"] not in cache.index["items"])
    for row in rows:
        if row["ID"] in cache.index["items"]:
            continue
        arrays, timing = encoders.process(row, Path(config["data_dir"]), cache_dir, batch, row_identity(row, cache.key))
        entry = cache.add(row, arrays)
        session_actual += timing["total"]
        session_predicted += predicted[row["ID"]]
        remaining -= predicted[row["ID"]]
        ratio = session_actual / max(session_predicted, 1e-9)
        with stats_path.open("a") as stream:
            stream.write(json.dumps({"ID": row["ID"], "windows": entry["windows"], "samples": entry["samples"],
                                     "timing": timing}) + "\n")
        atomic_json(progress_path, {"stage": "features", "done": len(cache.index["items"]), "total": len(rows),
                                    "last_id": row["ID"], "session_elapsed_seconds": time.monotonic() - began,
                                    "observed_over_pilot_ratio": ratio, "eta_seconds": max(0.0, remaining) * ratio,
                                    "memory": memory(device), "updated_at": now_iso()})
        print(f"features {len(cache.index['items'])}/{len(rows)} {row['ID']} windows={entry['windows']} "
              f"{timing['total']:.2f}s", flush=True)
    coverage = set(cache.index["items"])
    if coverage != set(by_id):
        raise PipelineError(f"Feature coverage is not exact: missing={len(set(by_id) - coverage)} "
                            f"extra={len(coverage - set(by_id))}")
    if encoders is not None:
        encoders.close()
    classifier_path = cache_dir / "voice_classifier_initial.pt"
    complete = {"rows": len(rows), "index_sha256": sha256_file(cache.index_path),
                "voice_classifier_sha256": sha256_file(classifier_path), "identity": identity,
                "session_seconds": time.monotonic() - began, "memory": memory(device), "device": info,
                "windows_total": sum(entry["windows"] for entry in cache.index["items"].values()),
                "finished_at": now_iso()}
    atomic_json(Path(config["features_dir"]) / "features_complete.json", complete)
    return complete


def compare_arrays(left: Mapping[str, np.ndarray], right: Mapping[str, np.ndarray]) -> dict[str, Any]:
    """Per-array comparison that cannot hide NaN/inf disagreements.

    Floating arrays: NaN positions must match exactly (a NaN on one side and a
    number on the other is a mismatch; NaN on both sides = matching unknown), any
    infinity is rejected, and max_abs is taken over positions finite on both sides.
    """
    report: dict[str, Any] = {}
    for name in sorted(set(left) | set(right)):
        if name == "identity":
            continue
        if name not in left or name not in right:
            report[name] = {"status": "missing"}
            continue
        a, b = np.asarray(left[name]), np.asarray(right[name])
        if a.shape != b.shape:
            report[name] = {"status": "shape_mismatch", "left": list(a.shape), "right": list(b.shape)}
            continue
        if a.dtype.kind == "f" or b.dtype.kind == "f":
            a64, b64 = a.astype(np.float64), b.astype(np.float64)
            nan_a, nan_b = np.isnan(a64), np.isnan(b64)
            infinite = int(np.isinf(a64).sum() + np.isinf(b64).sum())
            nan_mismatch = int((nan_a != nan_b).sum())
            finite = np.isfinite(a64) & np.isfinite(b64)
            difference = np.abs(a64[finite] - b64[finite])
            max_abs = float(difference.max()) if difference.size else 0.0
            entry = {"max_abs": max_abs, "nan_positions_left": int(nan_a.sum()),
                     "nan_positions_right": int(nan_b.sum()), "nan_mask_mismatches": nan_mismatch,
                     "infinite_values": infinite,
                     "bitwise_equal": bool(nan_mismatch == 0 and infinite == 0 and np.array_equal(a, b, equal_nan=True))}
            if nan_mismatch:
                entry["status"] = "nan_mask_mismatch"
            elif infinite or not np.isfinite(max_abs):
                entry["status"] = "nonfinite"
            else:
                entry["status"] = "compared"
            report[name] = entry
        elif a.dtype != b.dtype and not (a.dtype.kind in "biu" and b.dtype.kind in "biu"):
            report[name] = {"status": "dtype_mismatch", "left": str(a.dtype), "right": str(b.dtype)}
        else:
            report[name] = {"status": "compared", "equal": bool(np.array_equal(a, b))}
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("pilot", "full"))
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    config = load_config(args.config)
    if config["device"] != "cuda" or config["expected_gpu"] != "H100":
        raise PipelineError("The CLI runs only on the authorized H100 (device=cuda)")
    with StageLock(Path(config["job_root"]) / "locks" / f"features-{args.command}.lock", "dacon9504a100.features"):
        (run_pilot if args.command == "pilot" else run_full)(config)


if __name__ == "__main__":
    main()
