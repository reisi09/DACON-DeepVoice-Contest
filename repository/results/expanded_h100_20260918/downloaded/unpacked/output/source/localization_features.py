"""One shared backing-audio timeline for the bounded MERT/MIL/AST comparison."""
from __future__ import annotations

import argparse
import csv
import gc
import importlib.metadata
import json
import time
from pathlib import Path

import numpy as np
import torch

from eight_fixed_io import atomic_json
from mert12_submission import WINDOW, HOP, pin_torchaudio_loader, window_layout
from v6_data import sha256
from v6_runtime import baseline_module


def check_deadline(deadline: float) -> None:
    if time.time() >= deadline:
        raise TimeoutError("Feature extraction reached the original run deadline")


def frame_truth(row: dict, starts: np.ndarray, valid_samples: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Fractional waveform occupancy in each 25 ms Kaldi frame; never fill unknowns.

    Constructed partials have a documented real music bed. Whole-source music
    intervals are supplied explicitly by the dataset builder, not inferred here.
    """
    targets = np.full((len(starts), 1024), np.nan, np.float32)
    mask = np.zeros(targets.shape, bool)
    events = [event for event in json.loads(row.get("event_intervals") or "[]")
              if event.get("component") == "music"]
    real_bed = row.get("variant", "").startswith(("partial_", "sham"))
    for index, (start, valid) in enumerate(zip(starts, valid_samples, strict=True)):
        count = max(0, min(1024, (int(valid) - 400) // 160 + 1))
        mask[index, :count] = True
        for frame in range(count):
            left = int(start) + frame * 160
            right = left + 400
            known = np.ones(400, bool) if real_bed else np.zeros(400, bool)
            truth = np.zeros(400, np.float32)
            for event in events:
                a = max(left, round(float(event["start"]) * 16000))
                b = min(right, round(float(event["end"]) * 16000))
                if b > a and event.get("label") in (0, 1, "0", "1"):
                    known[a-left:b-left] = True
                    truth[a-left:b-left] = float(event["label"])
            if known.all():
                targets[index, frame] = truth.mean()
    return targets, mask


def select_batch(operation, candidates: tuple[int, ...], output: Path, deadline: float) -> int:
    """Measure real encoder work; require memory headroom and near-best throughput."""
    measurements = []
    for size in candidates:
        check_deadline(deadline)
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        try:
            operation(size)
            torch.cuda.synchronize()
            began = time.monotonic()
            for _ in range(2):
                operation(size)
            torch.cuda.synchronize()
            elapsed = time.monotonic() - began
            peak = torch.cuda.max_memory_allocated()
            measurements.append({"batch": size, "seconds": elapsed / 2,
                                 "windows_per_second": 2 * size / elapsed,
                                 "peak_allocated_bytes": peak, "status": "ok"})
        except torch.cuda.OutOfMemoryError:
            measurements.append({"batch": size, "status": "oom"})
            gc.collect()
            torch.cuda.empty_cache()
    limit = .8 * torch.cuda.get_device_properties(0).total_memory
    safe = [item for item in measurements if item["status"] == "ok" and item["peak_allocated_bytes"] <= limit]
    if not safe:
        raise RuntimeError("No encoder batch fits the measured VRAM safety margin")
    fastest = max(item["windows_per_second"] for item in safe)
    chosen = max(item["batch"] for item in safe if item["windows_per_second"] >= .9 * fastest)
    atomic_json(output, {"measurements": measurements, "chosen": chosen,
                         "policy": "largest within 90% fastest throughput and 80% VRAM"})
    return chosen


def prepare(args: argparse.Namespace) -> None:
    import librosa
    from transformers import AutoFeatureExtractor, AutoModel
    from ast_multitask import load_processor
    from localization_models import ASTLocalizer

    began = time.time()
    torch.set_num_threads(4)
    torch.manual_seed(41)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    gpu_name = getattr(args, 'gpu_name', 'A100')
    if not torch.cuda.is_available() or gpu_name not in torch.cuda.get_device_name(0):
        raise RuntimeError(f"The authorized {gpu_name} is required")
    cache, output = args.cache_dir, args.output_dir
    for folder in (cache, output, cache / "stems", cache / "features", cache / "prefix"):
        folder.mkdir(parents=True, exist_ok=True)
    if (cache / "index.json").exists():
        raise RuntimeError("Refuse unverified cache reuse; keep this run's original identity")
    with (args.dataset_dir / "manifest.csv").open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    if len({row["ID"] for row in rows}) != len(rows):
        raise ValueError("Duplicate dataset IDs")
    for row in rows:
        path = (args.dataset_dir / row["path"]).resolve()
        if not path.is_relative_to(args.dataset_dir.resolve()) or sha256(path) != row["audio_sha256"]:
            raise ValueError(f"Dataset audio identity changed: {row['ID']}")
    identity = {"manifest_sha256": sha256(args.dataset_dir / "manifest.csv"),
                "source_sha256": sha256(Path(__file__)), "torch": torch.__version__,
                "versions": {name: importlib.metadata.version(name) for name in
                             ("transformers", "torchaudio", "librosa", "numpy", "demucs", "soundfile")},
                "implementation_sha256": {name: sha256(Path(__file__).parent / name) for name in
                                          ("localization_models.py", "ast_multitask.py", "mert12_submission.py")},
                "assets_sha256": {name: sha256(args.assets / name) for name in
                                  ("baseline/script.py", "mert/config.json", "mert/modeling_MERT.py",
                                   "mert/preprocessor_config.json", "mert/pytorch_model.bin")},
                "ast_assets_sha256": {name: sha256(args.ast_assets / name) for name in
                                      ("config.json", "preprocessor_config.json", "model.safetensors")},
                "mert_dtype": "float32", "ast_compute": "bfloat16 autocast",
                "ast_cache_dtype": "float16 (lossy)", "window": WINDOW, "hop": HOP,
                "demucs_decoder": "torchaudio.load+demucs.audio.convert_audio",
                "frame_truth": "25ms waveform fake fraction, unknown frames excluded",
                "ast_padding_attention": "native attention includes right padding; pooling/loss exclude it"}
    asset_manifest = args.dataset_dir.parent / "asset_hashes.json"
    expected_assets = json.loads(asset_manifest.read_text())
    for name, expected in expected_assets.items():
        path = (args.assets / name).resolve()
        if not path.is_relative_to(args.assets.resolve()) or sha256(path) != expected:
            raise ValueError(f"Frozen branch asset changed: {name}")
    identity["all_frozen_assets_sha256"] = expected_assets
    atomic_json(output / "feature_identity.json", identity)
    base = baseline_module(args.assets)
    pin_torchaudio_loader(base)
    device = torch.device("cuda")
    separator = base.load_htdemucs_model().to(device).eval().requires_grad_(False)
    records, index = [], {"identity": identity, "items": {}}
    timings = {}
    stage_started = time.monotonic()
    for number, row in enumerate(rows, 1):
        check_deadline(args.deadline_epoch)
        raw = base.load_audio(args.dataset_dir / row["path"])
        with torch.inference_mode():
            voice, backing = base.separate_voice_and_music(args.dataset_dir / row["path"], separator, device)
        backing = np.pad(backing[:len(raw)], (0, max(0, len(raw) - len(backing))))
        voice = np.pad(voice[:len(raw)], (0, max(0, len(raw) - len(voice))))
        starts = window_layout(len(raw))
        valid = np.minimum(WINDOW, len(raw) - starts)
        targets, masks = frame_truth(row, starts, valid)
        np.savez(cache / "stems" / f"{row['ID']}.npz", backing=backing, voice=voice)
        np.savez(cache / "features" / f"{row['ID']}.npz", starts=starts, valid_samples=valid,
                 frame_targets=targets, frame_mask=masks)
        prefix = np.lib.format.open_memmap(cache / "prefix" / f"{row['ID']}.npy", mode="w+",
                                           dtype=np.float16, shape=(len(starts), 1214, 768))
        del prefix
        records.extend((row["ID"], i, int(start)) for i, start in enumerate(starts))
        index["items"][row["ID"]] = {"features": f"features/{row['ID']}.npz",
                                    "ast_prefix": f"prefix/{row['ID']}.npy"}
        if number % 20 == 0 or number == len(rows):
            atomic_json(output / "feature_progress.json", {"stage": "demucs", "done": number,
                        "total": len(rows), "elapsed_seconds": time.time() - began})
            print(f"Demucs {number}/{len(rows)}", flush=True)
    timings['demucs'] = time.monotonic() - stage_started
    del separator, raw, voice, backing
    gc.collect()
    torch.cuda.empty_cache()

    def audio_batch(selected):
        loaded = {}
        result = np.zeros((len(selected), WINDOW), np.float32)
        for i, (identifier, _, start) in enumerate(selected):
            if identifier not in loaded:
                with np.load(cache / "stems" / f"{identifier}.npz", allow_pickle=False) as saved:
                    loaded[identifier] = saved["backing"]
            piece = loaded[identifier][start:start + WINDOW]
            result[i, :len(piece)] = piece
        return result

    def save_arrays(selected, values, key):
        grouped = {}
        for (identifier, i, _), value in zip(selected, values, strict=True):
            grouped.setdefault(identifier, []).append((i, value))
        for identifier, items in grouped.items():
            path = cache / index["items"][identifier]["features"]
            with np.load(path, allow_pickle=False) as saved:
                arrays = dict(saved)
            arrays.setdefault(key, np.full((len(arrays["starts"]), values.shape[-1]), np.nan, np.float32))
            for i, value in items:
                arrays[key][i] = value
            np.savez(path, **arrays)

    sample = audio_batch(records[:256])
    if len(sample) < 256:
        sample = sample[np.arange(256) % len(sample)]
    processor = AutoFeatureExtractor.from_pretrained(args.assets / "mert", local_files_only=True)
    mert, loading = AutoModel.from_pretrained(args.assets / "mert", trust_remote_code=True,
                                             local_files_only=True, output_loading_info=True)
    if any(loading.values()):
        raise RuntimeError(f"MERT loading mismatch: {loading}")
    mert.to(device).eval().requires_grad_(False)
    def mert_batch(audio):
        resampled = [librosa.resample(item, orig_sr=16000, target_sr=24000, res_type="soxr_hq") for item in audio]
        values = processor(resampled, sampling_rate=24000, return_tensors="pt", padding=True).input_values.to(device)
        with torch.inference_mode():
            return mert(values, output_hidden_states=True).hidden_states[12].mean(1).float().cpu().numpy()
    batch = select_batch(lambda n: mert_batch(sample[:n]), (32, 64, 128, 256), output / "mert_batch_benchmark.json", args.deadline_epoch)
    stage_started = time.monotonic()
    for offset in range(0, len(records), batch):
        check_deadline(args.deadline_epoch)
        selected = records[offset:offset+batch]
        save_arrays(selected, mert_batch(audio_batch(selected)), "mert_windows")
        atomic_json(output / "feature_progress.json", {"stage": "mert", "done": min(offset+batch, len(records)),
                    "total": len(records), "batch": batch, "elapsed_seconds": time.time() - began})
    timings['mert'] = time.monotonic() - stage_started
    del mert, processor
    gc.collect()
    torch.cuda.empty_cache()
    reference_path = args.dataset_dir.parent / "mert_reference.npz"
    if reference_path.is_file():
        comparisons = {}
        with np.load(reference_path, allow_pickle=False) as reference:
            for identifier in reference.files:
                with np.load(cache / index["items"][identifier]["features"], allow_pickle=False) as saved:
                    actual = saved["mert_windows"]
                expected = reference[identifier]
                if actual.shape != expected.shape:
                    raise ValueError(f"Historical MERT window alignment changed: {identifier}")
                comparisons[identifier] = {"max_abs": float(np.abs(actual-expected).max()),
                                           "mean_abs": float(np.abs(actual-expected).mean())}
        atomic_json(output / "mert_historical_feature_replay.json", {
            "reference_sha256": sha256(reference_path), "files": comparisons,
            "interpretation": "cross-runtime diagnostic; training uses one shared current extraction"})

    ast = ASTLocalizer.from_pretrained(args.ast_assets).to(device).eval()
    processor = load_processor(args.ast_assets)
    def ast_batch(audio):
        values = processor([item for item in audio], sampling_rate=16000, return_tensors="pt").input_values.to(device)
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            hidden = ast.encode_prefix(values)
            identity["ast_prefix_actual_dtype"] = str(hidden.dtype)
            return hidden.to(torch.float16).cpu().numpy()
    batch = select_batch(lambda n: ast_batch(sample[:n]), (32, 64, 128, 256), output / "ast_prefix_batch_benchmark.json", args.deadline_epoch)
    stage_started = time.monotonic()
    for offset in range(0, len(records), batch):
        check_deadline(args.deadline_epoch)
        selected = records[offset:offset+batch]
        encoded = ast_batch(audio_batch(selected))
        for (identifier, i, _), value in zip(selected, encoded, strict=True):
            mapped = np.load(cache / index["items"][identifier]["ast_prefix"], mmap_mode="r+")
            mapped[i] = value
            mapped.flush()
            del mapped
        atomic_json(output / "feature_progress.json", {"stage": "ast_prefix", "done": min(offset+batch, len(records)),
                    "total": len(records), "batch": batch, "elapsed_seconds": time.time() - began})
    timings['ast_prefix'] = time.monotonic() - stage_started
    del ast, processor, encoded
    gc.collect()
    torch.cuda.empty_cache()

    # The fixed voice/presence branch is evaluated separately; it does not train any candidate.
    stage_started = time.monotonic()
    panns, voice_indices, music_indices = base.load_panns_model(device)
    panns.model.eval().requires_grad_(False)
    evaluation = [row for row in rows if row["split"] != "train"]
    frozen = {}
    for number, row in enumerate(evaluation, 1):
        check_deadline(args.deadline_epoch)
        with torch.inference_mode():
            vp, mp = base.predict_presence(panns, voice_indices, music_indices,
                                           base.load_audio(args.dataset_dir / row["path"]))
        frozen[row["ID"]] = {"VOICE_PRESENT_PROB": vp, "MUSIC_PRESENT_PROB": mp}
    del panns
    gc.collect()
    torch.cuda.empty_cache()
    df, fake_index = base.load_df_arena_model(device)
    df.eval().requires_grad_(False)
    for number, row in enumerate(evaluation, 1):
        check_deadline(args.deadline_epoch)
        with np.load(cache / "stems" / f"{row['ID']}.npz", allow_pickle=False) as saved:
            vocal = saved["voice"]
        frozen[row["ID"]]["VOICE_FAKE_PROB"] = base.predict_fake(df, fake_index, vocal, device)
        path = cache / index["items"][row["ID"]]["features"]
        with np.load(path, allow_pickle=False) as saved:
            arrays = dict(saved)
        np.savez(path, **arrays, **frozen[row["ID"]])
        if number % 20 == 0 or number == len(evaluation):
            atomic_json(output / "feature_progress.json", {"stage": "fixed_voice", "done": number,
                        "total": len(evaluation), "elapsed_seconds": time.time() - began})
    del df
    timings['fixed_branches'] = time.monotonic() - stage_started
    gc.collect()
    torch.cuda.empty_cache()
    for row in rows:
        record = index["items"][row["ID"]]
        with np.load(cache / record["features"], allow_pickle=False) as saved:
            if not np.isfinite(saved["mert_windows"]).all():
                raise ValueError(f"Unfilled MERT feature cache: {row['ID']}")
        record["features_sha256"] = sha256(cache / record["features"])
        record["ast_prefix_sha256"] = sha256(cache / record["ast_prefix"])
    index["elapsed_seconds"] = time.time() - began
    index['stage_seconds'] = timings
    index['fixed_branch_rows'] = len(evaluation)
    index["windows"] = len(records)
    atomic_json(output / "feature_identity.json", identity)
    atomic_json(cache / "index.json", index)
    atomic_json(output / "feature_cache_receipt.json", index)
    atomic_json(output / "feature_progress.json", {"stage": "completed", "files": len(rows),
                "windows": len(records), "elapsed_seconds": time.time() - began})
    print(f"Features complete: {len(rows)} files, {len(records)} windows", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("dataset-dir", "assets", "ast-assets", "cache-dir", "output-dir"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--deadline-epoch", type=float, required=True)
    parser.add_argument("--gpu-name", choices=("A100", "H100"), default="A100")
    prepare(parser.parse_args())


if __name__ == "__main__":
    main()
