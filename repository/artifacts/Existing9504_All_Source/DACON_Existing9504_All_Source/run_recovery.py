#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy", "soundfile", "torch"]
# ///
# How to run: uv run run_recovery.py --help
"""Run the bounded, inference-only Recovery V3 comparison."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from itertools import pairwise
from pathlib import Path
from typing import TypeVar

import numpy as np
import soundfile as sf
import torch

from graph_features import Extractor, atomic_json, digest
from recovery_fusion import CANDIDATES, candidate_predictions
from recovery_selection import decide_fresh, evaluate_old, lock_candidate
from recovery_sonics import SonicsPredictor

T = TypeVar("T")
RecoveryRecord = dict[str, str | dict]


@dataclass(frozen=True, slots=True)
class RecoveryError(ValueError):
    detail: str

    def __str__(self) -> str:
        return self.detail


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def write_csv(path: Path, rows: Sequence[Mapping[str, str | float]]) -> None:
    if not rows:
        raise RecoveryError(f"refusing empty CSV: {path}")
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def synchronized(device: torch.device, action: Callable[[], T]) -> tuple[T, float]:
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    started = time.perf_counter()
    value = action()
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    return value, time.perf_counter() - started


def runtime_signature(extractor: Extractor, baseline: Path, assets: Path) -> tuple[str, dict[str, str]]:
    files = [Path(__file__), Path("recovery_fusion.py"), Path("recovery_selection.py"),
             Path("recovery_sonics.py"), baseline / "script.py", assets / "config.json",
             assets / "pytorch_model.bin"]
    hashes = {str(path): digest(path) for path in files}
    vendor_candidates = (assets.parent / "sonics_vendor",
        Path("_workspace/recovery_v3/vendor/sonics-9156ffad151f797c71556923c4a02fa01fa8fc91"))
    vendor = next((path for path in vendor_candidates if path.is_dir()), None)
    if vendor is None:
        raise RecoveryError("pinned SONICS vendor source is missing")
    hashes.update({str(path): digest(path) for path in sorted(vendor.rglob("*.py"))})
    return hashlib.sha256((extractor.signature + json.dumps(hashes, sort_keys=True)).encode()).hexdigest(), hashes


def _g0(identifier: str, presence: np.ndarray, voice: Sequence[float], music: Sequence[float]) -> dict[str, str | float]:
    vp, mp = presence.max(axis=0)
    vf, mf = max(voice), max(music)
    return {"ID": identifier, "FILE_FAKE_PROB": round(max(vf * vp, mf * mp), 10),
            "VOICE_FAKE_PROB": round(vf, 10), "MUSIC_FAKE_PROB": round(mf, 10),
            "VOICE_PRESENT_PROB": round(float(vp), 10), "MUSIC_PRESENT_PROB": round(float(mp), 10)}


def extract_file(
    extractor: Extractor, sonics: SonicsPredictor, path: Path, cache: Path,
    signature: str, deadline: float, parity: bool = False,
) -> RecoveryRecord:
    """Compute one atomic record with exactly one Demucs separation."""
    source_hash = digest(path)
    target = cache / f"{hashlib.sha256((path.stem + source_hash + signature).encode()).hexdigest()}.json"
    if target.is_file():
        record = json.loads(target.read_text(encoding="utf-8"))
        required = {"signature", "source_sha256", "feature", "candidates", "timings"}
        if not required.issubset(record) or record["signature"] != signature or record["source_sha256"] != source_hash:
            raise RecoveryError(f"incomplete or stale recovery cache: {target}")
        if parity and "parity" not in record:
            raise RecoveryError(f"baseline parity missing from cached record: {target}")
        return record
    if time.time() >= deadline:
        raise TimeoutError(f"deadline reached before required file: {path.name}")
    base, device = extractor.base, extractor.device
    audio = base.load_audio(path)
    panns_value, panns_seconds = synchronized(device, lambda: extractor.panns.inference(base.make_panns_segments(audio)))
    panns_predictions = panns_value[0]
    presence = np.stack((panns_predictions[:, extractor.vi].max(1), panns_predictions[:, extractor.mi].max(1)), 1)
    stems, separation_seconds = synchronized(device, lambda: base.separate_voice_and_music(path, extractor.demucs, device))
    voice_audio, music_audio = stems
    voice_fake, voice_seconds = synchronized(device, lambda: extractor.fake_windows(voice_audio))
    music_fake, music_seconds = synchronized(device, lambda: extractor.fake_windows(music_audio))
    raw_result, raw_seconds = synchronized(device, lambda: sonics.predict(audio))
    stem_result, stem_seconds = synchronized(device, lambda: sonics.predict(music_audio))
    g0 = _g0(path.stem, presence, voice_fake, music_fake)
    candidates = candidate_predictions(g0, raw_result["mean_probability"], stem_result["mean_probability"])
    feature = {"ID": path.stem, "duration_seconds": audio.size / 16000, "presence": presence.tolist(),
               "voice_fake_windows": voice_fake, "music_fake_windows": music_fake,
               "sonics_raw": raw_result, "sonics_stem": stem_result}
    record = {"signature": signature, "source_sha256": source_hash, "feature": feature,
              "candidates": candidates, "timings": {"panns": panns_seconds, "separation": separation_seconds,
              "df_voice": voice_seconds, "df_music": music_seconds, "sonics_raw": raw_seconds,
              "sonics_stem": stem_seconds}}
    if parity:
        vp, mp = base.predict_presence(extractor.panns, extractor.vi, extractor.mi, audio)
        vf = base.predict_fake(extractor.df, extractor.fake_index, voice_audio, device)
        mf = base.predict_fake(extractor.df, extractor.fake_index, music_audio, device)
        expected = [base.combine_file_fake_score(vf, mf, vp, mp), vf, mf, vp, mp]
        actual = [float(g0[key]) for key in ("FILE_FAKE_PROB", "VOICE_FAKE_PROB", "MUSIC_FAKE_PROB",
                                             "VOICE_PRESENT_PROB", "MUSIC_PRESENT_PROB")]
        delta = max(abs(left - right) for left, right in zip(expected, actual, strict=True))
        if delta > 1e-5:
            raise AssertionError(f"G0 baseline parity failed for {path.name}: {delta}")
        record["parity"] = {"max_probability_delta": delta, "expected": expected, "actual": actual}
    atomic_json(target, record)
    return record


def manifest_paths(root: Path, rows: Sequence[Mapping[str, str]], expected: int | None = None) -> list[Path]:
    if expected is not None and len(rows) != expected:
        raise RecoveryError(f"required manifest coverage is {expected}, got {len(rows)}")
    if len({row["ID"] for row in rows}) != len(rows):
        raise RecoveryError("manifest IDs must be unique")
    paths = [root / row["path"] for row in rows]
    if any(path.stem != row["ID"] or not path.is_file() for path, row in zip(paths, rows, strict=True)):
        raise RecoveryError("manifest ID/path mismatch or missing audio")
    return paths


def predict_all(extractor: Extractor, sonics: SonicsPredictor, paths: Sequence[Path], cache: Path,
                signature: str, deadline: float, parity_ids: set[str] | None = None) -> tuple[list[dict], dict[str, list[dict]]]:
    records = []
    for index, path in enumerate(paths, 1):
        started = time.perf_counter()
        records.append(extract_file(extractor, sonics, path, cache, signature, deadline,
                                    parity=parity_ids is not None and path.stem in parity_ids))
        print(f"RECOVERY_FEATURE {index}/{len(paths)} {path.stem} {time.perf_counter() - started:.3f}s", flush=True)
    predictions = {name: [record["candidates"][name] for record in records] for name in CANDIDATES}
    return records, predictions


def main() -> None:
    parser = argparse.ArgumentParser()
    for name in ("data", "fresh", "baseline", "assets", "output"):
        parser.add_argument(f"--{name}", required=True, type=Path)
    parser.add_argument("--deadline", required=True, type=float)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    started = time.time()
    development_path = args.data / "development/manifest.csv"
    old_path = args.data / "locked_test/manifest.csv"
    dev = [row for row in read_csv(development_path) if row["split"] == "dev_select"]
    old = read_csv(old_path)
    if len(dev) != 122 or sum(row["split"] == "test_generator" for row in old) != 176 \
            or sum(row["split"] == "test_source" for row in old) != 98:
        raise RecoveryError("old evaluation requires dev_select=122, test_generator=176, test_source=98")
    old_paths = manifest_paths(args.data, [*dev, *old], 396)
    extractor = Extractor(args.baseline / "script.py", include_raw=False)
    try:
        sonics = SonicsPredictor(args.assets, str(extractor.device))
        signature, source_hashes = runtime_signature(extractor, args.baseline, args.assets)
        parity_ids = {path.stem for path in old_paths[:2]}
        old_records, old_predictions = predict_all(extractor, sonics, old_paths, args.output / "cache",
                                                    signature, args.deadline, parity_ids)
        atomic_json(args.output / "raw_features.json", [record["feature"] for record in old_records])
        for name in CANDIDATES:
            write_csv(args.output / f"{name}_predictions.csv", old_predictions[name])
        selection = evaluate_old(dev, old, old_predictions)
        atomic_json(args.output / "selection.json", selection)
        fresh_manifest = args.fresh / "manifest.csv"
        fresh_hash = digest(fresh_manifest)
        lock = {"selected": selection["selected"], "candidate_order": CANDIDATES,
                "formulas": {"music": "G0|max SONICS mean|0.5 blend", "file": "max(voice_present*voice_fake,music_present*music_fake)"},
                "thresholds": {"diagnostic_gain": .005, "eer_regression": .02, "auc_regression": .01,
                               "bootstrap_win_rate": .70, "bootstrap_p10": -.005, "bootstrap_resamples": 2000},
                "source_hashes": source_hashes, "development_manifest_sha256": digest(development_path),
                "old_manifest_sha256": digest(old_path), "fresh_manifest_sha256": fresh_hash,
                "selection_sha256": digest(args.output / "selection.json")}
        lock_hash = lock_candidate(args.output / "candidate_lock.json", lock)
        atomic_json(args.output / "candidate_lock_written.json", {"unix": time.time(),
                    "candidate_lock_sha256": lock_hash})
        if digest(fresh_manifest) != fresh_hash:
            raise RecoveryError("fresh manifest changed after candidate lock")
        atomic_json(args.output / "fresh_opened.json", {"unix": time.time(),
                    "candidate_lock_sha256": lock_hash, "fresh_manifest_sha256": fresh_hash})
        fresh_rows = read_csv(fresh_manifest)
        if any(row["split"] != "fresh_challenge" for row in fresh_rows):
            raise RecoveryError("fresh manifest contains a non-quality row")
        fresh_paths = manifest_paths(args.fresh, fresh_rows)
        durations = [sf.info(path).frames / sf.info(path).samplerate for path in fresh_paths]
        if any(seconds < 4 or seconds > 60.01 for seconds in durations):
            raise RecoveryError("fresh audio duration must be within 4-60 seconds")
        fresh_records, fresh_all = predict_all(extractor, sonics, fresh_paths, args.output / "fresh_cache",
                                                signature, args.deadline)
        atomic_json(args.output / "fresh_raw_features.json", [record["feature"] for record in fresh_records])
        selected = str(selection["selected"])
        fresh_predictions = {name: fresh_all[name] for name in (("G0",) if selected == "G0" else ("G0", selected))}
        for name, values in fresh_predictions.items():
            write_csv(args.output / f"{name}_fresh_predictions.csv", values)
        comparison = decide_fresh(args.output / "candidate_lock.json", fresh_manifest, fresh_rows, fresh_predictions)
        atomic_json(args.output / "comparison.json", comparison)
        stress_audio = extractor.base.load_audio(old_paths[0])
        stress_audio = np.tile(stress_audio, 960000 // stress_audio.size + 1)[:960000]
        stress_path = args.output / "stress_60s.wav"
        sf.write(stress_path, stress_audio, 16000)
        stress = extract_file(extractor, sonics, stress_path, args.output / "stress_cache", signature, args.deadline)
        first_audio, second_audio = extractor.base.load_audio(old_paths[0]), extractor.base.load_audio(old_paths[1])
        before = sonics.predict(first_audio)["mean_probability"]
        sonics.predict(second_audio)
        after = sonics.predict(first_audio)["mean_probability"]
        stress_trace = stress["feature"]["sonics_raw"]
        starts = stress_trace["starts"]
        continuous = starts[0] == 0 and all(right <= left + 80000 for left, right in pairwise(starts))
        if stress_trace["covered_samples"] != 960000 or starts[-1] + 80000 < 960000 or not continuous:
            raise AssertionError("60-second SONICS coverage is incomplete")
        if abs(before - after) > 1e-5:
            raise AssertionError("SONICS prediction changed with file order")
        coverage = {"old_complete": len(old_records) == 396, "fresh_complete": len(fresh_records) == len(fresh_rows),
                    "baseline_parity": [record["parity"] for record in old_records[:2]],
                    "stress_60s": stress_trace, "file_order_max_delta": abs(before - after),
                    "timings": {key: sum(record["timings"][key] for record in [*old_records, *fresh_records])
                                for key in old_records[0]["timings"]}}
        atomic_json(args.output / "coverage_timing.json", coverage)
        atomic_json(args.output / "finished.json", {"status": "finished", "elapsed_seconds": time.time() - started,
                    "candidate_lock_sha256": lock_hash, "development_selected": selection["development_selected"],
                    "old_veto_selected": selection["selected"], "final": comparison["final"],
                    "submission_status": "not_submitted"})
    finally:
        extractor.close()


if __name__ == "__main__":
    main()
