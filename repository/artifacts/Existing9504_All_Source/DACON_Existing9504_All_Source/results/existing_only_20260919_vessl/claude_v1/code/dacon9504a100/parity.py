"""Raw-audio reproduction of cached features and locked predictions (after the test lock).

A fresh process re-decodes a small representative set of original audio files
copied to a separate directory, recomputes the frozen-encoder features with the
same Encoders.process, and scores them with the locked heads.  It then compares
  * recomputed features vs the full cache (and pilot-cache vs full-cache features),
  * recomputed five-score predictions vs the val/test prediction CSVs.
These samples (audio + npz + expected predictions) go into the handoff so a later
packaging step can prove offline parity.  Nothing here selects or changes a model.
"""
from __future__ import annotations

import argparse
import csv
import shutil
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from full_bank_features import row_identity

from .common import PipelineError, StageLock, atomic_json, copy_with_hash, now_iso, read_json, safe_name, sha256_file
from .constants import SEEDS
from .dataset import read_manifest
from .evaluation import load_cached, load_heads, score_arrays, verify_lock
from .features import Encoders, compare_arrays, duration_of, identity_key, save_npz, stratum
from .gpu import configure_numerics, require_device

FEATURE_TOLERANCE = {"mert_windows": 1e-3, "voice_embeddings": 1e-3, "presence": 1e-4, "voice_original": 1e-4,
                     "window_truth": 0.0, "starts": 0.0, "valid_samples": 0.0, "samples": 0.0,
                     "normalization_gain": 1e-7, "source_peak": 1e-7, "normalized_rms": 1e-6, "voice_silent": 0.0}
PREDICTION_TOLERANCE = 1e-3
PREDICTION_COLUMNS = ("FILE_FAKE_PROB", "VOICE_FAKE_PROB", "MUSIC_FAKE_PROB", "VOICE_PRESENT_PROB",
                      "MUSIC_PRESENT_PROB", *(f"MUSIC_FAKE_PROB_seed{s}" for s in SEEDS),
                      *(f"VOICE_FAKE_PROB_seed{s}" for s in SEEDS), "VOICE_FAKE_PROB_df_arena_original_fc5")


def select_parity_rows(rows: Sequence[Mapping[str, str]], pilot_ids: Sequence[str], max_seconds: float) -> list[Mapping[str, str]]:
    """One short-to-median row per (split, stratum) for val/test, plus one pilot row per stratum."""
    chosen: dict[str, Mapping[str, str]] = {}
    for split in ("val", "test"):
        members = [r for r in rows if r["split"] == split and (duration_of(r) or 1e9) <= max_seconds]
        for name in sorted({stratum(r) for r in members}):
            group = sorted((r for r in members if stratum(r) == name), key=lambda r: (duration_of(r), r["ID"]))
            row = group[len(group) // 2]
            chosen[row["ID"]] = row
    pilot = [r for r in rows if r["ID"] in set(pilot_ids) and (duration_of(r) or 1e9) <= max_seconds]
    for name in sorted({stratum(r) for r in pilot}):
        group = sorted((r for r in pilot if stratum(r) == name), key=lambda r: (duration_of(r), r["ID"]))
        chosen[group[len(group) // 2]["ID"]] = group[len(group) // 2]
    return sorted(chosen.values(), key=lambda r: (r["split"], r["ID"]))


def status_for(report: Mapping[str, Any], tolerance: Mapping[str, float]) -> str:
    worst = "bitwise"
    for name, entry in report.items():
        if entry.get("status") != "compared":
            return "mismatch"
        if "equal" in entry and not entry["equal"]:
            return "mismatch"
        if "max_abs" in entry and not entry["bitwise_equal"]:
            if entry["max_abs"] > tolerance.get(name, 0.0):
                return "mismatch"
            worst = "within_tolerance"
    return worst


def read_predictions(path: Path) -> dict[str, dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return {row["ID"]: row for row in csv.DictReader(stream)}


def run_parity(config: Mapping[str, Any]) -> dict[str, Any]:
    device = config["device"]
    info = require_device(device, config["expected_gpu"])
    configure_numerics()
    lock = verify_lock(config)
    rows = read_manifest(Path(config["manifest"]))
    by_id = {row["ID"]: row for row in rows}
    cache_dir, pilot_dir = Path(config["cache_dir"]), Path(config["pilot_dir"])
    parity_dir = Path(config["parity_dir"])
    if parity_dir.exists():
        shutil.rmtree(parity_dir)
    (parity_dir / "recomputed").mkdir(parents=True)
    index = read_json(cache_dir / "index.json")
    key = identity_key(index["identity"])
    batch = int(index["identity"]["mert_batch"])
    pilot_index = read_json(pilot_dir / "cache" / "index.json")
    selected = select_parity_rows(rows, sorted(pilot_index["items"]), float(config.get("parity_max_seconds", 90)))
    data_copy = parity_dir / "data"
    for row in selected:
        copy_with_hash(Path(config["data_dir"]) / row["path"], data_copy / row["path"], expected_sha256=row["audio_sha256"])
    encoders = Encoders(Path(config["assets_dir"]), device)
    heads = load_heads(Path(config["train_dir"]), device, lock["checkpoints"])
    expected_predictions = {}
    for split in ("val", "test"):
        path = Path(config["eval_dir"]) / f"{split}_predictions.csv"
        if path.is_file():
            expected_predictions.update(read_predictions(path))
    samples = []
    for row in selected:
        arrays, timing = encoders.process(row, data_copy, parity_dir, batch, row_identity(row, key))
        name = safe_name(row["ID"])
        recomputed_path = parity_dir / "recomputed" / f"{name}.npz"
        save_npz(recomputed_path, arrays)
        cached, cached_sha = load_cached(cache_dir, row, index)
        feature_report = compare_arrays(arrays, cached)
        recomputed_scores = score_arrays(arrays, heads, device)
        reference = expected_predictions.get(row["ID"])
        if reference is None:  # train rows: score the cached features with the same locked heads
            reference = {k: str(v) for k, v in score_arrays(cached, heads, device).items()}
            reference_source = "locked heads on full-cache features"
        else:
            reference_source = f"{row['split']}_predictions.csv"
        differences = {column: abs(float(recomputed_scores[column]) - float(reference[column]))
                       for column in PREDICTION_COLUMNS}
        samples.append({"ID": row["ID"], "file_stem": name, "split": row["split"], "stratum": stratum(row),
                        "route": row["route"], "input_kind": row.get("input_kind", ""),
                        "audio_path": row["path"], "audio_sha256": row["audio_sha256"],
                        "duration_seconds": duration_of(row), "cache_npz_sha256": cached_sha,
                        "recomputed_npz_sha256": sha256_file(recomputed_path),
                        "feature_comparison": feature_report,
                        "feature_status": status_for(feature_report, FEATURE_TOLERANCE),
                        "recomputed_predictions": {c: float(recomputed_scores[c]) for c in PREDICTION_COLUMNS},
                        "reference_predictions": {c: float(reference[c]) for c in PREDICTION_COLUMNS},
                        "reference_source": reference_source,
                        "prediction_max_abs_difference": max(differences.values()),
                        "prediction_status": "bitwise" if max(differences.values()) == 0 else
                        ("within_tolerance" if max(differences.values()) <= PREDICTION_TOLERANCE else "mismatch"),
                        "timing": timing})
        print(f"parity {row['ID']}: features={samples[-1]['feature_status']} "
              f"predictions={samples[-1]['prediction_status']} max_diff={samples[-1]['prediction_max_abs_difference']:.3g}",
              flush=True)
    encoders.close()
    pilot_vs_full = []
    for identifier, entry in sorted(pilot_index["items"].items()):
        with np.load(pilot_dir / "cache" / entry["path"], allow_pickle=False) as data:
            pilot_arrays = {name: data[name] for name in data.files}
        cached, _ = load_cached(cache_dir, by_id[identifier], index)
        report = compare_arrays(pilot_arrays, cached)
        pilot_vs_full.append({"ID": identifier, "status": status_for(report, FEATURE_TOLERANCE), "comparison": report})
    statuses = [s["feature_status"] for s in samples] + [s["prediction_status"] for s in samples] + \
        [p["status"] for p in pilot_vs_full]
    overall = "mismatch" if "mismatch" in statuses else ("within_tolerance" if "within_tolerance" in statuses else "bitwise")
    receipt = {"status": overall, "device": info, "samples": samples, "pilot_vs_full": pilot_vs_full,
               "feature_tolerance": FEATURE_TOLERANCE, "prediction_tolerance": PREDICTION_TOLERANCE,
               "lock_sha256": sha256_file(Path(config["train_dir"]) / "candidate_lock.json"),
               "mert_batch": batch, "selection": "val/test median-duration row per stratum (<= parity_max_seconds) "
                                                 "+ one pilot row per stratum", "finished_at": now_iso()}
    atomic_json(parity_dir / "parity_receipt.json", receipt)
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    config = read_json(args.config)
    if config["device"] != "cuda" or config["expected_gpu"] != "A100":
        raise PipelineError("The CLI runs only on the authorized A100 (device=cuda)")
    with StageLock(Path(config["job_root"]) / "locks" / "parity.lock", "dacon9504a100.parity"):
        run_parity(config)


if __name__ == "__main__":
    main()
