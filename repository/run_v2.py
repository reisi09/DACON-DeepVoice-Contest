"""Run dev-only AST selection, then the sealed comparison and offline replay."""

import argparse
import gc
import importlib.metadata
import json
import shutil
import sys
import time
import zipfile
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

from ast_multitask import load_checkpoint
from graph_features import Extractor, atomic_json, digest, extract_cached
from graph_fusion import graph_predictions
from graph_scoring import SUBMISSION_COLUMNS, official_metrics
from package_submission import build_submission, verify_packaged
from package_v2_submission import build_ast_submission
from run_experiment import read_csv, write_csv
from train_v2 import predict_rows, read_development_manifest, train_candidates
from v2_inference import predict_audio_file
from v2_scoring import select_candidate


def checkpoint_hashes(directory: Path) -> dict[str, str]:
    """Bind selection to every byte needed by the exported inference model."""
    return {p.name: digest(p) for p in sorted(directory.iterdir()) if p.is_file()}


def lock_candidate(path: Path, record: dict) -> None:  # noqa: DICT_OK
    """A completed selection cannot be replaced during an interrupted run."""
    if path.exists():
        if json.loads(path.read_text()) != record:
            message = "A candidate lock already exists with different inputs or weights"
            raise ValueError(message)
        return
    atomic_json(path, record)


def open_locked_test(lock_path: Path, manifest: Path) -> list:
    """Read fresh labels only after verifying the development selection receipt."""
    lock = json.loads(lock_path.read_text())
    if lock["test_manifest_sha256"] != digest(manifest):
        message = "Locked test manifest changed after selection"
        raise ValueError(message)
    rows = read_csv(manifest)
    if not rows or any(r["split"] not in {"test_generator", "test_source"} for r in rows):
        message = "Locked evaluation requires fresh challenge roles only"
        raise ValueError(message)
    return rows


def baseline_predictions(baseline: Path, rows: list, data: Path, out: Path, deadline: float) -> list:
    extractor = Extractor(baseline / "script.py", include_raw=False)
    try:
        features = extract_cached(extractor, [data / r["path"] for r in rows], out / "g0_cache", deadline)
    finally:
        extractor.close()
    if len(features) != len(rows):
        message = "G0 comparison did not finish within the allocated deadline"
        raise TimeoutError(message)
    return [graph_predictions(feature)["G0"] for feature in features]


def compare_fresh(rows: list, predictions: dict) -> dict:  # noqa: DICT_OK
    report = {}
    for split in ("combined", "test_generator", "test_source"):
        chosen = rows if split == "combined" else [r for r in rows if r["split"] == split]
        report[split] = {}
        for name, predicted in predictions.items():
            by_id = {r["ID"]: r for r in predicted}
            report[split][name] = official_metrics(chosen, [by_id[r["ID"]] for r in chosen])
    return report


def package_replay(args, training: dict, selection: dict, deadline: float) -> dict:  # noqa: DICT_OK
    """Exercise fresh scripts on three real clips and a 60-second stress file."""
    data = args.output / "package_inputs"
    (data / "test").mkdir(parents=True, exist_ok=True)
    rows = read_csv(args.data / "development" / "manifest.csv")
    chosen = next(r for r in rows if r["split"] == "dev_select")
    audio, sr = sf.read(args.data / chosen["path"], dtype="float32")
    stress = np.tile(audio, int(np.ceil(60 * sr / len(audio))))[:60 * sr]
    sf.write(data / "test" / "stress_60s.wav", stress, sr)
    for p in sorted((args.baseline / "data" / "test").glob("*.wav")):
        shutil.copy2(p, data / "test" / p.name)
    paths = sorted((data / "test").glob("*.wav"))
    write_csv(data / "sample_submission.csv", [dict(zip(SUBMISSION_COLUMNS, [p.stem, 0, 0, 0, 0, 0])) for p in paths])
    selected = selection["selected"]
    names = ["AST_LAST_BLOCK"] + (["G0"] if selected == "G0" else [])
    if selected not in names:
        names.append(selected)
    receipts = {}
    for name in names:
        runtime_records = []
        destination = args.output / f"{name}_submission.zip"
        if name == "G0":
            build_submission(args.original_zip, destination, {"graph": "G0", "fusion": None})
            test_rows = [{"ID": p.stem, "path": str(p.relative_to(data))} for p in paths]
            reference = baseline_predictions(args.baseline, test_rows, data, args.output, deadline)
            runtime_records = [{k: v for k, v in json.loads(p.read_text()).items() if k != "feature"}
                               for p in (args.output / "g0_cache").glob("*.json")
                               if json.loads(p.read_text())["feature"]["ID"] == "stress_60s"]
        else:
            checkpoint = Path(training["candidates"][name]["checkpoint"])
            build_ast_submission(checkpoint, destination)
            mode = json.loads((checkpoint / "multitask_config.json").read_text())["mode"]
            model, processor = load_checkpoint(checkpoint, mode)
            model = model.to("cuda").eval()
            reference = []
            for p in paths:
                torch.cuda.reset_peak_memory_stats()
                torch.cuda.synchronize()
                started = time.perf_counter()
                probability = predict_audio_file(model, processor, p)
                torch.cuda.synchronize()
                reference.append(dict(zip(SUBMISSION_COLUMNS, [p.stem, *probability])))
                runtime_records.append({"ID": p.stem, "seconds": time.perf_counter() - started,
                    "max_allocated_gib": torch.cuda.max_memory_allocated() / 1024**3,
                    "max_reserved_gib": torch.cuda.max_memory_reserved() / 1024**3})
            del model, processor
            gc.collect()
            torch.cuda.empty_cache()
        fresh = args.work / f"unpacked_{name}"
        with zipfile.ZipFile(destination) as archive:
            archive.extractall(fresh)
        shutil.copytree(data, fresh / "data", dirs_exist_ok=True)
        start = time.perf_counter()
        packaged = verify_packaged(fresh, deadline)
        delta = max(abs(float(a[c]) - float(b[c])) for a, b in zip(reference, packaged, strict=True)
                    for c in SUBMISSION_COLUMNS[1:])
        if [r["ID"] for r in reference] != [r["ID"] for r in packaged] or delta > 1e-5:
            message = f"Fresh offline package parity failed: {name}, delta={delta}"
            raise AssertionError(message)
        shutil.copy2(fresh / "output" / "packaged_inference.log", args.output / f"{name}_package.log")
        write_csv(args.output / f"{name}_package_reference.csv", reference)
        receipts[name] = {"zip": str(destination), "zip_sha256": digest(destination),
            "max_probability_delta": delta, "script_seconds_including_model_load": time.perf_counter() - start,
            "stress_60s_included": True, "file_count": len(paths), "gpu": torch.cuda.get_device_name(),
            "reference_runtime": runtime_records, "L4_60_minute_compliance": "unverified"}
        shutil.rmtree(fresh)
        atomic_json(args.output / "packaged_verification.json", receipts)
    return receipts


def main() -> None:
    parser = argparse.ArgumentParser()
    for name in ("assets", "data", "baseline", "original-zip", "output", "work"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--deadline", type=float, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    start = time.time()
    if start >= args.deadline - 1200:
        message = "Insufficient GPU budget to begin the comparison"
        raise TimeoutError(message)
    atomic_json(args.output / "environment.json", {"started_unix": start, "python": sys.version,
        "gpu": torch.cuda.get_device_name(), "gpu_gib": torch.cuda.get_device_properties(0).total_memory / 1024**3,
        "packages": {k: importlib.metadata.version(k) for k in
                     ("torch", "torchaudio", "transformers", "numpy", "scikit-learn", "soundfile")}})
    development = args.data / "development" / "manifest.csv"
    rows = read_development_manifest(development, args.data)
    dev_select = [r for r in rows if r["split"] == "dev_select"]
    g0 = baseline_predictions(args.baseline, dev_select, args.data, args.output, args.deadline - 1200)
    write_csv(args.output / "G0_dev_predictions.csv", g0)
    training = train_candidates(args.assets, development, args.data, args.output / "training", args.deadline)
    if any("predictions" not in value for value in training["candidates"].values()):
        message = "The allocated budget did not permit all declared candidates to finish"
        raise TimeoutError(message)
    predictions = {"G0": g0, **{name: value["predictions"] for name, value in training["candidates"].items()}}
    selection = select_candidate(dev_select, predictions)
    atomic_json(args.output / "selection.json", selection)
    for name, value in predictions.items():
        write_csv(args.output / f"{name}_dev_predictions.csv", value)
    manifest = args.data / "locked_test" / "manifest.csv"
    lock = {"selected": selection["selected"], "development_manifest_sha256": digest(development),
        "test_manifest_sha256": digest(manifest), "original_zip_sha256": digest(args.original_zip),
        "checkpoints": {name: checkpoint_hashes(Path(value["checkpoint"])) for name, value in training["candidates"].items()},
        "selection_sha256": digest(args.output / "selection.json")}
    lock_candidate(args.output / "candidate_lock.json", lock)
    fresh_rows = open_locked_test(args.output / "candidate_lock.json", manifest)
    atomic_json(args.output / "test_opened.json", {"unix": time.time(), "candidate_lock_sha256": digest(args.output / "candidate_lock.json")})
    fresh = {"G0": baseline_predictions(args.baseline, fresh_rows, args.data, args.output, args.deadline - 600)}
    write_csv(args.output / "G0_test_predictions.csv", fresh["G0"])
    for name, value in training["candidates"].items():
        checkpoint = Path(value["checkpoint"])
        mode = json.loads((checkpoint / "multitask_config.json").read_text())["mode"]
        model, processor = load_checkpoint(checkpoint, mode)
        model = model.to("cuda").eval()
        fresh[name] = predict_rows(model, processor, fresh_rows, args.data, torch.device("cuda"), deadline=args.deadline - 600)
        write_csv(args.output / f"{name}_test_predictions.csv", fresh[name])
        del model, processor
        gc.collect()
        torch.cuda.empty_cache()
    comparison = {"selection": selection, "fresh_test": compare_fresh(fresh_rows, fresh),
        "scope": "external_source_disjoint_validation_not_DACON_leaderboard", "fresh_clips": len(fresh_rows)}
    atomic_json(args.output / "comparison.json", comparison)
    print("COMPARISON_FINISHED", json.dumps(comparison, ensure_ascii=False), flush=True)
    package_replay(args, training, selection, args.deadline - 120)
    atomic_json(args.output / "finished.json", {"status": "finished", "elapsed_seconds": time.time() - start,
        "selected": selection["selected"], "submission_status": "not_submitted_in_this_run"})
    print("V2_FINISHED", args.output, flush=True)


if __name__ == "__main__":
    main()
