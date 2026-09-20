"""Continue the six-epoch AST experiment with unchanged data and optimizer state."""
import argparse
import gc
import json
import os
import shutil
import time
import zipfile
from pathlib import Path

import torch

import train_v2
from ast_multitask import load_checkpoint
from graph_features import atomic_json, digest
from graph_scoring import PREDICTION_COLUMNS, official_metrics
from package_hybrid_submission import route_predictions
from package_submission import verify_packaged
from package_v2_submission import build_ast_submission
from run_experiment import read_csv, write_csv
from run_v2 import checkpoint_hashes, compare_fresh
from v2_scoring import select_candidate


def prediction_delta(reference, actual):
    if [r["ID"] for r in reference] != [r["ID"] for r in actual]:
        raise ValueError("prediction ID order changed")
    return max(abs(float(a[c]) - float(b[c])) for a, b in zip(reference, actual, strict=True)
               for c in PREDICTION_COLUMNS)


def run(primary: Path, data: Path, work: Path, output: Path, deadline: float):
    train_v2.seed_everything()
    train_v2.RESERVE_SECONDS = 300
    old_report = json.loads((primary / "training/training_report.json").read_text())["candidates"]["AST_LAST_BLOCK"]
    source = primary / "training/AST_LAST_BLOCK"
    checkpoint = work / "initial"
    shutil.copytree(source / "best", checkpoint)
    original_lock = json.loads((primary / "candidate_lock.json").read_text())
    expected = original_lock["checkpoints"]["AST_LAST_BLOCK"]
    assert checkpoint_hashes(checkpoint) == expected, "six-epoch checkpoint changed"
    manifest = data / "development/manifest.csv"
    assert digest(manifest) == old_report["config"]["manifest_sha256"]
    assert digest(data / "locked_test/manifest.csv") == original_lock["test_manifest_sha256"]
    old_config = old_report["config"]
    new_config = {**old_config, "epochs": 30, "patience": 5}
    train_v2.CONFIGS["AST_LAST_BLOCK"] = new_config
    training = output / "training"
    destination = training / "AST_LAST_BLOCK"
    train_v2.extend_checkpoint(source / "last.pt", destination / "last.pt", old_config, new_config)
    shutil.copy2(source / "best.pt", destination / "best.pt")
    protocol = {"start_unix": time.time(), "source_last_sha256": digest(source / "last.pt"),
                "source_checkpoint": expected, "old_config": old_config, "new_config": new_config,
                "source_epoch": old_report["epochs_completed"], "data_split": [633, 122, 176, 98],
                "selection": "development only; strict improvement, patience 5",
                "evaluation_scope": "previously observed validation; not a new untouched test",
                "architecture_learning_rate_augmentation_unchanged": True}
    atomic_json(output / "extension_protocol.json", protocol)
    model, processor = load_checkpoint(checkpoint, "last_block")
    model.to("cuda")
    rows = train_v2.read_development_manifest(manifest, data)
    dev = [r for r in rows if r["split"] == "dev_select"]
    reference = read_csv(primary / "AST_LAST_BLOCK_dev_predictions.csv")
    initial = train_v2.predict_rows(model, processor, dev, data, torch.device("cuda"), deadline=deadline)
    delta = prediction_delta(reference, initial)
    atomic_json(output / "resume_prediction_parity.json", {"max_delta": delta, "threshold": 1e-5})
    assert delta <= 1e-5, f"six-epoch resume differs: {delta}"
    print("RESUME_VERIFIED", delta, flush=True)
    model, result = train_v2.train_candidate("AST_LAST_BLOCK", model, processor, rows, data,
        training, deadline, digest(manifest), old_config["asset_model_sha256"])
    atomic_json(output / "training_report.json", result)
    g0_dev = read_csv(primary / "G0_dev_predictions.csv")
    selected = select_candidate(dev, {"G0": g0_dev, "AST_LAST_BLOCK": result["predictions"]})
    atomic_json(output / "selection.json", selected)
    lock = {"selected": selected["selected"], "selected_unix": time.time(),
            "best_epoch": result["best_epoch"], "checkpoint": checkpoint_hashes(Path(result["checkpoint"])),
            "selection_sha256": digest(output / "selection.json"),
            "test_manifest_sha256": digest(data / "locked_test/manifest.csv"),
            "scope": "epoch extension; original primary decision unchanged"}
    atomic_json(output / "candidate_lock.json", lock)
    test = read_csv(data / "locked_test/manifest.csv")
    predicted = train_v2.predict_rows(model, processor, test, data, torch.device("cuda"), deadline=deadline)
    write_csv(output / "AST_extended_test_predictions.csv", predicted)
    g0 = read_csv(primary / "G0_test_predictions.csv")
    old = read_csv(primary / "AST_LAST_BLOCK_test_predictions.csv")
    hybrid = route_predictions(g0, predicted)
    write_csv(output / "hybrid_extended_test_predictions.csv", hybrid)
    comparison = {"development": {"G0": official_metrics(dev, g0_dev),
        "AST_6": official_metrics(dev, reference), "AST_extended": official_metrics(dev, result["predictions"])},
        "reused_evaluation": compare_fresh(test, {"G0": g0, "AST_6": old,
            "AST_extended": predicted, "hybrid_6": route_predictions(g0, old), "hybrid_extended": hybrid}),
        "scope": "partial labels; reused validation, not leaderboard or untouched evaluation"}
    atomic_json(output / "comparison.json", comparison)
    inputs = work / "package_inputs"
    shutil.copytree(primary / "package_inputs", inputs)
    sample_rows = [{"ID": r["ID"], "path": "test/" + r["ID"] + ".wav"}
                   for r in read_csv(inputs / "sample_submission.csv")]
    expected_predictions = train_v2.predict_rows(model, processor, sample_rows, inputs,
                                               torch.device("cuda"), deadline=deadline)
    del model
    gc.collect()
    torch.cuda.empty_cache()
    package = work / "AST_extended_submission.zip"
    build_ast_submission(Path(result["checkpoint"]), package)
    unpacked = work / "package"
    with zipfile.ZipFile(package) as archive:
        archive.extractall(unpacked)
    shutil.copytree(inputs, unpacked / "data")
    guard = work / "network_guard"
    guard.mkdir()
    (guard / "sitecustomize.py").write_text("import socket\ndef deny(*a,**kw):\n    raise RuntimeError('offline replay forbids network')\nsocket.socket.connect=deny\nsocket.create_connection=deny\n")
    previous = os.environ.get("PYTHONPATH")
    os.environ["PYTHONPATH"] = str(guard)
    started = time.perf_counter()
    try:
        packaged = verify_packaged(unpacked, deadline - 60)
    finally:
        if previous is None:
            os.environ.pop("PYTHONPATH", None)
        else:
            os.environ["PYTHONPATH"] = previous
    delta = prediction_delta(expected_predictions, packaged)
    assert delta <= 1e-5, f"extended package differs: {delta}"
    shutil.copy2(package, output / package.name)
    assert digest(package) == digest(output / package.name)
    write_csv(output / "packaged_reference.csv", expected_predictions)
    write_csv(output / "packaged_predictions.csv", packaged)
    shutil.copy2(unpacked / "output/packaged_inference.log", output / "packaged_inference.log")
    atomic_json(output / "packaged_verification.json", {"max_delta": delta, "files": len(packaged),
        "network_blocked": True, "stress_60s_included": True, "seconds_including_drive_copy": time.perf_counter()-started,
        "zip_sha256": digest(package), "zip_bytes": package.stat().st_size, "L4_runtime": "unverified"})
    atomic_json(output / "finished.json", {"status": "finished", "best_epoch": result["best_epoch"],
        "epochs_completed": result["epochs_completed"], "best_dev_proxy": result["best_proxy"],
        "submission_status": "not_submitted"})
    print("EXTENSION_FINISHED", result["epochs_completed"], result["best_epoch"], result["best_proxy"], flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    for name in ("primary", "data", "work", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--deadline", type=float, required=True)
    args = parser.parse_args()
    run(args.primary, args.data, args.work, args.output, args.deadline)
