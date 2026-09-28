#!/usr/bin/env python3
"""Run only the receipt-bound, matched P1/P3 causal recheck."""

from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
import time
from dataclasses import replace
from pathlib import Path

import torch

from v4_batches import music_supervision_rows
from v4_data import load_manifest
from v4_model import load_model, parameter_groups
from v4_training_trials import TrialSpec, reset_training_seed, train_trial

SEED = 20_260_913
EPOCHS = 10
ENCODER_LR = 3e-5
HEAD_LR = 9e-5
PINNED_TRAIN_SHA256 = "df5c083d7deee6f5efa195c4fce2c8b4298f2e8ca743f2c7bce7f7129877c07f"
PINNED_DEV_SHA256 = "f78185faee0578a311eea420a1776256ccea0346d2152e77b2288db477c605d9"
SOURCE_FILES = tuple(Path(name) for name in ("recheck_v4_matched.py", "v4_model.py",
    "v4_model_components.py", "v4_training.py", "v4_training_data.py",
    "v4_training_trials.py", "v4_batches.py", "v4_data.py", "v4_evaluation.py",
    "graph_scoring.py", "v2_scoring.py", "recovery_sonics.py", "v2_data.py"))


def digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def atomic_json(path: Path, value: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def read_predictions(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def run(args: argparse.Namespace) -> dict:
    output: Path = args.output
    output.mkdir(parents=True, exist_ok=True)
    hard_deadline = min(args.deadline, args.allocation_start + 7_080)
    if time.time() >= hard_deadline:
        raise TimeoutError("matched recheck has no original allocation time remaining")
    if digest(args.g0) != args.g0_sha256:
        raise ValueError("G0 prediction SHA-256 does not match the pinned receipt")
    if digest(args.data / "train.csv") != PINNED_TRAIN_SHA256:
        raise ValueError("train.csv SHA-256 does not match the audited 1,116-row supervision source")
    if digest(args.data / "dev.csv") != PINNED_DEV_SHA256:
        raise ValueError("dev.csv SHA-256 does not match the audited 210-row evaluation source")
    train_rows = load_manifest(args.data, "train")
    dev_rows = load_manifest(args.data, "dev")
    eligible = music_supervision_rows(train_rows)
    if len(eligible) != 1_116 or len(dev_rows) != 210:
        raise ValueError("matched recheck requires exactly 1,116 eligible train and 210 dev rows")
    g0 = read_predictions(args.g0)
    if [row["ID"] for row in g0] != [row["ID"] for row in dev_rows]:
        raise ValueError("G0 IDs do not exactly match dev.csv order")
    model_asset = args.sonics / "pytorch_model.bin"
    contract = {"purpose": "fresh_matched_P1_P3_only", "legacy_p3_causal_eligible": False,
        "seed": SEED, "epochs_each": EPOCHS, "encoder_lr": ENCODER_LR, "head_lr": HEAD_LR,
        "scheduler": "one_epoch_warmup_then_30_epoch_cosine", "pooling": "mean_probability",
        "allocation_start_unix": args.allocation_start, "hard_deadline_unix": hard_deadline,
        "train_rows": len(train_rows), "eligible_train_rows": len(eligible), "dev_rows": len(dev_rows),
        "g0_sha256": args.g0_sha256, "train_csv_sha256": digest(args.data / "train.csv"),
        "dev_csv_sha256": digest(args.data / "dev.csv"), "model_sha256": digest(model_asset),
        "backend_determinism": {"deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
            "cudnn_deterministic": torch.backends.cudnn.deterministic,
            "cudnn_benchmark": torch.backends.cudnn.benchmark,
            "limitation": "shared seed/RNG/init controls pairing; bitwise CUDA equivalence is not claimed"},
        "source_hashes": {str(path): digest(path) for path in SOURCE_FILES}}
    atomic_json(output / "remedy_contract.json", contract)
    results = {}
    for name, augmented in (("P1", False), ("P3", True)):
        reset_training_seed(SEED)
        loaded = load_model("sonics", args.sonics, "cuda" if torch.cuda.is_available() else "cpu")
        groups = parameter_groups(loaded)
        spec = replace(TrialSpec(name, "sonics", ENCODER_LR, HEAD_LR, EPOCHS,
                                 p3=augmented, patience=0, seed=SEED),
                       contract_hash=digest(output / "remedy_contract.json"))
        results[name] = train_trial(loaded, (groups.encoder, groups.head), spec, train_rows,
            dev_rows, g0, args.data, output / name, hard_deadline)
        del loaded
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    receipts = {name: json.loads((output / name / "trial_receipt.json").read_text())
                for name in ("P1", "P3")}
    matched = (all(results[name].get("status") == "completed" and
                   int(results[name].get("epochs_completed", 0)) == EPOCHS for name in results)
        and results["P1"]["run_total_updates"] == results["P3"]["run_total_updates"]
        and results["P1"]["run_total_window_exposures"] == results["P3"]["run_total_window_exposures"]
        and receipts["P1"]["seed_rng_sha256"] == receipts["P3"]["seed_rng_sha256"]
        and receipts["P1"]["initial_weights_sha256"] == receipts["P3"]["initial_weights_sha256"])
    comparison = {"matched_complete": matched, "legacy_gpu_P3_excluded": True,
        "P1": results["P1"], "P3": results["P3"], "receipts": receipts}
    atomic_json(output / "matched_comparison.json", comparison)
    if not matched:
        raise RuntimeError("fresh P1/P3 arms did not satisfy the matched causal contract")
    return comparison


def main() -> None:
    parser = argparse.ArgumentParser()
    for name in ("data", "sonics", "g0", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--g0-sha256", required=True)
    parser.add_argument("--allocation-start", type=float, required=True)
    parser.add_argument("--deadline", type=float, required=True)
    args = parser.parse_args()
    try:
        run(args)
    except Exception:
        args.output.mkdir(parents=True, exist_ok=True)
        (args.output / "failure.txt").write_text(__import__("traceback").format_exc(), encoding="utf-8")
        raise


if __name__ == "__main__":
    main()
