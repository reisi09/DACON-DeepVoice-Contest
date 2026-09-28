"""Train the unchanged WindowMIL (music) and DF-Arena fc5 (voice) heads; lock on val.

Derived from source/full_bank_train.py (``train``).  The optimisation is the
original one; ``forward``/``step``/``predict``/``benchmark`` and the batching,
epoch order and loss weights are imported unchanged from full_bank_train /
full_bank_batches.  Changes versus the original:
  * H100 identity check instead of H100; three-split manifest.
  * This process opens feature caches for train and val rows only; test rows are
    never read before candidate_lock.json exists (counts recorded in the lock).
  * Normalisation statistics and loss weights come from train rows only (as before).
  * After selection, val predictions for all five columns, val EER thresholds and
    the six checkpoint hashes are written into an immutable candidate lock.
  * If the lock already exists the stage verifies it and never retrains.
"""
from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import random
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

from full_bank_batches import Item, epoch_order, loss_weights
from full_bank_features import row_identity, validate_cache
from full_bank_train import benchmark, predict, step
from graph_scoring import _roc_metrics
from localization_models import WindowMIL

from .common import PipelineError, StageLock, atomic_json, canonical_sha, now_iso, read_json, sha256_file
from .constants import SEEDS
from .dataset import read_manifest, validate_manifest
from .evaluation import checkpoint_names, evaluate_split, run_identity_for, validation_thresholds, verify_lock
from .features import identity_key
from .fingerprint import stage_fingerprint
from .gpu import configure_numerics, memory, release, require_device


class SplitAccess:
    """Counts every feature file this process opens, per split (lock evidence)."""

    def __init__(self) -> None:
        self.counts = {"train": 0, "val": 0, "test": 0}

    def record(self, split: str) -> None:
        self.counts[split] += 1


def read_items(rows: Sequence[Mapping[str, str]], cache_dir: Path, index: Mapping[str, Any], splits: Sequence[str],
               access: SplitAccess) -> list[Item]:
    """Original full_bank_train.read_items semantics for the requested splits only."""
    if "test" in splits:
        raise PipelineError("Training never reads test features")
    key = identity_key(index["identity"])
    result = []
    for row in rows:
        if row["split"] not in splits:
            continue
        entry = index["items"][row["ID"]]
        path = (cache_dir / entry["path"]).resolve()
        if not path.is_relative_to(cache_dir.resolve()):
            raise PipelineError("Feature path escapes cache")
        validate_cache(path, row_identity(row, key), entry["sha256"])
        access.record(row["split"])
        with np.load(path, allow_pickle=False) as data:
            music = row["route"] == "music"
            features = data["mert_windows" if music else "voice_embeddings"].copy()
            truth = data["window_truth"].copy() if music else np.full(len(features), np.nan, np.float32)
            silent = bool(data["voice_silent"])
        if features.ndim != 2 or not len(features) or not np.isfinite(features).all():
            raise PipelineError(f"Invalid route features: {row['ID']}")
        result.append(Item(row, features, truth, float(row["MUSIC_FAKE" if music else "VOICE_FAKE"]), silent))
    return result


def capture_rng(device: str) -> dict[str, Any]:
    """Python, NumPy, Torch CPU and CUDA generator states in torch.load(weights_only=True)-safe types."""
    version, internal, gauss = random.getstate()
    name, keys, position, has_gauss, cached = np.random.get_state()
    return {"python": {"version": version, "internal": list(internal), "gauss": gauss},
            "numpy": {"name": name, "keys": torch.from_numpy(np.asarray(keys, dtype=np.int64)),
                      "position": int(position), "has_gauss": int(has_gauss), "cached_gaussian": float(cached)},
            "torch_cpu": torch.get_rng_state().clone(),
            "torch_cuda": [s.clone() for s in torch.cuda.get_rng_state_all()] if device == "cuda" else []}


def restore_rng(state: Mapping[str, Any], device: str) -> None:
    python = state["python"]
    random.setstate((python["version"], tuple(python["internal"]), python["gauss"]))
    numpy_state = state["numpy"]
    np.random.set_state((numpy_state["name"], numpy_state["keys"].numpy().astype(np.uint32),
                         numpy_state["position"], numpy_state["has_gauss"], numpy_state["cached_gaussian"]))
    torch.set_rng_state(state["torch_cpu"])
    if device == "cuda":
        torch.cuda.set_rng_state_all(state["torch_cuda"])


def rng_digest(state: Mapping[str, Any]) -> str:
    digest = hashlib.sha256()
    digest.update(np.asarray(state["python"]["internal"], dtype=np.int64).tobytes())
    digest.update(state["numpy"]["keys"].numpy().tobytes())
    digest.update(str(state["numpy"]["position"]).encode())
    digest.update(state["torch_cpu"].numpy().tobytes())
    for cuda_state in state["torch_cuda"]:
        digest.update(cuda_state.numpy().tobytes())
    return digest.hexdigest()


def train_route_seed(route: str, seed: int, training: list[Item], validation: list[Item], weights, mean, scale,
                     voice: Mapping[str, Any], run_identity: Mapping[str, Any], output: Path, config, device: str
                     ) -> tuple[np.ndarray, dict[str, Any]]:
    """Original optimisation; resumable at every epoch boundary with identical results.

    The resume file carries model, optimizer, early-stopping state, the measured
    microbatch/window limit (never re-benchmarked on resume, because a different
    grouping changes floating-point accumulation) and the Python/NumPy/Torch
    CPU/CUDA generator states at the end of the last completed epoch.
    """
    fake_index = voice["fake_label_index"]
    checkpoint = output / f"{route}_{seed}.pt"
    resume = output / f"{route}_{seed}_resume.pt"
    state = None
    if resume.exists():
        state = torch.load(resume, map_location="cpu", weights_only=True)
        if state["identity"] != run_identity:
            raise PipelineError("Training resume belongs to a different run")
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = WindowMIL(mean, scale) if route == "music" else nn.Linear(1280, 2)
    if route == "voice":
        model.load_state_dict(voice["state_dict"])
    model.to(device)
    if state is None:
        override = config.get("train_benchmark_override")
        if override:
            microbatch, window_limit, measured = int(override["microbatch"]), int(override["window_limit"]), []
        else:
            before = capture_rng(device)  # benchmarking must not shift any generator
            microbatch, window_limit, measured = benchmark(model, training, route, fake_index, weights, device)
            restore_rng(before, device)
        atomic_json(output / f"{route}_{seed}_batch.json", {"microbatch": microbatch, "window_limit": window_limit,
                    "effective_batch": 256, "measurements": measured, "device_memory": memory(device)})
    else:
        microbatch, window_limit = int(state["batch"]["microbatch"]), int(state["batch"]["window_limit"])
    measured_batch = {"microbatch": microbatch, "window_limit": window_limit}
    window_limit = max(window_limit, max(len(x.features) for x in validation))
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3 if route == "music" else 1e-4, weight_decay=.01)
    best, stale, first_epoch, history = (float("inf"), float("inf")), 0, 1, []
    if state is not None:
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        best, stale, first_epoch, history = tuple(state["best"]), state["stale"], state["epoch"] + 1, state["history"]
        restore_rng(state["rng"], device)
        print(f"{route} seed={seed}: resumed after epoch {state['epoch']} (rng {rng_digest(state['rng'])[:12]})",
              flush=True)
    labels = np.asarray([x.label for x in validation])
    for epoch in range(first_epoch, int(config["max_epochs"]) + 1):
        if stale >= int(config["patience"]):
            break
        model.train()
        order = epoch_order(training, seed, epoch)
        epoch_loss, began = 0., time.monotonic()
        for offset in range(0, len(order), 256):
            batch = [training[i] for i in order[offset:offset + 256]]
            epoch_loss += step(model, optimizer, batch, route, fake_index, weights, microbatch, window_limit,
                               device) * len(batch)
        predictions = predict(model, validation, route, fake_index, microbatch, window_limit, device)
        eer = float(_roc_metrics(labels.astype(int).tolist(), predictions.tolist())[0])
        clipped = np.clip(predictions, 1e-7, 1 - 1e-7)
        bce = float(-(labels * np.log(clipped) + (1 - labels) * np.log1p(-clipped)).mean())
        improved = (eer, bce) < best
        if improved:
            best, stale = (eer, bce), 0
            payload = {"model": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
                       "route": route, "seed": seed, "epoch": epoch, "validation_eer": eer, "validation_bce": bce,
                       "identity": run_identity, "fake_label_index": fake_index, "voice_module": voice["module"],
                       "train_ids": [x.row["ID"] for x in training]}
            torch.save(payload, checkpoint.with_suffix(".tmp"))
            checkpoint.with_suffix(".tmp").replace(checkpoint)
        else:
            stale += 1
        rng = capture_rng(device)
        record = {"epoch": epoch, "loss": epoch_loss / len(training), "validation_eer": eer, "validation_bce": bce,
                  "rows_seen": len(order), "unique_rows_seen": len(set(order.tolist())),
                  "order_sha256": hashlib.sha256(order.tobytes()).hexdigest(), "improved": improved,
                  "rng_sha256_after_epoch": rng_digest(rng), "epoch_seconds": time.monotonic() - began}
        history.append(record)
        torch.save({"model": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
                    "optimizer": optimizer.state_dict(), "identity": run_identity, "epoch": epoch, "best": list(best),
                    "stale": stale, "history": history, "rng": rng, "batch": measured_batch},
                   resume.with_suffix(".tmp"))
        resume.with_suffix(".tmp").replace(resume)
        atomic_json(output / "progress.json", {"stage": "train", "route": route, "seed": seed, **record,
                                               "updated_at": now_iso()})
        atomic_json(output / f"{route}_{seed}_history.json", history)
        print(f"{route} seed={seed} epoch={epoch} val_eer={eer:.6f} val_bce={bce:.6f} seen={len(order)}", flush=True)
    best_state = torch.load(checkpoint, map_location=device, weights_only=True)
    model.load_state_dict(best_state["model"])
    prediction = predict(model, validation, route, fake_index, microbatch, window_limit, device)
    result = {"epoch": best_state["epoch"], "validation_eer": best_state["validation_eer"],
              "checkpoint_sha256": sha256_file(checkpoint), "epochs_completed": len(history),
              "microbatch": microbatch, "window_limit": window_limit}
    del model, optimizer
    release(device)
    return prediction, result


def train(config: Mapping[str, Any]) -> dict[str, Any]:
    device = config["device"]
    info = require_device(device, config["expected_gpu"])
    configure_numerics()
    if int(config["max_epochs"]) < 1 or int(config["patience"]) < 1:
        raise PipelineError("Convergence safeguards must be positive")
    output = Path(config["train_dir"])
    output.mkdir(parents=True, exist_ok=True)
    lock_path = output / "candidate_lock.json"
    if lock_path.exists():
        lock = verify_lock(config, info.get("name"))
        print("train: candidate lock exists and verifies; training is not repeated", flush=True)
        return lock
    manifest = Path(config["manifest"])
    rows = read_manifest(manifest)
    validate_manifest(rows, config["dataset_expectations"])
    cache_dir = Path(config["cache_dir"])
    complete = read_json(Path(config["features_dir"]) / "features_complete.json")
    index = read_json(cache_dir / "index.json")
    if sha256_file(cache_dir / "index.json") != complete["index_sha256"]:
        raise PipelineError("Feature index changed after extraction completed")
    if index["identity"]["manifest_sha256"] != sha256_file(manifest):
        raise PipelineError("Feature manifest mismatch")
    if set(index["items"]) != {row["ID"] for row in rows}:
        raise PipelineError("Feature cache coverage is not exactly the manifest")
    access = SplitAccess()
    began = time.monotonic()
    items = read_items(rows, cache_dir, index, ("train", "val"), access)
    voice = torch.load(cache_dir / "voice_classifier_initial.pt", map_location="cpu", weights_only=True)
    run_identity = run_identity_for(config, info.get("name"))
    identity_path = output / "run_identity.json"
    if identity_path.exists() and read_json(identity_path) != run_identity:
        raise PipelineError("Output directory belongs to a different training run")
    atomic_json(identity_path, run_identity)
    results: dict[str, Any] = {}
    for route in ("music", "voice"):
        training = [x for x in items if x.row["route"] == route and x.row["split"] == "train"]
        validation = [x for x in items if x.row["route"] == route and x.row["split"] == "val"]
        if not training or not validation or {x.label for x in validation} != {0., 1.}:
            raise PipelineError(f"Incomplete train/validation classes for {route}")
        if any(x.row["split"] != "train" for x in training):
            raise PipelineError("Non-train row in the optimizer set")
        weights = loss_weights(training)
        means = np.stack([x.features.mean(0) for x in training])
        mean, scale = torch.tensor(means.mean(0)), torch.tensor(np.maximum(means.std(0), 1e-6))
        seed_predictions = []
        for seed in SEEDS:
            prediction, result = train_route_seed(route, seed, training, validation, weights, mean, scale, voice,
                                                  run_identity, output, config, device)
            seed_predictions.append(prediction)
            results[f"{route}_{seed}"] = result
        ensemble = np.mean(seed_predictions, axis=0)
        ensemble_eer, ensemble_auc = _roc_metrics([int(x.label) for x in validation], ensemble.tolist())
        results[route + "_ensemble"] = {"eer": ensemble_eer, "auc": ensemble_auc,
                                        "train_rows": len(training), "validation_rows": len(validation),
                                        "normalization": "train-route file means (mean/std over train files)"}
        with (output / f"{route}_validation.csv").open("w", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(["ID", "label", "probability", *(f"seed{s}" for s in SEEDS)])
            for i, item in enumerate(validation):
                writer.writerow([item.row["ID"], item.label, ensemble[i], *(p[i] for p in seed_predictions)])
    del items
    gc.collect()
    atomic_json(output / "results.json", results)
    evaluation = evaluate_split(config, "val", lock=None)
    thresholds = validation_thresholds(evaluation["rows"], evaluation["predictions"])
    eval_dir = Path(config["eval_dir"])
    checkpoints = {name: sha256_file(output / name) for name in checkpoint_names()}
    train_ids = {route: sorted(r["ID"] for r in rows if r["split"] == "train" and r["route"] == route)
                 for route in ("music", "voice")}
    lock = {"selection": "per-route validation EER then BCE over epochs; seeds 41/42/43 equal probability mean; "
                         "FILE = max(voice*VOICE_PRESENT_PROB, music*MUSIC_PRESENT_PROB); no calibration",
            "checkpoints": checkpoints, "run_identity_sha256": canonical_sha(run_identity),
            "fingerprint": stage_fingerprint(config, "train"),
            "manifest_sha256": sha256_file(manifest), "feature_index_sha256": sha256_file(cache_dir / "index.json"),
            "voice_classifier_initial_sha256": sha256_file(cache_dir / "voice_classifier_initial.pt"),
            "results_sha256": sha256_file(output / "results.json"),
            "validation_predictions_sha256": sha256_file(eval_dir / "val_predictions.csv"),
            "validation_metrics_sha256": sha256_file(eval_dir / "val_metrics.json"),
            "validation_eer_thresholds": thresholds,
            "train_ids_sha256": {route: canonical_sha(ids) for route, ids in train_ids.items()},
            "feature_files_opened_before_lock": access.counts, "test_rows_read_before_lock": access.counts["test"],
            "official_submission": False, "independent_test": False, "training_seconds": time.monotonic() - began,
            "device": info, "locked_at": now_iso()}
    lock["lock_body_sha256"] = canonical_sha(lock)
    if lock_path.exists():
        raise PipelineError("candidate_lock.json appeared concurrently; refusing to overwrite")
    atomic_json(lock_path, lock)
    atomic_json(output / "complete.json", {"lock_sha256": sha256_file(lock_path), "results": results})
    print(f"train: locked {len(checkpoints)} checkpoints; val metrics written", flush=True)
    return lock


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    config = read_json(args.config)
    if config["device"] != "cuda" or config["expected_gpu"] != "H100" or config.get("train_benchmark_override"):
        raise PipelineError("The CLI runs only on the authorized H100 with measured VRAM batches")
    with StageLock(Path(config["job_root"]) / "locks" / "training.lock", "dacon9504a100.training"):
        train(config)


if __name__ == "__main__":
    main()
