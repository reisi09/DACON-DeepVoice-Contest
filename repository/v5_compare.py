#!/usr/bin/env python3
"""Execute the five V5 arms under one immutable Colab allocation deadline."""

from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import random
import shutil
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

from graph_features import Extractor
from graph_fusion import graph_predictions
from graph_scoring import SUBMISSION_COLUMNS
from v5_dataset import load_splits
from v5_experts import load_expert
from v5_features import (
    add_compositions,
    assemble_features,
    encode_expert_views,
    prepare_views,
)
from v5_fusion_training import train_fusion
from v5_reporting import combine_experts, predict_expert, record_predictions, write_json
from v5_training import TrainConfig, train_expert


def digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def reset_seed() -> None:
    random.seed(20260913)
    np.random.seed(20260913)
    torch.manual_seed(20260913)
    torch.cuda.manual_seed_all(20260913)


def release(model: torch.nn.Module) -> None:
    model.cpu()
    gc.collect()
    torch.cuda.empty_cache()


def record_temporal(output: Path, name: str, dev, probabilities, metadata) -> dict:
    trained = metadata["epochs_completed"] > 0 and metadata["updates"] > 0 and metadata["best_epoch"] > 0
    if not trained:
        return {"status": "budget_censored", "training": metadata}
    status = "evaluated" if metadata["status"] == "complete" else "evaluated_budget_censored"
    return {"status": status, "training": metadata,
            "metrics": record_predictions(output / f"{name}_predictions.csv", dev, probabilities)}


def baseline_predictions(work: Path, dev, deadline: float) -> np.ndarray:
    source = work / "G0_predictions.csv"
    if digest(source) != "bd1c27227829f36e9ae6191db7600920bf6649810f0896a791730b10555f0854":
        raise RuntimeError("Preserved G0 prediction receipt mismatch")
    with source.open(newline="") as stream:
        previous = {row["ID"]: row for row in csv.DictReader(stream)}
    missing = [row for row in dev if row.id not in previous]
    if len(missing) != 20:
        raise RuntimeError("Expected exactly the 20 new source-disjoint NSynth validation rows")
    extractor = Extractor(work / "baseline" / "script.py", include_raw=False)
    try:
        for row in missing:
            if time.time() >= deadline:
                raise TimeoutError("G0 added validation coverage exceeded the run deadline")
            result = graph_predictions(extractor.extract(row.path))["G0"]
            previous[row.id] = result
    finally:
        extractor.close()
    return np.asarray([[float(previous[row.id][name]) for name in SUBMISSION_COLUMNS[1:]] for row in dev])


def run_experts(work: Path, output: Path, train, dev, deadline: float, comparison):
    initial = time.time()
    available = deadline - initial
    plan = [(kind, arm, share) for kind, share in (("voice", .15), ("music", .065), ("presence", .065)) for arm in ("B1", "B2")]
    cumulative = 0.0
    best_paths, predictions = {}, {"B1": {}, "B2": {}}
    training_statuses = {"B1": {}, "B2": {}}
    for kind, arm, share in plan:
        cumulative += share
        stage_deadline = initial + available * cumulative
        job = output / "jobs" / f"{arm}_{kind}"
        local = work / "trials" / f"{arm}_{kind}"
        job.mkdir(parents=True, exist_ok=True)
        reset_seed()
        print(f"TRAIN_START {arm} {kind} stage_seconds={stage_deadline - time.time():.1f}", flush=True)
        model = load_expert(kind, work / "baseline", work / "sonics_assets", "cuda")
        write_json(job / "initialization.json", {"seed": 20260913, "kind": kind,
                   "pretrained_manifest_sha256": digest(work / "baseline/model/SHA256SUMS.txt"),
                   "train_manifest_sha256": digest(work / "v4_validation/train.csv"),
                   "requested_epochs": 60, "cosine_horizon_epochs": 60,
                   "stage_deadline": stage_deadline, "allocation_deadline": deadline})
        if kind == "voice":
            from v5_dataset import training_crop
            waveform, _ = training_crop(next(row for row in train if row.panel == "speech"), 0, model.window_samples, False, 20260913)
            model.eval()
            tensor = torch.from_numpy(waveform).cuda()
            with torch.inference_mode():
                expected = model.wrapper(input_values=tensor)["logits"]
                actual, _ = model(tensor.unsqueeze(0))
                delta = float((actual - (expected[:, :1] - expected[:, 1:2])).abs().max())
            if delta > 1e-4:
                raise RuntimeError(f"Actual pretrained DF adapter parity failed: {delta}")
            write_json(job / "adapter_parity.json", {"actual_pretrained_1b": True, "max_logit_delta": delta, "tolerance": 1e-4})
        result = train_expert(model, train, dev, TrainConfig(kind, local, deadline,
                              stage_deadline=stage_deadline - 90, augmented=arm == "B2"))
        write_json(job / "training.json", asdict(result))
        training_statuses[arm][kind] = result.status
        if result.best_path is None:
            release(model)
            raise RuntimeError(f"{arm}/{kind}: no completed and evaluated full-finetuned checkpoint; see training.json")
        model.load_state_dict(torch.load(result.best_path, map_location="cpu", weights_only=True), strict=True)
        destination = job / "best.pt"
        shutil.copy2(result.best_path, destination)
        best_paths[(arm, kind)] = destination
        write_json(job / "weights.json", {"sha256": digest(destination), "bytes": destination.stat().st_size,
                   "selected_path": str(destination), "epochs_completed": result.epochs_completed,
                   "updates": result.updates, "full_finetuning_gradient_audit": asdict(result.first_step_audit) if result.first_step_audit else None})
        infer_start = time.time()
        predictions[arm][kind] = predict_expert(model, dev, deadline - 120)
        np.save(job / "dev_probabilities.npy", predictions[arm][kind])
        write_json(job / "inference.json", {"seconds": time.time() - infer_start, "files": len(dev), "scope": "expert only; full-file coverage"})
        release(model)
        del model
        if len(predictions[arm]) == 3:
            probs = combine_experts(*(predictions[arm][name] for name in ("voice", "music", "presence")))
            complete = all(status == "completed" for status in training_statuses[arm].values())
            comparison[arm] = {"status": "evaluated" if complete else "evaluated_budget_censored",
                "component_training_status": training_statuses[arm],
                "metrics": record_predictions(output / f"{arm}_predictions.csv", dev, probs)}
            write_json(output / "comparison.json", comparison)
    return best_paths


def run_temporal(work: Path, output: Path, train, dev, deadline: float, best_paths, comparison) -> None:
    preparation_deadline = deadline - 300
    train, compositions = add_compositions(train, work / "compositions/train")
    write_json(output / "composition_ledger.json", compositions)
    train_views = prepare_views(train, work / "baseline", work / "views/train", preparation_deadline, training=True)
    dev_views = prepare_views(dev, work / "baseline", work / "views/dev", preparation_deadline, training=False)
    all_rows, all_views = [*train, *dev], {**train_views, **dev_views}
    roots = []
    for kind in ("voice", "music", "presence"):
        model = load_expert(kind, work / "baseline", work / "sonics_assets", "cuda")
        checkpoint = best_paths[("B2", kind)]
        model.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True), strict=True)
        root = work / "features" / f"B2_{kind}_{digest(checkpoint)[:16]}"
        encode_expert_views(model, all_rows, all_views, kind, root, preparation_deadline)
        roots.append(root)
        release(model)
        del model
    train_features, dev_features = assemble_features(train, roots), assemble_features(dev, roots)
    write_json(output / "feature_receipt.json", {"weights": {kind: digest(best_paths[("B2", kind)]) for kind in ("voice", "music", "presence")},
               "train_files": len(train), "dev_files": len(dev), "training_context_max_seconds": 20,
               "dev_full_file_coverage": True, "encoder_state": "frozen during cached fusion training"})
    for name, joint in (("B3", False), ("B4", True)):
        stage_deadline = time.time() + max(0, deadline - time.time() - 90) / (2 if name == "B3" else 1)
        model, probabilities, metadata = train_fusion(train_features, dev_features, joint,
                                                       output / name, stage_deadline, epochs=20)
        comparison[name] = record_temporal(output, name, dev, probabilities, metadata)
        if joint and metadata["best_epoch"] > 0:
            from v5_joint import adapt_joint, predict_joint
            if deadline - time.time() > 900:
                experts = {kind: load_expert(kind, work / "baseline", work / "sonics_assets", "cuda") for kind in ("voice", "music", "presence")}
                for kind, expert in experts.items():
                    expert.load_state_dict(torch.load(best_paths[("B2", kind)], map_location="cpu", weights_only=True), strict=True)
                adaptation = adapt_joint(experts, model, train, all_views, output / "B4_joint", deadline - 600)
                fresh = predict_joint(experts, model, dev, dev_views, deadline - 90)
                comparison["B4_joint"] = {"training": adaptation,
                    "status": "evaluated" if adaptation["status"] == "completed" else "evaluated_budget_censored",
                    "metrics": record_predictions(output / "B4_joint_predictions.csv", dev, fresh)}
                for kind, expert in experts.items():
                    torch.save(expert.state_dict(), output / "B4_joint" / f"{kind}.pt")
                    release(expert)
                torch.save(model.state_dict(), output / "B4_joint" / "fusion.pt")
            else:
                comparison["B4_joint"] = {"status": "not_run_budget", "reason": "Less than900s reserved for adaptation and freshfull-file validation"}
        write_json(output / "comparison.json", comparison)
        release(model)
        del model


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--deadline", type=float, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    train, dev, audit = load_splits(args.work / "v4_validation")
    write_json(args.output / "data_audit.json", audit)
    comparison = {name: {"status": "pending"} for name in ("B0", "B1", "B2", "B3", "B4")}
    comparison["protocol"] = {"requested_expert_epochs": 60, "max_expert_epochs": 90,
          "label_policy": "unknown masked independently", "new_submission": False,
          "baseline": "official G0 with preserved 210 predictions and fresh20NSynth",
          "B1": "Full-finetuned raw voice/music/presence experts with max pooling",
          "B2": "Same architecture with channel augmentation; matched source/seed and target horizon",
          "B3": "B2 raw/stem embeddings plus nonjoint temporal fusion",
          "B4": "B2 raw/stem embeddings plus joint nine-state temporal fusion",
          "comparison_boundary": "B0/B1 changes include expert architecture and input views; B1/B2 epoch mismatch is not an isolated augmentation estimate"}
    write_json(args.output / "comparison.json", comparison)
    start = time.time()
    probabilities = baseline_predictions(args.work, dev, args.deadline)
    comparison["B0"] = {"status": "evaluated", "fresh_files": 20, "cached_files": 210,
                         "seconds_added_files": time.time() - start,
                         "metrics": record_predictions(args.output / "B0_predictions.csv", dev, probabilities)}
    write_json(args.output / "comparison.json", comparison)
    best_paths = run_experts(args.work, args.output, train, dev, args.deadline, comparison)
    run_temporal(args.work, args.output, train, dev, args.deadline, best_paths, comparison)
    write_json(args.output / "finished.json", {"status": "comparison_finished", "unix": time.time(),
               "submitted": False, "all_five_evaluated": all(comparison[name]["status"] == "evaluated" for name in ("B0", "B1", "B2", "B3", "B4"))})
    print("V5_COMPARISON_SAVED", args.output, flush=True)


if __name__ == "__main__":
    main()
