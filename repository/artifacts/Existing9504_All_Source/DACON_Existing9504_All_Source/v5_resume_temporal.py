"""Resume only the cached B3/B4 temporal stage of a V5 comparison."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path
from typing import TypedDict

import numpy as np
import soundfile as sf
import torch

from v5_compare import digest, record_temporal, release
from v5_dataset import Example, load_splits
from v5_experts import ExpertKind, load_expert
from v5_features import (
    AudioViews,
    add_compositions,
    assemble_features,
    encode_expert_views,
    prepare_views,
    window_starts,
)
from v5_fusion import FusionContractError
from v5_fusion_training import train_fusion
from v5_reporting import write_json

CacheAudit = TypedDict("CacheAudit", {"valid": int, "absent": int, "partial": int, "invalid": int})


def _expected_context(row: Example, training: bool) -> tuple[int, float]:
    samples = sf.info(row.path).frames
    if training and samples > 320_000:
        offset = int(hashlib.sha256(row.id.encode()).hexdigest()[:16], 16) % (
            samples - 320_000 + 1
        )
        return 320_000, offset / 16_000
    return samples, 0.0


def inspect_cached_views(
    examples: list[Example], cache: Path, *, training: bool
) -> tuple[dict[str, AudioViews], list[Example], CacheAudit]:
    """Accept only complete 16 kHz trios aligned to the deterministic context."""
    views: dict[str, AudioViews] = {}
    missing: list[Example] = []
    audit: CacheAudit = {"valid": 0, "absent": 0, "partial": 0, "invalid": 0}
    for row in examples:
        paths = tuple(cache / f"{row.id}_{name}.wav" for name in ("raw", "voice", "music"))
        present = sum(path.is_file() for path in paths)
        if present == 0:
            audit["absent"] += 1
            missing.append(row)
            continue
        if present != len(paths):
            audit["partial"] += 1
            missing.append(row)
            continue
        frames, offset = _expected_context(row, training)
        try:
            infos = tuple(sf.info(path) for path in paths)
        except RuntimeError:
            audit["invalid"] += 1
            missing.append(row)
            continue
        if any(info.samplerate != 16_000 or info.frames != frames for info in infos):
            audit["invalid"] += 1
            missing.append(row)
            continue
        audit["valid"] += 1
        views[row.id] = AudioViews(paths[0], paths[1], paths[2], offset)
    return views, missing, audit


def _complete_views(
    cached: dict[str, AudioViews],
    missing: list[Example],
    baseline: Path,
    cache: Path,
    deadline: float,
    *,
    training: bool,
) -> dict[str, AudioViews]:
    if not missing:
        return cached
    if time.time() >= deadline:
        raise TimeoutError("View recovery reached the preparation deadline")
    generated = prepare_views(missing, baseline, cache, deadline, training=training)
    result = {**cached, **generated}
    if len(result) != len(cached) + len(missing):
        raise FusionContractError("View recovery did not produce the complete requested cohort")
    return result


def _valid_feature(path: Path, view: AudioViews) -> bool:
    if not path.is_file():
        return False
    starts = np.asarray(window_starts(sf.info(view.raw).frames, 64_600))
    expected_times = starts.astype(np.float32) / 16_000 + view.offset_seconds
    try:
        with np.load(path, allow_pickle=False) as data:
            features, times = data["features"], data["times"]
            return bool(
                features.ndim == 2
                and features.shape[0] == len(starts)
                and times.shape == expected_times.shape
                and np.array_equal(times, expected_times)
                and np.isfinite(features).all()
            )
    except (OSError, ValueError, KeyError):
        return False


def _ensure_features(
    kind: ExpertKind,
    checkpoint: Path,
    examples: list[Example],
    views: dict[str, AudioViews],
    work: Path,
    deadline: float,
) -> tuple[Path, dict[str, int]]:
    root = work / "features" / f"B2_{kind}_{digest(checkpoint)[:16]}"
    missing = [row for row in examples if not _valid_feature(root / f"{row.id}.npz", views[row.id])]
    audit = {"valid": len(examples) - len(missing), "missing_or_invalid": len(missing)}
    if not missing:
        return root, audit
    if time.time() >= deadline:
        raise TimeoutError(f"{kind} feature recovery reached the preparation deadline")
    model = load_expert(kind, work / "baseline", work / "sonics_assets", "cuda")
    temporary = root.parent / f".{root.name}.resume-{time.time_ns()}"
    try:
        model.load_state_dict(
            torch.load(checkpoint, map_location="cpu", weights_only=True), strict=True
        )
        encode_expert_views(model, missing, views, kind, temporary, deadline)
        root.mkdir(parents=True, exist_ok=True)
        for row in missing:
            (temporary / f"{row.id}.npz").replace(root / f"{row.id}.npz")
        temporary.rmdir()
    finally:
        release(model)
    remaining = [row.id for row in examples if not _valid_feature(root / f"{row.id}.npz", views[row.id])]
    if remaining:
        raise FusionContractError(f"{kind} feature cohort remains incomplete: {len(remaining)}")
    return root, audit


def _expert_checkpoints(expert_output: Path) -> tuple[dict[ExpertKind, Path], dict[str, dict[str, str | int]]]:
    """Pin B2 weights at their original output without copying them into a temporal run."""
    checkpoints: dict[ExpertKind, Path] = {}
    receipts: dict[str, dict[str, str | int]] = {}
    for kind in ("voice", "music", "presence"):
        checkpoint = expert_output / "jobs" / f"B2_{kind}" / "best.pt"
        receipt_path = expert_output / "jobs" / f"B2_{kind}" / "weights.json"
        if not checkpoint.is_file() or not receipt_path.is_file():
            raise FileNotFoundError(f"Missing pinned B2 {kind} weight or receipt")
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        expected = receipt.get("sha256")
        actual = digest(checkpoint)
        if not isinstance(expected, str) or actual != expected:
            raise FusionContractError(f"Pinned B2 {kind} weight receipt mismatch")
        checkpoints[kind] = checkpoint
        receipts[kind] = {"path": str(checkpoint), "sha256": actual, "bytes": checkpoint.stat().st_size}
    return checkpoints, receipts


def run(work: Path, output: Path, deadline: float, expert_output: Path | None = None) -> None:
    """Recover cached views/features, then run bounded B3 and B4 only."""
    if deadline <= time.time():
        raise TimeoutError("Immutable allocation deadline has already elapsed")
    comparison_path = output / "comparison.json"
    if not comparison_path.is_file():
        raise FileNotFoundError("Existing comparison.json is required to preserve B0/B1/B2")
    comparison = json.loads(comparison_path.read_text(encoding="utf-8"))
    origin = expert_output or output
    checkpoints, weight_receipts = _expert_checkpoints(origin)
    train, dev, _ = load_splits(work / "v4_validation")
    train, compositions = add_compositions(train, work / "compositions/train")
    preparation_deadline = deadline - 300
    train_cached, train_missing, train_audit = inspect_cached_views(
        train, work / "views/train", training=True
    )
    dev_cached, dev_missing, dev_audit = inspect_cached_views(
        dev, work / "views/dev", training=False
    )
    recovery_base = {
        "original_interrupted_stage": "prepare_views/train",
        "immutable_deadline": deadline,
        "preparation_encoding_deadline": preparation_deadline,
        "actual_cohort": {"train": len(train), "dev": len(dev)},
        "initial_view_cache": {"train": train_audit, "dev": dev_audit},
        "composition_count": len(compositions),
        "expert_output_origin": str(origin),
        "pinned_b2_weights": weight_receipts,
    }
    write_json(output / "temporal_recovery.json", {**recovery_base, "status": "recovering"})
    train_views = _complete_views(
        train_cached,
        train_missing,
        work / "baseline",
        work / "views/train",
        preparation_deadline,
        training=True,
    )
    dev_views = _complete_views(
        dev_cached,
        dev_missing,
        work / "baseline",
        work / "views/dev",
        preparation_deadline,
        training=False,
    )
    rows, views = [*train, *dev], {**train_views, **dev_views}
    roots: list[Path] = []
    feature_audit: dict[str, dict[str, int]] = {}
    for kind in ("voice", "music", "presence"):
        root, audit = _ensure_features(
            kind, checkpoints[kind], rows, views, work, preparation_deadline
        )
        roots.append(root)
        feature_audit[kind] = audit
    train_features = assemble_features(train, roots)
    dev_features = assemble_features(dev, roots)
    if len(train_features) != len(train) or len(dev_features) != len(dev):
        raise FusionContractError("Temporal feature assembly silently lost cohort members")
    write_json(
        output / "temporal_recovery.json",
        {
            **recovery_base,
            "status": "training",
            "final_view_cache": {"train": len(train_views), "dev": len(dev_views)},
            "initial_feature_cache": feature_audit,
        },
    )

    for name, joint in (("B3", False), ("B4", True)):
        reserve = max(0.0, deadline - time.time() - 60)
        stage_deadline = time.time() + reserve / (2 if name == "B3" else 1)
        model, probabilities, metadata = train_fusion(
            train_features,
            dev_features,
            joint,
            output / name,
            stage_deadline,
            epochs=20,
        )
        comparison[name] = record_temporal(output, name, dev, probabilities, metadata)
        write_json(comparison_path, comparison)
        release(model)
    comparison["B4_joint"] = {
        "status": "not_run_budget",
        "reason": "Recovery allocation reserved only cached B3/B4 temporal training",
    }
    write_json(comparison_path, comparison)
    write_json(
        output / "temporal_recovery.json",
        {
            **recovery_base,
            "status": "complete",
            "final_view_cache": {"train": len(train_views), "dev": len(dev_views)},
            "initial_feature_cache": feature_audit,
            "completed_unix": time.time(),
        },
    )
    write_json(
        output / "finished.json",
        {
            "status": "temporal_recovery_finished",
            "unix": time.time(),
            "submitted": False,
            "all_five_evaluated": all(
                comparison.get(name, {}).get("status") == "evaluated"
                for name in ("B0", "B1", "B2", "B3", "B4")
            ),
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--deadline", type=float, required=True)
    parser.add_argument("--expert-output", type=Path, help="Prior output retaining the pinned B2 checkpoints")
    args = parser.parse_args()
    run(args.work, args.output, args.deadline, args.expert_output)


if __name__ == "__main__":
    main()
