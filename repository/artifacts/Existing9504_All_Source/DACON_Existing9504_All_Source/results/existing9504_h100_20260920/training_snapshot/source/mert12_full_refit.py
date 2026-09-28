"""Full-data refit of the selected frozen MERT layer-12 head."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import time
from collections import Counter
from pathlib import Path
from typing import Final

import numpy as np
import torch
from torch.nn import functional as functional

from eight_heads import FrozenHead, balanced_draws

SEEDS: Final = (41, 42, 43)
STEPS: Final = 500
BATCH_SIZE: Final = 32
FEATURE_KEY: Final = "mert_l12"
EXPECTED_ROWS: Final = 1322


class RefitError(ValueError):
    """A refit input or completed artifact violates the locked recipe."""


def sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def atomic_json(path: Path, payload: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def eligible_rows(path: Path) -> list[dict[str, str | int]]:
    """Select every labeled original core-music row, regardless of former split."""
    with path.open(encoding="utf-8", newline="") as stream:
        source = list(csv.DictReader(stream))
    rows = [
        {**row, "label": int(row["MUSIC_FAKE"])}
        for row in source
        if row["probe_family"] == "core"
        and row["variant"] == "base"
        and row["MUSIC_PRESENT"] == "1"
        and row["MUSIC_FAKE"] in {"0", "1"}
    ]
    ids = [str(row["ID"]) for row in rows]
    if not rows or len(ids) != len(set(ids)) or {int(row["label"]) for row in rows} != {0, 1}:
        raise RefitError("Eligible full-data rows must be non-empty, unique, and contain both labels")
    return rows


def load_features(
    rows: list[dict[str, str | int]], directory: Path,
) -> tuple[np.ndarray, list[dict[str, str | int]], dict[str, str]]:
    """Load file means while binding every vector to its source ID and audio hash."""
    vectors: list[np.ndarray] = []
    lineage: list[dict[str, str | int]] = []
    asset_hashes: dict[str, str] | None = None
    for row in rows:
        path = directory / f"{row['ID']}.npz"
        if not path.is_file():
            raise RefitError(f"Missing feature record: {row['ID']}")
        with np.load(path, allow_pickle=False) as saved:
            if FEATURE_KEY not in saved.files or "valid_samples" not in saved.files or "identity_json" not in saved.files:
                raise RefitError(f"Incomplete MERT12 feature record: {row['ID']}")
            identity = json.loads(str(saved["identity_json"]))
            values = np.asarray(saved[FEATURE_KEY], dtype=np.float32)
            valid = np.asarray(saved["valid_samples"]) > 0
        if (identity.get("ID") != row["ID"] or identity.get("audio_sha256") != row["audio_sha256"]
                or values.ndim != 2 or values.shape[1] != 1024 or valid.shape != (len(values),)
                or not valid.any() or not np.isfinite(values[valid]).all()):
            raise RefitError(f"Feature/source identity or MERT12 values mismatch: {row['ID']}")
        current_assets = identity.get("baseline_assets_sha256")
        if not isinstance(current_assets, dict) or not current_assets:
            raise RefitError(f"Feature record lacks frozen-asset hashes: {row['ID']}")
        if asset_hashes is None:
            asset_hashes = current_assets
        elif current_assets != asset_hashes:
            raise RefitError(f"Feature records use different frozen assets: {row['ID']}")
        vectors.append(values[valid].mean(0))
        lineage.append({"ID": str(row["ID"]), "label": int(row["label"]),
                        "source": str(row["source"]), "generator": str(row["generator"]),
                        "parent_id": str(row["parent_id"]), "split": str(row["split"]),
                        "audio_sha256": str(row["audio_sha256"]), "feature_sha256": sha256(path),
                        "windows": int(valid.sum())})
    matrix = np.stack(vectors).astype(np.float32)
    if not np.isfinite(matrix).all():
        raise RefitError("Full-data feature matrix contains non-finite values")
    if asset_hashes is None:
        raise RefitError("No frozen-asset identity was loaded")
    return matrix, lineage, asset_hashes


def benchmark_batches(matrix: np.ndarray, labels: np.ndarray, output: Path) -> dict:  # noqa: DICT_OK
    """Measure head-only throughput at equal draws; never select on loss or quality."""
    results = []
    device = torch.device("cuda")
    x = torch.as_tensor(matrix, device=device)
    y = torch.as_tensor(labels, dtype=torch.float32, device=device)
    draws = 8192
    for batch_size in (32, 128, 256):
        torch.manual_seed(1)
        model = FrozenHead(matrix.shape[1], "mlp").to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-2)
        for _ in range(5):
            chosen = torch.arange(batch_size, device=device) % len(x)
            optimizer.zero_grad(set_to_none=True)
            logits, _ = model(x[chosen])
            functional.binary_cross_entropy_with_logits(logits, y[chosen]).backward()
            optimizer.step()
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        began = time.perf_counter()
        for step in range(draws // batch_size):
            chosen = (torch.arange(batch_size, device=device) + step * batch_size) % len(x)
            optimizer.zero_grad(set_to_none=True)
            logits, _ = model(x[chosen])
            functional.binary_cross_entropy_with_logits(logits, y[chosen]).backward()
            optimizer.step()
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - began
        results.append({"batch_size": batch_size, "equal_draws": draws, "seconds": elapsed,
                        "draws_per_second": draws / elapsed,
                        "peak_allocated_bytes": torch.cuda.max_memory_allocated()})
    report = {"scope": "throwaway frozen-head throughput only; not MERT extraction or quality selection",
              "warmup_steps": 5, "results": results, "final_recipe_batch_size": BATCH_SIZE}
    atomic_json(output / "batch_benchmark.json", report)
    return report


def coverage_draws(rows: list[dict[str, str | int]], seed: int) -> tuple[list[list[int]], dict]:
    """Preserve E6 draws except for a deterministic class-balanced coverage prefix."""
    original = balanced_draws(rows, steps=STEPS, batch_size=BATCH_SIZE, seed=seed,
                              mode="source_generator_parent")
    revised = [batch.copy() for batch in original]
    rng = np.random.default_rng(seed)
    replaced = 0
    for label in (0, 1):
        eligible = np.asarray([index for index, row in enumerate(rows) if int(row["label"]) == label])
        rng.shuffle(eligible)
        positions = [(batch_index, position) for batch_index, batch in enumerate(revised)
                     for position, index in enumerate(batch) if int(rows[index]["label"]) == label]
        for index, (batch_index, position) in zip(eligible.tolist(), positions, strict=False):
            revised[batch_index][position] = index
            replaced += 1
    before = Counter(index for batch in original for index in batch)
    after = Counter(index for batch in revised for index in batch)
    if len(after) != len(rows) or any(sum(int(rows[index]["label"]) == label for index in batch) != 16
                                      for batch in revised for label in (0, 1)):
        raise RefitError("Deterministic coverage schedule failed full exposure or class balance")
    return revised, {
        "seed": seed, "steps": STEPS, "batch_size": BATCH_SIZE, "replaced_prefix_slots": replaced,
        "before_unique": len(before), "before_zero_exposure_ids": [str(rows[i]["ID"]) for i in range(len(rows)) if i not in before],
        "after_unique": len(after), "after_zero_exposure_ids": [],
        "after_counts_by_id": {str(row["ID"]): after[index] for index, row in enumerate(rows)},
    }


def train_one(  # noqa: DICT_OK
    rows: list[dict[str, str | int]], matrix: np.ndarray, output: Path,
    seed: int, mean: np.ndarray, scale: np.ndarray,
) -> dict:
    """Run the E6 optimizer with only the declared coverage-prefix correction."""
    draws, coverage = coverage_draws(rows, seed)
    torch.manual_seed(seed)
    device = torch.device("cuda")
    model = FrozenHead(matrix.shape[1], "mlp").to(device)
    initial = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-2)
    x = torch.as_tensor(matrix, device=device)
    y = torch.as_tensor([int(row["label"]) for row in rows], dtype=torch.float32, device=device)
    first_gradient_l1: dict[str, float] = {}
    for step, indices in enumerate(draws):
        chosen = torch.as_tensor(indices, device=device)
        optimizer.zero_grad(set_to_none=True)
        logits, _ = model(x[chosen])
        loss = functional.binary_cross_entropy_with_logits(logits, y[chosen])
        loss.backward()
        if step == 0:
            first_gradient_l1 = {name: float(parameter.grad.abs().sum())
                                 for name, parameter in model.named_parameters() if parameter.grad is not None}
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
        optimizer.step()
    update_l2 = sum(float((value.detach().cpu() - initial[name]).square().sum())
                    for name, value in model.state_dict().items()) ** 0.5
    if update_l2 <= 0 or not first_gradient_l1 or any(value <= 0 for value in first_gradient_l1.values()):
        raise RefitError("Head gradient or optimizer update evidence failed")
    checkpoint = output / f"head_seed{seed}.pt"
    torch.save({"model": model.eval().cpu().state_dict(), "input_width": 1024, "seed": seed,
                "steps": STEPS, "batch_size": BATCH_SIZE, "mode": "source_generator_parent+coverage_prefix",
                "architecture": "mlp", "mean": torch.from_numpy(mean), "scale": torch.from_numpy(scale)}, checkpoint)
    coverage.update(checkpoint_sha256=sha256(checkpoint), parameter_update_l2=update_l2,
                    first_gradient_l1=first_gradient_l1, device="cuda",
                    torch_version=torch.__version__, cuda_version=torch.version.cuda)
    return coverage


def run(  # noqa: DICT_OK
    manifest: Path, features: Path, audit_receipt: Path, output: Path,
    device: str, run_benchmark: bool,
) -> dict:
    if device != "cuda" or not torch.cuda.is_available():
        raise RefitError("Actual CUDA training is required for the final refit")
    output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.cuda.reset_peak_memory_stats()
    rows = eligible_rows(manifest)
    audit = json.loads(audit_receipt.read_text(encoding="utf-8"))
    identifiers = [str(row["ID"]) for row in rows]
    if (len(identifiers) != EXPECTED_ROWS or audit.get("manifest_sha256") != sha256(manifest)
            or audit.get("eligible_ids") != identifiers):
        raise RefitError("Manifest digest or eligible ID order differs from the audited full-data lock")
    locked_audit = output / "joined_data_audit.json"
    locked_audit.write_bytes(audit_receipt.read_bytes())
    raw, lineage, asset_hashes = load_features(rows, features)
    mean, scale = raw.mean(0), raw.std(0)
    scale[scale < 1e-6] = 1.0
    matrix = (raw - mean) / scale
    training_manifest = output / "training_manifest.json"
    atomic_json(training_manifest, {"source_manifest": str(manifest), "source_manifest_sha256": sha256(manifest),
                                    "feature_key": FEATURE_KEY, "frozen_asset_sha256": asset_hashes, "rows": lineage})
    benchmark = benchmark_batches(matrix, np.asarray([int(row["label"]) for row in rows]), output) if run_benchmark else None
    receipts = [train_one(rows, matrix, output, seed, mean, scale) for seed in SEEDS]
    receipt = {"training_mode": "full_data_refit", "candidate": "E6_MERT_layer12",
               "seeds": list(SEEDS), "steps": STEPS, "batch_size": BATCH_SIZE,
               "optimizer": "AdamW(lr=1e-3,weight_decay=1e-2)", "architecture": "1024->128 GELU->1",
               "sampler": "source_generator_parent with deterministic within-class coverage prefix",
               "sampler_deviation": "E6 schedule had 109 zero-exposure rows; prefix replacement fixes coverage",
               "normalization": "mean/std fitted on every eligible full-data row; std<1e-6 replaced by 1",
               "file_pooling": "mean of every valid window; no audible mask", "independent_validation": False,
               "all_former_splits_are_training": True, "train_ids": [str(row["ID"]) for row in rows],
               "eligible_rows": len(rows), "training_manifest_sha256": sha256(training_manifest),
               "audit_receipt_sha256": sha256(locked_audit),
               "frozen_asset_sha256": asset_hashes, "batch_benchmark": benchmark, "checkpoints": receipts}
    atomic_json(output / "full_refit_receipt.json", receipt)
    atomic_json(output / "gpu_training_receipt.json", {"device_name": torch.cuda.get_device_name(),
                                                        "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
                                                        "checkpoints": receipts})
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--audit-receipt", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("cuda",), default="cuda")
    parser.add_argument("--benchmark-batches", action="store_true")
    args = parser.parse_args()
    receipt = run(args.manifest, args.features, args.audit_receipt, args.output,
                  args.device, args.benchmark_batches)
    print(json.dumps({"status": "complete", "eligible_rows": receipt["eligible_rows"],
                      "seeds": receipt["seeds"], "batch_size": receipt["batch_size"],
                      "output": str(args.output)}, indent=2))


if __name__ == "__main__":
    main()
