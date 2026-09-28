"""Stdlib fingerprints of everything a reusable stage result depends on.

A stored pilot receipt or candidate lock is reused only if the current
manifest, frozen assets, code, isolated environment and run configuration
reproduce its fingerprint exactly.  The same functions run in the notebook
kernel (before deciding to reuse) and in the child processes (before reading
the stored result), so both sides compare identical structures.
"""
from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from .common import canonical_sha, read_json, sha256_file

PACKAGE_DIR = Path(__file__).resolve().parent

# Code whose bytes determine cached features (identity of every .npz).
FEATURE_ORIGINAL_MODULES = ("full_bank_features.py", "mert12_submission.py", "v6_runtime.py",
                            "localization_features.py", "v6_data.py", "eight_fixed_io.py")
FEATURE_DERIVED_MODULES = ("features.py", "gpu.py", "common.py", "dataset.py", "fingerprint.py")
# Code whose bytes determine trained heads, predictions and locked thresholds.
TRAIN_ORIGINAL_MODULES = ("full_bank_train.py", "full_bank_batches.py", "full_bank_features.py",
                          "full_bank_evaluate.py", "localization_models.py", "eight_heads.py", "graph_scoring.py",
                          "eight_metrics.py")
TRAIN_DERIVED_MODULES = ("training.py", "evaluation.py", "features.py", "gpu.py", "common.py", "dataset.py",
                         "fingerprint.py")

CONFIG_KEYS = {
    "pilot": ("device", "expected_gpu", "cuda_visible_devices", "mert_batch_override", "dataset_expectations",
              "asset_hashes", "source_hashes", "runtime_manifest_sha256"),
    "train": ("device", "expected_gpu", "cuda_visible_devices", "max_epochs", "patience", "train_benchmark_override",
              "dataset_expectations", "asset_hashes", "source_hashes", "runtime_manifest_sha256"),
}


def feature_implementation(config: Mapping[str, Any]) -> dict[str, str]:
    source = Path(config["source_dir"])
    result = {f"source/{name}": sha256_file(source / name) for name in FEATURE_ORIGINAL_MODULES}
    result.update({f"dacon9504a100/{name}": sha256_file(PACKAGE_DIR / name) for name in FEATURE_DERIVED_MODULES})
    result["assets/baseline/script.py"] = sha256_file(Path(config["assets_dir"]) / "baseline" / "script.py")
    return result


def train_implementation(config: Mapping[str, Any]) -> dict[str, str]:
    source = Path(config["source_dir"])
    result = {f"source/{name}": sha256_file(source / name) for name in TRAIN_ORIGINAL_MODULES}
    result.update({f"dacon9504a100/{name}": sha256_file(PACKAGE_DIR / name) for name in TRAIN_DERIVED_MODULES})
    return result


def environment_summary(config: Mapping[str, Any]) -> dict[str, Any]:
    path = Path(config["env_receipt"])
    receipt = read_json(path) if path.is_file() else {}
    probe = receipt.get("probe", {})
    return {"lock_sha256": receipt.get("lock_sha256"),
            "packages_sha256": canonical_sha(receipt["packages"]) if "packages" in receipt else None,
            "torch": probe.get("torch"), "torch_cuda": probe.get("torch_cuda"),
            "gpu": (probe.get("gpu") or {}).get("name"),
            "ffmpeg_ld_library_path": receipt.get("ffmpeg_ld_library_path")}


def stage_fingerprint(config: Mapping[str, Any], stage: str) -> dict[str, Any]:
    implementation = feature_implementation(config)
    if stage == "train":
        implementation.update(train_implementation(config))
    value = {"stage": stage, "manifest_sha256": sha256_file(Path(config["manifest"])),
             "asset_hashes_sha256": sha256_file(Path(config["assets_dir"]) / "asset_hashes.json"),
             "implementation": implementation, "environment": environment_summary(config),
             "config": {key: config.get(key) for key in CONFIG_KEYS[stage]}}
    return json.loads(json.dumps(value, sort_keys=True))  # same types as a stored (JSON) copy


def differences(stored: Mapping[str, Any] | None, current: Mapping[str, Any]) -> list[str]:
    """Dotted paths that differ (empty list = identical)."""
    if not isinstance(stored, Mapping):
        return ["<no stored fingerprint>"]
    result = []
    for key in sorted(set(stored) | set(current)):
        left, right = stored.get(key), current.get(key)
        if isinstance(left, Mapping) and isinstance(right, Mapping):
            result += [f"{key}.{name}" for name in differences(left, right)]
        elif left != right:
            result.append(key)
    return result
