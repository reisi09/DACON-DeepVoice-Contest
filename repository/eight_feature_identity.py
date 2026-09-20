"""Exact source/model/provenance identity for shared feature records."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path


class FeatureIdentityError(RuntimeError):
    """A feature source or contributor cannot be bound to an exact hash."""


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def resolve_parent_hashes(root: Path, rows: list[dict[str, str]]) -> dict[str, dict[str, str]]:
    """Resolve derived parents from the manifest and retained sources from sources.csv."""
    hashes = {row["ID"]: row["audio_sha256"] for row in rows}
    sources = root / "source/dataset/sources.csv"
    if sources.exists():
        with sources.open(encoding="utf-8", newline="") as stream:
            hashes.update({row["ID"]: row["source_sha256"] for row in csv.DictReader(stream)})
    resolved: dict[str, dict[str, str]] = {}
    for row in rows:
        parents = json.loads(row["parent_ids"])
        missing = [identifier for identifier in parents if identifier not in hashes]
        if missing:
            raise FeatureIdentityError(f"Parent source hashes are unavailable for {row['ID']}: {missing}")
        resolved[row["ID"]] = {identifier: hashes[identifier] for identifier in parents}
    return resolved


def feature_identity(
    row: dict[str, str], assets: dict[str, str], anti: dict[str, str], parent_hashes: dict[str, str],
    profile: str, required: frozenset[str],
) -> dict[str, object]:
    """Bind one NPZ to its audio, contributors, code, models, views, and window frontend."""
    code = {
        name: file_sha256(Path(__file__).with_name(name))
        for name in (
            "eight_features.py", "eight_spectral.py", "eight_feature_schema.py",
            "eight_feature_identity.py", "eight_probes.py", "antideepfake_runtime.py",
        )
    }
    views = ["raw"]
    if any(key.endswith("_vocals") for key in required):
        views.append("recovered_vocals")
    if any(key.startswith("mert_") or key in {"stem_mask", "spectrum_stem"} for key in required):
        views.append("accompaniment")
    return {
        "ID": row["ID"], "audio_sha256": row["audio_sha256"], "source_path": row["path"],
        "window_samples": 64_600, "hop_samples": 32_300, "padding": "right_zero",
        "sample_rate": 16_000, "feature_profile": profile, "required_arrays": sorted(required),
        "mert_layers": sorted(int(key.removeprefix("mert_l")) for key in required if key.startswith("mert_l")),
        "anti_layers": sorted(int(key.removeprefix("anti_l")) for key in required if key.startswith("anti_l")),
        "views": views, "parent_sha256": parent_hashes,
        "code_sha256": code, "baseline_assets_sha256": assets, "antideepfake_sha256": anti,
    }
