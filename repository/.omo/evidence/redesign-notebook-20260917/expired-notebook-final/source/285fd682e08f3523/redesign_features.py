"""Public frozen-feature extraction and verified cache loading API."""

from __future__ import annotations

import hashlib
import json
import resource
import sys
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import TypedDict

from redesign_assets import validate_assets
from redesign_features_cache import (
    CacheError,
    FeatureRecord,
    FeatureRow,
    install_receipt,
    load_receipt,
    read_record,
    receipt_payload,
    record_path,
    save_record,
)


class ExtractionResult(TypedDict):
    status: str
    extracted: int
    cached: int
    records: int
    receipt_sha256: str
    manifest: str
    device: str
    elapsed_seconds: float
    windows: int
    peak_gpu_bytes: int
    peak_rss_bytes: int


def _rows(rows: Sequence[Mapping[str, str]]) -> list[FeatureRow]:
    parsed: list[FeatureRow] = []
    seen: set[str] = set()
    for source in rows:
        try:
            row = FeatureRow(ID=source["ID"], path=source["path"], audio_sha256=source["audio_sha256"])
        except KeyError as error:
            raise CacheError(f"Feature row lacks required field: {error.args[0]}") from error
        if not row["ID"] or row["ID"] in seen:
            raise CacheError(f"Feature row ID is empty or duplicated: {row['ID']!r}")
        if len(row["audio_sha256"]) != 64:
            raise CacheError(f"Feature row has invalid audio SHA256: {row['ID']}")
        seen.add(row["ID"])
        parsed.append(row)
    return parsed


def _audio_matches(row: FeatureRow) -> None:
    path = Path(row["path"])
    digest = hashlib.sha256()
    try:
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
    except FileNotFoundError as error:
        raise CacheError(f"Audio file is missing: {row['ID']}") from error
    if digest.hexdigest() != row["audio_sha256"]:
        raise CacheError(f"Audio checksum differs from row: {row['ID']}")


def _write_json(path: Path, value: Mapping[str, str | int | float]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".partial")
    temporary.write_text(json.dumps(dict(value), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def extract_features(
    rows: Sequence[Mapping[str, str]],
    assets: Path,
    cache: Path,
    *,
    device: str = "cuda",
    deadline: float | None = None,
    views: tuple[str, ...] = ("stem", "raw"),
) -> ExtractionResult:
    """Extract aligned full-coverage features, resuming only exact identity matches."""
    if views != ("stem", "raw"):
        raise CacheError("The portable redesign cache requires both ordered views: ('stem', 'raw')")
    parsed = _rows(rows)
    hashes = validate_assets(assets)
    payload = receipt_payload(assets, hashes)
    receipt = install_receipt(cache, payload)
    engine = None
    manifest_path = cache / "manifest.json"
    entries: dict[str, dict[str, str]] = {}
    if manifest_path.exists():
        try:
            previous = json.loads(manifest_path.read_text(encoding="utf-8"))
            if previous["receipt_sha256"] != receipt:
                raise CacheError("Existing cache manifest uses a different extraction receipt")
            entries = {str(key): dict(value) for key, value in previous["records"].items()}
        except (KeyError, TypeError, json.JSONDecodeError) as error:
            raise CacheError("Existing cache manifest is invalid") from error
    extracted = cached = 0
    windows = 0
    began = time.time()
    for index, row in enumerate(parsed):
        if deadline is not None and time.time() >= deadline:
            _write_json(cache / "progress.json", {"stage": "deadline", "completed": index, "total": len(parsed), "utc": time.time()})
            raise TimeoutError("Feature extraction deadline reached")
        _audio_matches(row)
        target = record_path(cache, row["ID"])
        previous_entry = entries.get(row["ID"])
        if previous_entry is not None and previous_entry.get("audio_sha256") != row["audio_sha256"]:
            raise CacheError(f"Existing cache ID has different audio: {row['ID']}")
        if previous_entry is not None and previous_entry.get("record") != str(target.relative_to(cache)):
            raise CacheError(f"Existing cache ID has a different record path: {row['ID']}")
        if target.exists():
            checksum = None if previous_entry is None else previous_entry.get("record_sha256")
            values = read_record(cache, row, receipt, checksum)
            cached += 1
        else:
            if engine is None:
                from redesign_features_runtime import FrozenEncoders

                if device == "cuda":
                    import torch

                    if not torch.cuda.is_available():
                        raise CacheError("CUDA was requested but is unavailable; CPU fallback is disabled")
                    torch.cuda.reset_peak_memory_stats()
                engine = FrozenEncoders(assets, device, deadline)
            values = engine.extract(Path(row["path"]))
            target, _ = save_record(cache, row, receipt, values)
            extracted += 1
        windows += len(values["starts"])
        entries[row["ID"]] = {
            "audio_sha256": row["audio_sha256"],
            "record": str(target.relative_to(cache)),
            "record_sha256": hashlib.sha256(target.read_bytes()).hexdigest(),
        }
        _write_json(cache / "progress.json", {
            "stage": "extracting", "completed": index + 1, "total": len(parsed),
            "current_id": row["ID"], "utc": time.time(), "elapsed_seconds": time.time() - began,
            "estimated_remaining_seconds": (time.time() - began) / (index + 1) * (len(parsed) - index - 1),
        })
    manifest = {
        "receipt_sha256": receipt,
        "records": entries,
    }
    temporary = manifest_path.with_suffix(".partial")
    temporary.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(manifest_path)
    peak_gpu = 0
    if device == "cuda" and engine is not None:
        peak_gpu = int(engine.torch.cuda.max_memory_allocated())
    peak_rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    peak_rss_bytes = int(peak_rss if sys.platform == "darwin" else peak_rss * 1024)
    return ExtractionResult(
        status="complete", extracted=extracted, cached=cached, records=len(parsed),
        receipt_sha256=receipt, manifest=str(manifest_path), device=device,
        elapsed_seconds=time.time() - began, windows=windows,
        peak_gpu_bytes=peak_gpu, peak_rss_bytes=peak_rss_bytes,
    )


def load_records(rows: Sequence[Mapping[str, str]], cache: Path) -> dict[str, FeatureRecord]:
    """Load only a complete manifest bound to current rows, source, config, and receipt."""
    parsed = _rows(rows)
    try:
        manifest = json.loads((cache / "manifest.json").read_text(encoding="utf-8"))
        receipt = str(manifest["receipt_sha256"])
        entries = manifest["records"]
    except (FileNotFoundError, KeyError, ValueError, json.JSONDecodeError) as error:
        raise CacheError("Feature cache manifest is missing or invalid") from error
    load_receipt(cache, receipt)
    if not {row["ID"] for row in parsed}.issubset(entries):
        raise CacheError("Requested feature IDs are absent from the cache manifest")
    result: dict[str, FeatureRecord] = {}
    for row in parsed:
        _audio_matches(row)
        entry = entries[row["ID"]]
        if entry.get("audio_sha256") != row["audio_sha256"]:
            raise CacheError(f"Manifest audio identity is stale: {row['ID']}")
        expected_path = str(record_path(cache, row["ID"]).relative_to(cache))
        if entry.get("record") != expected_path:
            raise CacheError(f"Manifest record path is invalid: {row['ID']}")
        result[row["ID"]] = read_record(cache, row, receipt, str(entry.get("record_sha256", "")))
    return result
