#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = []
# ///

# ─── How to run ───
# 1. Install uv (if not installed):
#      curl -LsSf https://astral.sh/uv/install.sh | sh
# 2. Run directly (no venv, no pip install needed):
#      uv run redesign_notebook_runtime.py
# 3. Or make executable and run:
#      chmod +x redesign_notebook_runtime.py && ./redesign_notebook_runtime.py
# ──────────────────
"""Small, standard-library runtime helpers for the portable redesign notebook."""

from __future__ import annotations

import copy
import csv
import hashlib
import json
import shutil
import stat
import time
import zipfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Final

GPU_CAP_SECONDS: Final = 4 * 60 * 60
EXPORT_RESERVE_SECONDS: Final = 15 * 60


class NotebookContractError(RuntimeError):
    """Raised when a notebook safety or provenance contract is not met."""


def sha256_file(path: Path) -> str:
    """Return the SHA-256 of one regular file."""
    if not path.is_file() or path.is_symlink():
        raise NotebookContractError(f"regular file required: {path}")
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def write_json(path: Path, value: dict[str, str | int | float | bool | None]) -> None:
    """Write a small JSON receipt atomically."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True), encoding="utf-8")
    temporary.replace(path)


def safe_extract_zip(archive: Path, destination: Path) -> list[str]:
    """Extract a ZIP after rejecting traversal, links, and duplicate members."""
    root = destination.resolve()
    destination.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive) as packed:
        members = packed.infolist()
        names = [member.filename for member in members]
        if len(names) != len(set(names)):
            raise NotebookContractError(f"duplicate ZIP member: {archive}")
        for member in members:
            relative = Path(member.filename)
            target = destination / relative
            mode = member.external_attr >> 16
            if (
                relative.is_absolute()
                or ".." in relative.parts
                or stat.S_ISLNK(mode)
                or not target.resolve().is_relative_to(root)
            ):
                raise NotebookContractError(f"unsafe ZIP member: {member.filename}")
        packed.extractall(destination)
    return names


@dataclass(frozen=True, slots=True)
class Budget:
    """Persistent four-hour GPU allocation with a reserved export window."""

    started_unix: float
    hard_deadline_unix: float
    stage_deadline_unix: float

    @classmethod
    def load_or_start(cls, receipt: Path, *, now: float | None = None) -> Budget:
        """Reuse the original allocation start instead of resetting it on rerun."""
        current = time.time() if now is None else now
        if receipt.exists():
            raw = json.loads(receipt.read_text(encoding="utf-8"))
            return cls(
                started_unix=float(raw["started_unix"]),
                hard_deadline_unix=float(raw["hard_deadline_unix"]),
                stage_deadline_unix=float(raw["stage_deadline_unix"]),
            )
        budget = cls(
            started_unix=current,
            hard_deadline_unix=current + GPU_CAP_SECONDS,
            stage_deadline_unix=current + GPU_CAP_SECONDS - EXPORT_RESERVE_SECONDS,
        )
        write_json(receipt, asdict(budget))
        return budget

    def require_stage_time(self, *, now: float | None = None) -> None:
        """Reject a costly stage once the export reserve has begun."""
        current = time.time() if now is None else now
        if current >= self.stage_deadline_unix:
            raise NotebookContractError("GPU stage deadline reached; reserve time for export")


def storage_receipt(root: Path) -> dict[str, str | int | float | bool | None]:
    """Capture CPU-side storage readiness without starting the GPU budget."""
    root.mkdir(parents=True, exist_ok=True)
    usage = shutil.disk_usage(root)
    probe = root / ".write_probe"
    probe.write_bytes(b"redesign-notebook")
    probe.unlink()
    return {
        "root": str(root.resolve()),
        "free_bytes": usage.free,
        "total_bytes": usage.total,
        "writable": True,
        "checked_unix": time.time(),
    }


def storage_plan(
    root: Path,
    manifest: Path,
    baseline_zip: Path | None,
) -> dict[str, str | int | float | bool | None]:
    """Estimate required space from configured inputs before extraction starts."""
    source_paths: set[Path] = set()
    if manifest.is_file():
        with manifest.open(newline="", encoding="utf-8") as stream:
            for row in csv.DictReader(stream):
                raw = row.get("path", "").strip()
                if raw:
                    path = Path(raw).expanduser()
                    source_paths.add(path if path.is_absolute() else manifest.parent / path)
    missing_sources = sum(not path.is_file() for path in source_paths)
    source_bytes = sum(path.stat().st_size for path in source_paths if path.is_file())
    archive_bytes = 0
    expanded_asset_bytes = 0
    if baseline_zip is not None and baseline_zip.is_file():
        archive_bytes = baseline_zip.stat().st_size
        with zipfile.ZipFile(baseline_zip) as packed:
            expanded_asset_bytes = sum(item.file_size for item in packed.infolist())
    cache_estimate = source_bytes * 12
    export_headroom = max(2 * 1024**3, source_bytes * 2) if source_paths else 0
    safety_headroom = 5 * 1024**3 if source_paths or baseline_zip is not None else 0
    required = source_bytes + archive_bytes + expanded_asset_bytes + cache_estimate + export_headroom + safety_headroom
    free = shutil.disk_usage(root).free
    return {
        "configured_source_count": len(source_paths),
        "missing_source_count": missing_sources,
        "source_bytes": source_bytes,
        "baseline_zip_bytes": archive_bytes,
        "expanded_asset_bytes": expanded_asset_bytes,
        "cache_estimate_bytes": cache_estimate,
        "cache_assumption": "12x configured audio bytes for dual-view features and checkpoints",
        "export_headroom_bytes": export_headroom,
        "safety_headroom_bytes": safety_headroom,
        "estimated_required_bytes": required,
        "free_bytes": free,
        "ready": missing_sources == 0 and required <= free,
    }


def require_storage(plan: dict[str, str | int | float | bool | None]) -> None:
    """Fail before preparation when configured inputs or estimated space are insufficient."""
    if int(plan["missing_source_count"]) > 0:
        raise NotebookContractError("configured manifest contains missing source files")
    if int(plan["estimated_required_bytes"]) > int(plan["free_bytes"]):
        raise NotebookContractError("estimated data, asset, cache, and export storage exceeds free space")


def record_stage(
    progress_path: Path,
    stage: str,
    status: str,
    *,
    detail: str,
) -> None:
    """Update the latest state of one notebook stage."""
    progress_path.parent.mkdir(parents=True, exist_ok=True)
    if progress_path.exists():
        raw = json.loads(progress_path.read_text(encoding="utf-8"))
    else:
        raw = {"contract": "redesign-notebook-v1", "stages": {}}
    raw["updated_unix"] = time.time()
    raw["stages"][stage] = {"status": status, "detail": detail, "unix": time.time()}
    temporary = progress_path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(raw, indent=2, sort_keys=True), encoding="utf-8")
    temporary.replace(progress_path)


def write_file_control(prediction_path: Path, output: Path, *, noisy_or: bool) -> dict[str, str | int | float | bool | None]:
    """Write a file-local max or noisy-OR control using the official component columns."""
    payload = json.loads(prediction_path.read_text(encoding="utf-8"))
    predictions = payload["predictions"]
    required = ("VOICE_FAKE_PROB", "MUSIC_FAKE_PROB", "VOICE_PRESENT_PROB", "MUSIC_PRESENT_PROB")
    for identifier, row in predictions.items():
        if not set(required) <= set(row):
            raise NotebookContractError(f"official prediction columns are missing: {identifier}")
    result = copy.deepcopy(payload)
    for row in result["predictions"].values():
        voice = float(row["VOICE_PRESENT_PROB"]) * float(row["VOICE_FAKE_PROB"])
        music = float(row["MUSIC_PRESENT_PROB"]) * float(row["MUSIC_FAKE_PROB"])
        row["FILE_FAKE_PROB"] = 1.0 - (1.0 - voice) * (1.0 - music) if noisy_or else max(voice, music)
    result["file_relation"] = "noisy_or" if noisy_or else "max"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    return {
        "output": str(output),
        "rows": len(predictions),
        "relation": result["file_relation"],
        "sha256": sha256_file(output),
    }


def export_results(inputs: tuple[Path, ...], archive: Path) -> dict[str, str | int | float | bool | None]:
    """Create a deterministic result ZIP and verify every member by readback hash."""
    members: list[tuple[Path, str]] = []
    for root in inputs:
        if not root.exists():
            continue
        if root.is_symlink():
            raise NotebookContractError(f"export input cannot be a symlink: {root}")
        paths = (root,) if root.is_file() else tuple(path for path in root.rglob("*") if path.is_file())
        for path in paths:
            if path.is_symlink() or path.resolve() == archive.resolve():
                continue
            relative = path.name if root.is_file() else f"{root.name}/{path.relative_to(root).as_posix()}"
            members.append((path, relative))
    if not members:
        raise NotebookContractError("no result files are ready to export")
    archive.parent.mkdir(parents=True, exist_ok=True)
    temporary = archive.with_suffix(archive.suffix + ".tmp")
    hashes = {relative: sha256_file(path) for path, relative in members}
    with zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as packed:
        for path, relative in sorted(members, key=lambda item: item[1]):
            packed.write(path, relative)
        packed.writestr("EXPORT_SHA256.json", json.dumps(hashes, indent=2, sort_keys=True))
    temporary.replace(archive)
    with zipfile.ZipFile(archive) as packed:
        bad = packed.testzip()
        if bad is not None:
            raise NotebookContractError(f"ZIP CRC readback failed: {bad}")
        for relative, expected in hashes.items():
            actual = hashlib.sha256(packed.read(relative)).hexdigest()
            if actual != expected:
                raise NotebookContractError(f"ZIP hash readback failed: {relative}")
    return {
        "archive": str(archive.resolve()),
        "archive_sha256": sha256_file(archive),
        "member_count": len(hashes),
        "bytes": archive.stat().st_size,
        "verified": True,
    }


def main() -> None:
    """Run one meaningful self-check for budget persistence and ZIP readback."""
    import tempfile

    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        receipt = root / "budget.json"
        first = Budget.load_or_start(receipt, now=100.0)
        second = Budget.load_or_start(receipt, now=999.0)
        if first != second or second.started_unix != 100.0:
            raise NotebookContractError("budget start was reset")
        payload = root / "result"
        payload.mkdir()
        (payload / "answer.json").write_text('{"ok": true}', encoding="utf-8")
        report = export_results((payload,), root / "result.zip")
        if not report["verified"]:
            raise NotebookContractError("export readback was not verified")
        print(json.dumps({"budget_persistence": True, "export": report}, indent=2))


if __name__ == "__main__":
    main()
