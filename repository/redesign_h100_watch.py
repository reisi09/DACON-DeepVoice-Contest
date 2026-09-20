"""Monitor the bounded redesign H100 run, verify its export, and pause compute."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import stat
import subprocess
import tempfile
import time
import warnings
import zipfile
from pathlib import Path, PurePosixPath


ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "results" / "redesign_h100_20260917"
RUN_METADATA = OUTPUT / "workspace_run.json"
CLI = ROOT / "_workspace" / "bin" / "vesslctl"
CLI_PREFIX = (str(CLI), "--org", "qwdkw", "--team", "Default")
EXPECTED_VOLUME = "objvol-a5gxdg7er70i"
EXPECTED_PREFIX = "redesign-h100-20260917/"
TERMINAL_STATUSES = {"completed", "exported", "failed"}
STOPPED_PHASES = {"standby", "stopped"}
POLL_SECONDS = 60
MAX_SECONDS = 4 * 60 * 60
KNOWN_REMOTE_FILES = {
    "input.zip", "redesign_h100_run.sh", "upload_receipt.json", "run_state.json",
    "results.zip", "results.zip.sha256", "export_receipt.json",
}


def _sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _digest(value: object, name: str) -> str:
    if not isinstance(value, str) or len(value) != 64 or any(
        character not in "0123456789abcdef" for character in value
    ):
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")
    return value


def _safe_member_name(name: str) -> str:
    if not name or "\\" in name or name.startswith("/"):
        raise ValueError(f"unsafe ZIP member path: {name!r}")
    path = PurePosixPath(name)
    if any(part in {"", ".", ".."} for part in path.parts) or path.is_absolute():
        raise ValueError(f"unsafe ZIP member path: {name!r}")
    normalized = path.as_posix()
    if normalized != name:
        raise ValueError(f"non-canonical ZIP member path: {name!r}")
    return normalized


def verify_archive(archive: Path, expected_sha256: str, target: Path) -> dict[str, object]:
    """Verify the whole ZIP and every declared regular member before extraction."""
    expected_sha256 = _digest(expected_sha256, "archive_sha256")
    actual_archive_sha256 = _sha256(archive)
    if actual_archive_sha256 != expected_sha256:
        raise ValueError("results.zip SHA-256 does not match run state")

    with zipfile.ZipFile(archive) as packed:
        if packed.testzip() is not None:
            raise ValueError("results.zip has a CRC failure")
        infos = packed.infolist()
        names: set[str] = set()
        for info in infos:
            name = _safe_member_name(info.filename)
            if name in names:
                raise ValueError(f"duplicate ZIP member: {name}")
            names.add(name)
            if info.is_dir():
                raise ValueError(f"directory ZIP member is not export data: {name}")
            file_type = (info.external_attr >> 16) & 0o170000
            if file_type not in (0, stat.S_IFREG):
                raise ValueError(f"link or special ZIP member is forbidden: {name}")

        manifest_name = "EXPORT_SHA256.json"
        if manifest_name not in names:
            raise ValueError("results.zip is missing EXPORT_SHA256.json")
        try:
            manifest = json.loads(packed.read(manifest_name))
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise ValueError("EXPORT_SHA256.json is invalid") from error
        if not isinstance(manifest, dict):
            raise ValueError("EXPORT_SHA256.json must be an object")
        payload_names = names - {manifest_name}
        if set(manifest) != payload_names:
            raise ValueError("EXPORT_SHA256.json must cover every payload member exactly once")
        for name, expected in manifest.items():
            _safe_member_name(name)
            expected = _digest(expected, f"member digest for {name}")
            if hashlib.sha256(packed.read(name)).hexdigest() != expected:
                raise ValueError(f"ZIP member SHA-256 mismatch: {name}")

        if target.exists():
            shutil.rmtree(target)
        target.mkdir(parents=True)
        target_root = target.resolve()
        for info in infos:
            destination = (target / info.filename).resolve()
            if not destination.is_relative_to(target_root):
                raise ValueError(f"ZIP member escapes extraction root: {info.filename}")
            destination.parent.mkdir(parents=True, exist_ok=True)
            with packed.open(info) as source, destination.open("xb") as sink:
                shutil.copyfileobj(source, sink)

    return {
        "archive_sha256": actual_archive_sha256,
        "archive_bytes": archive.stat().st_size,
        "files_verified": len(payload_names),
    }


def _load_run() -> dict[str, object]:
    run = json.loads(RUN_METADATA.read_text(encoding="utf-8"))
    if not isinstance(run, dict):
        raise ValueError("workspace_run.json must be an object")
    required = {"workspace", "pause_deadline_epoch", "object_volume", "object_prefix"}
    if not required <= run.keys():
        raise ValueError(f"workspace_run.json is missing {sorted(required - set(run))}")
    if not isinstance(run["workspace"], str) or not run["workspace"].startswith("wsp-"):
        raise ValueError("workspace_run.json has an invalid workspace slug")
    if run["object_volume"] != EXPECTED_VOLUME or run["object_prefix"] != EXPECTED_PREFIX:
        raise ValueError("workspace_run.json points at an unexpected object volume or prefix")
    deadline = float(run["pause_deadline_epoch"])
    started = float(run.get("started_epoch", time.time()))
    if not math.isfinite(deadline) or not math.isfinite(started):
        raise ValueError("run start and pause deadline must be finite")
    if deadline - started > MAX_SECONDS:
        raise ValueError("pause deadline exceeds four hours from run start")
    return run


def _call(arguments: list[str], timeout: int = 90) -> str:
    process = subprocess.run(
        [*CLI_PREFIX, *arguments], capture_output=True, text=True, timeout=timeout, check=False,
    )
    if process.returncode:
        command = " ".join(arguments[:2])
        raise RuntimeError(f"vesslctl {command} failed with exit code {process.returncode}")
    return process.stdout


def _download(run: dict[str, object], name: str, timeout: int = 90) -> Path:
    incoming = OUTPUT / "downloaded" / ".incoming" / name.replace("/", "_")
    if incoming.exists():
        shutil.rmtree(incoming)
    incoming.mkdir(parents=True)
    remaining = math.ceil(float(run["pause_deadline_epoch"]) - time.time())
    if remaining <= 0:
        raise TimeoutError(f"hard deadline reached before downloading {name}")
    listing = json.loads(_call([
        "volume", "ls", str(run["object_volume"]),
        "--prefix", str(run["object_prefix"]), "-o", "json",
    ], min(90, remaining)))
    if not isinstance(listing, list):
        raise RuntimeError("volume ls did not return a JSON list")
    prefix = str(run["object_prefix"])
    remote_files: set[str] = set()
    for entry in listing:
        if not isinstance(entry, dict) or entry.get("type") != "file":
            continue
        key = entry.get("key")
        if not isinstance(key, str) or not key.startswith(prefix):
            raise RuntimeError("volume ls returned a file outside the configured prefix")
        relative = key[len(prefix):]
        if not relative or "/" in relative:
            raise RuntimeError("redesign H100 object prefix must remain flat")
        remote_files.add(relative)
    if name not in remote_files:
        raise RuntimeError(f"remote object is not available: {name}")
    arguments = [
        "volume", "download", str(run["object_volume"]), str(incoming),
        "--remote-prefix", prefix, "--overwrite",
    ]
    for excluded in sorted((remote_files | KNOWN_REMOTE_FILES) - {name}):
        arguments.extend(("--exclude", excluded))
    remaining = math.ceil(float(run["pause_deadline_epoch"]) - time.time())
    if remaining <= 0:
        raise TimeoutError(f"hard deadline reached before downloading {name}")
    _call(arguments, min(timeout, remaining))
    downloaded = [path for path in incoming.rglob("*") if path.is_file()]
    if len(downloaded) != 1 or downloaded[0].name != name:
        raise RuntimeError(f"download of {name} produced unexpected files")
    destination = OUTPUT / "downloaded" / name
    destination.parent.mkdir(parents=True, exist_ok=True)
    os.replace(downloaded[0], destination)
    shutil.rmtree(incoming)
    return destination


def _read_state(run: dict[str, object]) -> dict[str, object]:
    state_path = _download(run, "run_state.json")
    state = json.loads(state_path.read_text(encoding="utf-8"))
    if not isinstance(state, dict) or state.get("status") not in {
        "running", "completed", "exported", "failed",
    }:
        raise ValueError("run_state.json has an invalid status")
    shutil.copy2(state_path, OUTPUT / "run_state.json")
    return state


def _verify_export(run: dict[str, object], state: dict[str, object]) -> dict[str, object]:
    state_digest = _digest(state.get("archive_sha256"), "run_state archive_sha256")
    archive = _download(run, "results.zip", timeout=300)
    sidecar = _download(run, "results.zip.sha256")
    sidecar_parts = sidecar.read_text(encoding="utf-8").strip().split()
    if not sidecar_parts:
        raise ValueError("results.zip.sha256 is empty")
    sidecar_digest = _digest(sidecar_parts[0], "sidecar")
    if sidecar_digest != state_digest:
        raise ValueError("results.zip.sha256 does not match run_state.json")
    receipt_path = _download(run, "export_receipt.json")
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    if not isinstance(receipt, dict):
        raise ValueError("export_receipt.json must be an object")
    if (
        receipt.get("status") not in {"exported", "partial_exported"}
        or receipt.get("archive_ready") is not True
        or receipt.get("archive_verified") is not True
        or receipt.get("archive_sha256") != state_digest
    ):
        raise ValueError("export receipt does not prove the downloaded archive")
    verification = verify_archive(archive, state_digest, OUTPUT / "downloaded" / "unpacked")
    if receipt.get("archive_bytes") != verification["archive_bytes"]:
        raise ValueError("export receipt archive size does not match results.zip")
    if receipt.get("export_member_count") != verification["files_verified"]:
        raise ValueError("export receipt member count does not match EXPORT_SHA256.json")
    return {
        "schema_version": 1,
        "status": state["status"],
        **verification,
        "export_receipt": receipt,
        "verified_at": time.time(),
    }


def _pause_and_verify(run: dict[str, object], reason: str) -> None:
    workspace = str(run["workspace"])
    requested_at = time.time()
    _call(["workspace", "pause", workspace])
    phase = "unknown"
    stopped = False
    for _ in range(30):
        detail = json.loads(_call(["workspace", "show", workspace, "-o", "json"]))
        phase = str(detail.get("workspace", {}).get("phase", "unknown")).lower()
        stopped = phase in STOPPED_PHASES and detail.get("workspace", {}).get("state") == "stopped"
        if stopped:
            break
        time.sleep(2)
    receipt = {
        "schema_version": 1,
        "reason": reason,
        "workspace": workspace,
        "pause_requested_at": requested_at,
        "verified_at": time.time(),
        "phase": phase,
        "pause_verified": stopped,
    }
    (OUTPUT / "resource_stop.json").write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8",
    )
    if not stopped:
        raise RuntimeError(f"workspace pause was not verified; observed phase {phase!r}")
    print(f"PAUSE_VERIFIED {reason} {phase}", flush=True)


def monitor() -> None:
    run = _load_run()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    while True:
        now = time.time()
        if now >= float(run["pause_deadline_epoch"]):
            try:
                _pause_and_verify(run, "four_hour_deadline")
                return
            except (
                OSError, RuntimeError, ValueError, subprocess.TimeoutExpired,
                json.JSONDecodeError,
            ) as error:
                print(
                    f"STOP_RETRY {int(time.time())} {type(error).__name__}: {error}",
                    flush=True,
                )
                time.sleep(10)
                continue
        try:
            state = _read_state(run)
            status = str(state["status"])
            if state.get("archive_ready") is True:
                try:
                    verification = _verify_export(run, state)
                except (
                    OSError, RuntimeError, ValueError, subprocess.TimeoutExpired,
                    json.JSONDecodeError, zipfile.BadZipFile,
                ) as error:
                    if status not in TERMINAL_STATUSES:
                        raise
                    failure = {
                        "schema_version": 1, "status": "rejected", "run_status": status,
                        "error_type": type(error).__name__, "error": str(error),
                        "verified_at": time.time(),
                    }
                    (OUTPUT / "download_verification.json").write_text(
                        json.dumps(failure, indent=2, sort_keys=True) + "\n", encoding="utf-8",
                    )
                    _pause_and_verify(run, f"terminal_export_verification_failed_{status}")
                    return
                (OUTPUT / "download_verification.json").write_text(
                    json.dumps(verification, indent=2, sort_keys=True) + "\n", encoding="utf-8",
                )
                _pause_and_verify(run, f"archive_verified_{status}")
                return
            if status in TERMINAL_STATUSES:
                _pause_and_verify(run, f"terminal_without_archive_{status}")
                return
        except (
            OSError, RuntimeError, ValueError, subprocess.TimeoutExpired,
            json.JSONDecodeError, zipfile.BadZipFile,
        ) as error:
            print(f"MONITOR_RETRY {int(time.time())} {type(error).__name__}: {error}", flush=True)
        remaining = float(run["pause_deadline_epoch"]) - time.time()
        time.sleep(max(0.0, min(POLL_SECONDS, remaining)))


def self_check() -> None:
    """Exercise valid, corrupt, traversal, link, and duplicate archive handling locally."""
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)

        def make(path: Path, entries: list[tuple[zipfile.ZipInfo | str, bytes]], manifest: dict) -> None:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", UserWarning)
                with zipfile.ZipFile(path, "w") as packed:
                    for name, data in entries:
                        packed.writestr(name, data)
                    packed.writestr("EXPORT_SHA256.json", json.dumps(manifest))

        payload = b"verified payload"
        digest = hashlib.sha256(payload).hexdigest()
        valid = root / "valid.zip"
        make(valid, [("result.json", payload)], {"result.json": digest})
        verified = verify_archive(valid, _sha256(valid), root / "valid")
        assert verified["files_verified"] == 1

        corrupt = root / "corrupt.zip"
        make(corrupt, [("result.json", payload)], {"result.json": "0" * 64})
        traversal = root / "traversal.zip"
        make(traversal, [("../escape.json", payload)], {"../escape.json": digest})
        link = root / "link.zip"
        link_info = zipfile.ZipInfo("link")
        link_info.create_system = 3
        link_info.external_attr = (stat.S_IFLNK | 0o777) << 16
        make(link, [(link_info, b"target")], {"link": hashlib.sha256(b"target").hexdigest()})
        duplicate = root / "duplicate.zip"
        make(
            duplicate,
            [("same.json", payload), ("same.json", payload)],
            {"same.json": digest},
        )
        for archive in (corrupt, traversal, link, duplicate):
            try:
                verify_archive(archive, _sha256(archive), root / archive.stem)
            except ValueError:
                continue
            raise AssertionError(f"unsafe archive was accepted: {archive.name}")

    original_load_run = globals()["_load_run"]
    original_pause = globals()["_pause_and_verify"]
    original_sleep = time.sleep
    pause_attempts = 0
    stop_sleeps: list[float] = []

    def expired_run() -> dict[str, object]:
        return {"pause_deadline_epoch": 0.0}

    def fail_once_then_stop(run: dict[str, object], reason: str) -> None:
        nonlocal pause_attempts
        pause_attempts += 1
        if pause_attempts == 1:
            raise RuntimeError("transient pause failure")
        assert reason == "four_hour_deadline"

    def capture_sleep(seconds: float) -> None:
        stop_sleeps.append(seconds)

    try:
        globals()["_load_run"] = expired_run
        globals()["_pause_and_verify"] = fail_once_then_stop
        time.sleep = capture_sleep
        monitor()
    finally:
        globals()["_load_run"] = original_load_run
        globals()["_pause_and_verify"] = original_pause
        time.sleep = original_sleep
    assert pause_attempts == 2 and stop_sleeps == [10]
    print(
        "SELF_CHECK_OK valid=1 corrupt_rejected=1 traversal_rejected=1 "
        "link_rejected=1 duplicate_rejected=1 deadline_pause_retry=1"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--self-check", action="store_true")
    arguments = parser.parse_args()
    if arguments.self_check:
        self_check()
    else:
        monitor()


if __name__ == "__main__":
    main()
