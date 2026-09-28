# /// script
# requires-python = ">=3.11"
# dependencies = []
# ///
# ─── How to run ───
# uv run mert12_a100_watch.py
"""Monitor the MERT12 A100 run, verify its export, and pause compute."""

from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
import subprocess
import time
import zipfile
from pathlib import Path
from typing import Final

import redesign_h100_watch as shared


ROOT: Final = Path(__file__).resolve().parent
OUTPUT: Final = ROOT / "results" / "mert12_a100_20260918"
WORKSPACE: Final = "wsp-5i3knd3gxeox"
PREFIX: Final = "mert12-a100-20260918/"
STOPPED: Final = {"standby", "stopped"}
VERIFY_PARTIAL_EXPORT: Final = shared._verify_export


def pause_and_verify(run: dict[str, object], reason: str) -> None:
    """Pause the fixed workspace and accept either VESSL stop-state field."""
    workspace = str(run["workspace"])
    if workspace != WORKSPACE:
        raise ValueError("refusing to pause an unexpected workspace")
    requested_at = time.time()
    shared._call(["workspace", "pause", workspace])
    observed = "unknown"
    for _ in range(30):
        detail = json.loads(shared._call(["workspace", "show", workspace, "-o", "json"]))
        workspace_detail = detail.get("workspace", {})
        states = {
            str(workspace_detail.get("state", "unknown")).lower(),
            str(workspace_detail.get("phase", "unknown")).lower(),
        }
        observed = "/".join(sorted(states))
        if workspace_detail.get("state") == "stopped" and workspace_detail.get("phase") == "standby":
            receipt = {
                "schema_version": 1,
                "reason": reason,
                "workspace": workspace,
                "pause_requested_at": requested_at,
                "verified_at": time.time(),
                "observed": observed,
                "pause_verified": True,
            }
            (OUTPUT / "resource_stop.json").write_text(
                json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8"
            )
            print(f"PAUSE_VERIFIED {reason} {observed}", flush=True)
            return
        time.sleep(2)
    raise RuntimeError(f"workspace pause was not verified; observed {observed!r}")


def download_flat(run: dict[str, object], name: str, timeout: int = 300) -> Path:
    """Download one top-level object while excluding the potentially multi-GB output tree."""
    shared._safe_member_name(name)
    incoming = OUTPUT / ".incoming" / name
    if incoming.exists():
        shutil.rmtree(incoming)
    incoming.mkdir(parents=True)
    remaining = math.ceil(float(run["pause_deadline_epoch"]) - time.time())
    if remaining <= 0:
        raise TimeoutError(f"hard deadline reached before downloading {name}")
    listing = json.loads(shared._call([
        "volume", "ls", str(run["object_volume"]),
        "--prefix", str(run["object_prefix"]), "-o", "json",
    ], min(90, remaining)))
    prefix = str(run["object_prefix"]); top_files: set[str] = set()
    for entry in listing:
        if not isinstance(entry, dict) or entry.get("type") != "file":
            continue
        key = entry.get("key")
        if not isinstance(key, str) or not key.startswith(prefix):
            raise RuntimeError("volume ls returned a file outside the configured prefix")
        relative = key[len(prefix):]
        if relative and "/" not in relative:
            top_files.add(relative)
    if name not in top_files:
        raise RuntimeError(f"remote object is not available: {name}")
    arguments = ["volume", "download", str(run["object_volume"]), str(incoming),
                 "--remote-prefix", prefix, "--overwrite",
                 "--exclude", "output", "--exclude", "output/**"]
    for excluded in sorted((top_files | shared.KNOWN_REMOTE_FILES) - {name}):
        arguments.extend(("--exclude", excluded))
    shared._call(arguments, min(timeout, remaining))
    downloaded = [path for path in incoming.rglob("*") if path.is_file()]
    if len(downloaded) != 1 or downloaded[0].name != name:
        raise RuntimeError(f"download of {name} produced unexpected files")
    destination = OUTPUT / "downloaded" / name
    destination.parent.mkdir(parents=True, exist_ok=True)
    os.replace(downloaded[0], destination)
    shutil.rmtree(incoming)
    return destination


def verify_submission_export(
    run: dict[str, object], state: dict[str, object]
) -> dict[str, object]:
    """Verify the single submission ZIP without creating a second multi-GB archive."""
    if state.get("status") == "failed":
        return VERIFY_PARTIAL_EXPORT(run, state)
    expected = shared._digest(state.get("archive_sha256"), "run_state archive_sha256")
    archive = OUTPUT / "downloaded" / "results.zip"
    if not archive.is_file() or shared._sha256(archive) != expected:
        archive = shared._download(run, "results.zip", timeout=900)
    sidecar = shared._download(run, "results.zip.sha256")
    if shared._digest(sidecar.read_text().split()[0], "sidecar") != expected:
        raise ValueError("results.zip sidecar does not match run state")
    receipt = json.loads(shared._download(run, "export_receipt.json").read_text())
    package = json.loads(shared._download(run, "package_receipt.json").read_text())
    if (
        receipt.get("archive_verified") is not True
        or receipt.get("archive_sha256") != expected
        or package.get("sha256") != expected
        or package.get("bytes") != archive.stat().st_size
        or package.get("crc_verified") is not True
    ):
        raise ValueError("export and package receipts do not bind the downloaded archive")
    if shared._sha256(archive) != expected:
        raise ValueError("downloaded submission archive SHA-256 mismatch")
    members = package.get("member_sha256")
    if not isinstance(members, dict):
        raise ValueError("package receipt has no member hash manifest")
    with zipfile.ZipFile(archive) as packed:
        infos = packed.infolist()
        names = [info.filename for info in infos]
        if len(names) != len(set(names)) or set(names) != set(members):
            raise ValueError("submission ZIP member set mismatch")
        if packed.testzip() is not None:
            raise ValueError("submission ZIP CRC failure")
        for info in infos:
            shared._safe_member_name(info.filename)
            with packed.open(info) as stream:
                actual = hashlib.file_digest(stream, "sha256").hexdigest()
            if actual != shared._digest(members[info.filename], info.filename):
                raise ValueError(f"submission ZIP member SHA-256 mismatch: {info.filename}")
    evidence_digest = shared._digest(state.get("evidence_sha256"), "evidence_sha256")
    evidence = shared._download(run, "evidence.zip", timeout=300)
    evidence_sidecar = shared._download(run, "evidence.zip.sha256")
    if shared._digest(evidence_sidecar.read_text().split()[0], "evidence sidecar") != evidence_digest:
        raise ValueError("evidence sidecar does not match run state")
    evidence_result = shared.verify_archive(
        evidence, evidence_digest, shared.OUTPUT / "downloaded" / "evidence"
    )
    return {
        "schema_version": 1,
        "status": state["status"],
        "archive_sha256": expected,
        "archive_bytes": archive.stat().st_size,
        "files_verified": len(members),
        "single_submission_archive": True,
        "evidence": evidence_result,
        "verified_at": time.time(),
    }


def main() -> None:
    shared.OUTPUT = OUTPUT
    shared.RUN_METADATA = OUTPUT / "workspace_run.json"
    shared.EXPECTED_PREFIX = PREFIX
    shared.POLL_SECONDS = 300
    shared.MAX_SECONDS = 2 * 60 * 60
    shared.KNOWN_REMOTE_FILES = {
        "bootstrap.sh",
        "source.zip",
        "features.zip",
        "full_data_manifest.csv",
        "joined_data_audit.csv",
        "joined_data_audit.json",
        "asset_hashes.json",
        "smoke_inputs.zip",
        "supplemental_real_source.wav",
        "DACON_MERT12_QA_Recovery.ipynb",
        "upload_manifest.json",
        "workspace_run.json",
        "run_state.json",
        "results.zip",
        "results.zip.sha256",
        "export_receipt.json",
        "package_receipt.json",
        "evidence.zip",
        "evidence.zip.sha256",
    }
    shared._pause_and_verify = pause_and_verify
    shared._download = download_flat
    shared._verify_export = verify_submission_export
    shared.monitor()


if __name__ == "__main__":
    main()
