# /// script
# requires-python = ">=3.11"
# dependencies = []
# ///
# ─── How to run ───
# uv run mert12_a100_launch.py prepare
# uv run mert12_a100_launch.py launch --release
"""Stage and explicitly release the existing A100 for the MERT12 refit."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
import time
import urllib.parse
from pathlib import Path
from typing import Final


ROOT: Final = Path(__file__).resolve().parent
WORK: Final = ROOT / "_workspace" / "mert12_a100_20260918"
UPLOAD: Final = WORK / "upload"
RESULTS: Final = ROOT / "results" / "mert12_a100_20260918"
CLI: Final = ROOT / "_workspace" / "bin" / "vesslctl"
CLI_PREFIX: Final = (str(CLI), "--org", "qwdkw", "--team", "Default")
WORKSPACE: Final = "wsp-5i3knd3gxeox"
EXPECTED_NAME: Final = "jolly-routine"
EXPECTED_RESOURCE: Final = "resourcespec-a100x1"
TERMINAL_HELPER: Final = WORK / "jupyter_terminal.mjs"
VOLUME: Final = "objvol-a5gxdg7er70i"
PREFIX: Final = "mert12-a100-20260918/"
MAX_SECONDS: Final = 2 * 60 * 60


def digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def call(arguments: list[str], *, capture: bool = False, input_text: str | None = None) -> str:
    process = subprocess.run(
        [*CLI_PREFIX, *arguments],
        check=True,
        capture_output=capture,
        input=input_text,
        text=True,
        timeout=900,
    )
    return process.stdout if capture else ""


def workspace_detail() -> dict[str, object]:
    payload = json.loads(call(["workspace", "show", WORKSPACE, "-o", "json"], capture=True))
    detail = payload.get("workspace")
    if not isinstance(detail, dict):
        raise RuntimeError("workspace show did not return workspace details")
    if detail.get("slug") != WORKSPACE or detail.get("name") != EXPECTED_NAME:
        raise RuntimeError("workspace identity mismatch")
    if detail.get("resourceSpecSlug") != EXPECTED_RESOURCE:
        raise RuntimeError("workspace is not the expected A100 SXM 80GB resource")
    return detail


def jupyter_endpoint() -> str:
    payload = json.loads(call(["workspace", "show", WORKSPACE, "-o", "json"], capture=True))
    endpoints = payload.get("endpoints")
    if not isinstance(endpoints, list):
        raise RuntimeError("workspace show did not return endpoints")
    matches = [entry for entry in endpoints if entry.get("name") == "jupyter"]
    if len(matches) != 1 or not isinstance(matches[0].get("httpUrl"), str):
        raise RuntimeError("workspace has no unique Jupyter endpoint")
    url = matches[0]["httpUrl"]
    parsed = urllib.parse.urlparse(url)
    if parsed.scheme != "https" or WORKSPACE not in (parsed.hostname or ""):
        raise RuntimeError("unexpected Jupyter endpoint")
    return url


def local_manifest() -> dict[str, dict[str, int | str]]:
    files = sorted(path for path in UPLOAD.iterdir() if path.is_file())
    required = {
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
    }
    missing = required - {path.name for path in files}
    if missing:
        raise FileNotFoundError(f"upload bundle is missing {sorted(missing)}")
    return {
        path.name: {"bytes": path.stat().st_size, "sha256": digest(path)}
        for path in files
        if path.name != "upload_manifest.json"
    }


def prepare() -> None:
    detail = workspace_detail()
    if str(detail.get("state", "")).lower() not in {"standby", "stopped"}:
        raise RuntimeError("refusing upload preparation while A100 compute is active")
    manifest = local_manifest()
    UPLOAD.joinpath("upload_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    call(["volume", "upload", VOLUME, str(UPLOAD), "--remote-prefix", PREFIX, "--overwrite"])
    readback = WORK / "readback"
    if readback.exists():
        shutil.rmtree(readback)
    readback.mkdir(parents=True)
    call(["volume", "download", VOLUME, str(readback), "--remote-prefix", PREFIX, "--overwrite"])
    found = {path.name: path for path in readback.rglob("*") if path.is_file()}
    expected = {**manifest, "upload_manifest.json": {
        "bytes": UPLOAD.joinpath("upload_manifest.json").stat().st_size,
        "sha256": digest(UPLOAD / "upload_manifest.json"),
    }}
    if set(found) != set(expected):
        raise RuntimeError("object-volume readback file set mismatch")
    for name, record in expected.items():
        if found[name].stat().st_size != record["bytes"] or digest(found[name]) != record["sha256"]:
            raise RuntimeError(f"object-volume readback mismatch: {name}")
    RESULTS.mkdir(parents=True, exist_ok=True)
    receipt = {
        "workspace": WORKSPACE,
        "workspace_state": detail.get("state"),
        "object_volume": VOLUME,
        "object_prefix": PREFIX,
        "files": expected,
        "gpu_started": False,
        "verified_at": time.time(),
    }
    RESULTS.joinpath("storage_preparation.json").write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def launch(release: bool) -> None:
    if not release:
        raise RuntimeError("launch requires the root operator's explicit --release gate")
    run_path = RESULTS / "workspace_run.json"
    if run_path.exists():
        raise RuntimeError("workspace_run.json already exists; refusing to reset the original deadline")
    listing = json.loads(call(["volume", "ls", VOLUME, "--prefix", PREFIX, "-o", "json"], capture=True))
    if any(entry.get("key") == PREFIX + "workspace_run.json" for entry in listing):
        raise RuntimeError("remote workspace_run.json exists; inspect the prior attempt")
    prepared = json.loads(RESULTS.joinpath("storage_preparation.json").read_text(encoding="utf-8"))
    if prepared.get("gpu_started") is not False:
        raise RuntimeError("invalid storage preparation receipt")
    expected = prepared.get("files")
    if not isinstance(expected, dict):
        raise RuntimeError("storage preparation receipt has no file manifest")
    current = {
        **local_manifest(),
        "upload_manifest.json": {
            "bytes": (UPLOAD / "upload_manifest.json").stat().st_size,
            "sha256": digest(UPLOAD / "upload_manifest.json"),
        },
    }
    if current != expected:
        raise RuntimeError("local upload bundle changed after verified preparation")
    detail = workspace_detail()
    if str(detail.get("state", "")).lower() not in {"standby", "stopped"}:
        raise RuntimeError("workspace is already active; refusing duplicate launch")
    if detail.get("initScript") not in {None, ""}:
        raise RuntimeError("workspace initScript is not empty; refusing stale automatic bootstrap")
    started = time.time()
    run = {
        "schema_version": 1,
        "workspace": WORKSPACE,
        "object_volume": VOLUME,
        "object_prefix": PREFIX,
        "started_epoch": started,
        "pause_deadline_epoch": started + MAX_SECONDS,
        "poll_seconds": 300,
    }
    run_path.write_text(
        json.dumps(run, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    call(["volume", "upload", VOLUME, str(run_path), "--remote-prefix", PREFIX, "--overwrite"])
    watch_log = RESULTS / "watch.log"
    with watch_log.open("ab") as log:
        watcher = subprocess.Popen(
            [sys.executable, str(ROOT / "mert12_a100_watch.py")],
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
    RESULTS.joinpath("watch_pid.txt").write_text(f"{watcher.pid}\n", encoding="utf-8")
    time.sleep(1)
    if watcher.poll() is not None:
        raise RuntimeError("watcher exited before workspace start")
    started_compute = False
    try:
        call(["workspace", "start", WORKSPACE])
        started_compute = True
        for _ in range(60):
            detail = workspace_detail()
            states = {
                str(detail.get("state", "unknown")).lower(),
                str(detail.get("phase", "unknown")).lower(),
            }
            if "running" in states:
                break
            time.sleep(5)
        else:
            raise TimeoutError("A100 did not reach running state within five minutes")
        command = (
            "set -euo pipefail; "
            "test -z \"$(pgrep -f '[m]ert12_full_refit.py' || true)\"; "
            f"nohup bash /root-1/{PREFIX}bootstrap.sh {started:.6f} {started + MAX_SECONDS:.6f} "
            f"> /root-1/{PREFIX}bootstrap.log 2>&1 & pid=$!; "
            "sleep 2; kill -0 \"$pid\"; echo MERT12_DISPATCHED_PID=$pid"
        )
        dispatch = subprocess.run(
            ["node", str(TERMINAL_HELPER)],
            input=json.dumps({"httpUrl": jupyter_endpoint(), "command": command}),
            capture_output=True,
            check=True,
            text=True,
            timeout=90,
        )
        dispatch_receipt = json.loads(dispatch.stdout)
        if not isinstance(dispatch_receipt.get("pid"), int):
            raise RuntimeError("Jupyter terminal dispatch returned no PID")
        RESULTS.joinpath("dispatch.json").write_text(
            json.dumps({**dispatch_receipt, "dispatched_at": time.time()}, indent=2) + "\n",
            encoding="utf-8",
        )
    except (OSError, RuntimeError, TimeoutError, subprocess.SubprocessError, json.JSONDecodeError):
        if started_compute:
            call(["workspace", "pause", WORKSPACE])
        raise


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("prepare", "launch"))
    parser.add_argument("--release", action="store_true")
    arguments = parser.parse_args()
    if arguments.action == "prepare":
        prepare()
    else:
        launch(arguments.release)


if __name__ == "__main__":
    main()
