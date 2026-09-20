# /// script
# requires-python = ">=3.11"
# dependencies = []
# ///
# Run: .venv/bin/python localization_a100_ops.py prepare
# Run: .venv/bin/python localization_a100_ops.py launch --release
"""Stage, release, monitor, and stop the existing A100 within one immutable deadline."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Final, TypedDict
from urllib.parse import urlparse

import redesign_h100_watch as shared


ROOT: Final = Path(__file__).resolve().parent
WORKSPACE: Final = "wsp-5i3knd3gxeox"
VOLUME: Final = "objvol-a5gxdg7er70i"
PREFIX: Final = "localization-a100-20260918/"
UPLOAD = ROOT / "_workspace/localization_a100_20260918/upload"
OUTPUT = ROOT / "results/localization_a100_20260918"
MAX_SECONDS: Final = 14_400
TERMINAL_HELPER: Final = ROOT / "_workspace/mert12_a100_20260918/jupyter_terminal.mjs"
ORIGINAL_LOAD_RUN: Final = shared._load_run
ORIGINAL_PAUSE: Final = shared._pause_and_verify
ENTRYPOINT = Path(__file__).resolve()
OBJECT_MOUNT = "/root-1"
ENDPOINT_SCHEME = "https"
ENDPOINT_HOST = None
DISPATCH_VIA_STARTUP = False


class FileDigest(TypedDict):
    bytes: int
    sha256: str


class Run(TypedDict):
    workspace: str
    object_volume: str
    object_prefix: str
    started_epoch: float
    pause_deadline_epoch: float
    poll_seconds: int


@dataclass(frozen=True, slots=True)
class Workspace:
    state: str
    phase: str
    jupyter: str


def configure() -> None:
    """Route the existing monitor and verifier to this run without editing old runs."""
    shared.OUTPUT = OUTPUT
    shared.RUN_METADATA = OUTPUT / "workspace_run.json"
    shared.EXPECTED_PREFIX = PREFIX
    shared.EXPECTED_VOLUME = VOLUME
    shared.MAX_SECONDS = MAX_SECONDS
    shared.POLL_SECONDS = 300
    shared.KNOWN_REMOTE_FILES = {
        "bootstrap.sh", "input.zip", "upload_manifest.json", "workspace_run.json",
        "run_state.json", "results.zip", "results.zip.sha256", "export_receipt.json",
        "progress.json",
    }
    shared._load_run = load_run
    shared._pause_and_verify = pause_and_verify


def load_run() -> Run:
    """Keep the original deadline even when resuming an expired monitor."""
    run = ORIGINAL_LOAD_RUN()
    started = float(run["started_epoch"])
    deadline = float(run["pause_deadline_epoch"])
    if run["workspace"] != WORKSPACE or not 0 < deadline - started <= MAX_SECONDS:
        raise ValueError("run identity or four-hour deadline is invalid")
    return {"workspace": WORKSPACE, "object_volume": VOLUME, "object_prefix": PREFIX,
            "started_epoch": started, "pause_deadline_epoch": deadline, "poll_seconds": 300}


def workspace_detail(*, stopped: bool = False) -> Workspace:
    """Whitelist identity/state fields; keep credential-bearing endpoints out of logs."""
    raw = json.loads(shared._call(["workspace", "show", WORKSPACE, "-o", "json"]))
    item, gpu = raw["workspace"], raw["billingInfo"]
    if (item["slug"] != WORKSPACE or item["name"] != "jolly-routine"
            or item["resourceSpecSlug"] != "resourcespec-a100x1"
            or gpu["resourceSpecAcceleratorName"] != "A100 SXM"
            or gpu["resourceSpecAcceleratorCount"] != 1 or item.get("initScript")):
        raise RuntimeError("unexpected A100 identity/resource or stale automatic bootstrap")
    if stopped and (item["state"] != "stopped" or item["phase"] != "standby"):
        raise RuntimeError("A100 must be stopped/standby before preparation or launch")
    endpoints = [entry["httpUrl"] for entry in raw.get("endpoints", [])
                 if entry.get("name") == "jupyter"]
    return Workspace(item["state"], item["phase"], endpoints[0] if len(endpoints) == 1 else "")


def bundle_manifest() -> dict[str, FileDigest]:
    """Bind runnable code and input bytes to the root-prepared upload manifest."""
    names = {"bootstrap.sh", "input.zip", "upload_manifest.json"}
    if {path.name for path in UPLOAD.iterdir()} != names:
        raise ValueError("upload directory must contain exactly bootstrap, input, and manifest")
    files = {name: FileDigest(bytes=(UPLOAD / name).stat().st_size,
                             sha256=shared._sha256(UPLOAD / name)) for name in names}
    declared = json.loads((UPLOAD / "upload_manifest.json").read_text())
    if declared != {name: files[name] for name in names - {"upload_manifest.json"}}:
        raise ValueError("upload manifest differs from current bootstrap/input bytes")
    return files


def fresh_prefix() -> None:
    """Refuse mixing new work with an existing run or non-bundle remote objects."""
    entries = json.loads(shared._call(["volume", "ls", VOLUME, "--prefix", PREFIX, "-o", "json"]))
    allowed = {PREFIX + name for name in ("bootstrap.sh", "input.zip", "upload_manifest.json")}
    if any(entry.get("type") != "file" or entry.get("key") not in allowed for entry in entries):
        raise RuntimeError("remote run prefix is already in use; preserve its original deadline")


def prepare() -> None:
    """Upload and read back all bytes while the existing A100 remains stopped."""
    configure()
    if (OUTPUT / "workspace_run.json").exists():
        raise FileExistsError("existing run metadata prevents preparation overwrite")
    files = bundle_manifest()
    workspace_detail(stopped=True)
    fresh_prefix()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    (OUTPUT / "storage_preparation.json").unlink(missing_ok=True)
    shared._call(["volume", "upload", VOLUME, str(UPLOAD), "--remote-prefix", PREFIX,
                  "--overwrite"], timeout=1800)
    with tempfile.TemporaryDirectory(prefix="readback-", dir=OUTPUT) as temporary:
        target = Path(temporary)
        shared._call(["volume", "download", VOLUME, str(target), "--remote-prefix", PREFIX,
                      "--overwrite"], timeout=1800)
        paths = [path for path in target.rglob("*") if path.is_file()]
        actual = {path.name: {"bytes": path.stat().st_size, "sha256": shared._sha256(path)}
                  for path in paths}
        if len(paths) != len(files) or actual != files:
            raise ValueError("uploaded bundle failed full byte/hash readback")
    workspace_detail(stopped=True)
    (OUTPUT / "storage_preparation.json").write_text(json.dumps({
        "workspace": WORKSPACE, "object_volume": VOLUME, "object_prefix": PREFIX,
        "files": files, "gpu_started": False, "verified_at": time.time(),
    }, indent=2, sort_keys=True) + "\n")
    print("STORAGE_VERIFIED A100_stopped=True", flush=True)


def pause_and_verify(run: Run, reason: str) -> None:
    """Permit stop requests only for this fixed workspace and verify stopped state."""
    if run["workspace"] != WORKSPACE:
        raise ValueError("refusing to pause a different workspace")
    ORIGINAL_PAUSE(run, reason)
    workspace_detail(stopped=True)


def endpoint_allowed(url: str) -> bool:
    endpoint = urlparse(url)
    return endpoint.scheme == ENDPOINT_SCHEME and (
        endpoint.hostname == ENDPOINT_HOST if ENDPOINT_HOST else WORKSPACE in (endpoint.hostname or ""))


def launch(*, release: bool = False, resume: bool = False) -> None:
    """Release a verified bundle once; startup/dispatch failures pause the A100."""
    if not release:
        raise RuntimeError("launch requires the root operator's --release gate")
    configure()
    run_path = OUTPUT / "workspace_run.json"
    if run_path.exists() and not resume:
        raise FileExistsError("existing run metadata cannot be replaced or extended")
    receipt = json.loads((OUTPUT / "storage_preparation.json").read_text())
    if (receipt.get("workspace") != WORKSPACE or receipt.get("object_volume") != VOLUME
            or receipt.get("object_prefix") != PREFIX or receipt.get("gpu_started") is not False
            or receipt.get("files") != bundle_manifest()):
        raise ValueError("verified storage preparation does not match this launch")
    workspace_detail(stopped=True)
    if resume:
        run = load_run()
        stop = OUTPUT / "resource_stop.json"
        if ((OUTPUT / "dispatch.json").exists() or not stop.exists()
                or json.loads(stop.read_text()).get("reason") != "launch_failure"
                or time.time() >= run['pause_deadline_epoch'] - 600):
            raise RuntimeError("Only an undispatched launch failure can resume within its original deadline")
        entries = json.loads(shared._call(["volume", "ls", VOLUME, "--prefix", PREFIX, "-o", "json"]))
        allowed = {PREFIX + name for name in ("bootstrap.sh", "input.zip", "upload_manifest.json", "workspace_run.json")}
        if any(entry.get('key') not in allowed for entry in entries):
            raise RuntimeError("Remote execution evidence prevents pre-dispatch resume")
        stop.replace(OUTPUT / 'launch_failure_stop.json')
        started = run['started_epoch']
    else:
        fresh_prefix()
        started = time.time()
        run = Run(workspace=WORKSPACE, object_volume=VOLUME, object_prefix=PREFIX,
                  started_epoch=started, pause_deadline_epoch=started + MAX_SECONDS, poll_seconds=300)
        with run_path.open("x") as stream:
            stream.write(json.dumps(run, indent=2, sort_keys=True) + "\n")
    shared._call(["volume", "upload", VOLUME, str(run_path), "--remote-prefix", PREFIX,
                  "--overwrite"], timeout=300)
    with (OUTPUT / "watch.log").open("ab") as log:
        watcher = subprocess.Popen([sys.executable, str(ENTRYPOINT), "watch"],
                                   stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    (OUTPUT / "watch_pid.txt").write_text(f"{watcher.pid}\n")
    start_requested = False
    dispatched = False
    try:
        time.sleep(1)
        if watcher.poll() is not None:
            raise RuntimeError("monitor exited before compute start")
        start_requested = True
        shared._call(["workspace", "start", WORKSPACE])
        for _ in range(60):
            detail = workspace_detail()
            if (detail.state == "running" or detail.phase == "running"
                    or (DISPATCH_VIA_STARTUP and detail.phase == 'active' and detail.state == 'initializing')):
                break
            time.sleep(5)
        else:
            raise TimeoutError("A100 did not start within five minutes")
        if DISPATCH_VIA_STARTUP:
            (OUTPUT / 'dispatch.json').write_text(json.dumps({
                'method': 'verified_workspace_init_script', 'dispatched_at': time.time(),
                'original_deadline_epoch': run['pause_deadline_epoch'],
                'observed_state': detail.state, 'runtime_execution_confirmed': False,
            }, indent=2) + '\n')
            dispatched = True
            return
        if not endpoint_allowed(detail.jupyter):
            raise RuntimeError("missing or unexpected A100 Jupyter endpoint")
        command = (
            "set -euo pipefail; "
            f"nohup bash {OBJECT_MOUNT}/{PREFIX}bootstrap.sh {started:.6f} {started + MAX_SECONDS:.6f} "
            f"> {OBJECT_MOUNT}/{PREFIX}bootstrap.log 2>&1 < /dev/null & pid=$!; "
            "sleep 2; kill -0 \"$pid\"; echo MERT12_DISPATCHED_PID=$pid"
        )
        result = subprocess.run(["node", str(TERMINAL_HELPER)],
                                input=json.dumps({"httpUrl": detail.jupyter, "command": command}),
                                capture_output=True, text=True, check=True, timeout=90)
        dispatch = json.loads(result.stdout)
        if type(dispatch.get("pid")) is not int:
            raise RuntimeError("Jupyter dispatch returned no bootstrap PID")
        (OUTPUT / "dispatch.json").write_text(json.dumps({
            "pid": dispatch["pid"], "terminal": dispatch["terminal"], "dispatched_at": time.time(),
        }, indent=2) + "\n")
        dispatched = True
    finally:
        if not dispatched:
            try:
                if start_requested:
                    pause_and_verify(run, "launch_failure")
            finally:
                watcher.terminate()
    print(f"A100_DISPATCHED deadline={run['pause_deadline_epoch']:.6f}", flush=True)


def watch() -> None:
    """Keep the deadline guard independent of blocking archive verification."""
    run = load_run()
    guard = subprocess.Popen([sys.executable, str(ENTRYPOINT), "guard"])
    (OUTPUT / "deadline_guard_pid.txt").write_text(f"{guard.pid}\n")
    finished = False
    try:
        shared.monitor()
        finished = True
    finally:
        if not finished:
            pause_and_verify(run, "monitor_failure")
        guard.terminate()


def deadline_guard() -> None:
    """Pause on the original clock even if the main monitor stalls in hashing."""
    run = load_run()
    time.sleep(max(0, run["pause_deadline_epoch"] - time.time()))
    receipt = OUTPUT / "resource_stop.json"
    try:
        stopped = receipt.exists() and json.loads(receipt.read_text()).get("pause_verified") is True
    except (OSError, ValueError):
        stopped = False
    if stopped:
        return
    while True:
        try:
            pause_and_verify(run, "four_hour_deadline_guard")
            return
        except (OSError, RuntimeError, ValueError, subprocess.SubprocessError) as error:
            print(f"DEADLINE_STOP_RETRY {type(error).__name__}", flush=True)
            time.sleep(10)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("prepare", "launch", "watch", "guard", "pause"))
    parser.add_argument("--release", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    configure()
    match args.action:
        case "prepare": prepare()
        case "launch": launch(release=args.release, resume=args.resume)
        case "watch": watch()
        case "guard": deadline_guard()
        case "pause": pause_and_verify(load_run(), "operator_request")


if __name__ == "__main__":
    main()
