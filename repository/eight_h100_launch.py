"""Package, preflight, upload, and explicitly release the existing H100 workspace."""

from __future__ import annotations

import argparse
import ast
import hashlib
import importlib
import json
import os
import shutil
import subprocess
import sys
import time
import zipfile
from pathlib import Path
from typing import Final


ROOT: Final = Path(__file__).resolve().parent
WORK: Final = ROOT / "_workspace" / "eight_h100_20260917"
RESULTS: Final = ROOT / "results" / "eight_h100_20260917"
CLI: Final = ROOT / "_workspace" / "bin" / "vesslctl"
CLI_PREFIX: Final = (str(CLI), "--org", "qwdkw", "--team", "Default")
WORKSPACE: Final = "wsp-9683432r72m7"
VOLUME: Final = "objvol-a5gxdg7er70i"
PREFIX: Final = "eight-h100-20260917/"
BOOTSTRAP_PREFIX: Final = "redesign-full-20260917/"
SOURCE: Final = WORK / "source.zip"


def digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def run(arguments: list[str], *, timeout: int = 600, capture: bool = False) -> str:
    process = subprocess.run(
        arguments,
        check=True,
        timeout=timeout,
        text=True,
        capture_output=capture,
    )
    return process.stdout if capture else ""


def cli(arguments: list[str], *, timeout: int = 600, capture: bool = False) -> str:
    return run([*CLI_PREFIX, *arguments], timeout=timeout, capture=capture)


def local_dependencies(entrypoints: list[Path]) -> list[Path]:
    pending = entrypoints[:]
    found: dict[str, Path] = {}
    while pending:
        path = pending.pop()
        if path.name in found:
            continue
        found[path.name] = path
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        names: set[str] = set()
        for node in ast.walk(tree):
            match node:
                case ast.Import(names=aliases):
                    names.update(alias.name.split(".")[0] for alias in aliases)
                case ast.ImportFrom(module=module) if module:
                    names.add(module.split(".")[0])
                case _:
                    pass
        for name in names:
            candidate = ROOT / f"{name}.py"
            if candidate.is_file() and candidate.name not in found:
                pending.append(candidate)
    return sorted(found.values(), key=lambda path: path.name)


def build_source() -> dict[str, int | str]:
    required = [
        ROOT / name
        for name in (
            "eight_data.py",
            "eight_contract.py",
            "eight_features.py",
            "eight_ablation.py",
            "eight_probes.py",
            "eight_lora_train.py",
            "eight_h100_run.py",
        )
    ]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise RuntimeError(f"source modules are not ready: {missing}")
    notebook = ROOT / "DACON_Eight_H100.ipynb"
    dataset = WORK / "dataset"
    if not notebook.is_file() or not (dataset / "manifest.csv").is_file():
        raise RuntimeError("notebook or prepared dataset manifest is missing")
    modules = local_dependencies(required)
    members: dict[str, Path] = {path.name: path for path in [*modules, notebook]}
    anti_source = ROOT / "_workspace" / "voice_music_h100_20260917" / "source" / "antideepfake"
    for path in sorted(anti_source.iterdir()):
        if path.is_file() and path.name != "model.safetensors":
            members[f"antideepfake/{path.name}"] = path
    for path in sorted(dataset.rglob("*")):
        if path.is_file() and not path.is_symlink():
            members[f"dataset/{path.relative_to(dataset).as_posix()}"] = path
    manifest = {name: digest(path) for name, path in members.items()}
    WORK.mkdir(parents=True, exist_ok=True)
    temporary_source = SOURCE.with_suffix(".zip.partial")
    with zipfile.ZipFile(
        temporary_source, "w", zipfile.ZIP_DEFLATED, allowZip64=True, compresslevel=1
    ) as packed:
        for name, path in sorted(members.items()):
            packed.write(path, name)
        packed.writestr("INPUT_SHA256.json", json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    with zipfile.ZipFile(temporary_source) as packed:
        if packed.testzip() is not None:
            raise RuntimeError("source ZIP CRC verification failed")
        for name, expected in manifest.items():
            if hashlib.sha256(packed.read(name)).hexdigest() != expected:
                raise RuntimeError(f"source ZIP member verification failed: {name}")
    temporary_source.replace(SOURCE)
    source_digest = digest(SOURCE)
    (WORK / "source.zip.sha256").write_text(f"{source_digest}  source.zip\n", encoding="utf-8")
    receipt = {
        "archive_sha256": source_digest,
        "archive_bytes": SOURCE.stat().st_size,
        "members": len(members),
        "modules": len(modules),
    }
    (RESULTS / "source_bundle.json").write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return receipt


def workspace_snapshot() -> dict[str, str | int | float]:
    raw = json.loads(cli(["workspace", "show", WORKSPACE, "-o", "json"], capture=True))
    workspace = raw["workspace"]
    billing = raw["billingInfo"]
    snapshot: dict[str, str | int | float] = {
        "workspace": workspace["slug"],
        "phase": workspace["phase"],
        "state": workspace["state"],
        "init_script": workspace["initScript"],
        "accelerator": billing["resourceSpecAcceleratorName"],
        "accelerator_count": billing["resourceSpecAcceleratorCount"],
        "hourly_compute_credits": billing["hourlyCreditForCompute"],
    }
    return snapshot


def base_config(started: int = 0) -> dict[str, int | str | list[int]]:
    return {
        "root": "/root/eight_h100_20260917",
        "shared": "/shared/eight-h100-20260917",
        "assets": "/root/redesign_h100_20260917/assets",
        "antideepfake": "/root/voice_music_h100_20260917/source/antideepfake",
        "started_epoch": started,
        "pause_deadline_epoch": started + 14_400 if started else 0,
        "export_reserve_seconds": 600,
        "seeds": [41, 42, 43],
        "head_steps": 500,
        "head_batch_size": 32,
        "lora_steps": 200,
        "lora_batch_size": 2,
        "notebook": "DACON_Eight_H100.ipynb",
        "kernel": "eight-h100",
    }


def upload_and_readback(local: Path, prefix: str) -> str:
    cli(["volume", "upload", VOLUME, str(local), "--remote-prefix", prefix, "--overwrite"], timeout=1800)
    incoming = WORK / "readback" / local.name
    if incoming.exists():
        shutil.rmtree(incoming)
    incoming.mkdir(parents=True)
    arguments = ["volume", "download", VOLUME, str(incoming), "--remote-prefix", prefix, "--overwrite"]
    # Keep single-object readbacks bounded when the prefix also contains retry directories.
    arguments.extend(("--exclude", "*/*"))
    listing = json.loads(cli(["volume", "ls", VOLUME, "--prefix", prefix, "-o", "json"], capture=True))
    remote_names = []
    for entry in listing:
        key = entry.get("key")
        if entry.get("type") == "file" and isinstance(key, str) and key.startswith(prefix):
            remote_names.append(key[len(prefix):])
    if local.name not in remote_names:
        raise RuntimeError(f"uploaded object is absent from listing: {prefix}{local.name}")
    for item in sorted(set(remote_names) - {local.name}):
        arguments.extend(("--exclude", item))
    cli(arguments, timeout=1800)
    matches = list(incoming.rglob(local.name))
    if len(matches) != 1 or digest(matches[0]) != digest(local):
        raise RuntimeError(f"remote readback hash mismatch: {local.name}")
    return digest(local)


def cpu_preflight(snapshot: dict[str, str | int | float]) -> dict[str, int | str | bool]:
    imported = []
    for name in (
        "imageio_ffmpeg",
        "nbformat",
        "nbclient",
        "ipykernel",
        "eight_contract",
        "eight_features",
        "eight_ablation",
        "eight_probes",
        "eight_lora_train",
    ):
        importlib.import_module(name)
        imported.append(name)
    assets = ROOT / "_workspace" / "redesign_full_20260917" / "assets"
    asset_manifest = json.loads((assets / "asset_hashes.json").read_text(encoding="utf-8"))
    for relative, expected in asset_manifest.items():
        path = assets / relative
        if not path.is_file() or digest(path) != expected:
            raise RuntimeError(f"local asset identity mismatch: {relative}")
    anti = ROOT / "_workspace" / "voice_music_h100_20260917"
    anti_source = anti / "source" / "antideepfake"
    model = anti / "model.safetensors"
    expected_model = (anti_source / "model.sha256").read_text(encoding="utf-8").split()[0]
    if digest(model) != expected_model:
        raise RuntimeError("local AntiDeepfake model identity mismatch")
    upload_receipt = json.loads(
        (ROOT / "results" / "voice_music_h100_20260917" / "model_upload_verification.json").read_text(
            encoding="utf-8"
        )
    )
    if (
        upload_receipt.get("sha256") != expected_model
        or upload_receipt.get("full_readback_verified") is not True
    ):
        raise RuntimeError("prior shared AntiDeepfake upload receipt is not verified")
    listing = json.loads(
        cli(
            ["volume", "ls", VOLUME, "--prefix", "voice-music-h100-20260917/", "-o", "json"],
            capture=True,
        )
    )
    keys = {entry.get("key") for entry in listing if entry.get("type") == "file"}
    if "voice-music-h100-20260917/model.safetensors" not in keys:
        raise RuntimeError("verified shared AntiDeepfake model object is absent")
    if snapshot["accelerator"] != "H100 SXM" or snapshot["accelerator_count"] != 1:
        raise RuntimeError("workspace does not provide exactly one H100 SXM")
    return {
        "status": "passed",
        "imports": len(imported),
        "imageio_ffmpeg_imported": True,
        "asset_files_verified": len(asset_manifest),
        "antideepfake_model_sha256": expected_model,
        "shared_antideepfake_receipt_verified": True,
        "workspace_h100_count": 1,
    }


def active_watcher_processes() -> list[dict[str, int | str]]:
    active: list[dict[str, int | str]] = []
    for pid_file in sorted((ROOT / "results").glob("*/watch_pid.txt")):
        try:
            pid = int(pid_file.read_text(encoding="utf-8").strip())
        except (OSError, ValueError):
            continue
        process = subprocess.run(
            ["ps", "-p", str(pid), "-o", "command="], capture_output=True, text=True, check=False
        )
        command = process.stdout.strip()
        if process.returncode == 0 and "h100_watch.py" in command:
            active.append({"pid": pid, "pid_file": str(pid_file), "command": command})
    return active


def prepare() -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)
    WORK.mkdir(parents=True, exist_ok=True)
    subprocess.run([sys.executable, str(ROOT / "build_eight_h100_notebook.py")], check=True)
    source_receipt = build_source()
    snapshot = workspace_snapshot()
    if snapshot["phase"] not in {"standby", "stopped"} or snapshot["accelerator_count"] != 1:
        raise RuntimeError("existing workspace is not safely stopped with one accelerator")
    if snapshot["init_script"] != "bash /shared/redesign-full-20260917/redesign_full_refit_run.sh":
        raise RuntimeError("workspace bootstrap target changed")
    (RESULTS / "workspace_before.json").write_text(
        json.dumps(snapshot, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    cpu_receipt = cpu_preflight(snapshot)
    upload_and_readback(SOURCE, PREFIX)
    upload_and_readback(WORK / "source.zip.sha256", PREFIX)
    config = WORK / "config.prelaunch.json"
    config.write_text(json.dumps(base_config(), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    receipt = {
        "status": "prepared_not_started",
        "workspace": snapshot,
        "source": source_receipt,
        "cpu_import_and_asset_preflight": cpu_receipt,
        "source_roundtrip_verified": True,
        "bootstrap_overwritten": False,
        "prepared_unix": time.time(),
    }
    (RESULTS / "preflight.json").write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def start(released_by: str) -> None:
    if released_by != "root":
        raise RuntimeError("GPU release must be attributed to root")
    preflight = json.loads((RESULTS / "preflight.json").read_text(encoding="utf-8"))
    if preflight.get("status") != "prepared_not_started":
        raise RuntimeError("prepare and readback verification must succeed before start")
    prior_run = RESULTS / "workspace_run.json"
    if prior_run.is_file():
        raise RuntimeError("refusing to reset an existing run deadline; use the recovery path")
    watchers = active_watcher_processes()
    if watchers:
        raise RuntimeError(f"another H100 deadline watcher is still alive: {watchers}")
    snapshot = workspace_snapshot()
    if snapshot["phase"] not in {"standby", "stopped"}:
        raise RuntimeError("workspace is not stopped at release time")
    started = int(time.time())
    config = WORK / "config.json"
    template = json.loads((WORK / "config.prelaunch.json").read_text(encoding="utf-8"))
    if template.get("started_epoch") != 0 or template.get("pause_deadline_epoch") != 0:
        raise RuntimeError("prelaunch config already contains a live deadline")
    template["started_epoch"] = started
    template["pause_deadline_epoch"] = started + 14_400
    config.write_text(json.dumps(template, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    upload_and_readback(config, PREFIX)

    override_dir = WORK / "bootstrap_override"
    override_dir.mkdir(parents=True, exist_ok=True)
    override = override_dir / "redesign_full_refit_run.sh"
    shutil.copy2(ROOT / "eight_h100_run.sh", override)
    os.chmod(override, 0o755)
    bootstrap_sha = upload_and_readback(override, BOOTSTRAP_PREFIX)
    cli(["workspace", "start", WORKSPACE], timeout=120)

    delays = (2, 3, 5, 8, 13, 21, 34, 55)
    observed = "unknown"
    for delay in delays:
        current = workspace_snapshot()
        observed = str(current["phase"])
        if observed not in {"standby", "stopped", "pending", "starting"}:
            break
        time.sleep(delay)
    if observed in {"standby", "stopped"}:
        raise RuntimeError(f"workspace did not leave stopped state after release: {observed}")

    run_metadata = {
        "workspace": WORKSPACE,
        "started_epoch": started,
        "pause_deadline_epoch": started + 14_400,
        "object_volume": VOLUME,
        "object_prefix": PREFIX,
        "remote_root": "/root/eight_h100_20260917",
        "status": "running",
        "max_gpu_hours": 4,
        "monitor_interval_seconds": 300,
        "bootstrap_sha256": bootstrap_sha,
        "released_by": released_by,
    }
    (RESULTS / "workspace_run.json").write_text(
        json.dumps(run_metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    log = (RESULTS / "watch.log").open("a", encoding="utf-8")
    process = subprocess.Popen(
        [sys.executable, str(ROOT / "eight_h100_watch.py")], stdout=log, stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    time.sleep(1)
    if process.poll() is not None:
        raise RuntimeError("watcher exited immediately after launch")
    identity = subprocess.run(
        ["ps", "-p", str(process.pid), "-o", "command="], check=True, capture_output=True, text=True
    ).stdout.strip()
    if "eight_h100_watch.py" not in identity:
        raise RuntimeError("watcher PID identity does not match eight_h100_watch.py")
    (RESULTS / "watch_pid.txt").write_text(f"{process.pid}\n", encoding="utf-8")
    print(f"STARTED workspace={WORKSPACE} watcher_pid={process.pid}")


def main() -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="action", required=True)
    subparsers.add_parser("prepare")
    start_parser = subparsers.add_parser("start")
    start_parser.add_argument("--released-by", required=True)
    arguments = parser.parse_args()
    match arguments.action:
        case "prepare":
            prepare()
        case "start":
            start(arguments.released_by)
        case unreachable:
            raise AssertionError(unreachable)


if __name__ == "__main__":
    main()
