"""Verify eight-run artifacts, stop compute, and restore the original bootstrap."""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import redesign_h100_watch as watch


ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "results" / "eight_h100_20260917"


def verify_required_artifacts() -> dict[str, int | bool]:
    unpacked = OUTPUT / "downloaded" / "unpacked" / "output"
    notebooks = list(unpacked.glob("*Executed.ipynb"))
    predictions = list(unpacked.rglob("*prediction*.json")) + list(unpacked.rglob("*prediction*.csv"))
    checkpoints = list(unpacked.rglob("*.pt")) + list(unpacked.rglob("*.safetensors"))
    receipt = {
        "executed_notebooks": len(notebooks),
        "prediction_files": len(predictions),
        "checkpoint_files": len(checkpoints),
        "required_artifacts_present": bool(notebooks and predictions and checkpoints),
    }
    (OUTPUT / "required_artifacts.json").write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return receipt


def restore_bootstrap() -> None:
    stopped = json.loads((OUTPUT / "resource_stop.json").read_text(encoding="utf-8"))
    if stopped.get("pause_verified") is not True:
        raise RuntimeError("refusing bootstrap restore before verified compute stop")
    original = ROOT / "_workspace" / "six_h100_20260917" / "original_bootstrap" / "redesign_full_refit_run.sh"
    if not original.is_file():
        raise RuntimeError("original bootstrap backup is missing")
    subprocess.run(
        [
            *watch.CLI_PREFIX,
            "volume",
            "upload",
            watch.EXPECTED_VOLUME,
            str(original),
            "--remote-prefix",
            "redesign-full-20260917/",
            "--overwrite",
        ],
        check=True,
    )
    with_path = ROOT / "_workspace" / "eight_h100_20260917" / "bootstrap_restore_readback"
    if with_path.exists():
        shutil.rmtree(with_path)
    with_path.mkdir(parents=True)
    listing = json.loads(
        subprocess.run(
            [*watch.CLI_PREFIX, "volume", "ls", watch.EXPECTED_VOLUME, "--prefix", "redesign-full-20260917/", "-o", "json"],
            check=True, capture_output=True, text=True,
        ).stdout
    )
    excluded = []
    for entry in listing:
        key = entry.get("key")
        if entry.get("type") == "file" and isinstance(key, str) and key.startswith("redesign-full-20260917/"):
            relative = key.removeprefix("redesign-full-20260917/")
            if relative != "redesign_full_refit_run.sh":
                excluded.extend(("--exclude", relative))
    subprocess.run(
        [
            *watch.CLI_PREFIX,
            "volume",
            "download",
            watch.EXPECTED_VOLUME,
            str(with_path),
            "--remote-prefix",
            "redesign-full-20260917/",
            "--overwrite",
            *excluded,
        ],
        check=True,
    )
    readbacks = list(with_path.rglob("redesign_full_refit_run.sh"))
    if len(readbacks) != 1 or watch._sha256(readbacks[0]) != watch._sha256(original):
        raise RuntimeError("restored bootstrap readback hash mismatch")
    (OUTPUT / "bootstrap_restored.json").write_text(
        json.dumps({"restored": True, "sha256": watch._sha256(original)}, indent=2) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    watch.OUTPUT = OUTPUT
    watch.RUN_METADATA = OUTPUT / "workspace_run.json"
    watch.EXPECTED_PREFIX = "eight-h100-20260917/"
    watch.KNOWN_REMOTE_FILES = {*watch.KNOWN_REMOTE_FILES, "*/*"}
    watch.POLL_SECONDS = 300
    watch.monitor()
    if (OUTPUT / "download_verification.json").exists():
        verify_required_artifacts()
    restore_bootstrap()


if __name__ == "__main__":
    main()
