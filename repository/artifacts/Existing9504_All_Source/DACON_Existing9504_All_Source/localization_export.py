# /// script
# requires-python = ">=3.11"
# dependencies = []
# ///
# Run: .venv/bin/python localization_export.py --help
"""Export evidence; require nine primary and three diagnostic runs for completion."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
import tempfile
import time
import zipfile
from pathlib import Path


def sha(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def completion(output: Path, exit_code: int) -> tuple[bool, str]:
    """A zero process exit alone does not prove that all candidates finished."""
    if exit_code:
        return False, f"workload_exit_{exit_code}"
    train = output / "train"
    primary = {f"{arm}_{seed}.pt" for arm in ("mean", "mil", "ast") for seed in (41, 42, 43)}
    controls = {f"original831_mean_{seed}.pt" for seed in (41, 42, 43)}
    expected = primary | controls
    try:
        results = json.loads((train / "results.json").read_text())
        lock = json.loads((train / "candidate_lock.json").read_text())
        config = json.loads((train / "config.json").read_text())
        if (results["status"] != "completed"
                or set(results["metrics"]) != {"mean", "mil", "ast", "original831_mean"}
                or config["diagnostic_controls"] != ["original831_mean"] or config["seeds"] != [41, 42, 43]
                or len(lock["checkpoints"]) != 9 or set(lock["checkpoints"]) != primary
                or len(lock["control_checkpoints"]) != 3 or set(lock["control_checkpoints"]) != controls
                or set(lock["checkpoint_sha256"]) != expected
                or set(lock["selected_updates"]) != {Path(name).stem for name in expected}
                or results["selected_updates"] != lock["selected_updates"]):
            return False, "candidate_evaluation_incomplete"
        for name in expected:
            if sha(train / name) != lock["checkpoint_sha256"][name]:
                return False, f"locked_checkpoint_changed:{name}"
    except (OSError, ValueError, KeyError, TypeError) as error:
        return False, f"completion_receipt_invalid:{type(error).__name__}"
    return True, "all_nine_primary_and_three_control_runs_locked_and_evaluated"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("output-dir", "object-dir"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--exit-code", type=int, required=True)
    parser.add_argument("--started-epoch", type=float, required=True)
    parser.add_argument("--deadline-epoch", type=float, required=True)
    parser.add_argument("--workspace", default="wsp-5i3knd3gxeox")
    args = parser.parse_args()
    if not (math.isfinite(args.started_epoch) and math.isfinite(args.deadline_epoch)
            and 0 < args.deadline_epoch - args.started_epoch <= 14400):
        raise ValueError("invalid original four-hour run interval")
    output, objects = args.output_dir, args.object_dir
    complete, reason = completion(output, args.exit_code)
    output.mkdir(parents=True, exist_ok=True)
    (output / "export_decision.json").write_text(json.dumps({
        "completed": complete, "reason": reason, "workload_exit_code": args.exit_code,
        "checked_at": time.time(),
    }, indent=2) + "\n")
    bootstrap_log = objects / "bootstrap.log"
    if bootstrap_log.is_file():
        shutil.copyfile(bootstrap_log, output / "bootstrap.log")
    files = sorted(path for path in output.rglob("*") if path.is_file())
    if any(path.is_symlink() for path in files):
        raise ValueError("export refuses symlinked output files")
    manifest = {"output/" + path.relative_to(output).as_posix(): sha(path) for path in files}
    with tempfile.TemporaryDirectory(prefix="localization-export-", dir=output.parent) as temporary:
        local_archive = Path(temporary) / "results.zip"
        with zipfile.ZipFile(local_archive, "w", zipfile.ZIP_STORED, allowZip64=True) as packed:
            for path in files:
                packed.write(path, "output/" + path.relative_to(output).as_posix())
            packed.writestr("EXPORT_SHA256.json", json.dumps(manifest, indent=2, sort_keys=True) + "\n")
        with zipfile.ZipFile(local_archive) as packed:
            if packed.testzip() is not None:
                raise ValueError("local export failed ZIP CRC verification")
        digest, size = sha(local_archive), local_archive.stat().st_size
        pending = objects / "results.zip.partial"
        shutil.copyfile(local_archive, pending)
        pending.replace(objects / "results.zip")
    (objects / "results.zip.sha256").write_text(f"{digest}  results.zip\n")
    receipt = {"schema_version": 1, "status": "exported" if complete else "partial_exported",
               "archive_ready": True, "archive_verified": True, "archive_sha256": digest,
               "archive_bytes": size, "export_member_count": len(files),
               "exit_code": args.exit_code, "completed_all_candidates": complete,
               "completion_reason": reason, "exported_at": time.time()}
    (objects / "export_receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    state = {**receipt, "status": "exported" if complete else "failed",
             "workspace": args.workspace, "started_epoch": args.started_epoch,
             "pause_deadline_epoch": args.deadline_epoch, "updated_at": time.time()}
    pending_state = objects / "run_state.json.partial"
    pending_state.write_text(json.dumps(state, indent=2) + "\n")
    pending_state.replace(objects / "run_state.json")
    print(f"EXPORT_READY status={state['status']} files={len(files)} sha256={digest}", flush=True)


if __name__ == "__main__":
    main()
