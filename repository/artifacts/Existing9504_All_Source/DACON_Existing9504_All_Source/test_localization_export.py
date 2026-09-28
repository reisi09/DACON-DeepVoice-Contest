# /// script
# requires-python = ">=3.11"
# dependencies = []
# ///
# Run: .venv/bin/python test_localization_export.py
"""Require primary/control locks and preserve verifiable partial results on failure."""

import hashlib
import json
import subprocess
import sys
import tempfile
import time
from pathlib import Path

from redesign_h100_watch import verify_archive


def main() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        output, objects = root / "output", root / "objects"
        train = output / "train"
        train.mkdir(parents=True)
        objects.mkdir()
        primary = [f"{arm}_{seed}.pt" for arm in ("mean", "mil", "ast") for seed in (41, 42, 43)]
        controls = [f"original831_mean_{seed}.pt" for seed in (41, 42, 43)]
        names = primary + controls
        for name in names:
            (train / name).write_bytes(name.encode())
        updates = {Path(name).stem: 125 for name in names}
        lock = {"checkpoints": primary, "control_checkpoints": controls, "selected_updates": updates,
                "checkpoint_sha256": {name: hashlib.sha256((train / name).read_bytes()).hexdigest()
                                      for name in names}}
        (train / "candidate_lock.json").write_text(json.dumps(lock))
        (train / "config.json").write_text(json.dumps({"diagnostic_controls": ["original831_mean"],
                                                       "seeds": [41, 42, 43]}))
        (train / "results.json").write_text(json.dumps({
            "status": "completed", "metrics": {arm: {} for arm in ("mean", "mil", "ast", "original831_mean")},
            "selected_updates": updates,
        }))
        started = time.time()
        command = [sys.executable, str(Path(__file__).with_name("localization_export.py")),
                   "--output-dir", str(output), "--object-dir", str(objects), "--exit-code", "0",
                   "--started-epoch", str(started), "--deadline-epoch", str(started + 14400)]
        for complete in (True, False):
            if not complete:
                (train / "original831_mean_43.pt").write_bytes(b"changed since candidate lock")
            subprocess.run(command, check=True, capture_output=True)
            state = json.loads((objects / "run_state.json").read_text())
            receipt = json.loads((objects / "export_receipt.json").read_text())
            assert state["status"] == ("exported" if complete else "failed")
            assert receipt["status"] == ("exported" if complete else "partial_exported")
            verified = verify_archive(objects / "results.zip", state["archive_sha256"], root / "verified")
            assert verified["files_verified"] == receipt["export_member_count"]
            assert (root / "verified/output/train/original831_mean_43.pt").read_bytes() == (train / "original831_mean_43.pt").read_bytes()
    print("EXPORT_CONTRACT_OK complete_primary9_control3 locked_control_corruption_partial full_payload_hashes")


if __name__ == "__main__":
    main()
