# /// script
# requires-python = ">=3.11"
# dependencies = []
# ///
# Run: .venv/bin/python test_localization_a100_ops.py
"""Protect paid-compute gates and immutable run deadlines without starting compute."""

from __future__ import annotations

import importlib
import json
import tempfile
from pathlib import Path
from unittest.mock import patch


def main() -> None:
    """Reject dangerous starts, changed uploads, and deadline resets."""
    ops = importlib.import_module("localization_a100_ops")
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        upload = root / "upload"
        output = root / "results"
        upload.mkdir()
        output.mkdir()
        (upload / "bootstrap.sh").write_text("#!/bin/bash\nexit 0\n")
        (upload / "input.zip").write_bytes(b"training input")
        files = {
            name: {"bytes": (upload / name).stat().st_size,
                   "sha256": ops.shared._sha256(upload / name)}
            for name in ("bootstrap.sh", "input.zip")
        }
        (upload / "upload_manifest.json").write_text(json.dumps(files))
        stopped = {
            "workspace": {"slug": ops.WORKSPACE, "name": "jolly-routine",
                          "resourceSpecSlug": "resourcespec-a100x1",
                          "state": "stopped", "phase": "standby", "initScript": ""},
            "billingInfo": {"resourceSpecAcceleratorName": "A100 SXM",
                            "resourceSpecAcceleratorCount": 1},
            "endpoints": [],
        }
        with patch.object(ops, "UPLOAD", upload), patch.object(ops, "OUTPUT", output):
            ops.configure()
            for field, value in (("slug", "wsp-wrong"), ("initScript", "bash old.sh"),
                                 ("state", "running")):
                detail = json.loads(json.dumps(stopped))
                detail["workspace"][field] = value
                with patch.object(ops.shared, "_call", return_value=json.dumps(detail)):
                    try:
                        ops.workspace_detail(stopped=True)
                    except RuntimeError:
                        continue
                    raise AssertionError(f"unsafe workspace accepted: {field}")
            bundle = ops.bundle_manifest()
            receipt = {"workspace": ops.WORKSPACE, "object_volume": ops.VOLUME,
                       "object_prefix": ops.PREFIX, "gpu_started": False, "files": bundle}
            (output / "storage_preparation.json").write_text(json.dumps(receipt))
            (upload / "bootstrap.sh").write_text("#!/bin/bash\necho changed\n")
            with patch.object(ops.shared, "_call") as cloud:
                try:
                    ops.launch(release=True)
                except (ValueError, RuntimeError):
                    pass
                else:
                    raise AssertionError("changed bootstrap passed the verified-bundle gate")
                assert not cloud.called, "cloud touched before detecting changed bootstrap"
            (upload / "bootstrap.sh").write_text("#!/bin/bash\nexit 0\n")
            receipt_path = output / "storage_preparation.json"
            calls: list[list[str]] = []
            corrupt_download = False

            def fake_cloud(arguments: list[str], timeout: int = 90) -> str:
                """Model storage transport and an ambiguous paid-compute start failure."""
                calls.append(arguments)
                if arguments[:2] == ["workspace", "show"]:
                    return json.dumps(stopped)
                if arguments[:2] == ["volume", "ls"]:
                    return "[]"
                if arguments[:2] == ["volume", "download"]:
                    for source in upload.iterdir():
                        content = source.read_bytes()
                        if corrupt_download and source.name == "bootstrap.sh":
                            content += b"corruption"
                        (Path(arguments[3]) / source.name).write_bytes(content)
                if arguments[:2] == ["workspace", "start"]:
                    raise RuntimeError("start response lost after request")
                return ""

            with patch.object(ops.shared, "_call", side_effect=fake_cloud):
                ops.prepare()
                assert json.loads(receipt_path.read_text())["files"] == ops.bundle_manifest()
                corrupt_download = True
                try:
                    ops.prepare()
                except ValueError:
                    pass
                else:
                    raise AssertionError("corrupted cloud readback was accepted")
                assert not receipt_path.exists(), "failed readback left a launchable receipt"
                assert not any(command[:2] == ["workspace", "start"] for command in calls)
                corrupt_download = False
                ops.prepare()
                assert json.loads(receipt_path.read_text())["files"] == ops.bundle_manifest()
                with patch.object(ops.subprocess, "Popen") as spawn, patch.object(ops.time, "sleep"):
                    spawn.return_value.poll.return_value = None
                    try:
                        ops.launch(release=True)
                    except RuntimeError:
                        pass
                    else:
                        raise AssertionError("ambiguous compute-start failure was swallowed")
                    assert any(command[:2] == ["workspace", "pause"] for command in calls)
                    assert json.loads((output / "resource_stop.json").read_text())["pause_verified"]
                    spawn.return_value.terminate.assert_called_once()
            run = {"workspace": ops.WORKSPACE, "object_volume": ops.VOLUME,
                   "object_prefix": ops.PREFIX, "started_epoch": 1000.0,
                   "pause_deadline_epoch": 15400.0}
            run_path = output / "workspace_run.json"
            run_path.write_text(json.dumps(run))
            original = run_path.read_bytes()
            with patch.object(ops.time, 'time', return_value=1100), patch.object(ops.time, 'sleep'), \
                 patch.object(ops.shared, '_call', side_effect=fake_cloud), \
                 patch.object(ops.subprocess, 'Popen') as spawn:
                spawn.return_value.poll.return_value = None
                try:
                    ops.launch(release=True, resume=True)
                except RuntimeError as error:
                    assert str(error) == 'start response lost after request'
                else:
                    raise AssertionError('resumed start failure was swallowed')
                assert run_path.read_bytes() == original, 'resume extended the original clock'
            with patch.object(ops.time, "time", return_value=50000):
                assert ops.load_run()["pause_deadline_epoch"] == 15400.0
            with patch.object(ops.shared, "_call") as cloud:
                try:
                    ops.launch(release=True)
                except (ValueError, RuntimeError, FileExistsError):
                    pass
                else:
                    raise AssertionError("duplicate launch reset the deadline")
                assert run_path.read_bytes() == original and not cloud.called
            (output / "resource_stop.json").unlink()
            with patch.object(ops.time, "time", return_value=16000), \
                 patch.object(ops.time, "sleep"), \
                 patch.object(ops, "pause_and_verify") as stop:
                ops.deadline_guard()
                stop.assert_called_once_with(ops.load_run(), "four_hour_deadline_guard")
            with patch.object(ops.shared, "monitor", side_effect=KeyError("unexpected state")), \
                 patch.object(ops.subprocess, "Popen") as spawn, \
                 patch.object(ops, "pause_and_verify") as stop:
                try:
                    ops.watch()
                except KeyError:
                    pass
                else:
                    raise AssertionError("unexpected monitor failure was hidden")
                stop.assert_called_once_with(ops.load_run(), "monitor_failure")
                spawn.return_value.terminate.assert_called_once()
            for deadline in (15401.0, 1000.0, float("nan")):
                run_path.write_text(json.dumps({**run, "pause_deadline_epoch": deadline}))
                try:
                    ops.load_run()
                except ValueError:
                    continue
                raise AssertionError(f"invalid deadline accepted: {deadline}")
    print("OPS_CONTRACT_OK wrong_target stale_bootstrap active_compute changed_bundle corrupt_readback "
          "startup_failure_pause immutable_deadline independent_deadline_guard monitor_failure_pause deadline_bounds")


if __name__ == "__main__":
    main()
