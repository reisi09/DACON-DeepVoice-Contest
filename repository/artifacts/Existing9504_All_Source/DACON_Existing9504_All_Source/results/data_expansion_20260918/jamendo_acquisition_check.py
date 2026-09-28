#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy", "soundfile", "imageio-ffmpeg"]
# ///
# How to run: PYTHONPATH=. .venv/bin/python results/data_expansion_20260918/jamendo_acquisition_check.py
"""Check real selection rights and the interrupted-download publication boundary."""
import json
import subprocess
import tempfile
from datetime import date
from pathlib import Path
from unittest.mock import patch
from urllib.parse import urlparse

import new_panel_data_jamendo_bank as bank


def main() -> None:
    """Reject forbidden selections and failed/resume-broken HTTP publication."""
    rows = json.loads((bank.BANK / "selection.json").read_text())
    assert len({row["id"] for row in rows}) == len(rows)
    for row in rows:
        assert row["audiodownload_allowed"] is True
        assert urlparse(row["license_ccurl"]).path.split("/")[2] in {"by", "by-sa", "by-nc", "by-nc-sa"}
        assert date(2008, 1, 1) <= date.fromisoformat(row["releasedate"]) < date(2016, 1, 1)
        assert urlparse(row["audiodownload"]).path == f"/download/track/{row['id']}/mp32/"
    with tempfile.TemporaryDirectory() as directory:
        target = Path(directory) / "response.mp3"
        calls = []
        def resume_response(command, **kwargs):
            calls.append(command)
            Path(command[-1]).write_bytes(b"full successful HTTP payload")
            return subprocess.CompletedProcess(command, 33 if len(calls) == 1 else 0, b"", b"resume unavailable")
        with patch.object(bank.subprocess, "run", side_effect=resume_response):
            bank.retrieve("https://prod-1.storage.jamendo.com/download/track/1/mp32/", target)
        assert calls[0][4:6] == ["-C", "-"]
        assert calls[1][4:6] == ["--connect-timeout", "20"] and "-C" not in calls[1]
        assert target.read_bytes() == b"full successful HTTP payload"
        failed = Path(directory) / "failed.mp3"
        def failed_response(command, **kwargs):
            Path(command[-1]).write_bytes(b"truncated HTTP payload")
            return subprocess.CompletedProcess(command, 28, b"", b"timeout")
        with patch.object(bank.subprocess, "run", side_effect=failed_response):
            try:
                bank.retrieve("https://prod-1.storage.jamendo.com/download/track/2/mp32/", failed)
            except subprocess.CalledProcessError:
                pass
            else:
                raise AssertionError("Timed-out response was accepted")
        assert not failed.exists() and failed.with_suffix(".mp3.part").read_bytes() == b"truncated HTTP payload"
    print(json.dumps({"selected_unique_rights_date_endpoint_checks": len(rows),
                      "resume_fallback_valid_arguments": True, "failed_response_not_published": True}))


if __name__ == "__main__":
    main()
