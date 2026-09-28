# /// script
# requires-python = ">=3.11"
# dependencies = ["pytest", "numpy", "imageio-ffmpeg"]
# ///
# How to run: .venv/bin/python -m pytest results/data_expansion_20260918/fma_bank_budget_checks.py -q
"""Protect authorized acquisition caps, restart charging, and exclusive ownership."""
from pathlib import Path
import csv
import fcntl
import json
import subprocess
import sys

import pytest

import new_panel_data_fma_bank as bank


@pytest.fixture
def isolated_bank(tmp_path, monkeypatch):
    """Keep acquisition probes isolated and fail if they try external downloading."""
    output = tmp_path / "results"
    output.mkdir()
    monkeypatch.setattr(bank, "BANK", tmp_path / "bank")
    monkeypatch.setattr(bank, "OUT", output)
    monkeypatch.setattr(bank, "METADATA", tmp_path / "metadata.csv")
    monkeypatch.setattr(bank, "LOCKS", tmp_path / "locks.json")
    monkeypatch.setattr(bank, "local_originals", lambda: {})
    row = {"track_id": "1", "artist_id": "fresh", "track_duration": "2:00", "track_bit_rate": "128000",
           "track_file": "not-networked.mp3", "license_url": "https://creativecommons.org/licenses/by/4.0/"}
    with bank.METADATA.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(row))
        writer.writeheader()
        writer.writerow(row)
    bank.LOCKS.write_text(json.dumps({"all_prior_seen": {"fma_artist_ids": []}}))
    def forbidden_download(*args):
        raise AssertionError("No acquisition is authorized during preparation checks")
    monkeypatch.setattr(bank, "download", forbidden_download)
    return row


@pytest.mark.parametrize("gigabytes", [10, 1.500000001])
def test_rejects_superseded_allocation_before_acquisition(isolated_bank, gigabytes):
    # Given the former allocation or one byte over the revised allocation,
    # when invoked, then it fails at the authorization boundary without downloading.
    with pytest.raises(bank.BankError, match="1.5"):
        bank.main(gigabytes)


def test_default_only_indexes_until_explicit_release(isolated_bank):
    # Given an eligible unseen artist, when run without a download option,
    # then no network attempt or reservation is made.
    bank.main()
    receipt = json.loads((bank.OUT / "fma_bank_receipt.json").read_text())
    assert receipt["network_attempts"] == 0
    assert not (bank.OUT / "fma_bank_reservations.jsonl").exists()


def test_restart_keeps_unresolved_bytes_reserved_and_rejects_duplicate_journal():
    # Given one interrupted30MB request and one completed4MB request,
    # when reconstructed, then the unreceipted request still costs30MB.
    reservations = [{"track_id": "a", "reserved_bytes": 30_000_000}, {"track_id": "b", "reserved_bytes": 30_000_000}]
    transfers = [{"track_id": "b", "transferred_bytes": 4_000_000}]
    assert bank.charged_bytes(reservations, transfers) == 34_000_000
    with pytest.raises(bank.BankError, match="Duplicate"):
        bank.charged_bytes(reservations + reservations[:1], transfers)


def test_interrupted_request_consumes_budget_after_restart(isolated_bank):
    # Given a reserved but unreceipted request near the cap, when resumed,
    # then its bytes cannot be spent a second time.
    bank.append_receipt(bank.OUT / "fma_bank_reservations.jsonl",
                        {"track_id": "interrupted", "reserved_bytes": 1_499_500_000})
    bank.main(1.5)
    receipt = json.loads((bank.OUT / "fma_bank_receipt.json").read_text())
    assert receipt["status"] == "complete_budget"
    assert receipt["charged_bytes_including_unresolved"] == 1_499_500_000
    assert receipt["network_attempts"] == 0


def test_retention_counts_partials_and_metadata_before_download(isolated_bank, monkeypatch):
    # Given retained bytes outside completed originals, when retention is exhausted,
    # then no request is dispatched even though the transfer budget is unused.
    bank.BANK.mkdir()
    (bank.BANK / "orphan.part").write_bytes(b"x" * 100)
    (bank.OUT / "fma_extra_evidence.json").write_bytes(b"y" * 100)
    monkeypatch.setattr(bank, "MAX_RETAINED", 2_000_150)
    bank.main(1.5)
    receipt = json.loads((bank.OUT / "fma_bank_receipt.json").read_text())
    assert receipt["status"] == "stopped_retention_cap"
    assert receipt["network_attempts"] == 0


def test_second_process_cannot_reach_metadata_or_network(tmp_path):
    # Given an exclusive lock held by another process, when the actual main starts,
    # then it is refused before reading even the deliberately missing metadata.
    target = tmp_path / "bank"
    target.mkdir()
    with (target / ".acquisition.lock").open("a") as stream:
        fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        code = "import new_panel_data_fma_bank as b; from pathlib import Path; "
        code += f"b.BANK=Path({str(target)!r}); b.OUT=Path({str(tmp_path / 'out')!r}); "
        code += f"b.METADATA=Path({str(tmp_path / 'missing.csv')!r}); b.main(0)"
        result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=False)
    assert result.returncode != 0 and "Another FMA acquisition process" in result.stderr
    assert not (tmp_path / "out").exists()


def test_repeated_size_rejections_stop_without_scanning_all_remaining_artists(isolated_bank, monkeypatch):
    # Given many otherwiseeligible artists rejected by curl's remaining byte cap,
    # when acquiring, then bounded no-progress termination stops after32attempts.
    with bank.METADATA.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(isolated_bank))
        writer.writeheader()
        for index in range(50):
            writer.writerow(dict(isolated_bank, track_id=str(index), artist_id=f"artist{index}"))
    def reject_size(row, cap, lock_fd):
        return {"track_id": row["track_id"], "transferred_bytes": 0, "curl_exit": 56,
                "path": str(bank.BANK / "absent.mp3"), "error": "Maximum file size exceeded"}
    monkeypatch.setattr(bank, "download", reject_size)
    bank.main(1.5)
    receipt = json.loads((bank.OUT / "fma_bank_receipt.json").read_text())
    assert receipt["status"] == "stopped_repeated_no_progress"
    assert receipt["network_attempts"] == 32
