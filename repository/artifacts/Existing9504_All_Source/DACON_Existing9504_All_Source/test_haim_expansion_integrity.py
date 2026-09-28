"""Protect storage reserve and frozen pilot audio while expanding the bank."""
from pathlib import Path
import pytest
import new_panel_data_haim_expansion as expansion

from new_panel_data_haim_expansion import check_capacity, reuse_original
from v2_data import DataIntegrityError, sha256_file


def test_batch_reservation_blocks_disk_and_bank_overrun() -> None:
    with pytest.raises(DataIntegrityError):
        check_capacity(retained=2_499_999_999, free=100 * 2**30, pending=1)
    with pytest.raises(DataIntegrityError):
        check_capacity(retained=0, free=70 * 2**30, pending=1)
    check_capacity(retained=0, free=100 * 2**30, pending=8)


def test_reuse_rejects_changed_pilot_before_linking(tmp_path: Path) -> None:
    source = tmp_path / "pilot.mp3"
    source.write_bytes(b"known pilot bytes")
    expected = sha256_file(source)
    target = tmp_path / "expansion.mp3"
    reuse_original(source, target, expected)
    assert target.stat().st_ino == source.stat().st_ino
    source.write_bytes(b"changed bytes")
    with pytest.raises(DataIntegrityError):
        reuse_original(source, tmp_path / "second.mp3", expected)


def test_reuse_rejects_a_corrupt_existing_target(tmp_path: Path) -> None:
    source, target = tmp_path / "pilot.mp3", tmp_path / "resumed.mp3"
    source.write_bytes(b"verified original")
    target.write_bytes(b"different valid audio could be here")
    with pytest.raises(DataIntegrityError):
        reuse_original(source, target, sha256_file(source))


def test_offline_acquire_refuses_missing_audio(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(expansion, "BANK", tmp_path)
    monkeypatch.setattr(expansion.subprocess, "run", lambda *args, **kwargs: pytest.fail("Offline guard attempted a network call"))
    candidate = expansion.Candidate("acestep", "missing", "missing.mp3", {})
    with pytest.raises(DataIntegrityError, match="disabled"):
        expansion.acquire(candidate)
