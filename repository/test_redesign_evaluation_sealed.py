import hashlib
import json
from pathlib import Path

import pytest

from graph_scoring import official_metrics
from redesign_evaluate import lock_candidate, open_sealed
from redesign_evaluation_common import artifact_rows
from test_redesign_evaluation import _artifact, _write

PANEL_IDENTITY = "a" * 64


def _lockable_report(prediction_path: Path, candidate: str) -> dict:
    return {
        "status": "provisional", "candidate": candidate,
        "candidate_sha256": hashlib.sha256(prediction_path.read_bytes()).hexdigest(),
        "comparison": {"statistical_eligible": True, "validation_eligible": True},
    }


def _lock(tmp_path: Path, prediction_path: Path, name: str = "lock.json") -> tuple[Path, dict]:
    report_path = _write(
        tmp_path / "report.json", _lockable_report(prediction_path, _artifact()["candidate"]),
    )
    lock_path = tmp_path / name
    lock = lock_candidate(
        prediction_path, report_path, lock_path, sealed_panel_identity=PANEL_IDENTITY,
    )
    return lock_path, lock


def test_static_identity_mismatch_is_recoverable_but_opened_lock_cannot_reopen(tmp_path: Path) -> None:
    prediction = _artifact()
    prediction_path = _write(tmp_path / "prediction.json", prediction)
    lock_path, lock = _lock(tmp_path, prediction_path)
    registry = tmp_path / "data" / ".sealed_access_registry"
    sealed = json.loads(json.dumps(prediction))
    sealed["upstream_identity"]["checkpoint_sha256"] = "f" * 64
    sealed_path = _write(tmp_path / "sealed.json", sealed)
    with pytest.raises(ValueError, match="model identity"):
        open_sealed(lock_path, sealed_path, tmp_path / "sealed-receipt.json",
                    panel_identity=PANEL_IDENTITY, access_registry=registry)
    opened = open_sealed(lock_path, prediction_path, tmp_path / "sealed-receipt.json",
                         panel_identity=PANEL_IDENTITY, access_registry=registry)
    assert opened["status"] == "opened"
    labels, predictions, _ = artifact_rows(prediction)
    assert opened["sealed_metrics"] == official_metrics(labels, predictions)
    with pytest.raises(FileExistsError, match="already opened"):
        open_sealed(lock_path, prediction_path, tmp_path / "second.json",
                    panel_identity=PANEL_IDENTITY, access_registry=registry)
    marker = json.loads(Path(f"{lock_path}.sealed-opened.json").read_text())
    assert marker["status"] == "opened"
    assert marker["lock_sha256"] == lock["lock_sha256"]


def test_post_claim_semantic_failure_consumes_sealed_panel(tmp_path: Path) -> None:
    prediction = _artifact()
    prediction_path = _write(tmp_path / "prediction.json", prediction)
    lock_path, _ = _lock(tmp_path, prediction_path)
    registry = tmp_path / "data" / ".sealed_access_registry"
    malformed = json.loads(json.dumps(prediction))
    malformed["labels"]["id-000"].pop("FILE_FAKE")
    with pytest.raises(ValueError, match="invalid label"):
        open_sealed(lock_path, _write(tmp_path / "malformed.json", malformed),
                    tmp_path / "receipt.json", panel_identity=PANEL_IDENTITY,
                    access_registry=registry)
    with pytest.raises(FileExistsError, match="already opened"):
        open_sealed(lock_path, prediction_path, tmp_path / "second.json",
                    panel_identity=PANEL_IDENTITY, access_registry=registry)
    marker = json.loads((registry / f"{PANEL_IDENTITY}.json").read_text())
    assert marker["status"] == "rejected"


def test_rejected_or_insufficient_report_cannot_create_candidate_lock(tmp_path: Path) -> None:
    prediction = _artifact()
    prediction_path = _write(tmp_path / "prediction.json", prediction)
    report = _lockable_report(prediction_path, prediction["candidate"])
    report["status"] = "rejected"
    report["comparison"]["validation_eligible"] = False
    with pytest.raises(ValueError, match="not eligible"):
        lock_candidate(prediction_path, _write(tmp_path / "report.json", report),
                       tmp_path / "lock.json", sealed_panel_identity=PANEL_IDENTITY)
    assert not (tmp_path / "lock.json").exists()


def test_same_panel_cannot_reopen_through_different_lock_and_output_paths(tmp_path: Path) -> None:
    prediction = _artifact()
    prediction_path = _write(tmp_path / "prediction.json", prediction)
    report_path = _write(
        tmp_path / "report.json", _lockable_report(prediction_path, prediction["candidate"]),
    )
    first_lock, second_lock = tmp_path / "run-a" / "lock.json", tmp_path / "run-b" / "lock.json"
    lock_candidate(prediction_path, report_path, first_lock, sealed_panel_identity=PANEL_IDENTITY)
    lock_candidate(prediction_path, report_path, second_lock, sealed_panel_identity=PANEL_IDENTITY)
    registry = tmp_path / "immutable-data" / ".sealed_access_registry"
    open_sealed(first_lock, prediction_path, tmp_path / "run-a" / "receipt.json",
                panel_identity=PANEL_IDENTITY, access_registry=registry)
    with pytest.raises(FileExistsError, match="sealed panel was already opened"):
        open_sealed(second_lock, prediction_path, tmp_path / "run-b" / "receipt.json",
                    panel_identity=PANEL_IDENTITY, access_registry=registry)
    assert not (second_lock.parent / "receipt.json").exists()
