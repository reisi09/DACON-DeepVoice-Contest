"""A selected checkpoint and its unseen evaluation rows must remain immutable."""

import json

import pytest

from graph_features import digest
from run_v2 import lock_candidate, open_locked_test


def test_lock_prevents_reselection_and_detects_changed_evaluation(tmp_path):
    manifest = tmp_path / "manifest.csv"
    manifest.write_text("ID,split\nfresh_a,test_generator\nfresh_b,test_source\n")
    lock = tmp_path / "candidate_lock.json"
    record = {"selected": "AST_LAST_BLOCK", "test_manifest_sha256": digest(manifest),
              "checkpoints": {"AST_LAST_BLOCK": {"model.safetensors": "original-weight-hash"}}}
    lock_candidate(lock, record)
    assert [r["ID"] for r in open_locked_test(lock, manifest)] == ["fresh_a", "fresh_b"]
    with pytest.raises(ValueError, match="different inputs or weights"):
        lock_candidate(lock, {**record, "selected": "G0"})
    assert json.loads(lock.read_text()) == record
    manifest.write_text("ID,split\nfresh_a,test_generator\nold_train,dev_train\n")
    with pytest.raises(ValueError, match="changed after selection"):
        open_locked_test(lock, manifest)


def test_fresh_labels_cannot_open_before_selection(tmp_path):
    manifest = tmp_path / "manifest.csv"
    manifest.write_text("ID,split\nfresh_a,test_source\n")
    with pytest.raises(FileNotFoundError):
        open_locked_test(tmp_path / "candidate_lock.json", manifest)
