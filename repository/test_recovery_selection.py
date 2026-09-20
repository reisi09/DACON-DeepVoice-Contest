from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from recovery_selection import CANDIDATES, decide_fresh, evaluate_old, lock_candidate


def row(identifier: str, split: str, label: int | str, group: str | None = None) -> dict[str, str | int]:
    return {"ID": identifier, "path": f"audio/{identifier}.wav", "group_id": group or identifier,
            "split": split, "source": "fixture", "FILE_FAKE": label, "VOICE_FAKE": label,
            "MUSIC_FAKE": label, "VOICE_PRESENT": 1, "MUSIC_PRESENT": 1}


def prediction(identifier: str, score: float) -> dict[str, str | float]:
    return {"ID": identifier, "FILE_FAKE_PROB": max(.8 * .3, .7 * score), "VOICE_FAKE_PROB": .3,
            "MUSIC_FAKE_PROB": score, "VOICE_PRESENT_PROB": .8,
            "MUSIC_PRESENT_PROB": .7}


def candidate_sets(rows: list[dict[str, str | int]], bad_old: bool = False) -> dict[str, list[dict[str, str | float]]]:
    result: dict[str, list[dict[str, str | float]]] = {name: [] for name in CANDIDATES}
    for item in rows:
        truth = int(item["FILE_FAKE"])
        base_score = .5
        good_score = .1 if truth == 0 else .9
        for name in CANDIDATES:
            score = base_score if name == "G0" else good_score
            if bad_old and item["split"] == "test_source" and name == "SONICS_RAW":
                score = .9 if truth == 0 else .1
            result[name].append(prediction(str(item["ID"]), score))
    return result


def test_failed_gate_candidate_is_not_promoted_by_ranking() -> None:
    dev = [row(f"d{i}", "dev_select", i % 2) for i in range(20)]
    old = [row(f"g{i}", "test_generator", i % 2) for i in range(20)]
    old += [row(f"s{i}", "test_source", i % 2) for i in range(20)]
    predictions = candidate_sets(dev + old, bad_old=True)

    result = evaluate_old(dev, old, predictions)

    assert result["development_selected"] == "SONICS_RAW"
    assert result["selected"] == "G0"
    assert result["candidates"]["SONICS_RAW"]["old_safety"]["passed"] is False
    assert "test_source" in result["candidates"]["SONICS_RAW"]["old_safety"]["failed_slices"]
    assert result["old_usage"] == "REUSED diagnostic safety veto; never ranking"


def test_partial_labels_remain_unknown_and_are_not_called_official() -> None:
    dev = [row(f"d{i}", "dev_select", i % 2) for i in range(20)]
    old = [row(f"o{i}", "test_generator", i % 2) for i in range(20)]
    for item in dev + old:
        item["VOICE_FAKE"] = ""
    result = evaluate_old(dev, old, candidate_sets(dev + old))

    serialized = json.dumps(result)
    g0 = result["candidates"]["G0"]["development"]
    assert g0["metrics"]["voice_eer"] is None
    assert g0["unknown_counts"]["voice_eer"] == len(dev)
    assert "known_subset_weighted_diagnostic" in g0
    assert "official_score" not in serialized and "proxy_official" not in serialized


def test_freeze_before_fresh_and_holdout_cannot_tune(tmp_path: Path) -> None:
    manifest = tmp_path / "manifest.csv"
    manifest.write_text("sealed,without,labels\n", encoding="utf-8")
    lock_path = tmp_path / "candidate_lock.json"
    lock_candidate(lock_path, {"selected": "SONICS_RAW", "fresh_manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest()})
    fresh = [row(f"f{i}", "fresh_challenge", i % 2) for i in range(20)]
    predictions = candidate_sets(fresh)
    predictions["SONICS_RAW"] = [prediction(str(item["ID"]), .5) for item in fresh]
    predictions["SONICS_STEM"] = [prediction(str(item["ID"]), .99 if int(item["FILE_FAKE"]) else .01) for item in fresh]

    decision = decide_fresh(lock_path, manifest, fresh, predictions)

    assert decision["final"] == "G0"
    assert decision["reason"] == "frozen_candidate_failed_fresh_gate"
    assert decision["evaluated"] == ["G0", "SONICS_RAW"]
    assert "SONICS_STEM" in decision["ignored_candidates"]
    manifest.write_text("changed\n", encoding="utf-8")
    with pytest.raises(ValueError, match="manifest changed after candidate lock"):
        decide_fresh(lock_path, manifest, fresh, predictions)


def test_owned_fields_and_lock_are_fail_closed(tmp_path: Path) -> None:
    dev = [row(f"d{i}", "dev_select", i % 2) for i in range(20)]
    old = [row(f"o{i}", "test_generator", i % 2) for i in range(20)]
    predictions = candidate_sets(dev + old)
    predictions["SONICS_RAW"][0]["VOICE_PRESENT_PROB"] = .8000000001
    with pytest.raises(ValueError, match="protected G0 field changed"):
        evaluate_old(dev, old, predictions)
    path = tmp_path / "lock.json"
    lock_candidate(path, {"selected": "G0"})
    with pytest.raises(ValueError, match="immutable"):
        lock_candidate(path, {"selected": "SONICS_RAW"})


def test_prior_ast_ranking_winner_remains_rejected_by_failed_gates() -> None:
    receipt = json.loads(Path("results/v2_colab_20260912_174330/selection.json").read_text())
    ranked = max(("AST_FROZEN", "AST_LAST_BLOCK"), key=lambda name: receipt["models"][name]["value"])
    assert ranked == "AST_LAST_BLOCK"
    assert receipt["models"][ranked]["gates"]["proxy_gain"] is True
    assert receipt["models"][ranked]["gates"]["passed"] is False
    assert receipt["selected"] == "G0"
