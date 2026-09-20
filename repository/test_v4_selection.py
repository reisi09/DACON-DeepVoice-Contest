"""Decision tests for V4 selection and sealed promotion."""

from __future__ import annotations

import json
import time
from pathlib import Path

import pytest

from graph_fusion import graph_predictions
from v4_evaluation import evaluate_development
from v4_selection import (
    SelectionError,
    accept_sealed,
    mark_sealed_opened,
    select_development,
    verify_weight_lock,
    write_recipe_lock,
    write_sealed_inference_receipt,
    write_weight_lock,
)


def _recipe(path: Path, eligible_ids: tuple[str, ...] = ("example",)) -> float:
    selection = path.with_name("selection.json")
    eligible = path.with_name("eligible.csv")
    selection.write_text(json.dumps({"selected": "P1", "candidates": {"P1": {
        "status": "completed", "gates": {"passed": True, "resource": True}}}}), encoding="utf-8")
    eligible.write_text("ID\n" + "\n".join(eligible_ids) + "\n", encoding="utf-8")
    write_recipe_lock(path, selection, eligible,
                      {"selected": "P1", "parent": "P1", "treatment": "mean",
                             "learning_rate": 1e-5, "scheduler": "fixed-30-epoch-cosine",
                             "epoch_count": 3, "stage_epochs": {"main": 3},
                             "data_hashes": {"train": "a"},
                             "source_hashes": {"source": "b"}, "pooling": "mean",
                             "seeds": [20260913], "gate_thresholds": {"D": 0.005}})
    return json.loads(path.read_text(encoding="utf-8"))["locked_unix"]


def _panel() -> tuple[list[dict[str, str | float]], list[dict[str, str | float]], dict[str, list[dict[str, str | float]]]]:
    rows: list[dict[str, str | float]] = []
    baseline: list[dict[str, str | float]] = []
    candidates = {name: [] for name in ("P1", "P2", "P3", "P4")}

    def add(identifier: str, group: str, panel: str, file_label: str, music_label: str,
            music_present: str, base_music: float, candidate_music: float) -> None:
        rows.append({"ID": identifier, "group_id": group, "split": "dev", "panel": panel,
                     "source": panel, "FILE_FAKE": file_label, "VOICE_FAKE": "",
                     "MUSIC_FAKE": music_label, "VOICE_PRESENT": "0",
                     "MUSIC_PRESENT": music_present})
        base = {"ID": identifier, "FILE_FAKE_PROB": base_music,
                "VOICE_FAKE_PROB": 0.0, "MUSIC_FAKE_PROB": base_music,
                "VOICE_PRESENT_PROB": 0.0, "MUSIC_PRESENT_PROB": 1.0}
        baseline.append(base)
        for name in candidates:
            score = candidate_music if name == "P1" else base_music
            candidates[name].append({**base, "FILE_FAKE_PROB": score,
                                     "MUSIC_FAKE_PROB": score})

    for group_index in range(10):
        group = f"artist-{group_index}"
        real_base = 0.15 + 0.06 * group_index
        fake_base = 0.35 + 0.035 * group_index
        add(f"real-{group_index}", group, "music_real", "0", "0", "1",
            real_base, 0.1)
        for provider in range(3):
            add(f"seen-{group_index}-{provider}", group, "music_seen", "1", "1", "1",
                fake_base + 0.002 * provider, 0.9)
        add(f"ood-{group_index}", group, "music_ood", "1", "1", "1",
            fake_base, 0.05)
        for index in range(4):
            label = str(index % 2)
            score = 0.8 if label == "1" else 0.2
            add(f"speech-{group_index}-{index}", group, "speech", label, "", "0", score, score)
        for index in range(8):
            label = str(index % 2)
            score = 0.8 if label == "1" else 0.2
            add(f"mixture-{group_index}-{index}", group, "mixture", label, label, "1", score, score)
        for index in range(4):
            label = str(index % 2)
            score = 0.8 if label == "1" else 0.2
            add(f"splice-{group_index}-{index}", group, "splice", label, label, "1", score, score)
    return rows, baseline, candidates


def test_attractive_seen_result_cannot_hide_negative_ood_result() -> None:
    # Given: P1 is perfect on seen music but reverses the OOD ordering.
    rows, baseline, candidates = _panel()

    # When: the fixed development gates and ranking are applied.
    report = select_development(rows, baseline, candidates,
                                {name: 10.0 for name in candidates},
                                {name: 100.0 for name in candidates}, 1_000.0)

    # Then: P1 cannot be promoted through its attractive seen mean.
    assert report["selected"] == "G0"
    assert report["candidates"]["P1"]["gates"]["ood_gain"] is False


def test_candidate_passes_only_with_both_music_panels_and_partial_score_stays_null() -> None:
    # Given: P1 improves both music panels while protected controlled outputs stay fixed.
    rows, baseline, candidates = _panel()
    for prediction in candidates["P1"]:
        if str(prediction["ID"]).startswith("ood-"):
            prediction["FILE_FAKE_PROB"] = 0.9
            prediction["MUSIC_FAKE_PROB"] = 0.9

    # When: development selection receives enough reserved refit time.
    report = select_development(rows, baseline, {"P1": candidates["P1"]},
                                {"P1": 10.0}, {"P1": 100.0}, 125.0)

    # Then: P1 is selected, while unknown labels still prevent an absolute score claim.
    assert report["selected"] == "P1"
    assert report["candidates"]["P1"]["evaluation"]["absolute"]["score"] is None


def test_refit_forecast_reserve_is_checked_before_ranking() -> None:
    # Given: a quality-eligible P1 needs 125 seconds with the 1.25 safety factor.
    rows, baseline, candidates = _panel()
    for prediction in candidates["P1"]:
        if str(prediction["ID"]).startswith("ood-"):
            prediction["FILE_FAKE_PROB"] = 0.9
            prediction["MUSIC_FAKE_PROB"] = 0.9

    # When: only 124.99 seconds remain before the recipe lock.
    report = select_development(rows, baseline, {"P1": candidates["P1"]},
                                {"P1": 10.0}, {"P1": 100.0}, 124.99)

    # Then: the candidate stays reported but cannot be selected.
    assert report["selected"] == "G0"
    assert report["candidates"]["P1"]["gates"]["resource"] is False


def test_d_uses_unrenormalized_file_and_music_weights() -> None:
    # Given: both G0 EERs are 0.4 and both candidate EERs are 0.0 on seen music.
    rows, baseline, candidates = _panel()

    # When: the V4 development objective is calculated.
    report = evaluate_development(rows, baseline, candidates["P1"])

    # Then: D is .45*.4 + .27*.4, without renormalizing those two weights.
    assert report["music"]["music_seen"]["D"] == pytest.approx(0.288)


def test_candidate_cannot_change_a_protected_g0_output() -> None:
    # Given: a candidate changes the preserved voice-fake output on one file.
    rows, baseline, candidates = _panel()
    candidates["P1"][0]["VOICE_FAKE_PROB"] = 0.1

    # When / Then: evaluation rejects the candidate before any ranking.
    with pytest.raises(ValueError, match="protected G0 output changed"):
        evaluate_development(rows, baseline, candidates["P1"])


def test_candidate_file_score_must_use_the_locked_max_formula() -> None:
    # Given: all component scores are valid but FILE_FAKE_PROB bypasses the fixed formula.
    rows, baseline, candidates = _panel()
    candidates["P1"][0]["FILE_FAKE_PROB"] = 0.123

    # When / Then: evaluation rejects the inconsistent file score.
    with pytest.raises(ValueError, match="FILE formula mismatch"):
        evaluate_development(rows, baseline, candidates["P1"])


def test_actual_rounded_graph_g0_is_within_file_formula_tolerance() -> None:
    # Given: graph_fusion rounds FILE and component probabilities independently.
    rows, baseline, candidates = _panel()
    feature = {"ID": "real-0", "starts": [0],
               "presence": [[0.98765432105, 0.87654321095]],
               "voice_fake": [0.12345678905], "music_fake": [0.23456789015],
               "raw_fake": [0.1], "stem_rms": [0.1, 0.1], "duration": 4.0}
    rounded_g0 = graph_predictions(feature)["G0"]
    recomputed = max(rounded_g0["VOICE_PRESENT_PROB"] * rounded_g0["VOICE_FAKE_PROB"],
                     rounded_g0["MUSIC_PRESENT_PROB"] * rounded_g0["MUSIC_FAKE_PROB"])
    assert rounded_g0["FILE_FAKE_PROB"] != recomputed
    baseline[0] = rounded_g0
    candidates["P1"][0] = rounded_g0.copy()

    # When / Then: the real G0 rounding path remains evaluable.
    evaluate_development(rows, baseline, candidates["P1"])


def test_recipe_lock_allows_resume_but_refuses_reselection(tmp_path: Path) -> None:
    # Given: a completed development choice has been locked.
    path = tmp_path / "recipe_lock.json"
    selection = tmp_path / "selection.json"
    eligible = tmp_path / "eligible.csv"
    selection.write_text(json.dumps({"selected": "P1", "candidates": {"P1": {
        "status": "completed", "gates": {"passed": True, "resource": True}}}}), encoding="utf-8")
    eligible.write_text("ID\nexample\n", encoding="utf-8")
    payload = {"selected": "P1", "parent": "P1", "treatment": "mean",
               "learning_rate": 1e-5, "scheduler": "fixed-30-epoch-cosine",
               "epoch_count": 3, "stage_epochs": {"main": 3},
               "data_hashes": {"train": "a"},
               "source_hashes": {"source": "b"}, "pooling": "mean",
               "seeds": [20260913], "gate_thresholds": {"D": 0.005}}
    first_hash = write_recipe_lock(path, selection, eligible, payload)

    # When: the same interrupted stage resumes, then a different winner is attempted.
    resumed_hash = write_recipe_lock(path, selection, eligible, payload)
    changed = {**payload, "learning_rate": 3e-5}

    # Then: the retry is stable and post-lock reselection is refused.
    assert resumed_hash == first_hash
    with pytest.raises(SelectionError, match="immutable lock changed"):
        write_recipe_lock(path, selection, eligible, changed)


def test_failed_or_resource_ineligible_candidate_cannot_be_recipe_locked(tmp_path: Path) -> None:
    # Given: development retained G0 and P2 failed its resource gate.
    selection = tmp_path / "selection.json"
    eligible = tmp_path / "eligible.csv"
    selection.write_text(json.dumps({"selected": "G0", "candidates": {"P2": {
        "status": "completed", "gates": {"passed": False, "resource": False}}}}), encoding="utf-8")
    eligible.write_text("ID\nexample\n", encoding="utf-8")
    payload = {"selected": "P2", "parent": "P2", "treatment": "mean",
               "learning_rate": 1e-6, "scheduler": "fixed-30-epoch-cosine",
               "epoch_count": 3, "stage_epochs": {"main": 3},
               "data_hashes": {"train": "a"}, "source_hashes": {"source": "b"},
               "pooling": "mean", "seeds": [20260913],
               "gate_thresholds": {"D": 0.005}}

    # When / Then: a caller cannot bypass selection by writing P2 directly.
    with pytest.raises(SelectionError, match="does not match the development winner"):
        write_recipe_lock(tmp_path / "recipe_lock.json", selection, eligible, payload)


def test_missing_prediction_id_refuses_evaluation() -> None:
    # Given: one otherwise valid prediction set omits a manifest ID.
    rows, baseline, candidates = _panel()

    # When / Then: exact prediction coverage is mandatory.
    with pytest.raises(ValueError, match="IDs must match"):
        evaluate_development(rows, baseline, candidates["P1"][:-1])


def test_incomplete_refit_exposure_cannot_create_weight_lock(tmp_path: Path) -> None:
    # Given: completion flags are true but one file missed a prescribed exposure.
    recipe = tmp_path / "recipe_lock.json"
    weights = tmp_path / "weights.pt"
    manifest = tmp_path / "refit_manifest.csv"
    code = tmp_path / "inference.py"
    weights.write_bytes(b"weights")
    manifest.write_text("ID,expected_exposures,actual_exposures\nexample,3,2\n", encoding="utf-8")
    code.write_text("pass\n", encoding="utf-8")
    started = _recipe(recipe)

    # When / Then: the weight lock checks the exposure ledger itself and rejects it.
    with pytest.raises(SelectionError, match="refit exposures are incomplete"):
        write_weight_lock(tmp_path / "weight_lock.json", recipe, weights, (code,), manifest,
                          {"completed": True, "exposures_complete": True,
                           "started_unix": started, "completed_unix": time.time(),
                           "prescribed_epochs": 3, "completed_epochs": 3})


def test_refit_manifest_must_cover_the_recipe_locked_eligible_union(tmp_path: Path) -> None:
    # Given: the recipe locked two eligible IDs but the refit ledger contains only one.
    recipe = tmp_path / "recipe_lock.json"
    weights = tmp_path / "weights.pt"
    manifest = tmp_path / "refit_manifest.csv"
    code = tmp_path / "inference.py"
    started = _recipe(recipe, ("example", "second"))
    weights.write_bytes(b"weights")
    manifest.write_text("ID,expected_exposures,actual_exposures\nexample,3,3\n", encoding="utf-8")
    code.write_text("pass\n", encoding="utf-8")

    # When / Then: a partial ID ledger cannot acquire a weight lock.
    with pytest.raises(SelectionError, match="refit manifest is incomplete"):
        write_weight_lock(tmp_path / "weight_lock.json", recipe, weights, (code,), manifest,
                          {"completed": True, "exposures_complete": True,
                           "started_unix": started, "completed_unix": time.time(),
                           "prescribed_epochs": 3, "completed_epochs": 3})


def test_changed_weight_hash_refuses_sealed_evaluation(tmp_path: Path) -> None:
    # Given: a complete refit is locked and its weights are later changed.
    recipe = tmp_path / "recipe_lock.json"
    weights = tmp_path / "weights.pt"
    manifest = tmp_path / "refit_manifest.csv"
    code = tmp_path / "inference.py"
    weights.write_bytes(b"locked weights")
    manifest.write_text("ID,expected_exposures,actual_exposures\nexample,3,3\n", encoding="utf-8")
    code.write_text("pass\n", encoding="utf-8")
    started = _recipe(recipe)
    lock = tmp_path / "weight_lock.json"
    write_weight_lock(lock, recipe, weights, (code,), manifest,
                      {"completed": True, "exposures_complete": True,
                       "started_unix": started, "completed_unix": time.time(),
                       "prescribed_epochs": 3, "completed_epochs": 3})
    weights.write_bytes(b"changed weights")

    # When / Then: sealed scoring cannot continue with substituted weights.
    with pytest.raises(SelectionError, match="weight hash changed"):
        verify_weight_lock(lock, weights)


def test_sealed_gate_cannot_promote_a_runner_up(tmp_path: Path) -> None:
    # Given: P1 is recipe-locked, fully refit, weight-locked, and sealed is opened.
    recipe = tmp_path / "recipe_lock.json"
    weights = tmp_path / "weights.pt"
    manifest = tmp_path / "refit_manifest.csv"
    code = tmp_path / "inference.py"
    weights.write_bytes(b"locked weights")
    manifest.write_text("ID,expected_exposures,actual_exposures\nexample,3,3\n", encoding="utf-8")
    code.write_text("pass\n", encoding="utf-8")
    started = _recipe(recipe)
    lock = tmp_path / "weight_lock.json"
    write_weight_lock(lock, recipe, weights, (code,), manifest,
                      {"completed": True, "exposures_complete": True,
                       "started_unix": started, "completed_unix": time.time(),
                       "prescribed_epochs": 3, "completed_epochs": 3})
    opened = tmp_path / "sealed_opened.json"
    mark_sealed_opened(opened, lock, weights)

    # When / Then: P2 cannot replace the frozen P1 after sealed data is opened.
    with pytest.raises(SelectionError, match="differs from the recipe lock"):
        accept_sealed([], "P2", lock, weights, opened, tmp_path / "unused.json")


def test_sealed_predictions_must_match_the_locked_inference_receipt(tmp_path: Path) -> None:
    # Given: P1 outputs are hashed after the weight lock and sealed-open receipt.
    recipe = tmp_path / "recipe_lock.json"
    weights = tmp_path / "weights.pt"
    manifest = tmp_path / "refit_manifest.csv"
    code = tmp_path / "inference.py"
    weights.write_bytes(b"locked weights")
    manifest.write_text("ID,expected_exposures,actual_exposures\nexample,3,3\n", encoding="utf-8")
    code.write_text("pass\n", encoding="utf-8")
    started = _recipe(recipe)
    lock = tmp_path / "weight_lock.json"
    write_weight_lock(lock, recipe, weights, (code,), manifest,
                      {"completed": True, "exposures_complete": True,
                       "started_unix": started, "completed_unix": time.time(),
                       "prescribed_epochs": 3, "completed_epochs": 3})
    opened = tmp_path / "sealed_opened.json"
    mark_sealed_opened(opened, lock, weights)
    inference_started = json.loads(opened.read_text(encoding="utf-8"))["opened_unix"]
    baseline_csv = tmp_path / "G0_sealed.csv"
    candidate_csv = tmp_path / "P1_sealed.csv"
    baseline_csv.write_text("ID\nexample\n", encoding="utf-8")
    candidate_csv.write_text("ID\nexample\n", encoding="utf-8")
    receipt = tmp_path / "sealed_inference.json"
    write_sealed_inference_receipt(receipt, lock, weights, opened, baseline_csv,
                                   candidate_csv, inference_started, time.time())
    candidate_csv.write_text("ID\nsubstituted\n", encoding="utf-8")

    # When / Then: scoring refuses predictions changed after locked inference.
    with pytest.raises(SelectionError, match="prediction hash changed"):
        accept_sealed([], "P1", lock, weights, opened, receipt)
