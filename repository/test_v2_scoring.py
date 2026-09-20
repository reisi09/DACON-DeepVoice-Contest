"""Decision tests for source-lineage selection and ROC diagnosis."""

from __future__ import annotations

from pathlib import Path

import pytest

from v2_scoring import _load_predictions, _read_csv, audit_forensics, select_candidate


def _row(identifier: str, role: str, group: str, file_fake: int) -> dict[str, str | int]:
    return {
        "ID": identifier,
        "role": role,
        "group_id": group,
        "FILE_FAKE": file_fake,
        "VOICE_FAKE": file_fake,
        "MUSIC_FAKE": file_fake,
        "VOICE_PRESENT": 1,
        "MUSIC_PRESENT": 1,
    }


def _prediction(identifier: str, score: float) -> dict[str, str | float]:
    return {
        "ID": identifier,
        "FILE_FAKE_PROB": score,
        "VOICE_FAKE_PROB": score,
        "MUSIC_FAKE_PROB": score,
        "VOICE_PRESENT_PROB": score,
        "MUSIC_PRESENT_PROB": score,
    }


def test_select_rejects_locked_test_rows_before_reading_scores() -> None:
    # Given: predictions that would otherwise make a candidate look perfect.
    rows = [_row("a", "dev_select", "voice:a", 0), _row("b", "test_generator", "voice:b", 1)]
    predictions = {"G0": [_prediction("a", 0.5), _prediction("b", 0.5)],
                   "AST_FROZEN": [_prediction("a", 0.0), _prediction("b", 1.0)]}
    # When / Then: selection must stop at the role boundary, not consume test scores.
    with pytest.raises(ValueError, match="selection accepts development roles only"):
        select_candidate(rows, predictions)


def test_audit_reports_equal_auc_and_worse_discrete_crossing_separately() -> None:
    # Given: the same ranking AUC but a threshold crossing with more FP/FN.
    rows = [_row("n1", "holdout", "n1", 0), _row("n2", "holdout", "n2", 0),
            _row("p1", "holdout", "p1", 1), _row("p2", "holdout", "p2", 1)]
    baseline = [_prediction("n1", 0.1), _prediction("n2", 0.8),
                _prediction("p1", 0.7), _prediction("p2", 0.9)]
    changed = [_prediction("n1", 0.2), _prediction("n2", 0.8),
               _prediction("p1", 0.7), _prediction("p2", 0.9)]
    # When: discrete ROC operating points and ranking AUC are diagnosed together.
    report = audit_forensics(rows, {"G0": baseline, "G3": changed})
    # Then: the report retains both facts rather than calling the whole model worse.
    comparison = report["roles"]["holdout"]["comparisons"]["G3"]["file"]
    assert comparison["auc_delta"] == 0.0
    assert comparison["candidate"]["eer"] >= comparison["baseline"]["eer"]


def test_existing_pilot_reproduces_all_five_graphs_and_known_non_monotonic_crossings() -> None:
    # Given: the completed 225-file pilot's cached feature and fusion outputs.
    rows = _read_csv(Path("validation/manifest.csv"))
    report = audit_forensics(rows, _load_predictions(Path("results/pilot_20260912_160943"), rows))
    # When: all five graphs are recomputed using the current nearest-point rule.
    holdout = report["roles"]["holdout"]
    # Then: G2 is retained and G1/G3 retain their distinct AUC/EER evidence.
    assert set(holdout["models"]) == {"G0", "G1", "G2", "G3", "G4"}
    g1_voice = holdout["comparisons"]["G1"]["voice"]
    g3_music = holdout["comparisons"]["G3"]["music"]
    assert g1_voice["baseline"]["eer"] == 0.041958041958041946
    assert g1_voice["candidate"]["eer"] == 0.08391608391608389
    assert g1_voice["auc_delta"] > 0
    assert g3_music["auc_delta"] == 0.0
    assert (g3_music["baseline"]["fp"], g3_music["candidate"]["fp"]) == (3, 4)


def test_connected_lineage_bootstrap_keeps_transitive_mixture_constituents_together() -> None:
    # Given: a mixture joins two otherwise separate parents; sampling parents independently leaks it.
    rows = [_row("a", "dev_select", "voice:a", 0), _row("b", "dev_select", "music:b", 1),
            _row("mix", "dev_select", "voice:a|music:b", 1), _row("c", "dev_select", "c", 0)]
    base = [_prediction("a", .1), _prediction("b", .8), _prediction("mix", .7), _prediction("c", .2)]
    candidate = [_prediction("a", .1), _prediction("b", .9), _prediction("mix", .9), _prediction("c", .2)]
    # When: development-only selection forms resampling units.
    result = select_candidate(rows, {"G0": base, "AST_FROZEN": candidate})
    # Then: a and b share the component created by mix, and its count is three units not four clips.
    units = result["models"]["AST_FROZEN"]["bootstrap"]["lineage_components"]
    assert units == 2


def test_partial_labels_are_normalized_only_over_computable_official_weights() -> None:
    # Given: voice authenticity is unknown everywhere, while file and presence remain known.
    rows = [_row("n", "dev_select", "n", 0), _row("p", "dev_select", "p", 1)]
    for row in rows:
        row["VOICE_FAKE"] = ""
    base = [_prediction("n", .6), _prediction("p", .4)]
    candidate = [_prediction("n", .1), _prediction("p", .9)]
    # When: scores are evaluated as the explicitly partial development proxy.
    result = select_candidate(rows, {"G0": base, "AST_FROZEN": candidate})
    # Then: no unavailable voice-EER weight is silently treated as a zero contribution.
    report = result["models"]["G0"]
    assert "voice_eer" not in report["proxy_weights"]
    assert report["metric_report"]["metrics"]["voice_eer"] is None


def test_selector_breaks_equal_passing_candidates_in_fixed_order() -> None:
    # Given: twenty source-disjoint development components and two equally perfect candidates.
    rows = [_row(f"r{index}", "dev_select", f"r{index}", index % 2) for index in range(20)]
    base = [_prediction(row["ID"], .9 if row["FILE_FAKE"] == 0 else .1) for row in rows]
    perfect = [_prediction(row["ID"], .1 if row["FILE_FAKE"] == 0 else .9) for row in rows]
    # When: both candidates pass the development proxy and component-bootstrap gates.
    result = select_candidate(rows, {"G0": base, "AST_FROZEN": perfect, "AST_LAST_BLOCK": perfect})
    # Then: equal evidence chooses the documented frozen-first deterministic order.
    assert result["selected"] == "AST_FROZEN"
