from __future__ import annotations

import pytest

from graph_fusion import Prediction
from recovery_fusion import CANDIDATES, FusionContractError, candidate_predictions


def _g0() -> Prediction:
    return {
        "ID": "same-file",
        "FILE_FAKE_PROB": 0.20,
        "VOICE_FAKE_PROB": 0.80,
        "MUSIC_FAKE_PROB": 0.30,
        "VOICE_PRESENT_PROB": 0.25,
        "MUSIC_PRESENT_PROB": 0.50,
    }


def test_candidates_change_only_music_and_recompute_file_head() -> None:
    # Given: fixed baseline heads and separate raw/stem SONICS probabilities.
    g0 = _g0()

    # When: the five predeclared candidates are created.
    rows = candidate_predictions(g0, raw_prob=0.90, stem_prob=0.40)

    # Then: G0 remains exact, while each replacement updates only its dependent file head.
    assert tuple(rows) == CANDIDATES
    assert rows["G0"] == g0
    assert rows["SONICS_RAW"]["MUSIC_FAKE_PROB"] == 0.90
    assert rows["SONICS_STEM"]["MUSIC_FAKE_PROB"] == 0.40
    assert rows["BLEND_RAW"]["MUSIC_FAKE_PROB"] == pytest.approx(0.60)
    assert rows["BLEND_STEM"]["MUSIC_FAKE_PROB"] == pytest.approx(0.35)
    for name in CANDIDATES[1:]:
        row = rows[name]
        assert row["VOICE_FAKE_PROB"] == g0["VOICE_FAKE_PROB"]
        assert row["VOICE_PRESENT_PROB"] == g0["VOICE_PRESENT_PROB"]
        assert row["MUSIC_PRESENT_PROB"] == g0["MUSIC_PRESENT_PROB"]
        assert row["FILE_FAKE_PROB"] == max(0.80 * 0.25, row["MUSIC_FAKE_PROB"] * 0.50)


def test_candidates_are_file_local_and_reject_invalid_probabilities() -> None:
    # Given: one baseline row and an unrelated row that would alter a ranked/global implementation.
    original = candidate_predictions(_g0(), raw_prob=0.10, stem_prob=0.20)
    unrelated = _g0()
    unrelated["ID"] = "other-file"
    unrelated["MUSIC_FAKE_PROB"] = 0.99

    # When: each file is fused independently.
    again = candidate_predictions(_g0(), raw_prob=0.10, stem_prob=0.20)
    candidate_predictions(unrelated, raw_prob=0.95, stem_prob=0.99)

    # Then: unrelated data cannot change this file, and invalid inputs fail at the boundary.
    assert again == original
    with pytest.raises(FusionContractError, match="finite"):
        candidate_predictions(_g0(), raw_prob=float("nan"), stem_prob=0.2)
