"""The five fixed, file-local Recovery V3 candidate formulas."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Final

from graph_fusion import Prediction

CANDIDATES: Final = ("G0", "SONICS_RAW", "SONICS_STEM", "BLEND_RAW", "BLEND_STEM")


@dataclass(frozen=True, slots=True)
class FusionContractError(ValueError):
    detail: str

    def __str__(self) -> str:
        return self.detail


def _probability(value: float, name: str) -> float:
    """Parse one probability at the file-local fusion boundary."""
    try:
        parsed = float(value)
    except (TypeError, ValueError) as error:
        raise FusionContractError(f"{name} must be a finite probability") from error
    if not math.isfinite(parsed) or not 0.0 <= parsed <= 1.0:
        raise FusionContractError(f"{name} must be a finite probability in [0, 1]")
    return parsed


def _row(g0: Prediction, music: float) -> Prediction:
    """Keep G0's voice and presence heads exact while replacing music authenticity."""
    voice_fake = _probability(g0["VOICE_FAKE_PROB"], "G0 voice fake probability")
    voice_present = _probability(g0["VOICE_PRESENT_PROB"], "G0 voice presence probability")
    music_present = _probability(g0["MUSIC_PRESENT_PROB"], "G0 music presence probability")
    return {
        "ID": g0["ID"],
        "FILE_FAKE_PROB": max(voice_fake * voice_present, music * music_present),
        "VOICE_FAKE_PROB": g0["VOICE_FAKE_PROB"],
        "MUSIC_FAKE_PROB": music,
        "VOICE_PRESENT_PROB": g0["VOICE_PRESENT_PROB"],
        "MUSIC_PRESENT_PROB": g0["MUSIC_PRESENT_PROB"],
    }


def candidate_predictions(g0: Prediction, raw_prob: float, stem_prob: float) -> dict[str, Prediction]:  # noqa: DICT_OK
    """Build exactly the declared candidates for one file without cross-file state."""
    if not isinstance(g0["ID"], str) or not g0["ID"]:
        raise FusionContractError("G0 ID must be a nonempty string")
    baseline = {name: _probability(g0[name], f"G0 {name}") for name in (
        "FILE_FAKE_PROB", "VOICE_FAKE_PROB", "MUSIC_FAKE_PROB", "VOICE_PRESENT_PROB", "MUSIC_PRESENT_PROB"
    )}
    raw = _probability(raw_prob, "SONICS raw probability")
    stem = _probability(stem_prob, "SONICS stem probability")
    g0_row: Prediction = {"ID": g0["ID"], **baseline}
    music = baseline["MUSIC_FAKE_PROB"]
    return {
        "G0": g0_row,
        "SONICS_RAW": _row(g0_row, raw),
        "SONICS_STEM": _row(g0_row, stem),
        "BLEND_RAW": _row(g0_row, 0.5 * music + 0.5 * raw),
        "BLEND_STEM": _row(g0_row, 0.5 * music + 0.5 * stem),
    }
