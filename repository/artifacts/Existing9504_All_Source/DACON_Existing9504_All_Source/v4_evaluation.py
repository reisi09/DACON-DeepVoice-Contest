"""Exact V4 metrics on fixed development and sealed panels."""

from __future__ import annotations

import random
from collections.abc import Mapping, Sequence
from math import isclose
from typing import Final

from graph_scoring import _roc_metrics, official_metrics, validate_predictions
from v2_scoring import _components

Scalar = str | int | float | bool | None
Row = Mapping[str, str | int | float | None]
Report = dict[str, Scalar | dict | list]
PANEL_COUNTS: Final = {
    "music_real": 10, "music_seen": 30, "music_ood": 10,
    "speech": 40, "mixture": 80, "splice": 40,
}
PROTECTED: Final = ("VOICE_FAKE_PROB", "VOICE_PRESENT_PROB", "MUSIC_PRESENT_PROB")
FILE_FORMULA_ATOL: Final = 2e-10


def _label(row: Row, name: str) -> int | None:
    value = row.get(name)
    if value in (None, ""):
        return None
    if value not in (0, 1, "0", "1"):
        raise ValueError(f"invalid binary label: {name}")
    return int(value)


def _prediction_maps(rows: Sequence[Row], baseline: Sequence[Row], candidate: Sequence[Row]) -> tuple[dict[str, Row], dict[str, Row]]:
    expected = [str(row["ID"]) for row in rows]
    parsed_base = validate_predictions(baseline, expected)
    parsed_candidate = validate_predictions(candidate, expected)
    base_map: dict[str, Row] = {row["ID"]: row for row in parsed_base}
    candidate_map: dict[str, Row] = {row["ID"]: row for row in parsed_candidate}
    for identifier in expected:
        base, changed = base_map[identifier], candidate_map[identifier]
        if any(changed[column] != base[column] for column in PROTECTED):
            raise ValueError(f"protected G0 output changed: {identifier}")
        expected_file = max(
            float(base["VOICE_PRESENT_PROB"]) * float(base["VOICE_FAKE_PROB"]),
            float(base["MUSIC_PRESENT_PROB"]) * float(changed["MUSIC_FAKE_PROB"]),
        )
        if not isclose(float(changed["FILE_FAKE_PROB"]), expected_file,
                       rel_tol=0.0, abs_tol=FILE_FORMULA_ATOL):
            raise ValueError(f"FILE formula mismatch: {identifier}")
    return base_map, candidate_map


def _head(rows: Sequence[Row], predictions: Mapping[str, Row], target: str,
          score: str, mask: str | None = None) -> dict[str, float | int | None]:
    targets: list[int] = []
    scores: list[float] = []
    unknown = 0
    for row in rows:
        if mask is not None:
            present = _label(row, mask)
            if present is None:
                unknown += 1
                continue
            if present == 0:
                continue
        truth = _label(row, target)
        if truth is None:
            unknown += 1
            continue
        targets.append(truth)
        scores.append(float(predictions[str(row["ID"])][score]))
    if len(set(targets)) != 2:
        return {"eer": None, "auc": None, "n": len(targets), "unknown": unknown}
    eer, auc = _roc_metrics(targets, scores)
    return {"eer": eer, "auc": auc, "n": len(targets), "unknown": unknown}


def _comparison(rows: Sequence[Row], baseline: Mapping[str, Row], candidate: Mapping[str, Row]) -> Report:
    base_file = _head(rows, baseline, "FILE_FAKE", "FILE_FAKE_PROB")
    changed_file = _head(rows, candidate, "FILE_FAKE", "FILE_FAKE_PROB")
    base_music = _head(rows, baseline, "MUSIC_FAKE", "MUSIC_FAKE_PROB", "MUSIC_PRESENT")
    changed_music = _head(rows, candidate, "MUSIC_FAKE", "MUSIC_FAKE_PROB", "MUSIC_PRESENT")
    values = (base_file["eer"], changed_file["eer"], base_music["eer"], changed_music["eer"])
    gain = None if any(value is None for value in values) else (
        .45 * (float(base_file["eer"]) - float(changed_file["eer"]))
        + .27 * (float(base_music["eer"]) - float(changed_music["eer"]))
    )
    return {"D": gain, "G0": {"file": base_file, "music": base_music},
            "candidate": {"file": changed_file, "music": changed_music}}


def _class_support(rows: Sequence[Row]) -> dict[str, int]:
    support = {"0": 0, "1": 0}
    for unit in _components(rows):
        labels = {_label(rows[index], "MUSIC_FAKE") for index in unit
                  if _label(rows[index], "MUSIC_PRESENT") == 1}
        for label in (0, 1):
            support[str(label)] += int(label in labels)
    return support


def _bootstrap(rows: Sequence[Row], baseline: Mapping[str, Row], candidate: Mapping[str, Row]) -> Report:
    units = _components(rows)
    rng = random.Random(20260913)
    deltas: list[float] = []
    for _ in range(2000):
        sampled = [rows[index] for unit in (rng.choice(units) for _ in units) for index in unit]
        delta = _comparison(sampled, baseline, candidate)["D"]
        if isinstance(delta, float):
            deltas.append(delta)
    deltas.sort()
    return {"seed": 20260913, "resamples": 2000, "lineage_components": len(units),
            "class_group_support": _class_support(rows), "valid_resamples": len(deltas),
            "valid_rate": len(deltas) / 2000,
            "win_rate": None if not deltas else sum(value > 0 for value in deltas) / len(deltas),
            "p10_delta": None if not deltas else deltas[(len(deltas) - 1) // 10]}


def _validate_panel(rows: Sequence[Row], split: str) -> None:
    counts = {panel: 0 for panel in PANEL_COUNTS}
    for row in rows:
        if row.get("split") != split:
            raise ValueError(f"evaluation rows must use split={split}")
        panel = row.get("panel")
        if panel not in counts:
            raise ValueError(f"unknown evaluation panel: {panel}")
        counts[str(panel)] += 1
    if counts != PANEL_COUNTS:
        raise ValueError(f"evaluation panel counts must be exactly {PANEL_COUNTS}")


def _evaluate(rows: Sequence[Row], baseline: Sequence[Row], candidate: Sequence[Row], split: str) -> Report:
    _validate_panel(rows, split)
    base_map, candidate_map = _prediction_maps(rows, baseline, candidate)
    real = [row for row in rows if row["panel"] == "music_real"]
    music_panels: dict[str, Report] = {}
    for panel in ("music_seen", "music_ood"):
        selected = [*real, *(row for row in rows if row["panel"] == panel)]
        comparison = _comparison(selected, base_map, candidate_map)
        comparison["bootstrap"] = _bootstrap(selected, base_map, candidate_map)
        music_panels[panel] = comparison
    controlled = {panel: _comparison(
        [row for row in rows if row["panel"] == panel], base_map, candidate_map)
        for panel in ("speech", "mixture", "splice")}
    absolute = official_metrics(rows, [candidate_map[str(row["ID"])] for row in rows])
    gains = [music_panels[panel]["D"] for panel in music_panels]
    aucs = [music_panels[panel]["candidate"]["music"]["auc"] for panel in music_panels]
    return {"split": split, "absolute": absolute, "music": music_panels,
            "controlled": controlled,
            "rank_score": None if any(value is None for value in gains) else min(gains),
            "music_auc_tiebreak": None if any(value is None for value in aucs) else min(aucs)}


def evaluate_development(rows: Sequence[Row], baseline: Sequence[Row], candidate: Sequence[Row]) -> Report:
    return _evaluate(rows, baseline, candidate, "dev")


def evaluate_sealed(rows: Sequence[Row], baseline: Sequence[Row], candidate: Sequence[Row]) -> Report:
    return _evaluate(rows, baseline, candidate, "sealed")
