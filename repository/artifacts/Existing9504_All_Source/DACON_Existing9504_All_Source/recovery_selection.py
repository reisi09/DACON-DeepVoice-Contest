"""Frozen candidate selection and one-way fresh acceptance for Recovery V3."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Final

from graph_scoring import validate_predictions
from v2_scoring import _bootstrap, _detail_report, _proxy

Scalar = str | int | float | None
Row = Mapping[str, Scalar]
PredictionSets = Mapping[str, Sequence[Row]]
CANDIDATES: Final = ("G0", "SONICS_RAW", "SONICS_STEM", "BLEND_RAW", "BLEND_STEM")
PROTECTED: Final = ("VOICE_FAKE_PROB", "VOICE_PRESENT_PROB", "MUSIC_PRESENT_PROB")
EVALUATED_HEADS: Final = ("file", "music")


@dataclass(frozen=True, slots=True)
class SelectionError(ValueError):
    detail: str

    def __str__(self) -> str:
        return self.detail


def _digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _maps(rows: Sequence[Row], prediction_sets: PredictionSets) -> dict[str, dict[str, Row]]:
    if tuple(prediction_sets) != CANDIDATES:
        raise SelectionError(f"candidate registry must be exactly {CANDIDATES}")
    expected = [str(row["ID"]) for row in rows]
    maps = {name: {row["ID"]: row for row in validate_predictions(values, expected)}
            for name, values in prediction_sets.items()}
    baseline = maps["G0"]
    for name in CANDIDATES[1:]:
        for identifier in expected:
            g0, candidate = baseline[identifier], maps[name][identifier]
            if any(candidate[field] != g0[field] for field in PROTECTED):
                raise SelectionError(f"protected G0 field changed: {name}/{identifier}")
            file_fake = max(float(g0["VOICE_PRESENT_PROB"]) * float(g0["VOICE_FAKE_PROB"]),
                            float(g0["MUSIC_PRESENT_PROB"]) * float(candidate["MUSIC_FAKE_PROB"]))
            if float(candidate["FILE_FAKE_PROB"]) != file_fake:
                raise SelectionError(f"FILE formula mismatch: {name}/{identifier}")
    return maps


def _report(rows: Sequence[Row], predictions: Mapping[str, Row]) -> dict[str, Scalar | dict[str, Scalar]]:
    proxy = _proxy(rows, predictions)
    metrics = proxy["metric_report"]
    details = _detail_report(rows, predictions)
    return {"known_subset_weighted_diagnostic": proxy["value"],
            "weights_used": proxy["proxy_weights"], "metrics": metrics["metrics"],
            "counts": metrics["counts"], "unknown_counts": metrics["unknown_counts"],
            "coverage": metrics["coverage"], "partial": metrics["partial"],
            "invalid": metrics["invalid"], "companion_auc": {
                head: details[head]["point"]["auc"] if details[head]["point"] is not None else None
                for head in EVALUATED_HEADS}}


def _relative_gates(base: Mapping[str, Scalar | dict[str, Scalar]],
                    current: Mapping[str, Scalar | dict[str, Scalar]], gain_required: bool) -> dict[str, Scalar]:
    base_metrics, metrics = base["metrics"], current["metrics"]
    base_auc, auc = base["companion_auc"], current["companion_auc"]
    gain = None
    if base["known_subset_weighted_diagnostic"] is not None and current["known_subset_weighted_diagnostic"] is not None:
        gain = float(current["known_subset_weighted_diagnostic"]) - float(base["known_subset_weighted_diagnostic"])
    gates: dict[str, Scalar] = {"diagnostic_gain": not gain_required or (gain is not None and gain >= .005),
        "diagnostic_gain_delta": gain}
    for head in EVALUATED_HEADS:
        eer_name = f"{head}_eer"
        gates[f"{head}_eer"] = (metrics[eer_name] is not None and base_metrics[eer_name] is not None
            and float(metrics[eer_name]) <= float(base_metrics[eer_name]) + .02)
        gates[f"{head}_companion_auc"] = (auc[head] is not None and base_auc[head] is not None
            and float(auc[head]) >= float(base_auc[head]) - .01)
    gates["passed"] = all(value is True for key, value in gates.items() if key != "diagnostic_gain_delta")
    return gates


def evaluate_old(dev_rows: Sequence[Row], old_rows: Sequence[Row], prediction_sets: PredictionSets) -> dict[str, Scalar | dict]:
    """Rank on dev only; reused old-domain roles can only veto candidates."""
    if any(row.get("split") != "dev_select" for row in dev_rows):
        raise SelectionError("development selection requires dev_select rows only")
    if any(row.get("split") not in ("test_generator", "test_source") for row in old_rows):
        raise SelectionError("old safety rows require test_generator/test_source")
    rows = [*dev_rows, *old_rows]
    maps = _maps(rows, prediction_sets)
    dev_ids = [str(row["ID"]) for row in dev_rows]
    dev_maps = {name: {identifier: values[identifier] for identifier in dev_ids} for name, values in maps.items()}
    base = _report(dev_rows, dev_maps["G0"])
    candidates: dict[str, dict] = {"G0": {"development": base}}
    passed: list[str] = []
    for name in CANDIDATES[1:]:
        development = _report(dev_rows, dev_maps[name])
        bootstrap = _bootstrap(dev_rows, dev_maps["G0"], dev_maps[name])
        dev_gates = _relative_gates(base, development, True)
        dev_gates.update(bootstrap_complete=bootstrap["valid_resamples"] == 2000,
            bootstrap_win_rate=bootstrap["win_rate"] is not None and bootstrap["win_rate"] >= .70,
            bootstrap_p10=bootstrap["p10_delta"] is not None and bootstrap["p10_delta"] >= -.005)
        dev_gates["passed"] = all(value is True for key, value in dev_gates.items()
                                      if key not in ("diagnostic_gain_delta", "passed"))
        slices: dict[str, dict] = {}
        for role in ("test_generator", "test_source"):
            selected = [row for row in old_rows if row["split"] == role]
            if selected:
                slices[role] = _relative_gates(_report(selected, maps["G0"]), _report(selected, maps[name]), False)
        failed_slices = [role for role, value in slices.items() if not value["passed"]]
        old_safety = {"usage": "REUSED diagnostic safety veto; never ranking", "slices": slices,
                      "failed_slices": failed_slices, "passed": bool(slices) and not failed_slices}
        candidates[name] = {"development": development, "development_gates": dev_gates,
                            "bootstrap": bootstrap, "old_safety": old_safety}
        if dev_gates["passed"]:
            passed.append(name)
    development_selected = max(passed, key=lambda name: (float(candidates[name]["development"]["known_subset_weighted_diagnostic"]),
                                                          -CANDIDATES.index(name)), default="G0")
    selected = development_selected
    reason = "all_gates_passed"
    if development_selected == "G0":
        reason = "no_candidate_passed_development_gates"
    elif not candidates[development_selected]["old_safety"]["passed"]:
        selected, reason = "G0", "development_winner_failed_old_safety_veto"
    return {"development_selected": development_selected, "selected": selected, "reason": reason,
            "old_usage": "REUSED diagnostic safety veto; never ranking", "candidates": candidates}


def lock_candidate(path: Path, payload: Mapping[str, Scalar | dict]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    serialized = json.dumps(payload, sort_keys=True, indent=2, allow_nan=False) + "\n"
    if path.is_file():
        if path.read_text(encoding="utf-8") != serialized:
            raise SelectionError("candidate lock is immutable")
        return _digest(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(serialized, encoding="utf-8")
    temporary.replace(path)
    return _digest(path)


def _fresh_maps(rows: Sequence[Row], prediction_sets: PredictionSets, selected: str) -> dict[str, dict[str, Row]]:
    names = tuple(prediction_sets)
    if names == CANDIDATES:
        return _maps(rows, prediction_sets)
    expected_names = ("G0",) if selected == "G0" else ("G0", selected)
    if names != expected_names:
        raise SelectionError("fresh predictions must contain G0 and the frozen candidate only")
    expected = [str(row["ID"]) for row in rows]
    maps = {name: {row["ID"]: row for row in validate_predictions(values, expected)}
            for name, values in prediction_sets.items()}
    if selected == "G0":
        return maps
    for identifier in expected:
        g0, candidate = maps["G0"][identifier], maps[selected][identifier]
        if any(candidate[field] != g0[field] for field in PROTECTED):
            raise SelectionError(f"protected G0 field changed: {selected}/{identifier}")
        expected_file = max(float(g0["VOICE_PRESENT_PROB"]) * float(g0["VOICE_FAKE_PROB"]),
                            float(g0["MUSIC_PRESENT_PROB"]) * float(candidate["MUSIC_FAKE_PROB"]))
        if float(candidate["FILE_FAKE_PROB"]) != expected_file:
            raise SelectionError(f"FILE formula mismatch: {selected}/{identifier}")
    return maps


def decide_fresh(lock_path: Path, manifest_path: Path, rows: Sequence[Row],
                 prediction_sets: PredictionSets) -> dict[str, Scalar | list[str] | dict]:
    """Accept or reject the frozen candidate without selecting on fresh labels."""
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    if _digest(manifest_path) != lock["fresh_manifest_sha256"]:
        raise SelectionError("fresh manifest changed after candidate lock")
    selected = lock["selected"]
    if selected not in CANDIDATES:
        raise SelectionError("locked candidate is outside the registry")
    maps = _fresh_maps(rows, prediction_sets, selected)
    ignored = [name for name in CANDIDATES if name not in ("G0", selected)]
    if selected == "G0":
        baseline = _report(rows, maps["G0"])
        return {"final": "G0", "accepted": False, "reason": "development_selected_G0",
                "evaluated": ["G0"], "ignored_candidates": ignored,
                "fresh": {"G0": baseline, "scope": "diagnostic_only_no_promotion"}}
    base, current = _report(rows, maps["G0"]), _report(rows, maps[selected])
    gates = _relative_gates(base, current, True)
    bootstrap = _bootstrap(rows, maps["G0"], maps[selected])
    gates.update(bootstrap_complete=bootstrap["valid_resamples"] == 2000,
        bootstrap_win_rate=bootstrap["win_rate"] is not None and bootstrap["win_rate"] >= .70,
        bootstrap_p10=bootstrap["p10_delta"] is not None and bootstrap["p10_delta"] >= -.005)
    provider_slices: dict[str, dict[str, Scalar]] = {}
    generated_sources = sorted({str(row["source"]) for row in rows
        if row.get("MUSIC_FAKE") in (1, "1") and row.get("MUSIC_PRESENT") in (1, "1")})
    for source in generated_sources:
        generated = [row for row in rows if row.get("source") == source
                     and row.get("MUSIC_FAKE") in (1, "1")]
        groups = {part for row in generated for part in str(row["group_id"]).split("|")}
        matching_real = [row for row in rows if row.get("MUSIC_FAKE") in (0, "0")
                         and groups.intersection(str(row["group_id"]).split("|"))]
        family = [*matching_real, *generated]
        known = {int(row["MUSIC_FAKE"]) for row in family if row.get("MUSIC_FAKE") in (0, 1, "0", "1")}
        if known == {0, 1}:
            provider_slices[source] = _relative_gates(_report(family, maps["G0"]), _report(family, maps[selected]), False)
    provider_passed = all(value["passed"] for value in provider_slices.values())
    gates["provider_families"] = provider_passed
    gates["passed"] = all(value is True for key, value in gates.items()
                               if key not in ("diagnostic_gain_delta", "passed"))
    accepted = bool(gates["passed"])
    return {"final": selected if accepted else "G0", "accepted": accepted,
            "reason": "frozen_candidate_accepted" if accepted else "frozen_candidate_failed_fresh_gate",
            "evaluated": ["G0", selected], "ignored_candidates": ignored,
            "fresh": {"G0": base, selected: current, "gates": gates, "bootstrap": bootstrap,
                      "provider_slices": provider_slices}}
