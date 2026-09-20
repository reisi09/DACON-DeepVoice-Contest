"""V4 development selection and one-way sealed acceptance."""

from __future__ import annotations

import csv
import hashlib
import json
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Final

from v4_evaluation import Report, Row, evaluate_development, evaluate_sealed

PIPELINES: Final = ("P1", "P2", "P3", "P4")
RECIPE_FIELDS: Final = {
    "selected", "parent", "treatment", "learning_rate", "scheduler", "epoch_count",
    "stage_epochs", "data_hashes", "source_hashes", "pooling", "seeds", "gate_thresholds",
}


@dataclass(frozen=True, slots=True)
class SelectionError(ValueError):
    detail: str

    def __str__(self) -> str:
        return self.detail


def _digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _write_immutable(path: Path, payload: Mapping[str, str | int | float | bool | dict | list]) -> str:
    serialized = json.dumps(payload, sort_keys=True, indent=2, allow_nan=False) + "\n"
    if path.is_file():
        if path.read_text(encoding="utf-8") != serialized:
            raise SelectionError(f"immutable lock changed: {path.name}")
        return _digest(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(serialized, encoding="utf-8")
    temporary.replace(path)
    return _digest(path)


def _write_timestamped(path: Path, payload: Mapping[str, str | int | float | bool | dict | list],
                       timestamp_key: str) -> str:
    if path.is_file():
        existing = json.loads(path.read_text(encoding="utf-8"))
        existing.pop(timestamp_key, None)
        if existing != dict(payload):
            raise SelectionError(f"immutable lock changed: {path.name}")
        return _digest(path)
    return _write_immutable(path, {**payload, timestamp_key: time.time()})


def _quality_gates(report: Report) -> dict[str, bool]:
    music = report["music"]
    controlled = report["controlled"]
    gates: dict[str, bool] = {}
    for panel, label in (("music_seen", "seen"), ("music_ood", "ood")):
        result = music[panel]
        base, candidate = result["G0"], result["candidate"]
        bootstrap = result["bootstrap"]
        gates[f"{label}_gain"] = result["D"] is not None and result["D"] >= .005
        for head in ("file", "music"):
            gates[f"{label}_{head}_eer"] = (
                candidate[head]["eer"] is not None and base[head]["eer"] is not None
                and candidate[head]["eer"] <= base[head]["eer"] + .01)
        gates[f"{label}_music_auc"] = (
            candidate["music"]["auc"] is not None and base["music"]["auc"] is not None
            and candidate["music"]["auc"] >= base["music"]["auc"] - .01)
        support = bootstrap["class_group_support"]
        gates[f"{label}_class_support"] = support["0"] >= 10 and support["1"] >= 10
        gates[f"{label}_bootstrap_valid"] = bootstrap["valid_rate"] >= .90
        gates[f"{label}_bootstrap_win"] = (
            bootstrap["win_rate"] is not None and bootstrap["win_rate"] >= .80)
        gates[f"{label}_bootstrap_p10"] = (
            bootstrap["p10_delta"] is not None and bootstrap["p10_delta"] >= 0)
    for panel in ("speech", "mixture"):
        base = controlled[panel]["G0"]["file"]["eer"]
        candidate = controlled[panel]["candidate"]["file"]["eer"]
        gates[f"{panel}_file_eer"] = (
            base is not None and candidate is not None and candidate <= base + .02)
    return gates


def select_development(rows: Sequence[Row], baseline: Sequence[Row],
                       candidates: Mapping[str, Sequence[Row]],
                       runtime_seconds: Mapping[str, float],
                       projected_refit_seconds: Mapping[str, float],
                       reserved_refit_seconds: float) -> Report:
    unknown = set(candidates) - set(PIPELINES)
    if unknown:
        raise SelectionError(f"unknown V4 pipelines: {sorted(unknown)}")
    reports: dict[str, dict] = {}
    eligible: list[str] = []
    for name in PIPELINES:
        if name not in candidates:
            reports[name] = {"status": "not_completed", "gates": {"passed": False}}
            continue
        if name not in runtime_seconds or name not in projected_refit_seconds:
            raise SelectionError(f"missing runtime forecast: {name}")
        evaluation = evaluate_development(rows, baseline, candidates[name])
        gates = _quality_gates(evaluation)
        gates["resource"] = 1.25 * projected_refit_seconds[name] <= reserved_refit_seconds
        gates["passed"] = all(gates.values())
        reports[name] = {"status": "completed", "evaluation": evaluation, "gates": gates,
                         "runtime_seconds": runtime_seconds[name],
                         "projected_refit_seconds": projected_refit_seconds[name]}
        if gates["passed"]:
            eligible.append(name)
    selected = max(eligible, key=lambda name: (
        reports[name]["evaluation"]["rank_score"],
        reports[name]["evaluation"]["music_auc_tiebreak"],
        -runtime_seconds[name], -PIPELINES.index(name)), default="G0")
    return {"selected": selected,
            "reason": "eligible_development_winner" if selected != "G0" else "no_candidate_passed",
            "candidate_order": list(PIPELINES), "candidates": reports}


def _read_ids(path: Path) -> list[str]:
    with path.open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    identifiers = [str(row.get("ID", "")) for row in rows]
    if not identifiers or any(not identifier for identifier in identifiers) or len(set(identifiers)) != len(identifiers):
        raise SelectionError(f"manifest IDs are incomplete: {path.name}")
    return identifiers


def _verify_recipe_inputs(recipe: Mapping[str, str | int | float | bool | dict | list]) -> list[str]:
    for name in ("selection", "eligible_manifest"):
        target = Path(str(recipe[name]))
        if not target.is_file() or _digest(target) != recipe[f"{name}_sha256"]:
            raise SelectionError(f"{name} hash changed after recipe lock")
    return _read_ids(Path(str(recipe["eligible_manifest"])))


def write_recipe_lock(path: Path, selection_path: Path, eligible_manifest: Path,
                      payload: Mapping[str, str | int | float | bool | dict | list]) -> str:
    selected = payload.get("selected")
    if selected not in PIPELINES:
        raise SelectionError("recipe lock requires one selected V4 pipeline")
    missing = RECIPE_FIELDS - set(payload)
    if missing:
        raise SelectionError(f"recipe lock fields are missing: {sorted(missing)}")
    if not selection_path.is_file() or not eligible_manifest.is_file():
        raise SelectionError("recipe lock input is missing")
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    selected_report = selection.get("candidates", {}).get(selected, {})
    gates = selected_report.get("gates", {})
    if selection.get("selected") != selected or selected_report.get("status") != "completed":
        raise SelectionError("recipe does not match the development winner")
    if gates.get("passed") is not True or gates.get("resource") is not True:
        raise SelectionError("development winner is not eligible for refit")
    stage_epochs = payload["stage_epochs"]
    if not isinstance(stage_epochs, dict) or not stage_epochs or any(
            type(value) is not int or value <= 0 for value in stage_epochs.values()):
        raise SelectionError("recipe stage epochs are invalid")
    if sum(stage_epochs.values()) != payload["epoch_count"]:
        raise SelectionError("recipe epoch count does not match its stages")
    _read_ids(eligible_manifest)
    locked = {**payload, "selection": str(selection_path),
              "selection_sha256": _digest(selection_path),
              "eligible_manifest": str(eligible_manifest),
              "eligible_manifest_sha256": _digest(eligible_manifest)}
    return _write_timestamped(path, locked, "locked_unix")


def _verify_refit_manifest(path: Path, expected_ids: Sequence[str], expected_exposures: int) -> None:
    with path.open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    required = {"ID", "expected_exposures", "actual_exposures"}
    if not rows or set(rows[0]) < required or [row["ID"] for row in rows] != list(expected_ids):
        raise SelectionError("refit manifest is incomplete")
    try:
        complete = all(int(row["expected_exposures"]) == expected_exposures
                       and int(row["actual_exposures"]) == expected_exposures
                       for row in rows)
    except ValueError as error:
        raise SelectionError("refit manifest has invalid exposure counts") from error
    if not complete:
        raise SelectionError("refit exposures are incomplete")


def write_weight_lock(path: Path, recipe_lock: Path, weights: Path,
                      inference_code: Sequence[Path], refit_manifest: Path,
                      refit_report: Mapping[str, bool | int | float | str]) -> str:
    if not refit_report.get("completed") or not refit_report.get("exposures_complete"):
        raise SelectionError("refit is incomplete")
    inputs = (recipe_lock, weights, refit_manifest, *inference_code)
    if not all(item.is_file() for item in inputs):
        raise SelectionError("weight lock input is missing")
    recipe = json.loads(recipe_lock.read_text(encoding="utf-8"))
    eligible_ids = _verify_recipe_inputs(recipe)
    started = refit_report.get("started_unix")
    completed = refit_report.get("completed_unix")
    prescribed_epochs = refit_report.get("prescribed_epochs")
    completed_epochs = refit_report.get("completed_epochs")
    if type(started) not in (int, float) or type(completed) not in (int, float):
        raise SelectionError("refit chronology is incomplete")
    if not recipe["locked_unix"] <= started <= completed <= time.time():
        raise SelectionError("refit chronology is invalid")
    if type(prescribed_epochs) is not int or prescribed_epochs <= 0 or completed_epochs != prescribed_epochs:
        raise SelectionError("refit epoch count is incomplete")
    _verify_refit_manifest(refit_manifest, eligible_ids, int(recipe["epoch_count"]))
    payload = {"recipe_lock": str(recipe_lock),
               "recipe_lock_sha256": _digest(recipe_lock), "weights": str(weights),
               "weights_sha256": _digest(weights), "refit_manifest": str(refit_manifest),
               "refit_manifest_sha256": _digest(refit_manifest),
               "inference_code": [{"path": str(item), "sha256": _digest(item)} for item in inference_code],
               "refit_report": dict(refit_report)}
    return _write_timestamped(path, payload, "created_unix")


def verify_weight_lock(path: Path, weights: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if str(weights) != payload["weights"] or _digest(weights) != payload["weights_sha256"]:
        raise SelectionError("weight hash changed after lock")
    for name, hash_name in (("recipe_lock", "recipe_lock_sha256"),
                            ("refit_manifest", "refit_manifest_sha256")):
        target = Path(payload[name])
        if not target.is_file() or _digest(target) != payload[hash_name]:
            raise SelectionError(f"{name} hash changed after lock")
    for item in payload["inference_code"]:
        target = Path(item["path"])
        if not target.is_file() or _digest(target) != item["sha256"]:
            raise SelectionError("inference code hash changed after lock")
    recipe = json.loads(Path(payload["recipe_lock"]).read_text(encoding="utf-8"))
    _verify_recipe_inputs(recipe)
    return payload


def mark_sealed_opened(path: Path, weight_lock: Path, weights: Path) -> str:
    verify_weight_lock(weight_lock, weights)
    return _write_timestamped(path, {"weight_lock_sha256": _digest(weight_lock)},
                              "opened_unix")


def write_sealed_inference_receipt(path: Path, weight_lock: Path, weights: Path,
                                   sealed_opened: Path, baseline_predictions: Path,
                                   candidate_predictions: Path, started_unix: float,
                                   completed_unix: float) -> str:
    locked = verify_weight_lock(weight_lock, weights)
    opened = json.loads(sealed_opened.read_text(encoding="utf-8"))
    if opened.get("weight_lock_sha256") != _digest(weight_lock):
        raise SelectionError("sealed-open receipt does not match weight lock")
    if not locked["created_unix"] <= opened["opened_unix"] <= started_unix <= completed_unix <= time.time():
        raise SelectionError("sealed inference chronology is invalid")
    if not baseline_predictions.is_file() or not candidate_predictions.is_file():
        raise SelectionError("sealed prediction output is missing")
    recipe = json.loads(Path(locked["recipe_lock"]).read_text(encoding="utf-8"))
    payload = {"weight_lock_sha256": _digest(weight_lock), "selected": recipe["selected"],
               "started_unix": started_unix, "completed_unix": completed_unix,
               "baseline_predictions": str(baseline_predictions),
               "baseline_predictions_sha256": _digest(baseline_predictions),
               "candidate_predictions": str(candidate_predictions),
               "candidate_predictions_sha256": _digest(candidate_predictions)}
    return _write_immutable(path, payload)


def _read_predictions(path: Path) -> list[Row]:
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return [dict(row) for row in csv.DictReader(stream)]


def accept_sealed(rows: Sequence[Row], selected: str, weight_lock: Path, weights: Path,
                  sealed_opened: Path, inference_receipt: Path) -> Report:
    if selected not in PIPELINES:
        raise SelectionError("sealed acceptance requires the frozen candidate")
    locked = verify_weight_lock(weight_lock, weights)
    recipe = json.loads(Path(locked["recipe_lock"]).read_text(encoding="utf-8"))
    if selected != recipe["selected"]:
        raise SelectionError("sealed candidate differs from the recipe lock")
    opened = json.loads(sealed_opened.read_text(encoding="utf-8"))
    if opened.get("weight_lock_sha256") != _digest(weight_lock):
        raise SelectionError("sealed-open receipt does not match weight lock")
    receipt = json.loads(inference_receipt.read_text(encoding="utf-8"))
    if receipt.get("weight_lock_sha256") != _digest(weight_lock) or receipt.get("selected") != selected:
        raise SelectionError("sealed inference receipt does not match locked candidate")
    if not locked["created_unix"] <= opened["opened_unix"] <= receipt["started_unix"] <= receipt["completed_unix"] <= time.time():
        raise SelectionError("sealed inference chronology is invalid")
    for name in ("baseline_predictions", "candidate_predictions"):
        target = Path(receipt[name])
        if not target.is_file() or _digest(target) != receipt[f"{name}_sha256"]:
            raise SelectionError("sealed prediction hash changed after inference")
    evaluation = evaluate_sealed(
        rows, _read_predictions(Path(receipt["baseline_predictions"])),
        _read_predictions(Path(receipt["candidate_predictions"])))
    gates = _quality_gates(evaluation)
    accepted = all(gates.values())
    return {"final": selected if accepted else "G0", "candidate": selected,
            "accepted": accepted, "reason": "sealed_passed" if accepted else "sealed_failed",
            "gates": {**gates, "passed": accepted}, "evaluation": evaluation}
