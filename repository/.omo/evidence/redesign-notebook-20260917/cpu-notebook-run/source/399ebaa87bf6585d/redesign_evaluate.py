"""Decision-quality comparison, candidate locking, and one-time sealed opening."""

from __future__ import annotations

import json
import os
from pathlib import Path

from graph_scoring import official_metrics
from redesign_evaluation_common import (
    METRIC_FOR_OUTPUT,
    artifact_rows,
    canonical_sha256,
    changed_outputs,
    class_group_counts,
    file_sha256,
    metric_delta,
    paired_bootstrap,
    read_json,
    subgroup_report,
    write_json,
)


def compare(
    control_path: Path, candidate_path: Path, output: Path, *, bootstrap: int = 2000,
    seed: int = 41,
) -> dict:
    """Compare paired artifacts and refuse unidentifiable or under-evidenced promotion."""
    control, candidate = read_json(control_path), read_json(candidate_path)
    control_labels, control_predictions, identifiers = artifact_rows(control)
    candidate_labels, candidate_predictions, candidate_ids = artifact_rows(candidate)
    if identifiers != candidate_ids or control["labels"] != candidate["labels"]:
        raise ValueError("paired comparison requires identical ordered IDs and labels")
    if control["groups"] != candidate["groups"]:
        raise ValueError("paired comparison requires identical groups")
    changed = changed_outputs(control, candidate, identifiers)
    control_metrics = official_metrics(control_labels, control_predictions)
    candidate_metrics = official_metrics(candidate_labels, candidate_predictions)
    delta = metric_delta(control_metrics, candidate_metrics, changed)
    report = {
        "schema_version": 1, "candidate": candidate.get("candidate"),
        "control_sha256": file_sha256(control_path),
        "candidate_sha256": file_sha256(candidate_path),
        "changed_outputs": changed, "control": control_metrics,
        "candidate_metrics": candidate_metrics, "comparison": None,
        "status": "rejected", "reason": None,
    }
    if delta is None:
        unknown = next(
            output_name for output_name in changed
            if METRIC_FOR_OUTPUT[output_name][0] in control_metrics["partial"]
            or METRIC_FOR_OUTPUT[output_name][0] in control_metrics["invalid"]
        )
        report["reason"] = f"changed output has incomplete or invalid labels: {unknown}"
        write_json(output, report)
        return report

    bootstrap_report = paired_bootstrap(control, candidate, changed, bootstrap, seed)
    counts = class_group_counts(candidate)
    subgroup_ids = {
        "natural": [key for key, value in candidate["strata"].items()
                    if value.get("domain") == "natural"],
        "singing": [key for key, value in candidate["strata"].items()
                    if value.get("kind") == "singing"],
        "telephone": [key for key, value in candidate["strata"].items()
                      if value.get("domain") == "telephone"],
    }
    subgroups = {
        name: subgroup_report(control, candidate, selected)
        for name, selected in subgroup_ids.items()
    }
    actual_l4 = candidate.get("runtime_evidence", {}).get("actual_l4_verified") is True
    complete_counts = all(min(classes.values()) >= 100 for classes in counts.values())
    core_pass = (
        delta >= 0.005 and bootstrap_report["ci95"] is not None
        and bootstrap_report["ci95"][0] > 0
        and bootstrap_report["invalid_fraction"] < 0.05
    )
    regression_failures = []
    for name, subgroup in subgroups.items():
        if subgroup is None:
            regression_failures.append(f"missing {name} subgroup")
            continue
        threshold = 0.01 if name == "natural" else 0.02
        for metric in ("file_eer", "voice_eer", "music_eer"):
            before = subgroup["control"]["metrics"].get(metric)
            after = subgroup["candidate"]["metrics"].get(metric)
            if before is not None and after is not None and after - before > threshold:
                regression_failures.append(f"{name} {metric} regression")
    for metric in ("voice_presence_auc", "music_presence_auc"):
        before = control_metrics["metrics"].get(metric)
        after = candidate_metrics["metrics"].get(metric)
        if before is not None and after is not None and before - after > 0.005:
            regression_failures.append(f"{metric} regression")
    changed_voice = "VOICE_FAKE_PROB" in changed
    stratum_counts_complete = not changed_voice or all(
        subgroups[name] is not None
        and min(subgroups[name]["class_group_counts"]["voice_eer"].values()) >= 30
        for name in ("singing", "telephone")
    )
    statistical_eligible = core_pass and not regression_failures
    validation_eligible = complete_counts and stratum_counts_complete
    report["comparison"] = {
        "mode": "full_score" if control_metrics["score"] is not None else "identifiable_delta",
        "delta": delta, "bootstrap": bootstrap_report, "class_group_counts": counts,
        "subgroups": subgroups, "actual_l4_verified": actual_l4,
        "regression_failures": regression_failures,
        "statistical_eligible": statistical_eligible,
        "validation_eligible": validation_eligible,
    }
    if not core_pass or regression_failures:
        report["reason"] = "statistical or subgroup regression gate failed"
    elif not complete_counts or not stratum_counts_complete or not actual_l4:
        report["status"] = "provisional"
        report["reason"] = "local result lacks final label-count and/or actual L4 evidence"
    else:
        report["status"] = "accepted"
        report["reason"] = "all promotion gates passed"
    write_json(output, report)
    return report


def lock_candidate(prediction_path: Path, report_path: Path, output: Path) -> dict:
    """Bind an evaluated candidate, report, and model identity before sealed access."""
    prediction, report = read_json(prediction_path), read_json(report_path)
    if report.get("candidate") != prediction.get("candidate"):
        raise ValueError("report candidate does not match prediction candidate")
    comparison = report.get("comparison")
    if (report.get("status") not in ("accepted", "provisional")
            or not isinstance(comparison, dict)
            or comparison.get("statistical_eligible") is not True
            or comparison.get("validation_eligible") is not True):
        raise ValueError("candidate report is not eligible for locking")
    if report.get("candidate_sha256") != file_sha256(prediction_path):
        raise ValueError("candidate report does not bind the prediction artifact")
    identity = prediction.get("upstream_identity")
    if not isinstance(identity, dict) or not identity:
        raise ValueError("candidate is missing upstream model identity")
    lock = {
        "schema_version": 1, "candidate": prediction.get("candidate"),
        "prediction_sha256": file_sha256(prediction_path),
        "report_sha256": file_sha256(report_path),
        "model_identity_sha256": canonical_sha256(identity),
        "report_status": report.get("status"),
    }
    lock["lock_sha256"] = canonical_sha256(lock)
    write_json(output, lock)
    return lock


def open_sealed(lock_path: Path, prediction_path: Path, output: Path) -> dict:
    """Consume a lock exactly once, including failed sealed-open attempts."""
    lock = read_json(lock_path)
    payload = {key: value for key, value in lock.items() if key != "lock_sha256"}
    if lock.get("lock_sha256") != canonical_sha256(payload):
        raise ValueError("candidate lock hash is invalid")
    sealed = read_json(prediction_path)
    required = {"labels", "predictions", "groups", "strata", "window_scores", "upstream_identity"}
    if not required <= sealed.keys():
        raise ValueError(f"sealed prediction artifact is incomplete: {sorted(required - sealed.keys())}")
    if sealed.get("candidate") != lock["candidate"]:
        raise ValueError("sealed prediction candidate does not match candidate lock")
    if canonical_sha256(sealed["upstream_identity"]) != lock["model_identity_sha256"]:
        raise ValueError("sealed prediction model identity does not match candidate lock")
    marker_path = Path(f"{lock_path}.sealed-opened.json")
    try:
        descriptor = os.open(marker_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    except FileExistsError as error:
        raise FileExistsError("sealed predictions were already opened for this lock") from error
    marker = {"lock_sha256": lock["lock_sha256"],
              "sealed_sha256": file_sha256(prediction_path), "status": "opening"}
    with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
        json.dump(marker, stream, indent=2, sort_keys=True)
        stream.write("\n")
    try:
        artifact_rows(sealed)
        marker.update({"status": "opened", "candidate": lock["candidate"]})
        write_json(marker_path, marker)
        write_json(output, marker)
        return marker
    except (KeyError, TypeError, ValueError):
        marker["status"] = "rejected"
        write_json(marker_path, marker)
        raise
