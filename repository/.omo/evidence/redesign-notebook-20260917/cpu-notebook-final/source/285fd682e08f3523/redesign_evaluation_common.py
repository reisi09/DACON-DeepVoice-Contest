"""Shared deterministic mechanics for redesign evaluation."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Final

import numpy as np

from graph_scoring import LABEL_COLUMNS, PREDICTION_COLUMNS, official_metrics

METRIC_FOR_OUTPUT: Final = {
    "FILE_FAKE_PROB": ("file_eer", 0.45, -1.0),
    "VOICE_FAKE_PROB": ("voice_eer", 0.18, -1.0),
    "MUSIC_FAKE_PROB": ("music_eer", 0.27, -1.0),
    "VOICE_PRESENT_PROB": ("voice_presence_auc", 0.05, 1.0),
    "MUSIC_PRESENT_PROB": ("music_presence_auc", 0.05, 1.0),
}


def read_json(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return value


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def canonical_sha256(value: dict | list[str]) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def upstream_family_identity(identity: dict) -> str:
    """Verify and return a checkpoint-independent upstream family digest."""
    family = identity.get("upstream_family_identity")
    if not isinstance(family, str) or len(family) != 64:
        raise ValueError("upstream family identity is missing or invalid")
    payload = identity.get("upstream_family_payload")
    if payload is None:
        seed_families = identity.get("seed_upstream_family_identity")
        if isinstance(seed_families, list) and seed_families:
            payload = seed_families
        else:
            names = ("training_manifest_sha256", "records_identity",
                     "model_recipe_identity", "source_sha256")
            if any(name not in identity for name in names):
                raise ValueError("upstream family payload is incomplete")
            payload = {name: identity[name] for name in names}
    if isinstance(payload, dict):
        for name, value in payload.items():
            if name in identity and identity[name] != value:
                raise ValueError(f"upstream family payload disagrees with {name}")
    if canonical_sha256(payload) != family:
        raise ValueError("upstream family identity hash does not match its payload")
    return family


def artifact_rows(artifact: dict) -> tuple[list[dict], list[dict], list[str]]:
    identifiers = list(artifact["predictions"])
    if set(artifact["labels"]) != set(identifiers):
        raise ValueError("labels and predictions IDs must match")
    labels, predictions = [], []
    for identifier in identifiers:
        label = artifact["labels"][identifier]
        prediction = artifact["predictions"][identifier]
        if set(label) != set(LABEL_COLUMNS) or set(prediction) != set(PREDICTION_COLUMNS):
            raise ValueError(f"invalid label or prediction columns: {identifier}")
        labels.append({"ID": identifier, **label})
        predictions.append({"ID": identifier, **prediction})
    return labels, predictions, identifiers


def changed_outputs(control: dict, candidate: dict, identifiers: list[str]) -> list[str]:
    return [
        column for column in PREDICTION_COLUMNS
        if any(float(control["predictions"][identifier][column])
               != float(candidate["predictions"][identifier][column])
               for identifier in identifiers)
    ]


def metric_delta(control_report: dict, candidate_report: dict, changed: list[str]) -> float | None:
    if control_report["score"] is not None and candidate_report["score"] is not None:
        return float(candidate_report["score"] - control_report["score"])
    delta = 0.0
    for output in changed:
        metric, weight, direction = METRIC_FOR_OUTPUT[output]
        if (metric in control_report["invalid"] or metric in candidate_report["invalid"]
                or metric in control_report["partial"] or metric in candidate_report["partial"]
                or control_report["metrics"][metric] is None
                or candidate_report["metrics"][metric] is None):
            return None
        delta += weight * direction * (
            float(candidate_report["metrics"][metric]) - float(control_report["metrics"][metric])
        )
    return delta


def components(artifact: dict) -> dict[str, str]:
    parent: dict[str, str] = {}

    def find(group: str) -> str:
        parent.setdefault(group, group)
        while parent[group] != group:
            parent[group] = parent[parent[group]]
            group = parent[group]
        return group

    for groups in artifact["groups"].values():
        if not groups:
            raise ValueError("paired bootstrap requires non-empty groups")
        first = find(groups[0])
        for group in groups[1:]:
            root = find(group)
            if root != first:
                parent[root] = first
    members: dict[str, list[str]] = {}
    for group in parent:
        members.setdefault(find(group), []).append(group)
    names = {root: "|".join(sorted(groups)) for root, groups in members.items()}
    return {identifier: names[find(groups[0])] for identifier, groups in artifact["groups"].items()}


def _resample(artifact: dict, selected_groups: list[str]) -> dict:
    rows_by_group: dict[str, list[str]] = {}
    for identifier, component in components(artifact).items():
        rows_by_group.setdefault(component, []).append(identifier)
    sampled = {"labels": {}, "predictions": {}}
    for occurrence, group in enumerate(selected_groups):
        for identifier in rows_by_group[group]:
            sampled_id = f"{occurrence}:{identifier}"
            sampled["labels"][sampled_id] = artifact["labels"][identifier]
            sampled["predictions"][sampled_id] = artifact["predictions"][identifier]
    return sampled


def paired_bootstrap(
    control: dict, candidate: dict, changed: list[str], count: int, seed: int,
) -> dict:
    group_keys = sorted(set(components(control).values()))
    if not group_keys:
        raise ValueError("paired bootstrap requires groups")
    rng = np.random.default_rng(seed)
    deltas: list[float] = []
    invalid = 0
    for _ in range(count):
        selected = [group_keys[index] for index in rng.integers(0, len(group_keys), len(group_keys))]
        control_sample, candidate_sample = _resample(control, selected), _resample(candidate, selected)
        control_labels, control_predictions, _ = artifact_rows(control_sample)
        candidate_labels, candidate_predictions, _ = artifact_rows(candidate_sample)
        delta = metric_delta(
            official_metrics(control_labels, control_predictions),
            official_metrics(candidate_labels, candidate_predictions), changed,
        )
        if delta is None:
            invalid += 1
        else:
            deltas.append(delta)
    return {
        "resamples": count, "valid": len(deltas), "invalid": invalid,
        "invalid_fraction": invalid / count,
        "ci95": None if not deltas else [float(np.quantile(deltas, 0.025)),
                                           float(np.quantile(deltas, 0.975))],
    }


def class_group_counts(artifact: dict, identifiers: list[str] | None = None) -> dict:
    counts, component_by_id = {}, components(artifact)
    selected = set(artifact["labels"] if identifiers is None else identifiers)
    specifications = (
        ("file_eer", "FILE_FAKE", None), ("voice_eer", "VOICE_FAKE", "VOICE_PRESENT"),
        ("music_eer", "MUSIC_FAKE", "MUSIC_PRESENT"),
        ("voice_presence_auc", "VOICE_PRESENT", None),
        ("music_presence_auc", "MUSIC_PRESENT", None),
    )
    for metric, label_name, presence_name in specifications:
        values = {"0": set(), "1": set()}
        for identifier, label in artifact["labels"].items():
            if identifier not in selected or (presence_name is not None and label[presence_name] != 1):
                continue
            value = label[label_name]
            if value in (0, 1):
                values[str(value)].add(component_by_id[identifier])
        counts[metric] = {key: len(groups) for key, groups in values.items()}
    return counts


def subgroup_report(control: dict, candidate: dict, identifiers: list[str]) -> dict | None:
    if not identifiers:
        return None
    control_subset = {"labels": {key: control["labels"][key] for key in identifiers},
                      "predictions": {key: control["predictions"][key] for key in identifiers}}
    candidate_subset = {"labels": {key: candidate["labels"][key] for key in identifiers},
                        "predictions": {key: candidate["predictions"][key] for key in identifiers}}
    labels, control_predictions, _ = artifact_rows(control_subset)
    _, candidate_predictions, _ = artifact_rows(candidate_subset)
    return {"rows": len(identifiers), "class_group_counts": class_group_counts(candidate, identifiers),
            "control": official_metrics(labels, control_predictions),
            "candidate": official_metrics(labels, candidate_predictions)}
