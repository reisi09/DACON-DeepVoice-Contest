"""OOF-only file relation fitting and file-independent application."""

from __future__ import annotations

import hashlib
import json
from math import exp, log
from pathlib import Path
from typing import Final

import numpy as np
from scipy.optimize import minimize

from graph_scoring import PREDICTION_COLUMNS
from redesign_evaluation_common import canonical_sha256, components, upstream_family_identity
from redesign_oof_validation import validate_oof

FEATURE_NAMES: Final = (
    "logit_voice_evidence", "logit_music_evidence", "logit_voice_fake",
    "logit_music_fake", "logit_voice_present", "logit_music_present",
    "evidence_gap", "max_window_voice_evidence", "max_window_music_evidence",
    "max_window_file_evidence", "top2mean_window_file_evidence",
    "mean_joint_presence",
)
EPSILON: Final = 1e-5
L2: Final = 1e-2


def _group_class_weights(targets: np.ndarray, group_names: list[str]) -> np.ndarray:
    """Give each FILE class equal mass and each connected group equal mass within class."""
    weights = np.zeros(len(targets), dtype=np.float64)
    for target in (0.0, 1.0):
        class_indices = np.flatnonzero(targets == target)
        class_groups = sorted({group_names[index] for index in class_indices})
        for group in class_groups:
            indices = [index for index in class_indices if group_names[index] == group]
            weights[indices] = 0.5 / len(class_groups) / len(indices)
    return weights


def _read(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return value


def _write(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _logit(probability: float) -> float:
    clipped = min(max(float(probability), EPSILON), 1 - EPSILON)
    return log(clipped / (1 - clipped))


def _sigmoid(value: float) -> float:
    if value >= 0:
        return 1 / (1 + exp(-value))
    exponential = exp(value)
    return exponential / (1 + exponential)


def _features(prediction: dict, windows: dict) -> tuple[float, np.ndarray]:
    for name in PREDICTION_COLUMNS:
        value = float(prediction[name])
        if not np.isfinite(value) or not 0 <= value <= 1:
            raise ValueError(f"bag probability outside finite [0, 1]: {name}")
    vf = float(prediction["VOICE_FAKE_PROB"])
    mf = float(prediction["MUSIC_FAKE_PROB"])
    vp = float(prediction["VOICE_PRESENT_PROB"])
    mp = float(prediction["MUSIC_PRESENT_PROB"])
    a, b = vp * vf, mp * mf
    window_values = [
        np.asarray(windows[name], dtype=np.float64)
        for name in ("voice_fake", "music_fake", "voice_present", "music_present")
    ]
    if any(array.ndim != 1 or array.size == 0 for array in window_values):
        raise ValueError("window_scores must contain four non-empty aligned 1d arrays")
    if len({array.size for array in window_values}) != 1:
        raise ValueError("window_scores arrays must be aligned")
    if any(not np.isfinite(array).all() or (array < 0).any() or (array > 1).any()
           for array in window_values):
        raise ValueError("window probability outside finite [0, 1]")
    window_vf, window_mf, window_vp, window_mp = window_values
    voice_evidence = window_vp * window_vf
    music_evidence = window_mp * window_mf
    file_evidence = np.maximum(voice_evidence, music_evidence)
    top = np.sort(file_evidence)[-2:]
    values = np.asarray((
        _logit(a), _logit(b), _logit(vf), _logit(mf), _logit(vp), _logit(mp),
        abs(a - b), float(voice_evidence.max()), float(music_evidence.max()),
        float(file_evidence.max()), float(top.mean()),
        float(np.mean((window_vp > 0.5) & (window_mp > 0.5))),
    ), dtype=np.float64)
    return max(a, b), values


def fit_relation(oof_path: Path, output: Path) -> dict:
    """Fit the fixed offset logistic relation using genuine train-only OOF rows."""
    artifact = _read(oof_path)
    identifiers = validate_oof(artifact)
    family_identity = upstream_family_identity(artifact["upstream_identity"])
    features, offsets, targets, baselines = [], [], [], []
    for identifier in identifiers:
        target = artifact["labels"][identifier].get("FILE_FAKE")
        if target not in (0, 1):
            raise ValueError(f"FILE_FAKE must be known for relation fitting: {identifier}")
        f0, row_features = _features(
            artifact["predictions"][identifier], artifact["window_scores"][identifier]
        )
        features.append(row_features)
        baselines.append(f0)
        offsets.append(_logit(f0))
        targets.append(float(target))
    matrix = np.stack(features)
    eligible = np.asarray([0 < value < 1 for value in baselines])
    target_array_all = np.asarray(targets)
    if set(target_array_all[eligible]) != {0.0, 1.0}:
        raise ValueError("eligible relation rows require both FILE_FAKE classes")
    mean = matrix[eligible].mean(axis=0)
    scale = matrix[eligible].std(axis=0)
    scale[scale == 0] = 1.0
    standardized = ((matrix - mean) / scale)[eligible]
    offset_array = np.asarray(offsets)[eligible]
    target_array = target_array_all[eligible]
    component_by_id = components(artifact)
    eligible_groups = [component_by_id[identifier] for identifier, keep
                       in zip(identifiers, eligible, strict=True) if keep]
    sample_weights = _group_class_weights(target_array, eligible_groups)

    def objective(parameters: np.ndarray) -> tuple[float, np.ndarray]:
        logits = offset_array + standardized @ parameters[:-1] + parameters[-1]
        losses = np.logaddexp(0.0, logits) - target_array * logits
        probabilities = 1 / (1 + np.exp(-np.clip(logits, -700, 700)))
        weighted_error = sample_weights * (probabilities - target_array)
        gradient = standardized.T @ weighted_error
        gradient = np.append(gradient + L2 * parameters[:-1], weighted_error.sum())
        loss = float(np.dot(sample_weights, losses)
                     + 0.5 * L2 * np.dot(parameters[:-1], parameters[:-1]))
        return loss, gradient

    zeros = np.zeros(len(FEATURE_NAMES) + 1)
    result = minimize(objective, zeros, jac=True, method="L-BFGS-B")
    if not result.success or not np.isfinite(result.fun):
        raise RuntimeError(f"relation optimization failed: {result.message}")
    relation = {
        "schema_version": 1, "feature_names": list(FEATURE_NAMES), "epsilon": EPSILON,
        "l2": L2, "mean": mean.tolist(), "scale": scale.tolist(),
        "weights": result.x[:-1].tolist(), "intercept": float(result.x[-1]),
        "upstream_identity": artifact["upstream_identity"],
        "upstream_identity_sha256": canonical_sha256(artifact["upstream_identity"]),
        "upstream_family_identity": family_identity,
        "oof_sha256": _sha256(oof_path),
        "fit": {
            "rows": len(identifiers), "eligible_rows": int(eligible.sum()),
            "fixed_endpoint_rows": int((~eligible).sum()),
            "irreducible_endpoint_contradictions": int(sum(
                (f0 == 0 and target == 1) or (f0 == 1 and target == 0)
                for f0, target in zip(baselines, targets, strict=True)
            )),
            "initial_loss": objective(zeros)[0], "final_loss": float(result.fun),
            "converged": True,
            "weighting": {
                "scheme": "equal FILE class and connected-group mass",
                "class_mass": {str(target): float(sample_weights[target_array == target].sum())
                               for target in (0.0, 1.0)},
                "connected_groups": {str(target): len({group for group, value in
                                                       zip(eligible_groups, target_array, strict=True)
                                                       if value == target})
                                     for target in (0.0, 1.0)},
            },
        },
    }
    _write(output, relation)
    return relation


def apply_relation(prediction_path: Path, relation_path: Path, output: Path) -> dict:
    """Apply a fitted relation independently to every prediction row."""
    artifact, relation = _read(prediction_path), _read(relation_path)
    if relation.get("feature_names") != list(FEATURE_NAMES):
        raise ValueError("relation feature order does not match the fixed 12-feature contract")
    if upstream_family_identity(artifact.get("upstream_identity", {})) != relation["upstream_family_identity"]:
        raise ValueError("prediction upstream family does not match relation fitting family")
    if set(artifact["predictions"]) != set(artifact["window_scores"]):
        raise ValueError("predictions and window_scores IDs must match")
    mean, scale = np.asarray(relation["mean"]), np.asarray(relation["scale"])
    weights = np.asarray(relation["weights"])
    updated = json.loads(json.dumps(artifact))
    for identifier, prediction in updated["predictions"].items():
        if set(prediction) != set(PREDICTION_COLUMNS):
            raise ValueError(f"prediction columns are invalid: {identifier}")
        f0, features = _features(prediction, updated["window_scores"][identifier])
        if f0 in (0.0, 1.0):
            prediction["FILE_FAKE_PROB"] = f0
        else:
            correction = float(((features - mean) / scale) @ weights + relation["intercept"])
            prediction["FILE_FAKE_PROB"] = _sigmoid(_logit(f0) + correction)
    updated["file_relation"] = {
        "relation_sha256": _sha256(relation_path), "oof_sha256": relation["oof_sha256"],
        "upstream_family_identity": relation["upstream_family_identity"],
        "application_upstream_identity_sha256": canonical_sha256(artifact["upstream_identity"]),
    }
    _write(output, updated)
    return updated
