"""Metrics for the eight controlled experiments."""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Final

import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve

DEFAULT_BOOTSTRAPS: Final = 1000


class MetricContractError(ValueError):
    """Metric inputs cannot support the requested estimate."""


def _arrays(labels: Sequence[int], scores: Sequence[float]) -> tuple[np.ndarray, np.ndarray]:
    y = np.asarray(labels, dtype=np.int64)
    s = np.asarray(scores, dtype=np.float64)
    if y.ndim != 1 or s.shape != y.shape or not len(y) or not np.isfinite(s).all():
        raise MetricContractError("labels and finite scores must be non-empty aligned vectors")
    if not set(y.tolist()) <= {0, 1}:
        raise MetricContractError("binary labels must be 0 or 1")
    return y, s


def eer(labels: Sequence[int], scores: Sequence[float]) -> dict[str, float | int | None]:
    """Published full-ROC nearest-point EER; high score means fake."""
    y, s = _arrays(labels, scores)
    if len(np.unique(y)) != 2:
        return {"eer": None, "threshold": None, "n": len(y)}
    fpr, tpr, thresholds = roc_curve(y, s, drop_intermediate=False)
    index = int(np.argmin(np.abs(fpr - (1.0 - tpr))))
    threshold = float(thresholds[index])
    if not np.isfinite(threshold):
        threshold = float(np.nextafter(np.max(s), np.inf))
    return {
        "eer": float((fpr[index] + 1.0 - tpr[index]) / 2.0),
        "threshold": threshold,
        "n": len(y),
    }


def threshold_at_fpr(
    labels: Sequence[int], scores: Sequence[float], target_fpr: float = 0.05,
) -> float:
    """Largest accepted fake threshold meeting a REAL-only FPR target."""
    y, s = _arrays(labels, scores)
    if not 0 <= target_fpr <= 1 or not np.any(y == 0):
        raise MetricContractError("target_fpr must be in [0,1] and REAL calibration rows are required")
    real = np.sort(s[y == 0])[::-1]
    allowed = int(np.floor(target_fpr * len(real)))
    return float(np.nextafter(real[allowed], np.inf)) if allowed < len(real) else float("-inf")


def binary_summary(
    labels: Sequence[int], scores: Sequence[float], threshold: float | None = None,
) -> dict[str, float | int | None]:
    """Return EER/AUC plus thresholded FPR and recall with exact denominators."""
    y, s = _arrays(labels, scores)
    base = eer(y, s)
    chosen = float(base["threshold"]) if threshold is None and base["threshold"] is not None else threshold
    negatives, positives = y == 0, y == 1
    return {
        **base,
        "decision_threshold": chosen,
        "auc": float(roc_auc_score(y, s)) if negatives.any() and positives.any() else None,
        "fpr": float(np.mean(s[negatives] >= chosen)) if negatives.any() and chosen is not None else None,
        "recall": float(np.mean(s[positives] >= chosen)) if positives.any() and chosen is not None else None,
        "negative_n": int(negatives.sum()),
        "positive_n": int(positives.sum()),
    }


def grouped_bootstrap_delta(
    rows: Sequence[Mapping[str, str | int | float]],
    control_key: str,
    candidate_key: str,
    *,
    group_key: str = "component_id",
    n_boot: int = DEFAULT_BOOTSTRAPS,
    seed: int = 41,
) -> dict[str, float | int | bool | list[float] | None]:
    """Paired EER delta CI by provenance-connected component."""
    if n_boot < 1 or not rows:
        raise MetricContractError("positive bootstrap count and rows are required")
    groups = sorted({str(row[group_key]) for row in rows})
    observed = _eer_delta(rows, control_key, candidate_key)
    if len(groups) < 5:
        return {"delta_eer": observed, "ci95": None, "components": len(groups),
                "resamples": 0, "valid": 0, "uncertain": True}
    by_group = {group: [row for row in rows if str(row[group_key]) == group] for group in groups}
    rng = np.random.default_rng(seed)
    deltas: list[float] = []
    for _ in range(n_boot):
        sampled = [row for group in rng.choice(groups, len(groups), replace=True) for row in by_group[str(group)]]
        delta = _eer_delta(sampled, control_key, candidate_key)
        if delta is not None:
            deltas.append(delta)
    return {
        "delta_eer": observed,
        "ci95": [float(np.quantile(deltas, 0.025)), float(np.quantile(deltas, 0.975))] if deltas else None,
        "components": len(groups),
        "resamples": n_boot,
        "valid": len(deltas),
        "uncertain": not deltas,
    }


def _eer_delta(
    rows: Sequence[Mapping[str, str | int | float]], control_key: str, candidate_key: str,
) -> float | None:
    labels = [int(row["label"]) for row in rows]
    control = eer(labels, [float(row[control_key]) for row in rows])["eer"]
    candidate = eer(labels, [float(row[candidate_key]) for row in rows])["eer"]
    return None if control is None or candidate is None else float(candidate) - float(control)


def stratum_summaries(
    rows: Sequence[Mapping[str, str | int | float]], score_key: str,
    dimensions: Sequence[str] = ("source", "codec", "generator", "parent_id"),
) -> dict[str, dict[str, dict[str, float | int | None]]]:
    """Exact per-source/codec/generator/parent denominators and worst valid EER."""
    result: dict[str, dict[str, dict[str, float | int | None]]] = {}
    for dimension in dimensions:
        values: dict[str, dict[str, float | int | None]] = {}
        for value in sorted({str(row.get(dimension, "")) for row in rows}):
            selected = [row for row in rows if str(row.get(dimension, "")) == value]
            values[value] = binary_summary(
                [int(row["label"]) for row in selected], [float(row[score_key]) for row in selected],
            )
        result[dimension] = values
    return result


def comparison_status(
    delta: Mapping[str, float | int | bool | list[float] | None],
    control_strata: Mapping[str, Mapping[str, Mapping[str, float | int | None]]],
    candidate_strata: Mapping[str, Mapping[str, Mapping[str, float | int | None]]],
) -> str:
    """Apply the predeclared CI/component/5-point stratum adoption rule."""
    ci = delta.get("ci95")
    if not isinstance(ci, list) or len(ci) != 2 or int(delta["components"]) < 5:
        return "inconclusive"
    regressions = []
    for dimension, values in control_strata.items():
        for value, control in values.items():
            candidate = candidate_strata.get(dimension, {}).get(value, {})
            if control.get("eer") is not None and candidate.get("eer") is not None:
                regressions.append(float(candidate["eer"]) - float(control["eer"]))
    if float(ci[0]) > 0:
        return "regression"
    if float(ci[1]) < 0 and not any(value > 0.05 for value in regressions):
        return "measured_improvement"
    return "inconclusive"
