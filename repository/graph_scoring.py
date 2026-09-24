"""Submission validation and DACON 236749's discrete ROC scoring rule.

Labels are 0/1/None under the official prediction names without ``_PROB``.
Partial submetrics expose coverage; an official total requires complete labels.
"""

from collections.abc import Mapping, Sequence
from itertools import groupby
from math import isfinite
from typing import Final, TypedDict

PREDICTION_COLUMNS: Final = (
    "FILE_FAKE_PROB", "VOICE_FAKE_PROB", "MUSIC_FAKE_PROB",
    "VOICE_PRESENT_PROB", "MUSIC_PRESENT_PROB",
)
SUBMISSION_COLUMNS: Final = ("ID", *PREDICTION_COLUMNS)
LABEL_COLUMNS: Final = tuple(name.removesuffix("_PROB") for name in PREDICTION_COLUMNS)
Row = Mapping[str, str | float | int | None]


class PredictionRow(TypedDict):
    ID: str
    FILE_FAKE_PROB: float
    VOICE_FAKE_PROB: float
    MUSIC_FAKE_PROB: float
    VOICE_PRESENT_PROB: float
    MUSIC_PRESENT_PROB: float


class MetricsReport(TypedDict):
    score: float | None
    metrics: dict[str, float | None]
    invalid: dict[str, str]
    partial: dict[str, str]
    counts: dict[str, int]
    unknown_counts: dict[str, int]
    coverage: dict[str, float]


def _identifier(row: Row) -> str:
    identifier = row.get("ID")
    if not isinstance(identifier, str) or not identifier or identifier != identifier.strip():
        message = f"ID must be a non-empty, unpadded string: {identifier!r}"
        raise ValueError(message)
    return identifier


def validate_predictions(rows: Sequence[Row], expected_ids: Sequence[str]) -> list[PredictionRow]:
    """Parse probabilities and enforce exact columns, IDs, and submission order."""
    if not rows or len(set(expected_ids)) != len(expected_ids):
        message = "Predictions must be non-empty and expected IDs must be unique"
        raise ValueError(message)
    identifiers = [_identifier(row) for row in rows]
    if len(set(identifiers)) != len(identifiers) or identifiers != list(expected_ids):
        message = "Prediction IDs must match the expected unique IDs in order"
        raise ValueError(message)
    parsed = []
    for row, identifier in zip(rows, identifiers, strict=True):
        if set(row) != set(SUBMISSION_COLUMNS):
            message = f"Prediction {identifier} must have exactly the six submission columns"
            raise ValueError(message)
        probabilities = []
        for column in PREDICTION_COLUMNS:
            try:
                probability = float(row[column])
            except (TypeError, ValueError) as error:
                message = f"Invalid probability for {identifier}/{column}"
                raise ValueError(message) from error
            if not isfinite(probability) or not 0 <= probability <= 1:
                message = f"Probability outside finite [0, 1] for {identifier}/{column}"
                raise ValueError(message)
            probabilities.append(probability)
        parsed.append(PredictionRow(
            ID=identifier, FILE_FAKE_PROB=probabilities[0], VOICE_FAKE_PROB=probabilities[1],
            MUSIC_FAKE_PROB=probabilities[2], VOICE_PRESENT_PROB=probabilities[3],
            MUSIC_PRESENT_PROB=probabilities[4],
        ))
    return parsed


def _label(row: Row, column: str) -> int | None:
    if row.get(column + "_mask", "1") in (0, "0", False, "False", "0.0"):
        return None
    value = row.get(column)
    if value is None or value == "":
        return None
    if value not in (0, 1, "0", "1"):
        message = f"Label must be 0, 1, or unknown for {_identifier(row)}/{column}"
        raise ValueError(message)
    return int(value)


def _roc_metrics(targets: Sequence[int], scores: Sequence[float]) -> tuple[float, float]:
    """Match sklearn's full tied-score ROC and the published nearest-point EER."""
    positives = sum(targets)
    negatives = len(targets) - positives
    points = [(0.0, 0.0)]
    true_positives = false_positives = 0
    ranked = sorted(zip(scores, targets, strict=True), reverse=True)
    for _, tied in groupby(ranked, key=lambda pair: pair[0]):
        for _, target in tied:
            true_positives += target
            false_positives += 1 - target
        points.append((false_positives / negatives, true_positives / positives))
    fpr, tpr = min(points, key=lambda point: abs(point[0] - (1 - point[1])))
    fnr = 1 - tpr
    eer = (fpr + fnr) / 2
    auc = sum(
        (right[0] - left[0]) * (right[1] + left[1]) / 2
        for left, right in zip(points, points[1:])
    )
    return eer, auc


def official_metrics(labels: Sequence[Row], predictions: Sequence[Row]) -> MetricsReport:
    """Report known-subset metrics, but require complete truth for an official total.

    Coverage is known eligible labels / (known eligible + unknown relevant labels).
    Unknown presence is relevant because it leaves component eligibility unknown.
    """
    parsed = validate_predictions(predictions, [_identifier(row) for row in labels])
    truth = [{column: _label(row, column) for column in LABEL_COLUMNS} for row in labels]
    report = MetricsReport(
        score=None, metrics={}, invalid={}, partial={}, counts={}, unknown_counts={}, coverage={},
    )
    specifications = (
        ("file_eer", "FILE_FAKE", "FILE_FAKE_PROB", None),
        ("voice_eer", "VOICE_FAKE", "VOICE_FAKE_PROB", "VOICE_PRESENT"),
        ("music_eer", "MUSIC_FAKE", "MUSIC_FAKE_PROB", "MUSIC_PRESENT"),
        ("voice_presence_auc", "VOICE_PRESENT", "VOICE_PRESENT_PROB", None),
        ("music_presence_auc", "MUSIC_PRESENT", "MUSIC_PRESENT_PROB", None),
    )
    for name, target_column, score_column, mask_column in specifications:
        targets = []
        scores = []
        missing = 0
        for row, prediction in zip(truth, parsed, strict=True):
            if mask_column is not None:
                if row[mask_column] is None:
                    missing += 1
                    continue
                if row[mask_column] == 0:
                    continue
            target = row[target_column]
            if target is None:
                missing += 1
                continue
            targets.append(target)
            scores.append(float(prediction[score_column]))
        report["counts"][name] = len(targets)
        report["unknown_counts"][name] = missing
        denominator = len(targets) + missing
        report["coverage"][name] = len(targets) / denominator if denominator else 0.0
        report["metrics"][name] = None
        if missing:
            report["partial"][name] = f"known subset only; unknown relevant labels: {missing}"
        if len(set(targets)) != 2:
            report["invalid"][name] = "both positive and negative labels are required"
            continue
        eer, auc = _roc_metrics(targets, scores)
        report["metrics"][name] = auc if name.endswith("auc") else eer
    if not report["invalid"] and not report["partial"]:
        values = [value for value in report["metrics"].values() if value is not None]
        file_eer, voice_eer, music_eer, voice_auc, music_auc = values
        report["score"] = (
            0.45 * (1 - file_eer) + 0.18 * (1 - voice_eer) + 0.27 * (1 - music_eer)
            + 0.05 * voice_auc + 0.05 * music_auc
        )
    return report
