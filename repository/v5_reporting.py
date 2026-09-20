"""Measured five-output comparisons, preserving incomplete-label boundaries."""

from __future__ import annotations

import csv
import json
import math
import time
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import torch

from graph_scoring import LABEL_COLUMNS, SUBMISSION_COLUMNS, official_metrics
from v5_dataset import Example, read_audio
from v5_features import window_starts


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False,
                                   default=str), encoding="utf-8")
    temporary.replace(path)


def labels_as_rows(examples: Sequence[Example]) -> list[dict[str, str]]:
    return [{"ID": row.id, "group_id": "|".join(row.groups), "panel": row.panel,
             **{name: "" if math.isnan(value) else str(int(value))
                for name, value in zip(LABEL_COLUMNS, row.labels, strict=True)}} for row in examples]


def prediction_rows(examples: Sequence[Example], probabilities: np.ndarray) -> list[dict[str, str | float]]:
    return [{"ID": row.id, **{name: float(value) for name, value in
             zip(SUBMISSION_COLUMNS[1:], values, strict=True)}}
            for row, values in zip(examples, probabilities, strict=True)]


def record_predictions(output: Path, examples: Sequence[Example], probabilities: np.ndarray):
    rows = prediction_rows(examples, probabilities)
    report = official_metrics(labels_as_rows(examples), rows)
    metrics = report["metrics"]
    values = list(metrics.values())
    report["known_subset_proxy"] = None if any(value is None for value in values) else (
        .45 * (1 - metrics["file_eer"]) + .18 * (1 - metrics["voice_eer"])
        + .27 * (1 - metrics["music_eer"]) + .05 * metrics["voice_presence_auc"]
        + .05 * metrics["music_presence_auc"])
    report["scope"] = "Local development known-label subsets; not a DACON leaderboard score"
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=SUBMISSION_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    panels = {}
    for panel in sorted({row.panel for row in examples}):
        indices = [index for index, row in enumerate(examples) if row.panel == panel]
        panels[panel] = official_metrics(labels_as_rows([examples[index] for index in indices]),
                                         [rows[index] for index in indices])
    report["panels"] = panels
    write_json(output.with_suffix(".metrics.json"), report)
    return report


def predict_expert(model: torch.nn.Module, examples: Sequence[Example], deadline: float) -> np.ndarray:
    """Whole-file max pooling on each component, with bounded inference batches."""
    device = next(model.parameters()).device
    model.eval()
    result = []
    with torch.inference_mode(), torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
        for row in examples:
            audio = read_audio(row.path)
            width = model.window_samples
            logits = []
            for start in window_starts(len(audio), width):
                if time.time() >= deadline:
                    raise TimeoutError("Whole-file prediction reached the immutable deadline")
                clip = np.pad(audio[start:start + width], (0, max(0, width - len(audio[start:]))))
                value, _ = model(torch.from_numpy(clip).unsqueeze(0).to(device))
                logits.append(value.float())
            result.append(torch.cat(logits).amax(0).sigmoid().cpu().numpy())
    return np.stack(result)


def combine_experts(voice: np.ndarray, music: np.ndarray, presence: np.ndarray) -> np.ndarray:
    vf, mf, vp, mp = voice[:, 0], music[:, 0], presence[:, 0], presence[:, 1]
    return np.stack((np.maximum(vp * vf, mp * mf), vf, mf, vp, mp), axis=1)
