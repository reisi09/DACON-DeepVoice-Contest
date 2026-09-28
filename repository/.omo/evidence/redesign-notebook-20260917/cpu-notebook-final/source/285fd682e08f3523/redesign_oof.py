"""Honest group OOF construction for redesign heads."""

import hashlib
from pathlib import Path
from typing import Literal, Protocol

import numpy as np

from redesign_fit import FitResult, Records, Rows, fit_music
from redesign_models import MusicResidual, StemMusicWrapper

Arm = Literal["A", "B", "C", "D"]


class Predictor(Protocol):
    def __call__(
        self, model: StemMusicWrapper | MusicResidual, rows: Rows, records: Records, *, device: str
    ) -> dict[str, dict[str, float]]: ...


def row_groups(row: dict[str, str]) -> tuple[str, ...]:
    groups = tuple(sorted(part for part in row["group_id"].split("|") if part))
    if not groups:
        raise ValueError(f"Missing group_id: {row['ID']}")
    return groups


def connected_groups(rows: Rows) -> dict[str, str]:
    """Collapse rows sharing any parent group into conservative components."""
    parents = list(range(len(rows)))

    def root(index: int) -> int:
        while parents[index] != index:
            parents[index] = parents[parents[index]]
            index = parents[index]
        return index

    seen: dict[str, int] = {}
    for index, row in enumerate(rows):
        for group in row_groups(row):
            if group in seen:
                parents[root(index)] = root(seen[group])
            seen[group] = index
    components: dict[int, list[str]] = {}
    for index, row in enumerate(rows):
        components.setdefault(root(index), []).extend(row_groups(row))
    names = {key: "|".join(sorted(set(groups))) for key, groups in components.items()}
    return {row["ID"]: names[root(index)] for index, row in enumerate(rows)}


def fold_groups(rows: Rows, folds: int, seed: int) -> list[list[str]]:
    components = sorted(set(connected_groups(rows).values()), key=lambda group: hashlib.sha256(
        f"{seed}:{group}".encode()
    ).hexdigest())
    if folds < 2 or len(components) < folds:
        raise ValueError("OOF requires at least one conservative group per fold")
    return [components[index::folds] for index in range(folds)]


def internal_earlystop(rows: Rows, seed: int) -> tuple[Rows, Rows]:
    groups = connected_groups(rows)
    ordered = sorted(set(groups.values()), key=lambda group: hashlib.sha256(
        f"early:{seed}:{group}".encode()
    ).hexdigest())
    count = max(1, len(ordered) // 5)
    early = set(ordered[:count])
    fitting = [row for row in rows if groups[row["ID"]] not in early]
    validation = [row for row in rows if groups[row["ID"]] in early]
    if not fitting or not validation:
        raise ValueError("Grouped internal early stopping produced an empty partition")
    return fitting, validation


def build_oof(
    rows: Rows,
    records: Records,
    output: Path,
    *,
    arm: Arm,
    seed: int,
    epochs: int,
    patience: int,
    batch_size: int,
    device: str,
    deadline: float | None,
    folds: int,
    predict: Predictor,
) -> tuple[dict[str, dict[str, float]], list[dict[str, str | int | list[str]]], dict[str, list[str]], dict[str, list[str]]]:
    """Refit each fold without outer holdout or inner early-stop groups."""
    components = connected_groups(rows)
    predictions: dict[str, dict[str, float]] = {}
    receipts: list[dict[str, str | int | list[str]]] = []
    held_out: dict[str, list[str]] = {}
    trained: dict[str, list[str]] = {}
    for fold, held_groups in enumerate(fold_groups(rows, folds, 41)):
        held = set(held_groups)
        outer_train = [row for row in rows if components[row["ID"]] not in held]
        outer_valid = [row for row in rows if components[row["ID"]] in held]
        fitting, early = internal_earlystop(outer_train, 41 + fold)
        result: FitResult = fit_music(
            fitting, records, early, output / f"fold{fold}", arm=arm, seed=seed,
            epochs=epochs, patience=patience, batch_size=batch_size, device=device,
            deadline=deadline,
        )
        fold_predictions = predict(result.model, outer_valid, records, device=device)
        train_partitions = {components[row["ID"]] for row in fitting}
        early_partitions = {components[row["ID"]] for row in early}
        if held & train_partitions or held & early_partitions:
            raise RuntimeError("Outer OOF group leaked into fitting or early stopping")
        train_groups = sorted({group for row in fitting for group in row_groups(row)})
        early_groups = sorted({group for row in early for group in row_groups(row)})
        held_atoms = sorted({group for row in outer_valid for group in row_groups(row)})
        for row in outer_valid:
            identifier = row["ID"]
            predictions[identifier] = fold_predictions[identifier]
            held_out[identifier] = sorted(row_groups(row))
            trained[identifier] = train_groups
        receipts.append({
            "fold": fold,
            "held_out_groups": held_atoms,
            "trained_groups": train_groups,
            "earlystop_groups": early_groups,
            "held_out_ids": sorted(row["ID"] for row in outer_valid),
            "train_ids": sorted(row["ID"] for row in fitting),
            "earlystop_ids": sorted(row["ID"] for row in early),
            "checkpoint": str(result.checkpoint),
            "checkpoint_sha256": _sha256(result.checkpoint),
            "selected_epoch": int(result.report["selected_epoch"]),
        })
    if set(predictions) != {row["ID"] for row in rows}:
        raise RuntimeError("OOF predictions do not cover every training row exactly once")
    return predictions, receipts, held_out, trained


def _sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()
