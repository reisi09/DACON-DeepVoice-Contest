"""Small I/O seam shared by fixed-feature experiments."""
from __future__ import annotations

import csv
import json
from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np

from eight_metrics import binary_summary, stratum_summaries

LABELS = ("FILE_FAKE", "VOICE_FAKE", "MUSIC_FAKE")
JsonValue = str | int | float | bool | None | list["JsonValue"] | dict[str, "JsonValue"]


class FixedIOError(ValueError):
    """Manifest, feature, or config input violates the fixed experiment contract."""


def atomic_json(path: Path, value: Mapping[str, JsonValue]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
    temporary.replace(path)


def read_config(path: str | Path) -> dict[str, JsonValue]:
    value = json.loads(Path(path).read_text())
    if not isinstance(value, dict):
        raise FixedIOError("config root must be an object")
    return value


def read_rows(root: Path) -> list[dict[str, str]]:
    with (root / "source/dataset/manifest.csv").open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    if not rows or len({row["ID"] for row in rows}) != len(rows):
        raise FixedIOError("manifest must have unique rows")
    component_by_id = connected_components(rows)
    group_split: dict[str, str] = {}
    for row in rows:
        row["component_id"] = component_by_id[row["ID"]]
        for group in row["group_id"].split("|"):
            if group in group_split and group_split[group] != row["split"]:
                raise FixedIOError(f"connected provenance leaked across splits: {group}")
            group_split[group] = row["split"]
    return rows


def connected_components(rows: Sequence[Mapping[str, str]]) -> dict[str, str]:
    """Resolve transitive provenance atoms once over the complete manifest."""
    parent: dict[str, str] = {}

    def find(atom: str) -> str:
        parent.setdefault(atom, atom)
        while parent[atom] != atom:
            parent[atom] = parent[parent[atom]]
            atom = parent[atom]
        return atom

    atoms_by_id: dict[str, list[str]] = {}
    for row in rows:
        atoms = [atom for atom in str(row["group_id"]).split("|") if atom]
        atoms.extend(str(atom) for atom in json.loads(row.get("parent_ids", "[]")))
        atoms = sorted(set(atoms))
        if not atoms:
            raise FixedIOError(f"missing provenance atoms: {row['ID']}")
        root = find(atoms[0])
        for atom in atoms[1:]:
            other = find(atom)
            if other != root:
                parent[other] = root
        atoms_by_id[row["ID"]] = atoms
    members: dict[str, list[str]] = {}
    for atom in parent:
        members.setdefault(find(atom), []).append(atom)
    names = {root: "|".join(sorted(atoms)) for root, atoms in members.items()}
    return {identifier: names[find(atoms[0])] for identifier, atoms in atoms_by_id.items()}


def label_for(row: Mapping[str, str]) -> int | None:
    column = "MUSIC_FAKE" if row["kind"] == "music" else "VOICE_FAKE" if row["kind"] == "speech" else "FILE_FAKE"
    return int(row[column]) if row[column] in ("0", "1") else None


def feature(root: Path, identifier: str, key: str) -> np.ndarray:
    with np.load(root / "output/features" / f"{identifier}.npz", allow_pickle=False) as archive:
        if key not in archive.files:
            raise FixedIOError(f"missing {key} feature: {identifier}")
        value = np.asarray(archive[key], dtype=np.float32)
        valid = np.asarray(archive["valid_samples"]) > 0
    if value.ndim != 2 or valid.shape != (len(value),) or not valid.any() or not np.isfinite(value[valid]).all():
        raise FixedIOError(f"invalid {key} feature: {identifier}")
    return value[valid].mean(axis=0)


def matrix(root: Path, rows: Sequence[Mapping[str, str]], key: str, width: int | None = None) -> np.ndarray:
    values = [feature(root, row["ID"], key) for row in rows]
    result = np.stack(values)
    if width is not None:
        if result.shape[1] < width:
            raise FixedIOError(f"{key} has fewer than {width} fixed dimensions")
        result = result[:, :width]
    return result


def standardize(train: np.ndarray, *others: np.ndarray) -> tuple[np.ndarray, ...]:
    mean, scale = train.mean(axis=0), train.std(axis=0)
    scale[scale < 1e-6] = 1.0
    return tuple((value - mean) / scale for value in (train, *others))


def model_rows(rows: Sequence[Mapping[str, str]]) -> list[dict[str, str | int]]:
    result = []
    for row in rows:
        label = label_for(row)
        if label is not None:
            result.append({**row, "label": label})
    return result


def evaluate(
    rows: Sequence[Mapping[str, str | int]], scores: Sequence[float], threshold: float | None = None,
) -> dict[str, JsonValue]:
    records = [{**row, "score": float(score), "component_id": str(row["component_id"]),
                "codec": str(row.get("variant", ""))}
               for row, score in zip(rows, scores, strict=True)]
    return {
        **binary_summary([int(row["label"]) for row in records], [float(row["score"]) for row in records], threshold),
        "strata": stratum_summaries(records, "score"),
        "rows": len(records),
        "components": len({str(row["component_id"]) for row in records}),
    }


def write_predictions(path: Path, rows: Sequence[Mapping[str, str | int]], columns: Mapping[str, Sequence[float]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    names = list(columns)
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["ID", "split", "label", "component_id", "source", "generator", "codec", *names])
        writer.writeheader()
        for index, row in enumerate(rows):
            writer.writerow({"ID": row["ID"], "split": row["split"], "label": row["label"],
                             "component_id": row["component_id"], "source": row["source"],
                             "generator": row.get("generator", ""), "codec": row.get("variant", ""),
                             **{name: float(columns[name][index]) for name in names}})
