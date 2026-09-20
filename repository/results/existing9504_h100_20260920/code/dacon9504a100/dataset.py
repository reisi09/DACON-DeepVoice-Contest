"""Three-split manifest contract: labels/masks, balance, exposure, and leakage.

Standard library only.  Every check reports *all* violations it finds before
raising, so a failed preflight shows the complete problem at once.
"""
from __future__ import annotations

import csv
import json
import re
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

from .common import PipelineError, safe_relative
from .constants import LABELS, SPLITS

HEX64 = re.compile(r"^[0-9a-f]{64}$")
ROUTE_LABEL = {"voice": "VOICE_FAKE", "music": "MUSIC_FAKE"}
ROUTE_PRESENT = {"voice": "VOICE_PRESENT", "music": "MUSIC_PRESENT"}
# Content-identity and provenance atoms.  balance_pair is a class-balancing pair,
# not provenance, and is intentionally excluded.
HASH_ATOMS = ("audio_sha256", "source_sha256", "pcm_sha256", "native_pcm_sha256", "pcm16k_sha256")
PRESERVED_COLUMNS = (*LABELS, *(f"{name}_mask" for name in LABELS), "group_id", "component_id",
                     "parent_id", "parent_ids", "historical_development_exposure", "route", "source",
                     "generator", "input_kind", "variant", "event_intervals", "split")


class DatasetError(PipelineError):
    """The runtime manifest violates the approved split/label contract."""


def read_manifest(path: Path) -> list[dict[str, str]]:
    with Path(path).open(encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames is None or len(reader.fieldnames) != len(set(reader.fieldnames)):
            raise DatasetError(f"Manifest header missing or duplicated: {path}")
        return [dict(row) for row in reader]


def is_singing(row: Mapping[str, str], singing_source: str = "CtrSVDD") -> bool:
    return row.get("route") == "voice" and (row.get("source") == singing_source
                                            or row.get("kind") == "a_cappella_singing")


def exposure_flag(row: Mapping[str, str]) -> bool:
    value = str(row.get("historical_development_exposure", "")).strip()
    if value not in ("True", "False"):
        raise DatasetError(f"historical_development_exposure must be True/False: {row.get('ID')}")
    return value == "True"


def _parent_list(value: str) -> list[str]:
    value = (value or "").strip()
    if not value:
        return []
    if value.startswith("["):
        parsed = json.loads(value)
        if not isinstance(parsed, list):
            raise DatasetError(f"parent_ids must be a JSON list: {value[:80]}")
        return [str(item) for item in parsed if str(item)]
    return [item for item in value.split("|") if item]


def provenance_atoms(row: Mapping[str, str]) -> list[str]:
    """Atoms whose sharing makes two rows the same provenance component.

    group_id parts and parent references share one namespace (as the original
    eight_fixed_io.connected_components did); parent references also bind to
    the referenced row ID.  Exact-content hashes bind byte/PCM duplicates.
    """
    atoms = [f"id:{row['ID']}"]
    atoms += [f"ref:{item}" for item in str(row.get("group_id", "")).split("|") if item]
    parents = _parent_list(row.get("parent_ids", ""))
    if row.get("parent_id"):
        parents.append(str(row["parent_id"]))
    for item in parents:
        atoms += [f"ref:{item}", f"id:{item}"]
    if row.get("component_id"):
        atoms.append(f"component:{row['component_id']}")
    if row.get("pair_id"):
        atoms.append(f"pair:{row['pair_id']}")
    for name in HASH_ATOMS:
        value = str(row.get(name, "")).strip().lower()
        if value:
            atoms.append(f"content:{value}")
    if row.get("duplicate_of"):
        atoms.append(f"id:{row['duplicate_of']}")
    return atoms


def leakage_components(rows: Sequence[Mapping[str, str]]) -> dict[str, str]:
    """Union-find over provenance and exact-content atoms (transitive closure)."""
    parent: dict[str, str] = {}

    def find(atom: str) -> str:
        parent.setdefault(atom, atom)
        while parent[atom] != atom:
            parent[atom] = parent[parent[atom]]
            atom = parent[atom]
        return atom

    first: dict[str, str] = {}
    for row in rows:
        atoms = provenance_atoms(row)
        root = find(atoms[0])
        for atom in atoms[1:]:
            other = find(atom)
            if other != root:
                parent[other] = root
        first[row["ID"]] = atoms[0]
    return {identifier: find(atom) for identifier, atom in first.items()}


def split_leaks(rows: Sequence[Mapping[str, str]], limit: int = 20) -> list[dict[str, Any]]:
    component = leakage_components(rows)
    members: dict[str, list[Mapping[str, str]]] = defaultdict(list)
    for row in rows:
        members[component[row["ID"]]].append(row)
    leaks = []
    for group in members.values():
        splits = sorted({row["split"] for row in group})
        if len(splits) > 1:
            leaks.append({"splits": splits, "rows": len(group),
                          "examples": [(row["ID"], row["split"]) for row in group[:6]]})
    leaks.sort(key=lambda item: -item["rows"])
    return leaks[:limit] + ([{"more_components": len(leaks) - limit}] if len(leaks) > limit else [])


def validate_manifest(rows: Sequence[dict[str, str]], expectations: Mapping[str, Any]) -> dict[str, Any]:
    """Check every contract the H100 run relies on; raise with the full error list."""
    errors: list[str] = []

    def fail(message: str) -> None:
        if len(errors) < 200:
            errors.append(message)

    identifiers = [row.get("ID", "") for row in rows]
    if len(rows) != expectations["rows"]:
        fail(f"rows={len(rows)} expected {expectations['rows']}")
    if len(set(identifiers)) != len(identifiers) or any(not value for value in identifiers):
        fail("IDs must be unique and non-empty")
    required = {"ID", "split", "route", "path", "audio_sha256", "group_id", "component_id",
                "historical_development_exposure", "source", *LABELS, *(f"{n}_mask" for n in LABELS)}
    missing_columns = required - set(rows[0]) if rows else required
    if missing_columns:
        raise DatasetError(f"Manifest lacks required columns: {sorted(missing_columns)}")
    audio_by_hash: dict[str, list[str]] = defaultdict(list)
    fake_route_mismatch = 0
    for row in rows:
        identifier = row["ID"]
        if "/" in identifier or identifier in (".", "..") or "\x00" in identifier or identifier != identifier.strip():
            fail(f"Unsafe ID for cache filenames: {identifier!r}")
        if row["split"] not in SPLITS:
            fail(f"Invalid split {row['split']!r}: {identifier}")
        if row["route"] not in ROUTE_LABEL:
            fail(f"Invalid route {row['route']!r}: {identifier}")
            continue
        for name in LABELS:
            value, mask = row[name], row[f"{name}_mask"]
            if value not in ("", "0", "1") or mask not in ("0", "1"):
                fail(f"Label/mask format {name}={value!r} mask={mask!r}: {identifier}")
            elif (mask == "1") != (value != ""):
                fail(f"Mask disagrees with label presence {name}: {identifier}")
        if row["FILE_FAKE"] == "":
            fail(f"FILE_FAKE unknown: {identifier}")
        target, present = ROUTE_LABEL[row["route"]], ROUTE_PRESENT[row["route"]]
        if row[target] == "" or row[f"{target}_mask"] != "1":
            fail(f"Route target {target} unknown: {identifier}")
        if row[present] != "1" or row[f"{present}_mask"] != "1":
            fail(f"Route presence {present} not known-present: {identifier}")
        if row[target] in ("0", "1") and row["FILE_FAKE"] in ("0", "1") and row[target] != row["FILE_FAKE"]:
            fake_route_mismatch += 1
        digest = row["audio_sha256"].strip()
        if not HEX64.match(digest):
            fail(f"audio_sha256 is not lowercase SHA-256: {identifier}")
        if row.get("source_sha256") and row["source_sha256"] != digest:
            fail(f"audio_sha256 != source_sha256: {identifier}")
        try:
            relative = safe_relative(row["path"])
        except PipelineError as error:
            fail(str(error))
        else:
            if len(relative.parts) != 2 or relative.parts[0] != "audio" or not relative.name.startswith(digest):
                fail(f"path must be audio/<audio_sha256><extension>: {identifier} -> {row['path']}")
        try:
            exposure_flag(row)
        except DatasetError as error:
            fail(str(error))
        if row.get("event_intervals"):
            try:
                json.loads(row["event_intervals"])
            except json.JSONDecodeError:
                fail(f"event_intervals is not JSON: {identifier}")
        audio_by_hash[digest].append(row["split"])
    if errors:
        raise DatasetError("Manifest contract violations:\n  " + "\n  ".join(errors))

    counts = Counter(row["split"] for row in rows)
    per_class = {split: Counter(row["FILE_FAKE"] for row in rows if row["split"] == split) for split in SPLITS}
    routes = {split: Counter(row["route"] for row in rows if row["split"] == split) for split in SPLITS}
    route_class = {split: {route: dict(Counter(row[ROUTE_LABEL[route]] for row in rows
                                               if row["split"] == split and row["route"] == route))
                           for route in ROUTE_LABEL} for split in SPLITS}
    exposure = {split: sum(exposure_flag(row) for row in rows if row["split"] == split) for split in SPLITS}
    singing_source = expectations.get("singing_source", "CtrSVDD")
    singing = {split: sum(is_singing(row, singing_source) for row in rows if row["split"] == split)
               for split in SPLITS}
    for split in SPLITS:
        if counts[split] != expectations["splits"][split]:
            fail(f"{split} rows={counts[split]} expected {expectations['splits'][split]}")
        expected_class = expectations["per_class"][split]
        if per_class[split].get("0", 0) != expected_class or per_class[split].get("1", 0) != expected_class:
            fail(f"{split} FILE_FAKE balance {dict(per_class[split])} expected {expected_class}/{expected_class}")
        if dict(routes[split]) != {k: v for k, v in expectations["routes"][split].items() if v}:
            fail(f"{split} routes {dict(routes[split])} expected {expectations['routes'][split]}")
        if exposure[split] != expectations["historical_exposure"][split]:
            fail(f"{split} historical exposure {exposure[split]} expected {expectations['historical_exposure'][split]}")
        if singing[split] != expectations["singing"][split]:
            fail(f"{split} singing rows {singing[split]} expected {expectations['singing'][split]}")
        for route, classes in route_class[split].items():
            if classes and set(classes) != {"0", "1"}:
                fail(f"{split}/{route} lacks both route classes: {classes}")
    leaks = split_leaks(rows)
    if leaks:
        fail(f"Provenance/content components span splits: {json.dumps(leaks)[:3000]}")
    cross_split_duplicates = sum(1 for splits in audio_by_hash.values() if len(set(splits)) > 1)
    if cross_split_duplicates:
        fail(f"Identical audio bytes in different splits: {cross_split_duplicates}")
    if errors:
        raise DatasetError("Manifest contract violations:\n  " + "\n  ".join(errors))
    components = leakage_components(rows)
    cells = Counter((row["split"], row["route"], row.get("source", ""), row.get("generator", ""),
                     row[ROUTE_LABEL[row["route"]]]) for row in rows)
    return {
        "rows": len(rows),
        "splits": dict(counts),
        "file_fake_by_split": {split: dict(value) for split, value in per_class.items()},
        "routes_by_split": {split: dict(value) for split, value in routes.items()},
        "route_label_by_split": route_class,
        "historical_exposure_by_split": exposure,
        "singing_by_split": singing,
        "leakage_components": len(set(components.values())),
        "cross_split_components": 0,
        "within_split_duplicate_audio_groups": sum(1 for splits in audio_by_hash.values() if len(splits) > 1),
        "file_fake_differs_from_route_label": fake_route_mismatch,
        "unknown_labels_by_split": {split: {name: sum(row[name] == "" for row in rows if row["split"] == split)
                                            for name in LABELS} for split in SPLITS},
        "extensions": dict(Counter(Path(row["path"]).suffix.lower() for row in rows)),
        "cells": [{"split": s, "route": r, "source": src, "generator": g, "label": lab, "rows": n}
                  for (s, r, src, g, lab), n in sorted(cells.items())],
    }


def cross_check_split_files(rows: Sequence[Mapping[str, str]], split_files: Mapping[str, Path]) -> dict[str, Any]:
    """Prove the runtime manifest preserved the reviewed split CSVs' labels/masks/groups/exposure."""
    by_id = {row["ID"]: row for row in rows}
    report: dict[str, Any] = {}
    errors = []
    for split, path in split_files.items():
        with Path(path).open(encoding="utf-8", newline="") as stream:
            reference = list(csv.DictReader(stream))
        ids = [row["ID"] for row in reference]
        manifest_ids = sorted(identifier for identifier, row in by_id.items() if row["split"] == split)
        if sorted(ids) != manifest_ids:
            errors.append(f"{path.name}: ID set differs from manifest split {split}")
            continue
        columns = [name for name in PRESERVED_COLUMNS if name in reference[0] and name in rows[0]]
        differing = Counter()
        for row in reference:
            for name in columns:
                if row[name] != by_id[row["ID"]][name]:
                    differing[name] += 1
        if differing:
            errors.append(f"{path.name}: preserved columns changed {dict(differing)}")
        report[split] = {"file": path.name, "rows": len(reference), "columns_compared": columns}
    if errors:
        raise DatasetError("Split file cross-check failed:\n  " + "\n  ".join(errors))
    return report


def rows_for(rows: Iterable[Mapping[str, str]], splits: Sequence[str]) -> list[Mapping[str, str]]:
    return [row for row in rows if row["split"] in splits]
