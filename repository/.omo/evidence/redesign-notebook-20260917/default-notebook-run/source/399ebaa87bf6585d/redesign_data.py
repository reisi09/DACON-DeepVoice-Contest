# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy", "soundfile"]
# ///
# How to run: `.venv/bin/python redesign_data.py template sources.csv` or `build --sources sources.csv --output data`.
"""CPU provenance validation, lineage-safe splitting, and paired data preparation."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
import tempfile
from collections import Counter
from pathlib import Path
from typing import Final

from redesign_data_validation import DataContractError, LABELS, SOURCE_FIELDS, read_rows, validate_row, write_source_template
from redesign_mix_audio import make_paired_views

OUTPUT_FIELDS: Final = SOURCE_FIELDS + ("group_id", "split", "pair_id", "view", "event_intervals", "duration_seconds")
def _sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _assign_groups(rows: list[dict[str, str]], parents: dict[str, list[str]], seed: int) -> None:
    roots = list(range(len(rows)))

    def root(index: int) -> int:
        while roots[index] != index:
            roots[index] = roots[roots[index]]
            index = roots[index]
        return index

    identity_owner: dict[str, int] = {}
    for index, row in enumerate(rows):
        identities = [f"lineage:{row['ID']}", *(f"lineage:{item}" for item in parents[row["ID"]])]
        identities += [f"audio:{row['audio_sha256']}", f"pcm:{row['_pcm_sha256']}"]
        identities += ([f"artist:{row['artist_id']}"] if row["artist_id"] else [])
        identities += ([f"speaker:{row['speaker_id']}"] if row["speaker_id"] else [])
        for identity in identities:
            owner = identity_owner.get(identity)
            if owner is not None:
                roots[root(index)] = root(owner)
            identity_owner[identity] = index
    components: dict[int, list[int]] = {}
    for index in range(len(rows)):
        components.setdefault(root(index), []).append(index)
    ordered = sorted(components.values(), key=lambda members: hashlib.sha256(
        f"{seed}|{min(rows[index]['ID'] for index in members)}".encode()).hexdigest())
    count = len(ordered)
    holdout = max(1, round(count * 0.1)) if count >= 3 else 0
    capacities = {"train": count - 2 * holdout, "validation": holdout, "sealed": holdout}
    assigned = Counter[str]()
    stratum_counts: dict[str, Counter[str]] = {}
    global_strata = Counter(
        f"{rows[index]['kind']}:{rows[index]['FILE_FAKE']}:{rows[index]['source']}:{rows[index]['generator']}"
        for members in ordered for index in members
    )
    ratios = {"train": 0.8, "validation": 0.1, "sealed": 0.1}
    for members in ordered:
        strata = Counter(
            f"{rows[index]['kind']}:{rows[index]['FILE_FAKE']}:{rows[index]['source']}:{rows[index]['generator']}"
            for index in members
        )
        choices = [split for split in capacities if assigned[split] < capacities[split]]
        split = min(choices, key=lambda name: (
            sum((stratum_counts.get(key, Counter())[name] + amount - global_strata[key] * ratios[name]) ** 2 for key, amount in strata.items()),
            assigned[name] / max(capacities[name], 1), name,
        ))
        assigned[split] += 1
        for key, amount in strata.items():
            stratum_counts.setdefault(key, Counter())[split] += amount
        token = hashlib.sha256("|".join(sorted(rows[index]["ID"] for index in members)).encode()).hexdigest()[:20]
        for index in members:
            rows[index]["group_id"] = f"source-{token}"
            rows[index]["split"] = split


def _write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=OUTPUT_FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _lineage_overlap(rows: list[dict[str, str]]) -> dict[str, list[str]]:
    ownership: dict[str, set[str]] = {}
    for row in rows:
        for group in row["group_id"].split("|"):
            ownership.setdefault(group, set()).add(row["split"])
    return {group: sorted(splits) for group, splits in ownership.items() if len(splits) > 1}


def verify_lineage(rows: list[dict[str, str]]) -> dict[str, int]:
    """Reject cross-split parents and broken pure/mixed exposure pairs."""
    by_id = {row["ID"]: row for row in rows}
    pairs: dict[str, list[dict[str, str]]] = {}
    for row in rows:
        if row["pair_id"]:
            pairs.setdefault(row["pair_id"], []).append(row)
        for parent_id in json.loads(row["parent_ids"]):
            parent = by_id.get(parent_id)
            if parent is None:
                continue
            if parent["split"] != row["split"]:
                raise DataContractError(f"cross-split parent: {row['ID']} -> {parent_id}")
            if not set(parent["group_id"].split("|")) <= set(row["group_id"].split("|")):
                raise DataContractError(f"parent group missing from child: {row['ID']} -> {parent_id}")
    for pair_id, pair_rows in pairs.items():
        views = {row["view"]: row for row in pair_rows}
        if set(views) != {"pure", "mixed"} or len(pair_rows) != 2:
            raise DataContractError(f"pair {pair_id} must contain one pure and one mixed row")
        pure, mixed = views["pure"], views["mixed"]
        if (pure["split"], pure["group_id"], pure["duration_seconds"], pure["MUSIC_FAKE"]) != (
            mixed["split"], mixed["group_id"], mixed["duration_seconds"], mixed["MUSIC_FAKE"]
        ):
            raise DataContractError(f"pair {pair_id} changes split, group, duration, or music truth")
        pure_events, mixed_events = json.loads(pure["event_intervals"]), json.loads(mixed["event_intervals"])
        pure_music = next((event for event in pure_events if event["component"] == "music"), None)
        mixed_music = next((event for event in mixed_events if event["component"] == "music"), None)
        mixed_voice = next((event for event in mixed_events if event["component"] == "voice"), None)
        duration = float(pure["duration_seconds"])
        if pure_music != mixed_music or mixed_voice is None:
            raise DataContractError(f"pair {pair_id} changes music event exposure")
        if (
            pure_music["label"] != int(pure["MUSIC_FAKE"])
            or mixed_voice["label"] != int(mixed["VOICE_FAKE"])
            or mixed["FILE_FAKE"] != str(max(int(mixed["VOICE_FAKE"]), int(mixed["MUSIC_FAKE"])))
            or not (0.0 <= pure_music["start"] < pure_music["end"] <= duration)
            or not (0.0 <= mixed_voice["start"] < mixed_voice["end"] <= duration)
        ):
            raise DataContractError(f"pair {pair_id} has false component or interval truth")
        pure_parents, mixed_parents = json.loads(pure["parent_ids"]), json.loads(mixed["parent_ids"])
        if len(pure_parents) != 1 or pure_parents[0] not in mixed_parents:
            raise DataContractError(f"pair {pair_id} changes accompaniment parent")
    return {"rows": len(rows), "pairs": len(pairs)}


def _verified_existing(output: Path, manifest_sha256: str, seed: int, diagnostic: bool, make_mixtures: bool) -> dict[str, str | int | bool | dict[str, int] | list[dict[str, str]]] | None:
    if not output.exists():
        return None
    try:
        protocol = json.loads((output / "protocol.json").read_text())
        audit = json.loads((output / "audit.json").read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise DataContractError(f"existing output is incomplete: {output}: {error}") from error
    expected_status = "diagnostic-only" if diagnostic else "not declared historical"
    if (protocol.get("source_manifest_sha256"), protocol.get("seed"), protocol.get("historical_status"), protocol.get("make_mixtures")) != (
        manifest_sha256, seed, expected_status, make_mixtures
    ):
        raise DataContractError(f"existing output identity differs: {output}")
    if audit.get("protocol_sha256") != _sha256(output / "protocol.json"):
        raise DataContractError(f"existing protocol checksum differs: {output}")
    rows = [row for split in ("train", "validation", "sealed") for row in read_rows(output / f"{split}.csv")]
    expected_manifests = audit.get("split_manifest_sha256", {})
    if not isinstance(expected_manifests, dict) or any(
        expected_manifests.get(split) != _sha256(output / f"{split}.csv")
        for split in ("train", "validation", "sealed")
    ):
        raise DataContractError(f"existing split manifest checksum differs: {output}")
    for row in rows:
        path = Path(row["path"])
        if not path.is_file() or _sha256(path) != row["audio_sha256"]:
            raise DataContractError(f"existing output audio differs: {row['ID']}")
    verify_lineage(rows)
    return audit


def prepare_data(
    manifest: Path,
    output: Path,
    *,
    seed: int = 41,
    diagnostic: bool = False,
    make_mixtures: bool = True,
) -> dict[str, str | int | bool | dict[str, int] | list[dict[str, str]]]:
    """Validate sources, split connected lineage, and materialize paired views."""
    manifest_digest = _sha256(manifest)
    existing = _verified_existing(output, manifest_digest, seed, diagnostic, make_mixtures)
    if existing is not None:
        return existing
    raw_rows = read_rows(manifest)
    valid: list[dict[str, str]] = []
    parent_map: dict[str, list[str]] = {}
    quarantined: list[dict[str, str]] = []
    seen: set[str] = set()
    for position, raw in enumerate(raw_rows, start=2):
        try:
            row, parents = validate_row(raw, manifest.parent)
            if row["ID"] in seen:
                raise DataContractError(f"duplicate ID {row['ID']}")
            seen.add(row["ID"])
            valid.append(row)
            parent_map[row["ID"]] = parents
        except DataContractError as error:
            quarantined.append({"line": str(position), "ID": raw.get("ID", ""), "reason": str(error)})
    if not valid:
        raise DataContractError("manifest has no valid source rows")
    _assign_groups(valid, parent_map, seed)
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output.name}.", dir=output.parent))
    try:
        generated: list[dict[str, str]] = []
        state_counts: dict[str, int] = {split: 0 for split in ("train", "validation", "sealed")}
        state_counts["skipped_sample_rate_mismatch"] = 0
        if make_mixtures:
            generated, state_counts = make_paired_views(valid, staging / "audio", seed)
            for row in generated:
                row["path"] = str((output / "audio" / Path(row["path"]).name).resolve())
        rows = valid + generated
        lineage = verify_lineage(rows)
        overlap = _lineage_overlap(rows)
        if overlap:
            raise DataContractError(f"lineage crosses splits: {overlap}")
        for split in ("train", "validation", "sealed"):
            _write_csv(staging / f"{split}.csv", [row for row in rows if row["split"] == split])
        unknown = {label: sum(row[label] == "" for row in rows) for label in LABELS}
        strata: dict[str, dict[str, int]] = {}
        for row in valid:
            key = f"{row['kind']}:{row['FILE_FAKE']}:{row['source']}:{row['generator']}"
            strata.setdefault(key, {split: 0 for split in ("train", "validation", "sealed")})[row["split"]] += 1
        protocol = {
            "schema_version": 1, "split_policy": "connected parent/artist/speaker groups, deterministic stratified 8:1:1",
            "provider_grouping": False, "seed": seed, "source_manifest_sha256": manifest_digest,
            "paired_views": "same pair_id/group/split/duration/music gain; choose pure or mixed, never concatenate exposures",
            "mix_modes": ["simultaneous", "sequential", "sparse-0.5s"], "make_mixtures": make_mixtures,
            "historical_status": "diagnostic-only" if diagnostic else "not declared historical",
        }
        (staging / "protocol.json").write_text(json.dumps(protocol, indent=2) + "\n")
        audit = {
            "manifest_sha256": manifest_digest, "protocol_sha256": _sha256(staging / "protocol.json"),
            "seed": seed, "diagnostic": diagnostic, "input_rows": len(raw_rows),
            "accepted_sources": len(valid), "generated_rows": len(generated),
            "split_rows": dict(Counter(row["split"] for row in rows)),
            "independent_source_groups": dict(Counter(row["split"] for row in valid if row["ID"] == min(
                candidate["ID"] for candidate in valid if candidate["group_id"] == row["group_id"]
            ))),
            "strata_counts": strata, "paired_state_slots": state_counts, "unknown_labels": unknown,
            "lineage_overlap": overlap, "verified_pairs": lineage["pairs"], "quarantined": quarantined,
            "coverage_gate": "diagnostic_only" if diagnostic else "unmet" if any(state_counts[split] < 25 for split in ("train", "validation", "sealed")) else "screening_met",
        }
        audit["split_manifest_sha256"] = {
            split: _sha256(staging / f"{split}.csv") for split in ("train", "validation", "sealed")
        }
        (staging / "audit.json").write_text(json.dumps(audit, indent=2) + "\n")
        staging.replace(output)
        return audit
    finally:
        if staging.exists():
            shutil.rmtree(staging)


def main() -> None:
    parser = argparse.ArgumentParser()
    subcommands = parser.add_subparsers(dest="command", required=True)
    template = subcommands.add_parser("template")
    template.add_argument("output", type=Path)
    build = subcommands.add_parser("build")
    build.add_argument("--sources", required=True, type=Path)
    build.add_argument("--output", required=True, type=Path)
    build.add_argument("--seed", type=int, default=41)
    build.add_argument("--diagnostic", action="store_true")
    args = parser.parse_args()
    if args.command == "template":
        write_source_template(args.output)
    else:
        result = prepare_data(args.sources, args.output, seed=args.seed, diagnostic=args.diagnostic)
        print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
