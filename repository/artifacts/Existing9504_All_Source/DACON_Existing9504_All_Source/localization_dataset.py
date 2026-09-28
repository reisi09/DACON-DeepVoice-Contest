# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy", "soundfile", "imageio-ffmpeg"]
# ///
# Run: .venv/bin/python localization_dataset.py
"""Assemble existing split-local music, matched train edits, and optional new confirmation audio."""
# noqa: SIZE_OK -- Keep this single offline manifest builder and its integrity audit together.
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import json
from pathlib import Path
import shutil
from typing import Final

import numpy as np
import soundfile as sf

from eight_audio import SR, load, manifest_row, replace_event, rms_match, write
from eight_contract import LABELS, MANIFEST_FIELDS, ManifestRow, read_manifest, sha256_file, write_manifest

ROOT: Final = Path(__file__).resolve().parent
SOURCE: Final = ROOT / "_workspace/eight_h100_20260917/dataset/manifest.csv"
OUTPUT: Final = ROOT / "_workspace/localization_a100_20260918/dataset"


class DatasetError(ValueError):
    """Audio or contributor truth cannot support the declared construction."""


def construct_training_partials(
    rows: list[ManifestRow], source: Path, output: Path, *, pairs: int = 64, seed: int = 41,
) -> list[ManifestRow]:
    """Make same-boundary fake/sham pairs from unextended eight-second training audio."""
    cohort = [row for row in rows if row["split"] == "train" and row["variant"] == "base"
              and row["MUSIC_PRESENT"] == "1" and float(row["duration_seconds"]) == 8]
    real = sorted((row for row in cohort if row["MUSIC_FAKE"] == "0" and row["FILE_FAKE"] == "0"
                   and row["source"].startswith("FMA")), key=lambda row: row["ID"])
    fake = sorted((row for row in cohort if row["MUSIC_FAKE"] == "1" and row["source"] in {"suno", "udio"}),
                  key=lambda row: row["ID"])
    if pairs < 1 or len(real) < 2 or not fake:
        raise DatasetError("Need two real FMA and one generated SONICS eight-second training parents")
    rng = np.random.default_rng(seed)
    rng.shuffle(real)
    rng.shuffle(fake)
    additions: list[ManifestRow] = []
    receipt = []
    for ordinal in range(pairs):
        bed_row, sham_row, fake_row = real[ordinal % len(real)], real[(ordinal + 1) % len(real)], fake[ordinal % len(fake)]
        bed, sham, generated = [load(source / row["path"]) for row in (bed_row, sham_row, fake_row)]
        if any(len(audio) != 8 * SR for audio in (bed, sham, generated)):
            raise DatasetError("Actual donor duration differs from the eight-second manifest")
        seconds, position = (.25, .5, 1, 2)[ordinal % 4], ("start", "mid", "end")[ordinal % 3]
        length = int(seconds * SR)
        start = {"start": 0, "mid": (len(bed) - length) // 2, "end": len(bed) - length}[position]
        offset = (len(bed) - length) // 2
        inserts = (generated[offset:offset + length], sham[offset:offset + length])
        peak = max(float(np.abs(bed).max()), *(float(np.abs(rms_match(part, bed[start:start + length])).max())
                                              for part in inserts))
        gain = min(1.0, .9 / max(peak, 1e-8))
        pair = "localization_train:" + hashlib.sha256(
            f"{seed}:{ordinal}:{bed_row['ID']}:{fake_row['ID']}:{sham_row['ID']}".encode()).hexdigest()[:20]
        for label, donor, insert in ((1, fake_row, inserts[0]), (0, sham_row, inserts[1])):
            identifier = hashlib.sha256(f"{pair}:{label}".encode()).hexdigest()[:24]
            path = output / "audio" / f"localization_{identifier}.wav"
            changed = replace_event(bed * gain, insert * gain, start)
            if not (np.array_equal(changed[:start], (bed * gain)[:start])
                    and np.array_equal(changed[start + length:], (bed * gain)[start + length:])):
                raise DatasetError("Event processing changed samples outside the declared interval")
            write(path, changed)
            atoms = set(bed_row["group_id"].split("|")) | set(donor["group_id"].split("|"))
            events = [{"component": "music", "start": start / SR, "end": (start + length) / SR, "label": label}]
            additions.append(manifest_row(f"localization_{identifier}", path, [bed_row["ID"], donor["ID"]], atoms,
                                          "train", "constructed_music", donor["generator"] if label else "REAL",
                                          "music", "partial_fake" if label else "sham", "partial", pair,
                                          (str(label), "", str(label), "", "1"), events, output))
        receipt.append({"pair_id": pair, "bed_id": bed_row["ID"], "fake_id": fake_row["ID"], "sham_id": sham_row["ID"],
                        "event_seconds": seconds, "position": position, "start_sample": start,
                        "end_sample": start + length, "insert_start_sample": offset, "common_gain": gain,
                        "fade_samples": 160, "seed": seed})
    output.mkdir(parents=True, exist_ok=True)
    (output / "construction_receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    return additions


def _copy_audio(source: Path, row: ManifestRow, output: Path) -> ManifestRow:
    path = Path(row["path"])
    original = path if path.is_absolute() else source / path
    if sha256_file(original) != row["audio_sha256"]:
        raise DatasetError(f"Audio differs from declared hash: {row['ID']}")
    target = output / "audio" / f"{row['ID']}.wav"
    target.parent.mkdir(parents=True, exist_ok=True)
    if not target.exists():
        shutil.copy2(original, target)
    elif sha256_file(target) != row["audio_sha256"]:
        raise DatasetError(f"Existing destination differs from input: {target}")
    return ManifestRow(**{**row, "path": str(target.relative_to(output))})


def presence_rows(csv_path: Path, output: Path) -> list[ManifestRow]:
    """Resample original presence controls without stretching, repetition, or authenticity inference."""
    with csv_path.open(newline="") as stream:
        records = list(csv.DictReader(stream))
    result: list[ManifestRow] = []
    for record in records:
        original = ROOT / record["path"]
        if sha256_file(original) != record["source_sha256"]:
            raise DatasetError(f"Presence source hash mismatch: {record['ID']}")
        audio = load(original)
        gain = min(1.0, .95 / max(float(np.abs(audio).max()), 1e-8))
        identifier = record["ID"] + "_presence"
        path = output / "audio" / f"{identifier}.wav"
        write(path, audio * gain)
        music = record["MUSIC_PRESENT"]
        events = [{"component": "music", "start": 0, "end": len(audio) / SR, "label": 0}] if music == "1" else []
        result.append(manifest_row(identifier, path, [record["ID"]], set(record["group_id"].split("|")),
                                   "confirmation_presence", record["source"], "REAL", "music" if music == "1" else "speech",
                                   "presence_control", "presence", "", (record["FILE_FAKE"], record["VOICE_FAKE"], "0" if music == "1" else "",
                                                                            record["VOICE_PRESENT"], music), events, output))
        record["resample_common_gain"] = str(gain)
        record["derived_ID"] = identifier
    (output / f"{csv_path.stem}_metadata.json").write_text(json.dumps(records, indent=2) + "\n")
    return result


def audit_dataset(output: Path) -> None:
    """Check actual audio, complete parent provenance, and contributor separation across splits."""
    with (output / "sources.csv").open(newline="") as stream:
        source_rows = list(csv.DictReader(stream))
    sources = {row["ID"]: row for row in source_rows}
    if len(sources) != len(source_rows):
        raise DatasetError("Duplicate contributor source ID")
    rows = read_manifest(output / "manifest.csv")
    by_id = {row["ID"]: row for row in rows}
    if len(by_id) != len(rows):
        raise DatasetError("Duplicate unified manifest ID")
    atoms: dict[str, set[str]] = defaultdict(set)
    hashes: dict[str, set[str]] = defaultdict(set)
    source_hashes: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        parents = json.loads(row["parent_ids"])
        if (not isinstance(parents, list) or not parents or row["parent_id"] not in parents
                or any(not isinstance(parent, str) or not parent for parent in parents)):
            raise DatasetError(f"Missing or malformed parent provenance: {row['ID']}")
        if (row["split"] not in {"train", "dev", "calibration", "test", "confirmation", "confirmation_external", "confirmation_presence"}
                or any(row[label] not in {"", "0", "1"} for label in LABELS)
                or row["MUSIC_PRESENT"] not in {"0", "1"}
                or (row["MUSIC_PRESENT"] == "1" and row["MUSIC_FAKE"] not in {"0", "1"})
                or (row["MUSIC_PRESENT"] == "0" and row["MUSIC_FAKE"] != "")):
            raise DatasetError(f"Invalid split or music/component truth: {row['ID']}")
        path = output / row["path"]
        info = sf.info(path)
        duration = float(row["duration_seconds"])
        if (info.samplerate != SR or info.channels != 1 or not np.isfinite(duration) or duration <= 0
                or abs(info.frames / SR - duration) > 1e-7
                or sha256_file(path) != row["audio_sha256"]):
            raise DatasetError(f"Audio identity, rate, or duration mismatch: {row['ID']}")
        members = set(row["group_id"].split("|"))
        if not members or "" in members:
            raise DatasetError(f"Empty contributor identity: {row['ID']}")
        for atom in members:
            atoms[atom].add(row["split"])
        hashes[row["audio_sha256"]].add(row["split"])
        for parent in parents:
            provenance = by_id.get(parent, sources.get(parent))
            if not provenance or not set(provenance["group_id"].split("|")).issubset(members):
                raise DatasetError(f"Unresolved/truncated contributor: {row['ID']} <- {parent}")
            if parent in by_id and by_id[parent]["split"] != row["split"]:
                raise DatasetError(f"Cross-split parent: {row['ID']} <- {parent}")
            if parent in sources:
                if not sources[parent].get("source_sha256"):
                    raise DatasetError(f"Missing original source hash: {parent}")
                source_hashes[sources[parent]["source_sha256"]].add(row["split"])
        for event in json.loads(row["event_intervals"]):
            if (event["component"] != "music" or event["label"] not in (0, 1)
                    or not 0 <= event["start"] < event["end"] <= info.frames / SR + 1e-7):
                raise DatasetError(f"Unsupported or out-of-bounds music truth: {row['ID']}")
    overlaps = {atom: sorted(splits) for atom, splits in atoms.items() if len(splits) > 1}
    duplicates = {key: sorted(splits) for key, splits in hashes.items() if len(splits) > 1}
    source_duplicates = {key: sorted(splits) for key, splits in source_hashes.items() if len(splits) > 1}
    ids_path = output / "original_music_ids.json"
    original_ids = {identifier for cohort in json.loads(ids_path.read_text()).values() for identifier in cohort} if ids_path.exists() else set()
    original = [row for row in rows if row["ID"] in original_ids]
    fresh = [row for row in rows if row["ID"] not in original_ids and row["split"] == "train" and row["variant"] == "base"]
    original_atoms = {atom for row in original for atom in row["group_id"].split("|")}
    fresh_atoms = {atom for row in fresh for atom in row["group_id"].split("|")}
    original_hashes = {sources[parent]["source_sha256"] for row in original for parent in json.loads(row["parent_ids"]) if parent in sources}
    fresh_hashes = {sources[parent]["source_sha256"] for row in fresh for parent in json.loads(row["parent_ids"]) if parent in sources}
    report = {"rows": len(rows), "splits": dict(Counter(row["split"] for row in rows)),
              "variants": dict(Counter(row["variant"] for row in rows)), "provenance_atoms": len(atoms),
              "cross_split_atoms": overlaps, "cross_split_audio_hashes": duplicates,
              "cross_split_original_source_hashes": source_duplicates,
              "additional_train_rows": len(fresh),
              "additional_train_original_atom_overlap": sorted(original_atoms & fresh_atoms),
              "additional_train_original_source_hash_overlap": sorted(original_hashes & fresh_hashes),
              "manifest_sha256": sha256_file(output / "manifest.csv"),
              "all_audio_hashes_verified": True, "all_parent_provenance_resolved": True}
    (output / "dataset_audit.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    if overlaps or duplicates or source_duplicates or original_atoms & fresh_atoms or original_hashes & fresh_hashes:
        raise DatasetError("Contributor/audio overlap detected across dataset splits; see dataset_audit.json")


def validate_partition(rows: list[ManifestRow], path: Path) -> None:
    """Verify source membership, not just source counts, against the pre-score lock."""
    partition = json.loads(path.read_text())
    for part, expected in (("train", "expected_new_train"), ("confirmation_external", "expected_confirmation_external")):
        if dict(Counter(row["source"] for row in rows if row["split"] == part)) != partition[expected]:
            raise DatasetError(f"Additional {part} population differs from the pre-score partition")
    external_parents = {parent for row in rows if row["source"] == "JamendoMaxCaps"
                        and row["split"] == "confirmation_external" for parent in json.loads(row["parent_ids"])}
    if external_parents != set(partition["jamendo_external_source_ids"]) or partition["selection_uses_model_outputs"] is not False:
        raise DatasetError("External parent assignment differs from the frozen pre-score partition")


def build_dataset(
    manifest: Path = SOURCE, output: Path = OUTPUT, *, confirmation_csv: Path | None = None,
    confirmation_sources: Path | None = None, additional_training_csv: Path | None = None,
    additional_training_sources: Path | None = None, additional_partition: Path | None = None,
    presence_csv: tuple[Path, ...] = (),
) -> None:
    """Preserve original splits; only constructed train rows and disjoint confirmation rows are appended."""
    selected = [row for row in read_manifest(manifest) if row["probe_family"] == "core" and row["variant"] == "base"
                and row["MUSIC_PRESENT"] == "1" and row["MUSIC_FAKE"] in {"0", "1"}]
    if Counter(row["split"] for row in selected) != {"train": 831, "dev": 164, "calibration": 200, "test": 127}:
        raise DatasetError("Original 1322-row music split differs from the locked population")
    copied = [_copy_audio(manifest.parent, row, output) for row in selected]
    (output / "original_music_ids.json").write_text(json.dumps(
        {split: [row["ID"] for row in selected if row["split"] == split]
         for split in ("train", "dev", "calibration", "test")}, indent=2) + "\n")
    for row in copied:
        row["event_intervals"] = json.dumps([{"component": "music", "start": 0,
                                             "end": float(row["duration_seconds"]), "label": int(row["MUSIC_FAKE"])}])
    rows = copied + construct_training_partials(selected, manifest.parent, output)
    with (manifest.parent / "sources.csv").open(newline="") as stream:
        original_sources = list(csv.DictReader(stream))
    referenced = {parent for row in selected for parent in json.loads(row["parent_ids"])}
    sources = [row for row in original_sources if row["ID"] in referenced]
    for incoming_csv, incoming_sources, split in ((additional_training_csv, additional_training_sources, "train"),
                                                  (confirmation_csv, confirmation_sources, "confirmation")):
        if incoming_csv is None:
            continue
        if incoming_sources is None:
            raise DatasetError(f"Source provenance CSV is required alongside additional {split} audio")
        with incoming_csv.open(newline="") as stream:
            incoming = list(csv.DictReader(stream))
        allowed = {"train", "confirmation_external"} if split == "train" else {split}
        if not incoming or any(row["split"] not in allowed for row in incoming):
            raise DatasetError(f"Incoming rows must exclusively use the declared {split} split")
        if split == "train" and any(not np.isfinite(float(row["duration_seconds"]))
                                    or (row["split"] == "train" and float(row["duration_seconds"]) != 8)
                                    or (row["split"] == "confirmation_external" and float(row["duration_seconds"]) < 4)
                                    or row["variant"] != "base"
                                    or row["MUSIC_PRESENT"] != "1" or row["MUSIC_FAKE"] not in {"0", "1"} for row in incoming):
            raise DatasetError("Additional base music must be eight seconds for train or at least four natural seconds for confirmation")
        rows.extend(_copy_audio(ROOT, ManifestRow(**{key: row[key] for key in MANIFEST_FIELDS}), output) for row in incoming)
        with incoming_sources.open(newline="") as stream:
            incoming_provenance = list(csv.DictReader(stream))
        by_parent = {row["ID"]: row for row in incoming_provenance}
        if any(parent not in by_parent or by_parent[parent].get("split") != row["split"]
               for row in incoming for parent in json.loads(row["parent_ids"])):
            raise DatasetError("Additional source partition differs from the imported view partition")
        sources.extend(incoming_provenance)
        shutil.copy2(incoming_csv, output / f"{split}_metadata.csv")
        if split == "train" and additional_partition is not None:
            validate_partition(incoming, additional_partition)
            shutil.copy2(additional_partition, output / "additional_partition.json")
            summary = []
            for source_name, part in sorted({(row["source"], row["split"]) for row in incoming}):
                cohort = [row for row in incoming if row["source"] == source_name and row["split"] == part]
                group_atoms = {atom for row in cohort for atom in row["group_id"].split("|")}
                summary.append({"source": source_name, "split": part, "rows": len(cohort),
                                "labels": dict(Counter(row["MUSIC_FAKE"] for row in cohort)),
                                "generators": dict(Counter(row["generator"] for row in cohort)),
                                "group_atoms": len(group_atoms), "artist_atoms": sum(atom.startswith("artist:") for atom in group_atoms),
                                "reference_atoms": sum(atom.startswith("reference:") for atom in group_atoms)})
            (output / "additional_data_receipt.json").write_text(json.dumps({"sources": summary,
                "source_csv_sha256": sha256_file(incoming_sources), "manifest_csv_sha256": sha256_file(incoming_csv),
                "partition_sha256": sha256_file(additional_partition)}, indent=2) + "\n")
    for csv_path in presence_csv:
        rows.extend(presence_rows(csv_path, output))
        with csv_path.open(newline="") as stream:
            sources.extend(csv.DictReader(stream))
    source_fields = sorted({key for row in sources for key in row})
    with (output / "sources.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=source_fields)
        writer.writeheader()
        writer.writerows(sources)
    write_manifest(output / "manifest.csv", rows)
    assumptions = {
        "source_manifest": str(manifest), "source_manifest_sha256": sha256_file(manifest),
        "original_filewide_music_truth": "Source metadata identifies fully genuine or fully generated music; not timestamp-annotated listening truth.",
        "constructed_interval_truth": "The declared edited interval has its donor's music label; all remaining audio is the genuine FMA bed.",
        "constructed_voice_truth": "Voice presence and voice authenticity are unknown in both arms.",
        "sham_file_truth": "Genuine FMA donors are FILE_FAKE=0; identical event processing does not synthesize voice.",
        "construction": "64 train-only fake/sham pairs; four event lengths each used 16 times; no source padding/repetition.",
        "confirmation_csv": str(confirmation_csv) if confirmation_csv else None,
        "additional_training_csv": str(additional_training_csv) if additional_training_csv else None,
        "additional_partition": str(additional_partition) if additional_partition else None,
        "presence_csv": [str(path) for path in presence_csv],
    }
    (output / "provenance_assumptions.json").write_text(json.dumps(assumptions, indent=2) + "\n")
    audit_dataset(output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=SOURCE)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--confirmation-csv", type=Path)
    parser.add_argument("--confirmation-sources", type=Path)
    parser.add_argument("--additional-training-csv", type=Path)
    parser.add_argument("--additional-training-sources", type=Path)
    parser.add_argument("--additional-partition", type=Path)
    parser.add_argument("--presence-csv", type=Path, nargs="*", default=[])
    options = parser.parse_args()
    build_dataset(options.manifest, options.output, confirmation_csv=options.confirmation_csv,
                  confirmation_sources=options.confirmation_sources, additional_training_csv=options.additional_training_csv,
                  additional_training_sources=options.additional_training_sources, additional_partition=options.additional_partition,
                  presence_csv=tuple(options.presence_csv))
    print((options.output / "dataset_audit.json").read_text())
