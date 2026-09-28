#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy==2.4.6", "imageio-ffmpeg==0.6.0"]
# ///
# How to run: `uv run prepare_v2_validation.py --sources _workspace/v2_sources/source_inventory.json`.
"""Build source-disjoint development and sealed challenge manifests."""
from __future__ import annotations

import argparse
import csv
import json
import shutil
import zipfile
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Final

import numpy as np

from v2_data import Labels, decode_audio, pcm_sha256, sha256_file, stable_key, write_wav

ROOT: Final = Path(__file__).resolve().parent
FIELDS: Final = ("ID", "path", "group_id", "split", "source", "FILE_FAKE", "VOICE_FAKE", "MUSIC_FAKE", "VOICE_PRESENT", "MUSIC_PRESENT")
ROLES: Final = ("dev_train", "dev_select", "test_generator", "test_source")


class SplitIntegrityError(RuntimeError):
    """A data row violates the sealed split contract."""


@dataclass(frozen=True, slots=True)
class ManifestRow:
    ID: str
    path: str
    group_id: str
    split: str
    source: str
    FILE_FAKE: str
    VOICE_FAKE: str
    MUSIC_FAKE: str
    VOICE_PRESENT: str
    MUSIC_PRESENT: str
    pcm_sha256: str

    def csv_record(self) -> dict[str, str]:
        data = asdict(self)
        data.pop("pcm_sha256")
        return data


def write_manifest(path: Path, rows: list[ManifestRow]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(row.csv_record() for row in rows)


def read_manifest(path: Path) -> list[ManifestRow]:
    with path.open() as handle:
        return [ManifestRow(**row, pcm_sha256="") for row in csv.DictReader(handle)]


def audit_rows(rows: list[ManifestRow]) -> None:
    roles: dict[str, set[str]] = {}
    for row in rows:
        if row.split not in ROLES:
            raise SplitIntegrityError(f"unknown role: {row.split}")
        for group in row.group_id.split("|"):
            roles.setdefault(group, set()).add(row.split)
        values = (row.FILE_FAKE, row.VOICE_FAKE, row.MUSIC_FAKE, row.VOICE_PRESENT, row.MUSIC_PRESENT)
        if any(value not in {"", "0", "1"} for value in values):
            raise SplitIntegrityError("labels must be 0, 1, or explicit unknown")
    if any(len(value) != 1 for value in roles.values()):
        raise SplitIntegrityError("atomic source group crosses roles")
    hashes = [row.pcm_sha256 for row in rows]
    if len(hashes) != len(set(hashes)):
        raise SplitIntegrityError("decoded PCM is duplicated")


def _old_rows(old: Path) -> list[ManifestRow]:
    raw = list(csv.DictReader((old / "manifest.csv").open()))
    parent: dict[str, str] = {}

    def find(item: str) -> str:
        parent.setdefault(item, item)
        while parent[item] != item:
            parent[item] = parent[parent[item]]
            item = parent[item]
        return item

    def union(left: str, right: str) -> None:
        left, right = find(left), find(right)
        if left != right:
            parent[right] = left

    for row in raw:
        groups = row["group_id"].split("|")
        for group in groups[1:]:
            union(groups[0], group)
        find(groups[0])
    components: dict[str, list[str]] = {}
    for group in parent:
        components.setdefault(find(group), []).append(group)
    role_by_group = {}
    for groups in components.values():
        role = "dev_select" if int(stable_key("|".join(sorted(groups)))[:8], 16) % 5 == 0 else "dev_train"
        role_by_group.update({group: role for group in groups})
    result = []
    for row in raw:
        source = old / row["path"]
        voice_fake = "" if row["source"].startswith("mixture:") and row["VOICE_FAKE"] == "0" and row["MUSIC_FAKE"] == "1" else row["VOICE_FAKE"]
        result.append(ManifestRow(row["ID"], f"audio/{row['ID']}.wav", row["group_id"], role_by_group[row["group_id"].split("|")[0]], row["source"], row["FILE_FAKE"], voice_fake, row["MUSIC_FAKE"], row["VOICE_PRESENT"], row["MUSIC_PRESENT"], pcm_sha256(decode_audio(source))))
    return result


def _role_for_source(item: dict[str, str | dict[str, str]], test_voice_generators: set[str]) -> str:
    provider, kind, group = str(item["provider"]), str(item["source_kind"]), str(item["group_id"])
    if provider in test_voice_generators | {"suno", "udio", "elevenlabs"}:
        return "test_generator"
    bucket = int(stable_key(group)[:8], 16) % 20
    if bucket < 3:
        return "test_source"
    if bucket < 6:
        return "dev_select"
    return "dev_train"


def _select_new(items: list[dict[str, str | dict[str, str]]], test_voice_generators: set[str]) -> list[tuple[dict[str, str | dict[str, str]], str]]:
    music_test_groups = {str(item["group_id"]) for item in items if item["provider"] in {"suno", "udio", "elevenlabs"}}
    voice_test_groups = {str(item["group_id"]) for item in items if item["provider"] in test_voice_generators}
    generator_test_groups = music_test_groups | voice_test_groups
    categorized: dict[str, list[dict[str, str | dict[str, str]]]] = {role: [] for role in ROLES}
    for item in items:
        kind, group = str(item["source_kind"]), str(item["group_id"])
        role = "test_generator" if group in generator_test_groups else _role_for_source(item, test_voice_generators)
        categorized[role].append(item)
    selected: list[tuple[dict[str, str | dict[str, str]], str]] = []
    for item in items:
        role = "test_generator" if str(item["group_id"]) in generator_test_groups else _role_for_source(item, test_voice_generators)
        selected.append((item, role))
    return selected


def _copy_rows(selected: list[tuple[dict[str, str | dict[str, str]], str]], sources: Path, output: Path) -> list[ManifestRow]:
    rows = []
    for item, role in selected:
        source = sources.parent / str(item["prepared_path"])
        identifier = str(item["id"])
        destination = output / "audio" / f"{identifier}.wav"
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, destination)
        labels = item["labels"]
        if not isinstance(labels, dict):
            raise SplitIntegrityError("inventory labels must be an object")
        rows.append(ManifestRow(identifier, f"audio/{identifier}.wav", str(item["group_id"]), role, f"{item['source_kind']}/{item['provider']}", *(str(labels[name]) for name in FIELDS[5:]), str(item["pcm_sha256"])))
    return rows


def pure_voice_groups(rows: list[ManifestRow], role: str) -> list[str]:
    return sorted({row.group_id for row in rows if row.split == role and row.VOICE_PRESENT == "1" and row.MUSIC_PRESENT == "0"}, key=stable_key)


def paired_music_groups(rows: list[ManifestRow], role: str) -> list[str]:
    groups = {row.group_id for row in rows if row.split == role and row.MUSIC_PRESENT == "1" and row.VOICE_PRESENT == ""}
    return sorted((group for group in groups if {row.MUSIC_FAKE for row in rows if row.group_id == group} == {"0", "1"}), key=stable_key)


def mixture_voice_label(voice_fake: str, music_fake: str) -> str:
    return "" if voice_fake == "0" and music_fake == "1" else voice_fake


def _mixtures(rows: list[ManifestRow], output: Path) -> list[ManifestRow]:
    result = []
    for role in ROLES:
        voice_groups = pure_voice_groups(rows, role)[:2]
        music_groups = paired_music_groups(rows, role)[:2]
        if len(voice_groups) < 2 or len(music_groups) < 2:
            raise SplitIntegrityError(f"mixture bank requires two voice/music groups in {role}")
        for voice_group, music_group in zip(voice_groups, music_groups, strict=True):
            for voice_fake in ("0", "1"):
                voice = next(row for row in rows if row.group_id == voice_group and row.VOICE_FAKE == voice_fake)
                for music_fake in ("0", "1"):
                    music = next(row for row in rows if row.group_id == music_group and row.MUSIC_FAKE == music_fake)
                    foreground, background = decode_audio(output / voice.path), decode_audio(output / music.path)
                    length = min(len(foreground), len(background))
                    foreground, background = foreground[:length], background[:length]
                    for snr in (0, 10):
                        gain = np.sqrt(np.mean(foreground ** 2) / max(float(np.mean(background ** 2)), 1e-12)) / 10 ** (snr / 20)
                        audio = foreground + gain * background
                        identifier = stable_key(f"mix|{voice.ID}|{music.ID}|{snr}")[:20]
                        path = output / "audio" / f"{identifier}.wav"
                        write_wav(path, audio)
                        result.append(ManifestRow(identifier, f"audio/{identifier}.wav", voice_group + "|" + music_group, role, f"mixture/SNR={snr}", str(max(int(voice_fake), int(music_fake))), mixture_voice_label(voice_fake, music_fake), music_fake, "1", "1", pcm_sha256(decode_audio(path))))
    return result


def _label_audit(rows: list[ManifestRow]) -> dict[str, dict[str, list[str] | int]]:
    report = {}
    for name in FIELDS[5:]:
        values = [getattr(row, name) for row in rows]
        report[name] = {"classes": sorted(set(values) - {""}), "unknown": values.count("")}
    return report


def connected_count(rows: list[ManifestRow], role: str) -> int:
    parent: dict[str, str] = {}

    def find(item: str) -> str:
        parent.setdefault(item, item)
        while parent[item] != item:
            parent[item] = parent[parent[item]]
            item = parent[item]
        return item

    for row in rows:
        if row.split != role:
            continue
        groups = row.group_id.split("|")
        root = find(groups[0])
        for group in groups[1:]:
            parent[find(group)] = root
    return len({find(group) for group in parent})


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sources", type=Path, required=True)
    parser.add_argument("--old", type=Path, default=ROOT / "validation")
    parser.add_argument("--output", type=Path, default=ROOT / "v2_validation")
    parser.add_argument("--zip", dest="zip_path", type=Path, default=ROOT / "_workspace/v2_validation.zip")
    parser.add_argument("--audit", type=Path)
    args = parser.parse_args()
    inventory = json.loads(args.sources.read_text())
    output: Path = args.output.resolve()
    old_rows = _old_rows(args.old.resolve())
    copied_old = []
    for row in old_rows:
        destination = output / row.path
        destination.parent.mkdir(parents=True, exist_ok=True)
        write_wav(destination, decode_audio(args.old.resolve() / "audio" / f"{row.ID}.wav"))
        copied_old.append(replace(row, pcm_sha256=pcm_sha256(decode_audio(destination))))
    test_voice_generators = set(inventory["acquisition"]["test_voice_generators"])
    rows = copied_old + _copy_rows(_select_new(inventory["sources"], test_voice_generators), args.sources.resolve(), output)
    rows += _mixtures(rows, output)
    audit_rows(rows)
    used_paths = {row.path for row in rows}
    for path in (output / "audio").glob("*"):
        if str(path.relative_to(output)) not in used_paths:
            path.unlink()
    development = sorted([row for row in rows if row.split.startswith("dev_")], key=lambda row: row.ID)
    locked = sorted([row for row in rows if row.split.startswith("test_")], key=lambda row: row.ID)
    if not 400 <= len(rows) <= 1100:
        raise SplitIntegrityError(f"final bank has {len(rows)} rows outside [400, 1100]")
    if any(connected_count(locked, role) < 20 for role in ("test_generator", "test_source")):
        raise SplitIntegrityError("each challenge requires at least 20 connected lineage components")
    labels = _label_audit(locked)
    if any(value["classes"] != ["0", "1"] for value in labels.values()):
        raise SplitIntegrityError("combined locked challenge lacks both classes for an official target")
    development_path = output / "development/manifest.csv"
    locked_path = output / "locked_test/manifest.csv"
    write_manifest(development_path, development)
    write_manifest(locked_path, locked)
    pcm_path = output / "pcm_sha256.csv"
    with pcm_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=("ID", "pcm_sha256"))
        writer.writeheader()
        writer.writerows({"ID": row.ID, "pcm_sha256": row.pcm_sha256} for row in sorted(rows, key=lambda item: item.ID))
    audit = {"rows": len(rows), "mixtures": sum(row.source.startswith("mixture/") for row in rows), "role_counts": {role: sum(row.split == role for row in rows) for role in ROLES}, "atomic_groups": {role: len({part for row in rows if row.split == role for part in row.group_id.split("|")}) for role in ROLES}, "connected_components": {role: connected_count(rows, role) for role in ROLES}, "voice_authenticity_masked_for_generated_music": sum(row.VOICE_FAKE == "" and row.MUSIC_FAKE == "1" and row.VOICE_PRESENT == "1" for row in rows), "locked_label_coverage": labels, "development_manifest_sha256": sha256_file(development_path), "locked_manifest_sha256": sha256_file(locked_path), "pcm_index_sha256": sha256_file(pcm_path)}
    (output / "split_audit.json").write_text(json.dumps(audit, indent=2) + "\n")
    if args.audit:
        args.audit.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(output / "split_audit.json", args.audit)
    args.zip_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(args.zip_path, "w", zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(output.rglob("*")):
            if path.is_file():
                info = zipfile.ZipInfo("v2_validation/" + str(path.relative_to(output)), (2026, 9, 13, 0, 0, 0))
                info.compress_type = zipfile.ZIP_DEFLATED
                archive.writestr(info, path.read_bytes())
        info = zipfile.ZipInfo("v2_validation/DATA_SOURCES_V2.md", (2026, 9, 13, 0, 0, 0))
        info.compress_type = zipfile.ZIP_DEFLATED
        archive.writestr(info, (ROOT / "DATA_SOURCES_V2.md").read_bytes())
    print(json.dumps({**audit, "zip": str(args.zip_path), "zip_sha256": sha256_file(args.zip_path)}))


if __name__ == "__main__":
    main()
