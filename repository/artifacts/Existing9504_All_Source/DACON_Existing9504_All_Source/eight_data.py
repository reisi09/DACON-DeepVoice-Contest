"""Build the provenance-regrouped CPU dataset for the approved all-eight study.

Run: .venv/bin/python eight_data.py prepare|check [repository-root]
"""
from __future__ import annotations

import csv
import hashlib
import json
import shutil
import sys
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Final

import numpy as np

from eight_audio import core, deterministic_noise, load, manifest_row, partial_rows, semantic_audio_check, transform, write
from eight_contract import ContractError, ManifestRow, adversarial_check, read_manifest, sha256_file, validate_manifest, write_manifest

ROOT: Final = Path(__file__).resolve().parent
LABELS: Final = ("FILE_FAKE", "VOICE_FAKE", "MUSIC_FAKE", "VOICE_PRESENT", "MUSIC_PRESENT")


@dataclass(frozen=True, slots=True)
class Source:
    identifier: str
    path: Path
    source_hash: str
    atoms: tuple[str, ...]
    source: str
    generator: str
    kind: str
    labels: tuple[str, str, str, str, str]
    historical: bool


def _speaker(value: str) -> str:
    return "speaker:" + value.removeprefix("mlaad_").replace(":", "/")


def _sources(repo: Path) -> list[Source]:
    sources: list[Source] = []
    prepared = repo / "_workspace/redesign_20260917/input-preparation"
    for name in ("train", "validation", "sealed"):
        for row in csv.DictReader((prepared / f"prepared/{name}.csv").open()):
            if row["source"] == "nsynth":
                continue
            parents = json.loads(row["parent_ids"])
            atoms = {"legacy:" + row["group_id"], *("reference:" + value for value in parents)}
            if row["artist_id"]:
                atoms.add("artist:fma:" + row["artist_id"])
            if row["speaker_id"]:
                atoms.add(_speaker(row["speaker_id"]))
            sources.append(Source(row["ID"], prepared / row["path"], row["audio_sha256"], tuple(sorted(atoms)),
                                  row["source"], row["generator"], row["kind"],
                                  tuple(row[key] for key in LABELS), True))
    manifests = (
        repo / "results/six_h100_20260917/fresh_fma_manifest.csv",
        repo / "results/voice_music_h100_20260917/fresh_fma_manifest.csv",
        repo / "results/six_h100_20260917/fresh_sonics_manifest.csv",
        repo / "results/voice_music_h100_20260917/fresh_sonics_manifest.csv",
        repo / "results/six_h100_20260917/fresh_speech_manifest.csv",
    )
    for manifest in manifests:
        for row in csv.DictReader(manifest.open()):
            path = repo / row["path"]
            atoms: set[str] = set()
            if row["source"] == "FMA":
                atoms.add("artist:fma:" + row["artist_id"])
                atoms.add("reference:fma:" + row["track_id"])
                generator, kind = "", "mixed"
            elif row["source"] == "MLAAD":
                atoms.update({_speaker(row["group_id"]), "reference:mlaad:" + row["original_file"]})
                generator, kind = row["provider"], "speech"
            else:
                atoms.add("reference:sonics:" + row["id"])
                generator, kind = row["source"], "mixed"
            sources.append(Source(row["ID"], path, row["source_sha256"], tuple(sorted(atoms)), row["source"],
                                  generator, kind, tuple(row.get(key, "") for key in LABELS), False))
    retained: list[Source] = []
    seen: set[str] = set()
    for source in sources:
        if source.source_hash not in seen:
            if sha256_file(source.path) != source.source_hash:
                raise ContractError(f"source changed: {source.path}")
            retained.append(source)
            seen.add(source.source_hash)
    return retained


def _split_sources(sources: list[Source]) -> tuple[dict[int, str], dict[int, int]]:
    parent = list(range(len(sources)))
    def find(item: int) -> int:
        while parent[item] != item:
            parent[item] = parent[parent[item]]
            item = parent[item]
        return item
    owners: dict[str, int] = {}
    for index, source in enumerate(sources):
        for atom in source.atoms:
            owner = owners.setdefault(atom, index)
            a, b = find(index), find(owner)
            if a != b:
                parent[b] = a
    components = {index: find(index) for index in range(len(sources))}
    atoms_by_component: dict[int, set[str]] = defaultdict(set)
    for index, component in components.items():
        atoms_by_component[component].update(sources[index].atoms)
    speaker_components = {component for component, atoms in atoms_by_component.items()
                          if any(atom.startswith("speaker:") for atom in atoms)}
    fresh = sorted((component for component in speaker_components
                    if any(not sources[index].historical for index, value in components.items() if value == component)),
                   key=lambda value: hashlib.sha256("|".join(sorted(atoms_by_component[value])).encode()).hexdigest())
    historical = sorted(speaker_components - set(fresh),
                        key=lambda value: hashlib.sha256("|".join(sorted(atoms_by_component[value])).encode()).hexdigest())
    forced = dict(zip(fresh, ("train", "calibration", "dev", "test"), strict=True))
    forced.update(dict(zip(historical, ("train", "train", "calibration", "calibration", "dev", "test"), strict=True)))
    splits: dict[int, str] = {}
    for component, atoms in atoms_by_component.items():
        fraction = int(hashlib.sha256("|".join(sorted(atoms)).encode()).hexdigest()[:8], 16) / 2**32
        splits[component] = forced.get(component, "train" if fraction < .60 else "calibration" if fraction < .75
                                       else "dev" if fraction < .875 else "test")
    return splits, components


def _components(rows: list[ManifestRow]) -> int:
    owners: dict[str, int] = {}
    parent = list(range(len(rows)))
    def find(item: int) -> int:
        while parent[item] != item:
            parent[item] = parent[parent[item]]
            item = parent[item]
        return item
    for index, row in enumerate(rows):
        for atom in row["group_id"].split("|"):
            owner = owners.setdefault(atom, index)
            a, b = find(index), find(owner)
            if a != b:
                parent[b] = a
    return len({find(index) for index in range(len(rows))})


def _write_sources(path: Path, sources: list[Source]) -> None:
    fields = ("ID", "source_sha256", "group_id", "source", "generator", "kind", "historical")
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for source in sources:
            writer.writerow(dict(zip(fields, (source.identifier, source.source_hash, "|".join(source.atoms), source.source,
                                              source.generator, source.kind, str(source.historical).lower()), strict=True)))


def _base(repo: Path, dataset: Path) -> tuple[list[ManifestRow], list[Source]]:
    sources = _sources(repo)
    splits, components = _split_sources(sources)
    rows: list[ManifestRow] = []
    started = time.monotonic()
    for index, source in enumerate(sources):
        split = splits[components[index]]
        identifier = "core_" + hashlib.sha256((source.identifier + source.source_hash).encode()).hexdigest()[:20]
        audio = core(source.path)
        base_path = dataset / "audio" / f"{identifier}_base.wav"
        write(base_path, audio)
        pair = "codec:" + identifier
        rows.append(manifest_row(identifier + "_base", base_path, [source.identifier], set(source.atoms), split, source.source,
                         source.generator, source.kind, "base", "core", pair, source.labels, [], dataset))
        codecs = ("telephone", "mp3_32k") if split == "train" else ("telephone", "mp3_32k", "opus_telephone")
        for codec in codecs:
            path = dataset / "audio" / f"{identifier}_{codec}.wav"
            write(path, transform(audio, codec))
            rows.append(manifest_row(identifier + "_" + codec, path, [source.identifier], set(source.atoms), split,
                             source.source, source.generator, source.kind, codec, "core", pair, source.labels, [], dataset))
        if (index + 1) % 100 == 0 or index + 1 == len(sources):
            print(json.dumps({"stage": "core_codecs", "done": index + 1, "total": len(sources),
                              "elapsed_seconds": round(time.monotonic() - started, 1)}), flush=True)
    return rows, sources


def _swaps(rows: list[ManifestRow], dataset: Path) -> list[ManifestRow]:
    base = [row for row in rows if row["variant"] == "base"]
    additions: list[ManifestRow] = []
    selected: dict[tuple[str, str], ManifestRow] = {}
    for row in sorted((item for item in base if item["kind"] == "speech"), key=lambda item: item["ID"]):
        speaker = next(atom for atom in row["group_id"].split("|") if atom.startswith("speaker:"))
        selected.setdefault((speaker, row["VOICE_FAKE"]), row)
    per_split: dict[str, int] = defaultdict(int)
    for vocal in sorted(selected.values(), key=lambda row: (row["split"], row["ID"])):
        same = sorted((row for row in base if row["split"] == vocal["split"] and row["source"] == "FMA"),
                      key=lambda row: row["group_id"])
        start = per_split[vocal["split"]]
        backings = (same[start % len(same)], same[(start + 1) % len(same)])
        per_split[vocal["split"]] += 2
        voice = load(dataset / vocal["path"])
        fake_music = sorted((row for row in base if row["split"] == vocal["split"] and row["MUSIC_FAKE"] == "1"),
                            key=lambda row: row["group_id"])
        fake_backing = fake_music[(per_split[vocal["split"]] // 2 - 1) % len(fake_music)]
        candidates = (("fma_a", load(dataset / backings[0]["path"])[:len(voice)], backings[0], ("0", "1")),
                      ("fma_b", load(dataset / backings[1]["path"])[:len(voice)], backings[1], ("0", "1")),
                      ("sonics_fake", load(dataset / fake_backing["path"])[:len(voice)], fake_backing, ("1", "1")))
        for background_name, background, backing, music in (*candidates,
            ("noise", deterministic_noise(len(voice), vocal["ID"]), None, ("", "0"))):
            extra_atoms = set(backing["group_id"].split("|")) if backing else set()
            voice_rms = max(float(np.sqrt(np.mean(voice ** 2))), 1e-8)
            background = background * (voice_rms / max(float(np.sqrt(np.mean(background ** 2))), 1e-8))
            for level, ratio in (("minus6dB", .501187), ("0dB", 1.0)):
                audio = voice + background * ratio
                identifier = f"e2_{vocal['ID']}_{background_name}_{level}"
                path = dataset / "audio" / (identifier + ".wav")
                write(path, audio)
                exact_voice = background_name == "noise"
                voice_fake = vocal["VOICE_FAKE"] if exact_voice or vocal["VOICE_FAKE"] == "1" else ""
                file_fake = "1" if music[0] == "1" else vocal["FILE_FAKE"]
                labels = (file_fake, voice_fake, music[0], "1", music[1])
                additions.append(manifest_row(identifier, path, [vocal["ID"], backing["ID"]] if backing else [vocal["ID"]],
                                      set(vocal["group_id"].split("|")) | extra_atoms, vocal["split"], "constructed:E2",
                                      "", "mixed", f"swap_{background_name}_{level}", "swap", "e2:" + vocal["ID"],
                                      labels, [], dataset))
    return additions


def prepare(root: str | Path = ROOT) -> dict[str, int | dict[str, int]]:
    repo = Path(root)
    remote = repo / "source/dataset"
    if (remote / "manifest.csv").exists():
        return validate_manifest(remote)
    dataset = repo / "_workspace/eight_h100_20260917/dataset"
    if dataset.exists():
        shutil.rmtree(dataset)
    dataset.mkdir(parents=True)
    rows, sources = _base(repo, dataset)
    rows.extend(_swaps(rows, dataset))
    rows.extend(partial_rows(rows, dataset))
    rows.sort(key=lambda row: row["ID"])
    write_manifest(dataset / "manifest.csv", rows)
    _write_sources(dataset / "sources.csv", sources)
    audit = validate_manifest(dataset)
    source_base = [row for row in rows if row["probe_family"] == "core" and row["variant"] == "base"]
    music_base = [row for row in source_base if row["kind"] != "speech"]
    speaker_split = {next(atom for atom in row["group_id"].split("|") if atom.startswith("speaker:")): row["split"]
                     for row in source_base if row["kind"] == "speech"}
    role_counts = {split: {"sources": sum(row["split"] == split for row in source_base),
                           "generators": dict(Counter(row["generator"] or "REAL_OR_UNKNOWN" for row in source_base if row["split"] == split)),
                           "file_labels": dict(Counter(row["FILE_FAKE"] or "unknown" for row in source_base if row["split"] == split)),
                           "voice_unknown": sum(row["split"] == split and row["VOICE_FAKE"] == "" for row in source_base)}
                   for split in ("train", "calibration", "dev", "test")}
    receipt = {**audit, "manifest_sha256": sha256_file(dataset / "manifest.csv"), "source_records": len(sources),
               "connected_source_components": _components(source_base), "connected_music_components": _components(music_base),
               "historically_exposed_source_records": sum(source.historical for source in sources),
               "historical_exposure": "all canonical train/validation/sealed rows are diagnostic; no unbiased M330 claim",
               "prior_project_exposure": "all canonical and fresh-cohort source records were available to prior project work",
               "excluded": {"NSynth": "explicit study exclusion"}, "partial_cells_per_speaker": 36,
               "partial_repeats_per_split": {"train": 2, "calibration": 3, "dev": 2, "test": 2}, "role_counts": role_counts,
               "source_biases": ["canonical Echoes variants share reference parents", "FMA is artist-grouped and vocals are unverified",
                                   "fresh FMA is all real while fresh SONICS is all fake", "MLAAD speech is source-annotated and historically exposed",
                                   "test speech has two connected speakers, so population uncertainty is wide and diagnostic"]}
    (dataset / "data_receipt.json").write_text(json.dumps(receipt, indent=2, sort_keys=True))
    (dataset / "speaker_split.json").write_text(json.dumps(speaker_split, indent=2, sort_keys=True))
    result = repo / "results/eight_h100_20260917"
    result.mkdir(parents=True, exist_ok=True)
    shutil.copy2(dataset / "data_receipt.json", result / "data_receipt.json")
    return audit


def self_check(root: str | Path = ROOT) -> dict[str, bool | int | float]:
    dataset = Path(root) / "_workspace/eight_h100_20260917/dataset"
    validate_manifest(dataset)
    return adversarial_check(dataset) | semantic_audio_check(dataset, read_manifest(dataset / "manifest.csv"))


if __name__ == "__main__":
    command = sys.argv[1] if len(sys.argv) > 1 else "prepare"
    root = Path(sys.argv[2]) if len(sys.argv) > 2 else ROOT
    print(json.dumps(prepare(root) if command == "prepare" else self_check(root), sort_keys=True))
