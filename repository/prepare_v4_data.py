#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy==2.4.6", "imageio-ffmpeg==0.6.0"]
# ///
# How to run: `.venv/bin/python prepare_v4_data.py`.
"""Build full-source V4 training data and fixed disjoint evaluation panels."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
import zipfile
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Final

import numpy as np

from prepare_v2_sources import ECHOES_REV, MLAAD_REV, NSYNTH_REV, _normalized
from v2_data import Labels, decode_audio, pcm_sha256, sha256_file, stable_key, write_wav
from v4_data_acquisition import (
    DEV_HELDOUT,
    SEALED_HELDOUT,
    TRAIN_PROVIDERS,
    acquire_echoes_variant,
    acquire_fma,
    acquire_speech_pair,
    assign_music_roles,
    by_artist,
    load_echoes,
    load_echoes_variants,
    old_music_tracks,
    prior_identities,
    prior_speech_speakers,
    round_robin_providers,
    select_speech_roles,
    select_training_tracks,
    speech_candidates,
)
from v4_data_augmentation import crop_or_tile, load_audio, mix_at_snr, splice, write_derived

ROOT: Final = Path(__file__).resolve().parent
FIELDS: Final = ("ID", "path", "group_id", "split", "source", "FILE_FAKE", "VOICE_FAKE", "MUSIC_FAKE", "VOICE_PRESENT", "MUSIC_PRESENT", "panel")
PRIMARY_DURATIONS: Final = (4, 10, 30, 60)
MAX_PART_BYTES: Final = 95 * 1024 * 1024


class V4IntegrityError(RuntimeError):
    """A built row violates the fixed V4 data contract."""


@dataclass(frozen=True, slots=True)
class SourceSpec:
    ID: str
    raw_path: Path
    group_id: str
    split: str
    source: str
    panel: str
    labels: Labels
    provider: str
    content_id: str
    source_url: str
    revision: str
    license: str
    attribution: str
    artist_id: str = ""
    speaker_id: str = ""
    parents: tuple[str, ...] = ()
    variant_type: str = "source"


@dataclass(frozen=True, slots=True)
class BuiltRow:
    ID: str
    path: str
    full_path: str
    group_id: str
    split: str
    source: str
    panel: str
    FILE_FAKE: str
    VOICE_FAKE: str
    MUSIC_FAKE: str
    VOICE_PRESENT: str
    MUSIC_PRESENT: str
    provider: str
    content_id: str
    source_url: str
    revision: str
    license: str
    attribution: str
    artist_id: str
    speaker_id: str
    parent_ids: tuple[str, ...]
    variant_type: str
    source_sha256: str
    source_pcm_sha256: str
    pcm_sha256: str
    original_duration_seconds: float
    evaluation_duration_seconds: float
    evaluation_offset_seconds: float
    segments: tuple[dict[str, str | float], ...] = ()

    def manifest_record(self) -> dict[str, str]:
        return {name: str(getattr(self, name)) for name in FIELDS}

    def provenance_record(self) -> dict[str, str | float | tuple[str, ...] | tuple[dict[str, str | float], ...]]:
        return asdict(self)


def _labels(file_fake: str, voice_fake: str, music_fake: str, voice_present: str, music_present: str) -> Labels:
    return Labels(file_fake, voice_fake, music_fake, voice_present, music_present)


def _materialize(spec: SourceSpec, output: Path) -> BuiltRow:
    audio = decode_audio(spec.raw_path, limit_seconds=None)
    if len(audio) < 64000:
        raise V4IntegrityError(f"source shorter than four seconds: {spec.raw_path}")
    full = output / "audio/full" / spec.split / f"{spec.ID}.wav"
    write_wav(full, audio)
    stored = decode_audio(full, limit_seconds=None)
    relative = str(full.relative_to(output))
    return BuiltRow(
        spec.ID, relative, relative, spec.group_id, spec.split, spec.source, spec.panel,
        spec.labels.FILE_FAKE, spec.labels.VOICE_FAKE, spec.labels.MUSIC_FAKE,
        spec.labels.VOICE_PRESENT, spec.labels.MUSIC_PRESENT, spec.provider,
        spec.content_id, spec.source_url, spec.revision, spec.license, spec.attribution,
        spec.artist_id, spec.speaker_id, spec.parents, spec.variant_type,
        sha256_file(spec.raw_path), pcm_sha256(audio), pcm_sha256(stored),
        len(audio) / 16000, len(stored) / 16000, 0.0,
    )


def _evaluation_excerpt(row: BuiltRow, output: Path, seconds: float) -> BuiltRow:
    audio = load_audio(output / row.full_path)
    samples = min(len(audio), int(round(seconds * 16000)))
    maximum = len(audio) - samples
    offset = int(stable_key("eval|" + row.ID)[:16], 16) % (maximum + 1)
    excerpt = audio[offset:offset + samples]
    path = output / "audio/eval" / row.split / f"{row.ID}.wav"
    write_wav(path, excerpt)
    stored = load_audio(path)
    return replace(row, path=str(path.relative_to(output)), pcm_sha256=pcm_sha256(stored), evaluation_duration_seconds=len(stored) / 16000, evaluation_offset_seconds=offset / 16000)


def _existing_inventory(root: Path) -> tuple[list[dict[str, str]], dict[tuple[str, str, str], Path]]:
    rows = json.loads((root / "_workspace/v2_sources/source_inventory.json").read_text())["sources"]
    paths: dict[tuple[str, str, str], Path] = {}
    for row in rows:
        key = (str(row["content_id"]), str(row["provider"]), str(row["source_kind"]))
        paths[key] = root / "_workspace/v2_sources" / str(row["original_path"])
    return rows, paths


def _train_specs(root: Path, cache: Path, pairs: list[dict[str, str]], generated: dict[tuple[str, str], dict[str, str]], entries: dict[str, dict[str, int | str]], variants: dict[str, list[dict[str, str]]], excluded_artists: set[str], speaker_lookup: dict[str, str]) -> tuple[list[SourceSpec], dict[str, int | tuple[str, ...]]]:
    inventory, existing = _existing_inventory(root)
    old_tracks = old_music_tracks(root, pairs)
    tracks, included_providers = select_training_tracks(pairs, generated, old_tracks, excluded_artists)
    specs: list[SourceSpec] = []
    for row in inventory:
        if row["source_kind"] in {"fma_real", "echoes_tta"}:
            continue
        labels = Labels(**row["labels"])
        specs.append(SourceSpec(
            str(row["id"]), root / "_workspace/v2_sources" / str(row["original_path"]), str(row["group_id"]), "train", str(row["source_kind"]), "speech" if str(row["source_kind"]).startswith("mlaad") else "music_real", labels, str(row["provider"]), str(row["content_id"]), str(row["source_url"]), str(row["revision"]), str(row["license"]), str(row["attribution"]), speaker_id=speaker_lookup.get(str(row["content_id"]), "") if str(row["source_kind"]).startswith("mlaad") else "", parents=(str(row["group_id"]),),
        ))
    for item in tracks:
        content = item["echoes_original_audio"]
        fma_key = (content, "FMA", "fma_real")
        if fma_key in existing:
            real_path = existing[fma_key]
            real_url = next(row["source_url"] for row in inventory if (row["content_id"], row["provider"], row["source_kind"]) == fma_key)
        else:
            real_path, real_url = acquire_fma(item, cache)
        real_id = stable_key("v4|real|" + content)[:24]
        license_name = item["license_title"] + " " + item["license_url"]
        specs.append(SourceSpec(real_id, real_path, item["group_id"], "train", "FMA/FMA", "music_real", _labels("0", "", "0", "", "1"), "FMA", content, str(real_url), "FMA metadata snapshot", license_name, item["artist_name"], item["artist_id"], parents=(item["group_id"],)))
        for meta in variants[_normalized(content)]:
            if meta["generator"] not in included_providers:
                continue
            source_url = f"Echoes.zip::Echoes/{meta['path_in_dataset']}"
            existing_row = next((row for row in inventory if row["source_url"] == source_url), None)
            if existing_row is None:
                raw, source_url = acquire_echoes_variant(meta, entries, cache)
            else:
                raw = root / "_workspace/v2_sources" / str(existing_row["original_path"])
            identifier = stable_key("v4|" + meta["path_in_dataset"])[:24]
            specs.append(SourceSpec(identifier, raw, item["group_id"], "train", f"Echoes/{meta['generator']}/{meta['type']}", "music_seen", _labels("1", "", "1", "", "1"), meta["generator"], content, source_url, ECHOES_REV, "CC-BY-SA-4.0", f"Echoes; {meta['generator']}", item["artist_id"], parents=(item["group_id"],), variant_type=meta["type"]))
    provider_groups = {provider: len({spec.group_id for spec in specs if spec.provider == provider}) for provider in included_providers}
    if len(tracks) != 70 or any(count < 20 for count in provider_groups.values()):
        raise V4IntegrityError(f"training quota failure: artists={len(tracks)} providers={provider_groups}")
    return specs, {"artists": len(tracks), "providers": included_providers, "provider_groups": provider_groups, "reused_inventory_sources": len(inventory)}


def _music_panel_specs(role: str, artists: list[str], heldout: str, grouped: dict[str, list[dict[str, str]]], generated: dict[tuple[str, str], dict[str, str]], entries: dict[str, dict[str, int | str]], cache: Path) -> tuple[list[SourceSpec], list[dict[str, str | tuple[str, ...]]]]:
    specs: list[SourceSpec] = []
    selections: list[dict[str, str | tuple[str, ...]]] = []
    for index, artist in enumerate(artists):
        viable = [item for item in grouped[artist] if (_normalized(item["echoes_original_audio"]), heldout) in generated and sum((_normalized(item["echoes_original_audio"]), provider) in generated for provider in TRAIN_PROVIDERS) >= 3]
        item = min(viable, key=lambda value: stable_key(value["group_id"] + "|" + value["echoes_original_audio"]))
        seen = round_robin_providers(item, generated, index)
        real, real_url = acquire_fma(item, cache)
        specs.append(SourceSpec(stable_key(f"v4|{role}|real|{item['echoes_original_audio']}")[:24], real, item["group_id"], role, "FMA/FMA", "music_real", _labels("0", "", "0", "", "1"), "FMA", item["echoes_original_audio"], real_url, "FMA metadata snapshot", item["license_title"] + " " + item["license_url"], item["artist_name"], artist, parents=(item["group_id"],)))
        for provider in (*seen, heldout):
            meta = generated[(_normalized(item["echoes_original_audio"]), provider)]
            raw, source_url = acquire_echoes_variant(meta, entries, cache)
            panel = "music_ood" if provider == heldout else "music_seen"
            specs.append(SourceSpec(stable_key(f"v4|{role}|{provider}|{item['echoes_original_audio']}")[:24], raw, item["group_id"], role, f"Echoes/{provider}/TTA", panel, _labels("1", "", "1", "", "1"), provider, item["echoes_original_audio"], source_url, ECHOES_REV, "CC-BY-SA-4.0", f"Echoes; {provider}", artist, parents=(item["group_id"],), variant_type="TTA"))
        selections.append({"role": role, "artist_id": artist, "group_id": item["group_id"], "content_id": item["echoes_original_audio"], "heldout": heldout, "seen_providers": seen})
    return specs, selections


def _speech_panel_specs(root: Path, cache: Path, excluded_groups: set[str], excluded_speakers: set[str]) -> tuple[list[SourceSpec], dict[str, list[tuple[str, str, str, str, str, str]]], list[str]]:
    candidates = speech_candidates(root, cache, excluded_groups, excluded_speakers)
    initial_dev, initial_sealed = select_speech_roles(candidates)
    specs: list[SourceSpec] = []
    roles: dict[str, list[tuple[str, str, str, str, str, str]]] = {}
    skips: list[str] = []
    for role, initial in (("dev", initial_dev), ("sealed", initial_sealed)):
        speakers = {row[3] for row in initial}
        pool = sorted((row for row in candidates if row[3] in speakers), key=lambda row: stable_key(row[2] + "|" + row[0]))
        rows: list[tuple[str, str, str, str, str, str]] = []
        used_groups: set[str] = set()
        for candidate in pool:
            provider, filename, group, speaker, real_url, fake_url = candidate
            if group in used_groups:
                continue
            real, fake = acquire_speech_pair(candidate, cache)
            if min(len(decode_audio(real, limit_seconds=None)), len(decode_audio(fake, limit_seconds=None))) < 64000:
                skips.append(f"{role},{provider},{filename},decoded shorter than 4 seconds")
                continue
            rows.append(candidate)
            used_groups.add(group)
            if len(rows) == 20:
                break
        if len(rows) != 20:
            raise V4IntegrityError(f"only {len(rows)} valid speech pairs for {role}")
        roles[role] = rows
        for provider, filename, group, speaker, real_url, fake_url in rows:
            real, fake = acquire_speech_pair((provider, filename, group, speaker, real_url, fake_url), cache)
            specs.extend((
                SourceSpec(stable_key(f"v4|{role}|speech-real|{filename}")[:24], real, group, role, "MLAAD/M-AILABS", "speech", _labels("0", "0", "", "1", "0"), "M-AILABS", filename, real_url, MLAAD_REV, "M-AILABS permissive notice", filename, speaker_id=speaker, parents=(group,)),
                SourceSpec(stable_key(f"v4|{role}|speech-fake|{provider}|{filename}")[:24], fake, group, role, f"MLAAD/{provider}", "speech", _labels("1", "1", "", "1", "0"), provider, filename, fake_url, MLAAD_REV, "CC-BY-NC-4.0", provider, speaker_id=speaker, parents=(group,)),
            ))
    return specs, roles, skips


def _group_rows(rows: list[BuiltRow], split: str, prefix: str) -> dict[str, list[BuiltRow]]:
    grouped: dict[str, list[BuiltRow]] = defaultdict(list)
    for row in rows:
        if row.split == split and row.source.startswith(prefix):
            grouped[row.group_id].append(row)
    return grouped


def _derivatives(rows: list[BuiltRow], output: Path, role: str) -> list[BuiltRow]:
    music = _group_rows(rows, role, "FMA/")
    for row in rows:
        if row.split == role and row.source.startswith("Echoes/"):
            music[row.group_id].append(row)
    speech = _group_rows(rows, role, "MLAAD/")
    music_groups = sorted(music, key=stable_key)
    speech_groups = sorted(speech, key=stable_key)
    derived: list[BuiltRow] = []
    snrs = (-5.0, 0.0, 5.0)
    for index, speech_group in enumerate(speech_groups):
        music_group = music_groups[index % len(music_groups)]
        music_real = next(row for row in music[music_group] if row.MUSIC_FAKE == "0")
        music_fake = next(row for row in music[music_group] if row.panel == "music_ood")
        voice_real = next(row for row in speech[speech_group] if row.VOICE_FAKE == "0")
        voice_fake = next(row for row in speech[speech_group] if row.VOICE_FAKE == "1")
        for tag, music_row, voice_row in (("rr", music_real, voice_real), ("rf", music_real, voice_fake), ("fr", music_fake, voice_real), ("ff", music_fake, voice_fake)):
            identifier = stable_key(f"v4|{role}|mix|{index}|{tag}")[:24]
            a = crop_or_tile(load_audio(output / music_row.full_path), 160000, identifier + "m")
            b = crop_or_tile(load_audio(output / voice_row.full_path), 160000, identifier + "v")
            path = output / "audio/eval" / role / f"{identifier}.wav"
            write_derived(path, mix_at_snr(a, b, snrs[index % 3]))
            music_fake_label = music_row.MUSIC_FAKE
            voice_fake_label = "1" if voice_row.VOICE_FAKE == "1" else ("0" if music_fake_label == "0" else "")
            file_fake = "1" if "1" in {music_fake_label, voice_fake_label} else "0"
            groups = tuple(sorted({music_group, speech_group}))
            stored = load_audio(path)
            derived.append(BuiltRow(identifier, str(path.relative_to(output)), "", "|".join(groups), role, f"mixture/{tag}", "mixture", file_fake, voice_fake_label, music_fake_label, "1", "1", "controlled", identifier, "local controlled mixture", "v4 protocol", "parent licenses", f"SNR {snrs[index % 3]:g} dB", "", "", (music_row.ID, voice_row.ID), "mixture", sha256_file(path), pcm_sha256(stored), pcm_sha256(stored), 10.0, 10.0, 0.0))
    for index in range(20):
        first_group = music_groups[index % len(music_groups)]
        second_group = music_groups[(index + 1) % len(music_groups)]
        first_real = next(row for row in music[first_group] if row.MUSIC_FAKE == "0")
        second_real = next(row for row in music[second_group] if row.MUSIC_FAKE == "0")
        second_fake = next(row for row in music[second_group] if row.panel == "music_ood")
        for tag, second, fake in (("real_real", second_real, "0"), ("real_fake", second_fake, "1")):
            identifier = stable_key(f"v4|{role}|splice|{index}|{tag}")[:24]
            audio, boundary = splice(load_audio(output / first_real.full_path), load_audio(output / second.full_path), 480000)
            path = output / "audio/eval" / role / f"{identifier}.wav"
            write_derived(path, audio)
            stored = load_audio(path)
            groups = tuple(sorted({first_group, second_group}))
            segments = ({"start_seconds": 0.0, "end_seconds": boundary / 16000, "MUSIC_FAKE": "0", "VOICE_FAKE": ""}, {"start_seconds": boundary / 16000, "end_seconds": 30.0, "MUSIC_FAKE": fake, "VOICE_FAKE": ""})
            derived.append(BuiltRow(identifier, str(path.relative_to(output)), "", "|".join(groups), role, f"splice/{tag}", "splice", fake, "", fake, "", "1", "controlled", identifier, "local controlled splice", "v4 protocol", "parent licenses", "50 ms equal-power-free linear crossfade", "", "", (first_real.ID, second.ID), "splice", sha256_file(path), pcm_sha256(stored), pcm_sha256(stored), 30.0, 30.0, 0.0, segments))
    return derived


def validate_rows(rows: list[BuiltRow], prior_groups: set[str], dev_artists: set[str], sealed_artists: set[str], train_providers: tuple[str, ...]) -> dict[str, int | dict[str, int]]:
    ids = [row.ID for row in rows]
    if len(ids) != len(set(ids)):
        raise V4IntegrityError("duplicate IDs")
    role_groups = {role: {part for row in rows if row.split == role for part in row.group_id.split("|")} for role in ("train", "dev", "sealed")}
    if role_groups["dev"] & role_groups["sealed"] or dev_artists & sealed_artists:
        raise V4IntegrityError("development and sealed parent groups overlap")
    if (role_groups["dev"] | role_groups["sealed"]) & role_groups["train"]:
        raise V4IntegrityError("fresh panel parent group leaked into training")
    if (role_groups["dev"] | role_groups["sealed"]) & prior_groups:
        raise V4IntegrityError("fresh panels overlap V2/V3 parent groups")
    if any(row.split == "train" and row.provider in {DEV_HELDOUT, SEALED_HELDOUT} for row in rows):
        raise V4IntegrityError("held-out music provider leaked into training")
    provider_groups = {provider: len({row.group_id for row in rows if row.split == "train" and row.provider == provider}) for provider in train_providers}
    if len(provider_groups) < 6 or any(count < 20 for count in provider_groups.values()):
        raise V4IntegrityError(f"post-acquisition provider quota failed: {provider_groups}")
    for role in ("dev", "sealed"):
        counts = Counter(row.panel for row in rows if row.split == role)
        expected = {"music_real": 10, "music_seen": 30, "music_ood": 10, "speech": 40, "mixture": 80, "splice": 40}
        if dict(counts) != expected:
            raise V4IntegrityError(f"{role} panel counts {dict(counts)} != {expected}")
    if any(row.MUSIC_FAKE == "1" and row.VOICE_FAKE == "0" and row.panel in {"mixture", "splice"} for row in rows):
        raise V4IntegrityError("generated music cannot establish VOICE_FAKE=0")
    return {"files": len(rows), "splits": dict(Counter(row.split for row in rows)), "train_provider_groups": provider_groups}


def _write_csv(rows: list[BuiltRow], output: Path, split: str) -> None:
    with (output / f"{split}.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(row.manifest_record() for row in sorted((item for item in rows if item.split == split), key=lambda item: item.ID))


def _archive(paths: list[Path], archive: Path, root: Path) -> dict[str, str | int | list[dict[str, str | int]]]:
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED, compresslevel=6, allowZip64=True) as handle:
        for path in sorted(paths):
            handle.write(path, path.relative_to(root.parent))
    parts: list[dict[str, str | int]] = []
    with archive.open("rb") as source:
        index = 0
        while block := source.read(MAX_PART_BYTES):
            part = archive.with_suffix(archive.suffix + f".part{index:03d}")
            part.write_bytes(block)
            parts.append({"name": part.name, "path": str(part), "bytes": len(block), "sha256": sha256_file(part)})
            index += 1
    return {"name": archive.name, "bytes": archive.stat().st_size, "sha256": sha256_file(archive), "parts": parts}


def _write_archives(rows: list[BuiltRow], output: Path, workspace: Path) -> list[dict[str, str | int | list[dict[str, str | int]]]]:
    workspace.mkdir(parents=True, exist_ok=True)
    results = []
    for split in ("train", "dev", "sealed"):
        relative_paths = {row.path for row in rows if row.split == split}
        relative_paths.update(row.full_path for row in rows if row.split == split and row.full_path)
        files = [output / f"{split}.csv"] + [output / path for path in relative_paths]
        results.append(_archive(files, workspace / f"v4_{split}.zip", output))
    metadata = [path for path in output.rglob("*") if path.is_file() and "audio" not in path.relative_to(output).parts]
    results.append(_archive(metadata, workspace / "v4_metadata.zip", output))
    return results


def _protocol(root: Path, output: Path, dev: list[str], sealed: list[str], train_info: dict[str, int | tuple[str, ...]], source_artifacts: list[Path]) -> None:
    ast = json.loads((root / "_workspace/v2_discovery/ast_assets_receipt.json").read_text())
    protocol = {
        "schema_version": 4,
        "selection_basis": "source availability, licenses, hashes and parent identity only; no model predictions",
        "music_roles": {"dev_provider": DEV_HELDOUT, "sealed_provider": SEALED_HELDOUT, "dev_artists": dev, "sealed_artists": sealed, "assignment": "exclusive-first then SHA-256 ordered shared artists", "training": train_info},
        "exclusions": {"prior_sources": "V2 source_inventory.json plus V3 recovery provenance", "heldout_from_training": [DEV_HELDOUT, SEALED_HELDOUT], "derivatives_inherit_parent_union": True},
        "labels": {"unknown": "", "rule": "source metadata and controlled parent logic only", "generated_song_voice": "unknown unless fake speech is retained"},
        "evaluation": {"primary_seconds": PRIMARY_DURATIONS, "music_files_per_role": 50, "speech_files_per_role": 40, "mixture_files_per_role": 80, "splice_files_per_role": 40},
        "models": {"ast": {"revision": ast["model_revision"], "asset_sha256": ast["sha256"]}, "sonics": {"checkpoint_revision": "a349d427b681fa15e3d0e22d1708108322f967b1", "source_commit": "9156ffad151f797c71556923c4a02fa01fa8fc91"}},
        "resources": [
            {"name": "Echoes", "revision": ECHOES_REV, "license": "CC-BY-SA-4.0", "permitted_use": "attribution/share-alike research and adaptation"},
            {"name": "FMA", "revision": "local metadata snapshot", "license": "per-track license retained in provenance", "permitted_use": "only license_eligible tracks"},
            {"name": "MLAAD-tiny", "revision": MLAAD_REV, "license": "CC-BY-NC-4.0 generated; M-AILABS original notice", "permitted_use": "non-commercial research"},
            {"name": "NSynth", "revision": NSYNTH_REV, "license": "CC-BY-4.0", "permitted_use": "attribution research and adaptation"},
        ],
        "source_receipts": [{"path": str(path.relative_to(root)), "bytes": path.stat().st_size, "sha256": sha256_file(path)} for path in source_artifacts],
    }
    (output / "protocol.json").write_text(json.dumps(protocol, indent=2, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=ROOT / "v4_validation")
    parser.add_argument("--workspace", type=Path, default=ROOT / "_workspace/v4_data")
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    output, workspace = args.output.resolve(), args.workspace.resolve()
    cache = workspace / "sources"
    output.mkdir(parents=True, exist_ok=True)
    pairs, generated, entries = load_echoes(ROOT)
    variants = load_echoes_variants(ROOT)
    prior_groups, prior_artists, _ = prior_identities(ROOT)
    dev_artists, sealed_artists = assign_music_roles(pairs, generated, prior_artists)
    grouped = by_artist(pairs)
    music_specs: list[SourceSpec] = []
    selections: list[dict[str, str | tuple[str, ...]]] = []
    for role, artists, heldout in (("dev", dev_artists, DEV_HELDOUT), ("sealed", sealed_artists, SEALED_HELDOUT)):
        specs, selected = _music_panel_specs(role, artists, heldout, grouped, generated, entries, cache)
        music_specs.extend(specs)
        selections.extend(selected)
    prior_speakers, prior_speech_rows, speaker_lookup = prior_speech_speakers(ROOT, cache)
    train_specs, train_info = _train_specs(ROOT, cache, pairs, generated, entries, variants, prior_artists | set(dev_artists) | set(sealed_artists), speaker_lookup)
    speech_specs, speech_roles, acquisition_skips = _speech_panel_specs(ROOT, cache, prior_groups | {spec.group_id for spec in train_specs}, prior_speakers)
    all_specs = train_specs + music_specs + speech_specs
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        rows = list(pool.map(lambda spec: _materialize(spec, output), all_specs))
    for role in ("dev", "sealed"):
        groups = sorted({row.group_id for row in rows if row.split == role and row.panel.startswith("music_")}, key=stable_key)
        for index, group in enumerate(groups):
            group_rows = [row for row in rows if row.split == role and row.group_id == group and row.panel.startswith("music_")]
            seconds = min(float(PRIMARY_DURATIONS[index % len(PRIMARY_DURATIONS)]), *(row.original_duration_seconds for row in group_rows))
            for row in group_rows:
                rows[rows.index(row)] = _evaluation_excerpt(row, output, seconds)
    rows.extend(_derivatives(rows, output, "dev"))
    rows.extend(_derivatives(rows, output, "sealed"))
    integrity = validate_rows(rows, prior_groups, set(dev_artists), set(sealed_artists), tuple(train_info["providers"]))
    for split in ("train", "dev", "sealed"):
        _write_csv(rows, output, split)
    source_artifacts = [ROOT / "_workspace/v2_sources/source_inventory.json", ROOT / "_workspace/v2_discovery/echoes_fma_unique.json", ROOT / "_workspace/v2_discovery/archive_probes/echoes_manifest.csv", ROOT / "_workspace/v2_discovery/archive_probes/echoes_entries.json", ROOT / "_workspace/v2_discovery/nsynth_repo.json", ROOT / "_workspace/v2_discovery/ast_assets_receipt.json", ROOT / "_workspace/recovery_v3/sonics_repo_tree.json", ROOT / "_workspace/v4_data/source_receipts/ECHOES_README.md", ROOT / "_workspace/v2_sources/licenses/MLAAD_LICENSE", ROOT / "_workspace/v2_sources/licenses/MLAAD_original_LICENSE", ROOT / "_workspace/v2_sources/licenses/MLAAD_README.md"]
    license_dir = output / "licenses"
    license_dir.mkdir(exist_ok=True)
    for path in source_artifacts[-4:]:
        shutil.copy2(path, license_dir / path.name)
    _protocol(ROOT, output, dev_artists, sealed_artists, train_info, source_artifacts)
    provenance = {"schema_version": 4, "selection": selections, "speech_roles": {role: [{"provider": row[0], "filename": row[1], "group_id": row[2], "speaker_id": row[3]} for row in role_rows] for role, role_rows in speech_roles.items()}, "acquisition_skips": acquisition_skips, "sources": [row.provenance_record() for row in sorted(rows, key=lambda item: item.ID)]}
    (output / "provenance.json").write_text(json.dumps(provenance, indent=2, ensure_ascii=False) + "\n")
    (output / "integrity_report.json").write_text(json.dumps(integrity, indent=2) + "\n")
    dev_speakers = {row[3] for row in speech_roles["dev"]}
    sealed_speakers = {row[3] for row in speech_roles["sealed"]}
    data_protocol_audit = {"resources_with_pins_and_terms": 4, "selection_uses_predictions": False, "dev_artists": len(dev_artists), "sealed_artists": len(sealed_artists), "role_artist_overlap": sorted(set(dev_artists) & set(sealed_artists)), "prior_artist_overlap": sorted((set(dev_artists) | set(sealed_artists)) & prior_artists), "prior_speech_rows_with_metadata": prior_speech_rows, "prior_speakers": sorted(prior_speakers), "dev_speakers": sorted(dev_speakers), "sealed_speakers": sorted(sealed_speakers), "speech_speaker_overlap": sorted((dev_speakers & sealed_speakers) | (dev_speakers & prior_speakers) | (sealed_speakers & prior_speakers)), "source_receipts": len(source_artifacts)}
    (output / "data_protocol_audit.json").write_text(json.dumps(data_protocol_audit, indent=2) + "\n")
    (output / "data_integrity.json").write_text(json.dumps({**integrity, "manifest_sha256": {split: sha256_file(output / f"{split}.csv") for split in ("train", "dev", "sealed")}, "provenance_sha256": sha256_file(output / "provenance.json")}, indent=2) + "\n")
    (output / "duration_audit.csv").write_text("ID,split,original_duration_seconds,evaluation_duration_seconds,evaluation_offset_seconds,source_pcm_sha256,pcm_sha256\n" + "".join(f"{row.ID},{row.split},{row.original_duration_seconds:.6f},{row.evaluation_duration_seconds:.6f},{row.evaluation_offset_seconds:.6f},{row.source_pcm_sha256},{row.pcm_sha256}\n" for row in sorted(rows, key=lambda item: item.ID)))
    label_audit = {"unknown_voice_generated_music": sum(row.MUSIC_FAKE == "1" and row.VOICE_FAKE == "" for row in rows), "all_real_splices": sum(row.panel == "splice" and row.FILE_FAKE == "0" for row in rows), "fake_splices": sum(row.panel == "splice" and row.FILE_FAKE == "1" for row in rows)}
    (output / "label_audit.json").write_text(json.dumps(label_audit, indent=2) + "\n")
    archives = _write_archives(rows, output, workspace)
    (workspace / "final_drive_inputs.json").write_text(json.dumps(archives, indent=2) + "\n")
    print(json.dumps({**integrity, "archives": [{"name": row["name"], "bytes": row["bytes"], "parts": len(row["parts"])} for row in archives]}))


if __name__ == "__main__":
    main()
