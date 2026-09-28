"""Pinned source selection and acquisition for the V4 data bank."""
from __future__ import annotations

import csv
import json
import urllib.parse
from collections import Counter, defaultdict
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Final

from prepare_v2_sources import ECHOES_REV, MLAAD_REV, _echoes_member, _normalized, fetch
from v2_data import stable_key

TRAIN_PROVIDERS: Final = (
    "acestep", "audioldm", "brev", "diffrhythm", "elevenlabs",
    "musicgen", "producer", "songgen", "suno", "udio",
)
DEV_HELDOUT: Final = "stableaudio"
SEALED_HELDOUT: Final = "mubert"
SPEECH_PROVIDERS: Final = ("Cartesia.ai (Sonic-3)", "Higgs-Audio-V2", "Chatterbox Multilingual", "Kani-TTS-370M")


class V4SourceError(RuntimeError):
    """A pinned source cannot satisfy the V4 acquisition contract."""


def load_echoes(root: Path) -> tuple[list[dict[str, str]], dict[tuple[str, str], dict[str, str]], dict[str, dict[str, int | str]]]:
    pairs = json.loads((root / "_workspace/v2_discovery/echoes_fma_unique.json").read_text())
    metadata = csv.DictReader((root / "_workspace/v2_discovery/archive_probes/echoes_manifest.csv").open())
    generated = {(_normalized(row["original_audio"]), row["generator"]): row for row in metadata if row["type"] == "TTA"}
    raw_entries = json.loads((root / "_workspace/v2_discovery/archive_probes/echoes_entries.json").read_text())
    entries = {str(item["filename"]): item for item in raw_entries}
    return pairs, generated, entries


def load_echoes_variants(root: Path) -> dict[str, list[dict[str, str]]]:
    """Return every ATA/TTA row keyed by normalized reference title."""
    variants: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in csv.DictReader((root / "_workspace/v2_discovery/archive_probes/echoes_manifest.csv").open()):
        if row["type"] in {"ATA", "TTA"}:
            variants[_normalized(row["original_audio"])].append(row)
    return {key: sorted(rows, key=lambda row: stable_key(row["path_in_dataset"])) for key, rows in variants.items()}


def prior_identities(root: Path) -> tuple[set[str], set[str], set[str]]:
    inventory = json.loads((root / "_workspace/v2_sources/source_inventory.json").read_text())["sources"]
    v3 = json.loads((root / "recovery_validation/provenance.json").read_text())["sources"]
    groups = {part for row in (*inventory, *v3) for part in str(row["group_id"]).split("|")}
    artists = {group.removeprefix("fma_artist_") for group in groups if group.startswith("fma_artist_")}
    contents = {str(row["content_id"]) for row in (*inventory, *v3)}
    return groups, artists, contents


def _speaker_id(original_file: str) -> str:
    parts = Path(original_file).parts
    if len(parts) > 4 and parts[1] == "by_book":
        return f"mlaad_{parts[0]}_speaker:{parts[3]}"
    raise V4SourceError(f"MLAAD original_file has no speaker path: {original_file}")


def prior_speech_speakers(root: Path, cache: Path) -> tuple[set[str], int, dict[str, str]]:
    """Resolve every prior fake-speech utterance to its actual MLAAD speaker metadata."""
    inventory = json.loads((root / "_workspace/v2_sources/source_inventory.json").read_text())["sources"]
    v3 = json.loads((root / "recovery_validation/provenance.json").read_text())["sources"]
    prior = [(str(row["provider"]), str(row["content_id"])) for row in inventory if row["source_kind"] == "mlaad_fake"]
    prior.extend((str(row["source"]).removeprefix("MLAAD/"), str(row["content_id"])) for row in v3 if str(row["source"]).startswith("MLAAD/") and row["VOICE_FAKE"] == "1")
    base = f"https://huggingface.co/datasets/mueller91/MLAAD-tiny/resolve/{MLAAD_REV}/"
    lookup: dict[tuple[str, str], str] = {}
    for provider in sorted({provider for provider, _ in prior}):
        url = base + urllib.parse.quote(f"fake/en/{provider}/meta.csv", safe="/")
        path = fetch(url, cache / "metadata" / f"mlaad_en_{stable_key(provider)[:12]}.csv")
        for row in csv.DictReader(path.open(), delimiter="|"):
            lookup[(provider, Path(row["path"]).name)] = _speaker_id(row["original_file"])
    missing = [item for item in prior if item not in lookup]
    if missing:
        raise V4SourceError(f"missing prior MLAAD speaker metadata for {len(missing)} utterances")
    by_filename: dict[str, str] = {}
    for (_, filename), speaker in lookup.items():
        if filename in by_filename and by_filename[filename] != speaker:
            raise V4SourceError(f"conflicting speaker metadata for {filename}")
        by_filename[filename] = speaker
    return {lookup[item] for item in prior}, len(prior), by_filename


def by_artist(items: Iterable[dict[str, str]]) -> dict[str, list[dict[str, str]]]:
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for item in items:
        grouped[item["artist_id"]].append(item)
    return {artist: sorted(rows, key=lambda row: stable_key(row["group_id"] + "|" + row["echoes_original_audio"])) for artist, rows in grouped.items()}


def assign_music_roles(
    pairs: Sequence[dict[str, str]], generated: dict[tuple[str, str], dict[str, str]], excluded_artists: set[str], count: int = 10,
) -> tuple[list[str], list[str]]:
    """Assign held-out artists exclusive-first, then shared by stable hash."""
    eligible: dict[str, set[str]] = defaultdict(set)
    for item in pairs:
        artist = item["artist_id"]
        if artist in excluded_artists:
            continue
        providers = {provider for provider in (*TRAIN_PROVIDERS, DEV_HELDOUT, SEALED_HELDOUT) if (_normalized(item["echoes_original_audio"]), provider) in generated}
        if len(providers & set(TRAIN_PROVIDERS)) < 3:
            continue
        for heldout in (DEV_HELDOUT, SEALED_HELDOUT):
            if heldout in providers:
                eligible[artist].add(heldout)
    dev_only = sorted((artist for artist, roles in eligible.items() if roles == {DEV_HELDOUT}), key=stable_key)
    sealed_only = sorted((artist for artist, roles in eligible.items() if roles == {SEALED_HELDOUT}), key=stable_key)
    shared = sorted((artist for artist, roles in eligible.items() if roles == {DEV_HELDOUT, SEALED_HELDOUT}), key=stable_key)
    dev = dev_only[:count]
    sealed = sealed_only[:count]
    for artist in shared:
        if len(dev) < count:
            dev.append(artist)
        elif len(sealed) < count:
            sealed.append(artist)
    if len(dev) != count or len(sealed) != count or set(dev) & set(sealed):
        raise V4SourceError(f"exclusive-first assignment cannot form {count}+{count} artists")
    return dev, sealed


def round_robin_providers(item: dict[str, str], generated: dict[tuple[str, str], dict[str, str]], index: int, count: int = 3) -> tuple[str, ...]:
    available = [provider for provider in TRAIN_PROVIDERS if (_normalized(item["echoes_original_audio"]), provider) in generated]
    if len(available) < count:
        raise V4SourceError(f"{item['group_id']} has only {len(available)} training providers")
    start = index % len(TRAIN_PROVIDERS)
    ordered = [provider for step in range(len(TRAIN_PROVIDERS)) for provider in [TRAIN_PROVIDERS[(start + step) % len(TRAIN_PROVIDERS)]] if provider in available]
    return tuple(ordered[:count])


def select_training_tracks(
    pairs: Sequence[dict[str, str]], generated: dict[tuple[str, str], dict[str, str]], old_artist_tracks: dict[str, dict[str, str]], excluded_artists: set[str], new_count: int = 10,
) -> tuple[list[dict[str, str]], tuple[str, ...]]:
    grouped = by_artist(pairs)
    chosen = list(old_artist_tracks.values())
    candidates: list[dict[str, str]] = []
    for artist, items in grouped.items():
        if artist in excluded_artists or artist in old_artist_tracks:
            continue
        viable = [item for item in items if len({p for p in TRAIN_PROVIDERS if (_normalized(item["echoes_original_audio"]), p) in generated}) >= 3]
        if viable:
            candidates.append(max(viable, key=lambda item: (sum((_normalized(item["echoes_original_audio"]), p) in generated for p in TRAIN_PROVIDERS), stable_key(item["group_id"]))))
    chosen.extend(sorted(candidates, key=lambda item: stable_key(item["group_id"]))[:new_count])
    if len(chosen) != len(old_artist_tracks) + new_count:
        raise V4SourceError("not enough disjoint new training artists")
    counts = Counter(provider for item in chosen for provider in TRAIN_PROVIDERS if (_normalized(item["echoes_original_audio"]), provider) in generated)
    included = tuple(provider for provider in TRAIN_PROVIDERS if counts[provider] >= 20)
    if len(included) < 6:
        raise V4SourceError(f"only {len(included)} providers meet the 20-group quota")
    return chosen, included


def old_music_tracks(root: Path, pairs: Sequence[dict[str, str]]) -> dict[str, dict[str, str]]:
    inventory = json.loads((root / "_workspace/v2_sources/source_inventory.json").read_text())["sources"]
    old_contents = {row["content_id"] for row in inventory if row["source_kind"] == "fma_real"}
    matches = {item["artist_id"]: item for item in pairs if item["echoes_original_audio"] in old_contents}
    if len(matches) != 60:
        raise V4SourceError(f"expected 60 prior FMA artists, found {len(matches)}")
    return matches


def acquire_fma(item: dict[str, str], cache: Path) -> tuple[Path, str]:
    url = "https://files.freemusicarchive.org/storage-freemusicarchive-org/" + urllib.parse.quote(item["track_file"], safe="/")
    return fetch(url, cache / "fma" / f"{item['track_id']}.mp3"), url


def acquire_echoes(
    item: dict[str, str], provider: str, generated: dict[tuple[str, str], dict[str, str]], entries: dict[str, dict[str, int | str]], cache: Path,
) -> tuple[Path, str]:
    meta = generated[(_normalized(item["echoes_original_audio"]), provider)]
    archive_name = "Echoes/" + meta["path_in_dataset"]
    output = cache / "echoes" / meta["path_in_dataset"]
    return _echoes_member(entries[archive_name], output), f"Echoes.zip::{archive_name}"


def acquire_echoes_variant(meta: dict[str, str], entries: dict[str, dict[str, int | str]], cache: Path) -> tuple[Path, str]:
    archive_name = "Echoes/" + meta["path_in_dataset"]
    output = cache / "echoes" / meta["path_in_dataset"]
    return _echoes_member(entries[archive_name], output), f"Echoes.zip::{archive_name}"


def speech_candidates(root: Path, cache: Path, excluded_groups: set[str], excluded_speakers: set[str] | None = None) -> list[tuple[str, str, str, str, str, str]]:
    base = f"https://huggingface.co/datasets/mueller91/MLAAD-tiny/resolve/{MLAAD_REV}/"
    candidates: list[tuple[str, str, str, str, str, str]] = []
    for provider in SPEECH_PROVIDERS:
        meta_url = base + urllib.parse.quote(f"fake/de/{provider}/meta.csv", safe="/")
        meta_path = fetch(meta_url, cache / "metadata" / f"mlaad_de_{stable_key(provider)[:12]}.csv")
        entries = csv.DictReader(meta_path.open(), delimiter="|")
        for entry in sorted(entries, key=lambda value: stable_key(value["path"])):
            filename = Path(entry["path"]).name
            group = "mlaad:" + Path(filename).stem
            speaker = _speaker_id(entry["original_file"])
            if group not in excluded_groups and speaker not in (excluded_speakers or set()) and float(entry["duration"]) >= 4.0 and filename.endswith(".wav"):
                fake_url = base + urllib.parse.quote(f"fake/de/{provider}/{filename}", safe="/")
                real_url = base + urllib.parse.quote(f"original/de/{filename}", safe="/")
                candidates.append((provider, filename, group, speaker, real_url, fake_url))
    return candidates


def select_speech_roles(candidates: Sequence[tuple[str, str, str, str, str, str]], count: int = 20) -> tuple[list[tuple[str, str, str, str, str, str]], list[tuple[str, str, str, str, str, str]]]:
    """Choose complete role panels with speaker-disjoint development and sealed sets."""
    speakers = sorted({row[3] for row in candidates}, key=stable_key)
    by_speaker = {speaker: sorted((row for row in candidates if row[3] == speaker), key=lambda row: stable_key(row[2] + "|" + row[0])) for speaker in speakers}
    dev: list[tuple[str, str, str, str, str, str]] = []
    sealed: list[tuple[str, str, str, str, str, str]] = []
    for speaker in speakers:
        target = dev if len(dev) < count else sealed
        used_groups = {row[2] for row in (*dev, *sealed)}
        target.extend(row for row in by_speaker[speaker] if row[2] not in used_groups and len(target) < count)
        if len(sealed) == count:
            break
    if len(dev) != count or len(sealed) != count or {row[3] for row in dev} & {row[3] for row in sealed}:
        raise V4SourceError(f"speaker-disjoint MLAAD roles cannot form {count}+{count} pairs")
    return dev, sealed


def acquire_speech_pair(candidate: tuple[str, str, str, str, str, str], cache: Path) -> tuple[Path, Path]:
    provider, filename, _, _, real_url, fake_url = candidate
    real = fetch(real_url, cache / "mlaad/original/de" / filename)
    fake = fetch(fake_url, cache / "mlaad/fake/de" / provider / filename)
    return real, fake
