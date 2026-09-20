#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy==2.4.6", "imageio-ffmpeg==0.6.0"]
# ///
# How to run: `.venv/bin/python prepare_recovery_data.py`.
"""Acquire the source-disjoint V3 recovery challenge bank."""
from __future__ import annotations

import argparse
import csv
import json
import os
import re
import subprocess
import urllib.parse
import zipfile
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Final

import imageio_ffmpeg

from prepare_v2_sources import ECHOES_REV, MLAAD_REV, _echoes_member, _normalized, fetch, real_voice_labels, fake_voice_labels
from v2_data import DataIntegrityError, Labels, decode_audio, first_valid, pcm_sha256, sha256_file, stable_key, write_wav

ROOT: Final = Path(__file__).resolve().parent
MUSIC_PROVIDERS: Final = ("suno", "udio", "elevenlabs")
SPEECH_PROVIDERS: Final = ("Cartesia.ai (Sonic-3)", "Higgs-Audio-V2", "MeloTTS", "FireRedTTS-2.0")
FIELDS: Final = ("ID", "path", "group_id", "split", "source", "FILE_FAKE", "VOICE_FAKE", "MUSIC_FAKE", "VOICE_PRESENT", "MUSIC_PRESENT")


class RecoveryIntegrityError(RuntimeError):
    """A recovery source fails source-disjoint or label integrity."""


@dataclass(frozen=True, slots=True)
class FreshRow:
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
    content_id: str
    source_url: str
    revision: str
    source_sha256: str
    pcm_sha256: str
    original_duration_seconds: float
    retained_duration_seconds: float
    license: str
    short_pcm_sha256: str = ""

    def manifest_record(self) -> dict[str, str]:
        return {name: str(getattr(self, name)) for name in FIELDS}


def music_real_labels() -> Labels:
    return Labels("0", "", "0", "", "1")


def music_fake_labels() -> Labels:
    return Labels("1", "", "1", "", "1")


def _mlaad_url(base: str, path: str) -> str:
    return base + urllib.parse.quote(path, safe="/")


def _duration(path: Path) -> float:
    result = subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(), "-hide_banner", "-i", str(path), "-f", "null", "-"], capture_output=True, text=True, check=False)
    match = re.search(r"Duration: (\d+):(\d+):(\d+(?:\.\d+)?)", result.stderr)
    if match is None:
        raise DataIntegrityError(f"missing source duration: {path}")
    hours, minutes, seconds = match.groups()
    return int(hours) * 3600 + int(minutes) * 60 + float(seconds)


def _retain(fresh: Path, recovery: Path, original: Path, *, kind: str, group: str, provider: str, content: str, url: str, revision: str, license_name: str, labels: Labels) -> FreshRow:
    audio = decode_audio(original, limit_seconds=60)
    if len(audio) < 64000:
        raise DataIntegrityError(f"audio shorter than four seconds: {original}")
    identifier = stable_key(f"{kind}|{provider}|{content}")[:20]
    prepared = fresh / "audio" / f"{identifier}.wav"
    write_wav(prepared, audio)
    destination = recovery / "audio" / prepared.name
    destination.parent.mkdir(parents=True, exist_ok=True)
    if not destination.exists():
        os.link(prepared, destination)
    return FreshRow(identifier, f"audio/{identifier}.wav", group, "fresh_challenge", f"{kind}/{provider}", labels.FILE_FAKE, labels.VOICE_FAKE, labels.MUSIC_FAKE, labels.VOICE_PRESENT, labels.MUSIC_PRESENT, content, url, revision, sha256_file(original), pcm_sha256(decode_audio(prepared, limit_seconds=60)), _duration(original), len(audio) / 16000, license_name, pcm_sha256(decode_audio(prepared)))


def _old_exclusions() -> tuple[set[str], set[str], set[str]]:
    inventory = json.loads((ROOT / "_workspace/v2_sources/source_inventory.json").read_text())["sources"]
    groups = {part for row in inventory for part in row["group_id"].split("|")}
    contents = {str(row["content_id"]) for row in inventory}
    pcms = {str(row["pcm_sha256"]) for row in inventory}
    for manifest in (ROOT / "validation/manifest.csv", ROOT / "v2_validation/development/manifest.csv", ROOT / "v2_validation/locked_test/manifest.csv"):
        for row in csv.DictReader(manifest.open()):
            groups.update(row["group_id"].split("|")); contents.add(row["ID"])
    pcms.update(row["pcm_sha256"] for row in csv.DictReader((ROOT / "v2_validation/pcm_sha256.csv").open()))
    return groups, contents, pcms


def _music_candidates(old_groups: set[str], old_contents: set[str]) -> tuple[list[dict[str, str]], dict[tuple[str, str], dict[str, str]], dict[str, dict[str, int | str]]]:
    candidates = json.loads((ROOT / "_workspace/v2_discovery/echoes_fma_unique.json").read_text())
    metadata = list(csv.DictReader((ROOT / "_workspace/v2_discovery/archive_probes/echoes_manifest.csv").open()))
    generated = {(_normalized(row["original_audio"]), row["generator"]): row for row in metadata if row["type"] == "TTA"}
    entries = {str(item["filename"]): item for item in json.loads((ROOT / "_workspace/v2_discovery/archive_probes/echoes_entries.json").read_text())}
    eligible = [item for item in candidates if item["group_id"] not in old_groups and item["track_id"] not in old_contents and item["echoes_original_audio"] not in old_contents and all((_normalized(item["echoes_original_audio"]), provider) in generated for provider in MUSIC_PROVIDERS)]
    selected: list[dict[str, str]] = []
    artists: set[str] = set()
    for item in sorted(eligible, key=lambda value: stable_key(value["group_id"] + "|" + value["echoes_original_audio"])):
        if item["artist_id"] not in artists:
            selected.append(item); artists.add(item["artist_id"])
    return selected, generated, entries


def _acquire_music(fresh: Path, recovery: Path, old_groups: set[str], old_contents: set[str], skips: list[str]) -> list[FreshRow]:
    candidates, generated, entries = _music_candidates(old_groups, old_contents)
    def load(item: dict[str, str]) -> list[FreshRow] | None:
        content, group = item["echoes_original_audio"], item["group_id"]
        try:
            url = "https://files.freemusicarchive.org/storage-freemusicarchive-org/" + urllib.parse.quote(item["track_file"], safe="/")
            original = fetch(url, fresh / "originals/fma" / f"{item['track_id']}.mp3")
            rows = [_retain(fresh, recovery, original, kind="FMA", group=group, provider="FMA", content=item["track_id"], url=url, revision="FMA metadata snapshot", license_name=item["license_title"] + " " + item["license_url"], labels=music_real_labels())]
            for provider in MUSIC_PROVIDERS:
                meta = generated[(_normalized(content), provider)]
                archive_name = "Echoes/" + meta["path_in_dataset"]
                path = fresh / "originals/echoes" / meta["path_in_dataset"]
                _echoes_member(entries[archive_name], path)
                rows.append(_retain(fresh, recovery, path, kind="Echoes", group=group, provider=provider, content=content, url=f"Echoes.zip::{archive_name}", revision=ECHOES_REV, license_name="CC-BY-SA-4.0", labels=music_fake_labels()))
            return rows
        except (subprocess.CalledProcessError, DataIntegrityError, KeyError) as error:
            skips.append(f"music {item['artist_id']}:{item['track_id']} {type(error).__name__}: {error}")
            return None
    loaded = first_valid(candidates, load, 12)
    if len(loaded) != 12:
        raise RecoveryIntegrityError(f"only {len(loaded)} source-disjoint music artists acquired")
    return [row for quartet in loaded for row in quartet]


def _acquire_speech(fresh: Path, recovery: Path, old_groups: set[str], old_contents: set[str], skips: list[str]) -> list[FreshRow]:
    base = f"https://huggingface.co/datasets/mueller91/MLAAD-tiny/resolve/{MLAAD_REV}/"
    api = f"https://huggingface.co/api/datasets/mueller91/MLAAD-tiny/tree/{MLAAD_REV}/"
    rows: list[FreshRow] = []; used = set(old_groups)
    for provider in SPEECH_PROVIDERS:
        listing = fetch(api + urllib.parse.quote(f"fake/en/{provider}", safe="/") + "?limit=1000", fresh / "metadata" / f"mlaad_{stable_key(provider)[:12]}.json")
        candidates = []
        for entry in sorted((item for item in json.loads(listing.read_text()) if item["type"] == "file"), key=lambda value: stable_key(value["path"])):
            filename = Path(entry["path"]).name; group = "mlaad:" + Path(filename).stem
            if group not in used and filename not in old_contents:
                candidates.append((entry, filename, group))
        def load(candidate: tuple[dict[str, str | int], str, str]) -> tuple[str, list[FreshRow]] | None:
            entry, filename, group = candidate
            try:
                fake_url = _mlaad_url(base, str(entry["path"])); real_url = _mlaad_url(base, f"original/en/{filename}")
                real = fetch(real_url, fresh / "originals/original/en" / filename)
                fake = fetch(fake_url, fresh / "originals" / entry["path"])
                return group, [_retain(fresh, recovery, real, kind="MLAAD", group=group, provider="M-AILABS", content=filename, url=real_url, revision=MLAAD_REV, license_name="M-AILABS permissive notice", labels=real_voice_labels()), _retain(fresh, recovery, fake, kind="MLAAD", group=group, provider=provider, content=filename, url=fake_url, revision=MLAAD_REV, license_name="CC-BY-NC-4.0", labels=fake_voice_labels())]
            except (subprocess.CalledProcessError, DataIntegrityError) as error:
                skips.append(f"speech {provider}:{filename} {type(error).__name__}: {error}")
                return None
        loaded = first_valid(candidates, load, 5)
        if len(loaded) != 5:
            raise RecoveryIntegrityError(f"only {len(loaded)} valid MLAAD pairs for {provider}")
        for group, pair in loaded:
            used.add(group); rows.extend(pair)
    for name in ("LICENSE", "original/LICENSE", "README.md"):
        fetch(base + name, fresh / "licenses" / ("MLAAD_" + name.replace("/", "_")))
    return rows


def audit_rows(rows: list[FreshRow], old_groups: set[str], old_contents: set[str], old_pcms: set[str]) -> None:
    groups = {row.group_id for row in rows}; contents = {row.content_id for row in rows}; pcms = [row.pcm_sha256 for row in rows]
    if groups & old_groups or contents & old_contents or {row.short_pcm_sha256 for row in rows} & old_pcms:
        raise RecoveryIntegrityError("source group, content, or PCM overlaps the prior corpus")
    if len(pcms) != len(set(pcms)) or any(not row.license or row.retained_duration_seconds > 60.001 or row.original_duration_seconds + 0.02 < row.retained_duration_seconds for row in rows):
        raise RecoveryIntegrityError("duplicate PCM, missing license, or invalid retained duration")
    music = [row for row in rows if row.group_id.startswith("fma_artist_")]
    for row in music:
        if (row.VOICE_FAKE, row.VOICE_PRESENT, row.MUSIC_PRESENT) != ("", "", "1"):
            raise RecoveryIntegrityError("music rows must retain unknown voice labels")
    for group in {row.group_id for row in music}:
        group_rows = [row for row in music if row.group_id == group]
        sources = {row.source for row in group_rows}
        if sources != {"FMA/FMA", *(f"Echoes/{provider}" for provider in MUSIC_PROVIDERS)}:
            raise RecoveryIntegrityError("music artists must have real plus all requested providers")
        if len(group_rows) != 4 or any(row.MUSIC_FAKE != ("0" if row.source == "FMA/FMA" else "1") for row in group_rows):
            raise RecoveryIntegrityError("music artists must balance verified real and fake rows")
    speech = [row for row in rows if row.group_id.startswith("mlaad:")]
    for group in {row.group_id for row in speech}:
        group_rows = [row for row in speech if row.group_id == group]
        labels = {(row.FILE_FAKE, row.VOICE_FAKE, row.VOICE_PRESENT, row.MUSIC_PRESENT) for row in group_rows}
        if len(group_rows) != 2 or labels != {("0", "0", "1", "0"), ("1", "1", "1", "0")}:
            raise RecoveryIntegrityError("speech pairs require verified real and fake labels")


def _prune_audio(fresh: Path, recovery: Path, rows: list[FreshRow]) -> None:
    used = {Path(row.path).name for row in rows}
    for directory in (fresh / "audio", recovery / "audio"):
        for path in directory.glob("*.wav"):
            if path.name not in used:
                path.unlink()


def _write_products(rows: list[FreshRow], recovery: Path, fresh: Path, archive: Path, skips: list[str]) -> dict[str, int | str]:
    recovery.mkdir(parents=True, exist_ok=True)
    _prune_audio(fresh, recovery, rows)
    with (recovery / "manifest.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS); writer.writeheader(); writer.writerows(row.manifest_record() for row in sorted(rows, key=lambda value: value.ID))
    provenance = {"schema_version": 1, "selection": {"music_providers": MUSIC_PROVIDERS, "speech_providers": SPEECH_PROVIDERS, "selection": "stable SHA-256 ordering; source availability only"}, "sources": [asdict(row) for row in sorted(rows, key=lambda value: value.ID)], "source_artifacts": [{"path": str(path.relative_to(ROOT)), "sha256": sha256_file(path), "bytes": path.stat().st_size} for path in (ROOT / "_workspace/v2_sources/source_inventory.json", ROOT / "_workspace/v2_discovery/echoes_fma_unique.json", ROOT / "_workspace/v2_discovery/archive_probes/echoes_manifest.csv", ROOT / "_workspace/v2_discovery/archive_probes/echoes_entries.json")]}
    (recovery / "provenance.json").write_text(json.dumps(provenance, indent=2, ensure_ascii=False) + "\n")
    (fresh / "acquisition_skips.log").write_text("\n".join(skips) + ("\n" if skips else ""))
    report = {"files": len(rows), "music_artists": len({row.group_id for row in rows if row.group_id.startswith("fma_artist_")}), "music_provider_counts": dict(Counter(row.source for row in rows if row.group_id.startswith("fma_artist_"))), "speech_pairs": len({row.group_id for row in rows if row.group_id.startswith("mlaad:")}), "duration_seconds": round(sum(row.retained_duration_seconds for row in rows), 3), "skipped": len(skips), "manifest_sha256": sha256_file(recovery / "manifest.csv"), "provenance_sha256": sha256_file(recovery / "provenance.json")}
    (recovery / "integrity_report.json").write_text(json.dumps(report, indent=2) + "\n")
    archive.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as handle:
        for path in sorted(recovery.rglob("*")):
            if path.is_file(): handle.write(path, path.relative_to(recovery.parent))
    report["archive_sha256"] = sha256_file(archive); report["archive_bytes"] = archive.stat().st_size
    (recovery / "integrity_report.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--fresh", type=Path, default=ROOT / "_workspace/recovery_v3/fresh_sources"); parser.add_argument("--output", type=Path, default=ROOT / "recovery_validation"); parser.add_argument("--archive", type=Path, default=ROOT / "_workspace/recovery_v3/recovery_validation.zip")
    args = parser.parse_args(); fresh, recovery = args.fresh.resolve(), args.output.resolve(); fresh.mkdir(parents=True, exist_ok=True)
    old_groups, old_contents, old_pcms = _old_exclusions(); skips: list[str] = []
    rows = _acquire_music(fresh, recovery, old_groups, old_contents, skips) + _acquire_speech(fresh, recovery, old_groups, old_contents, skips)
    if len(rows) != 88 or Counter(row.source for row in rows if row.source.startswith("Echoes/")) != Counter({f"Echoes/{provider}": 12 for provider in MUSIC_PROVIDERS}) or Counter(row.source for row in rows if row.source.startswith("MLAAD/") and row.source != "MLAAD/M-AILABS") != Counter({f"MLAAD/{provider}": 5 for provider in SPEECH_PROVIDERS}):
        raise RecoveryIntegrityError("fresh bank is not balanced to the declared music and speech targets")
    audit_rows(rows, old_groups, old_contents, old_pcms); print(json.dumps(_write_products(rows, recovery, fresh, args.archive.resolve(), skips)))


if __name__ == "__main__":
    main()
