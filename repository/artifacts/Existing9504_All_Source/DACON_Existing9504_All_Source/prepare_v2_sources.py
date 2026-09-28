#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy==2.4.6", "pyarrow==21.0.0", "imageio-ffmpeg==0.6.0"]
# ///
# How to run: `uv run prepare_v2_sources.py --output _workspace/v2_sources`.
"""Acquire the exact licensed audio retained for the V2 experiment."""
from __future__ import annotations

import argparse
import csv
import json
import shutil
import subprocess
import unicodedata
import urllib.parse
from pathlib import Path
from collections.abc import Sequence
from typing import Final

import numpy as np
import pyarrow.parquet as pq

from v2_data import DataIntegrityError as SourceIntegrityError
from v2_data import Labels, SourceRow, decode_audio, extract_local_member, first_valid, pcm_sha256, sha256_file, stable_key, write_wav

ROOT: Final = Path(__file__).resolve().parent
MLAAD_REV: Final = "9143e5ea709575ebab6bec52840a1043aada7bb1"
ECHOES_REV: Final = "14b0c76c6a691c42fadfab9fb6a4eb1ee8c628a2"
NSYNTH_REV: Final = "5ba57c8114d9d4d7cf076c4cb3c8f508f105fca8"
MLAAD_GENERATORS: Final = (
    "Cartesia.ai (Sonic-3)", "Higgs-Audio-V2", "MeloTTS", "FireRedTTS-2.0",
    "Index-TTS-1.5", "Index-TTS-2.0", "Llasa-1B-Multilingual", "Mars5",
    "OpenVoiceV2", "VoxCPM-0.5B",
)
MLAAD_FALLBACKS: Final = ("Llasa-8B", "RVC")
DEV_MUSIC_PROVIDERS: Final = ("acestep", "musicgen", "audioldm")
TEST_MUSIC_PROVIDERS: Final = ("suno", "udio", "elevenlabs")
def real_voice_labels() -> Labels:
    return Labels("0", "0", "", "1", "0")


def fake_voice_labels() -> Labels:
    return Labels("1", "1", "", "1", "0")


def fetch(url: str, path: Path) -> Path:
    if path.exists():
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    part = path.with_suffix(path.suffix + ".part")
    subprocess.run(["curl", "-fLsS", "--retry", "3", "--max-time", "180", url, "-o", str(part)], check=True)
    part.replace(path)
    return path


def validate_inventory(rows: list[SourceRow], old_groups: set[str], old_pcm: set[str]) -> None:
    groups = [part for row in rows for part in row.group_id.split("|")]
    repeated_pcm = len({row.pcm_sha256 for row in rows}) != len(rows)
    if set(groups) & old_groups or {row.pcm_sha256 for row in rows} & old_pcm or repeated_pcm:
        raise SourceIntegrityError("source group or decoded PCM overlaps another retained source")
    if any(not row.license or not row.source_sha256 for row in rows):
        raise SourceIntegrityError("every retained source requires a license and checksum")
    known = {"mlaad_real", "mlaad_fake", "fma_real", "echoes_tta", "nsynth"}
    if {row.source_kind for row in rows} - known:
        raise SourceIntegrityError("unknown source kind")


def distinct_artists(items: Sequence[dict[str, str]], excluded: set[str], count: int) -> list[dict[str, str]]:
    selected: list[dict[str, str]] = []
    artists = set(excluded)
    for item in items:
        artist = item["artist_id"]
        if artist in artists:
            continue
        selected.append(item)
        artists.add(artist)
        if len(selected) == count:
            break
    return selected


def _normalized(value: str) -> str:
    plain = unicodedata.normalize("NFKD", value).encode("ascii", "ignore").decode().lower()
    return "".join(char for char in plain if char.isalnum())


def _retain(output: Path, original: Path, kind: str, group: str, provider: str, content: str, url: str, revision: str, license_name: str, attribution: str, labels: Labels) -> SourceRow:
    audio = decode_audio(original)
    if len(audio) < 64000:
        raise SourceIntegrityError(f"audio shorter than four seconds: {original}")
    identifier = stable_key(f"{kind}|{provider}|{content}")[:20]
    prepared = output / "prepared" / f"{identifier}.wav"
    write_wav(prepared, audio)
    return SourceRow(identifier, str(prepared.relative_to(output)), str(original.relative_to(output)), kind, group, provider, content, url, revision, sha256_file(original), pcm_sha256(decode_audio(prepared)), license_name, attribution, labels)


def acquire_speech(output: Path, old_groups: set[str]) -> tuple[list[SourceRow], list[str]]:
    base = f"https://huggingface.co/datasets/mueller91/MLAAD-tiny/resolve/{MLAAD_REV}/"
    api = f"https://huggingface.co/api/datasets/mueller91/MLAAD-tiny/tree/{MLAAD_REV}/"
    rows: list[SourceRow] = []
    used_groups: set[str] = set()
    accepted_generators: list[str] = []
    for generator in (*MLAAD_GENERATORS, *MLAAD_FALLBACKS):
        listing_path = fetch(api + urllib.parse.quote(f"fake/en/{generator}", safe="/") + "?limit=1000", output / "metadata" / f"mlaad_{stable_key(generator)[:12]}.json")
        entries = [item for item in json.loads(listing_path.read_text()) if item["type"] == "file"]
        entries.sort(key=lambda item: stable_key(item["path"]))
        candidates = []
        for entry in entries:
            filename = Path(entry["path"]).name
            group = "mlaad:" + Path(filename).stem
            if group in old_groups or group in used_groups:
                continue
            candidates.append((entry, filename, group))

        def load_pair(candidate: tuple[dict[str, str | int], str, str]) -> tuple[str, tuple[SourceRow, SourceRow]] | None:
            entry, filename, group = candidate
            fake_url = base + urllib.parse.quote(entry["path"], safe="/")
            real_name = f"original/en/{filename}"
            real_url = base + urllib.parse.quote(real_name, safe="/")
            fake_path = output / "originals" / entry["path"]
            real_path = output / "originals" / real_name
            try:
                fetch(fake_url, fake_path)
                fetch(real_url, real_path)
                pair = (
                    _retain(output, real_path, "mlaad_real", group, "M-AILABS", filename, real_url, MLAAD_REV, "M-AILABS permissive notice", filename, real_voice_labels()),
                    _retain(output, fake_path, "mlaad_fake", group, generator, filename, fake_url, MLAAD_REV, "CC-BY-NC-4.0", generator, fake_voice_labels()),
                )
            except (subprocess.CalledProcessError, SourceIntegrityError):
                return None
            return group, pair

        accepted_pairs = first_valid(candidates, load_pair, 20)
        accepted = len(accepted_pairs)
        if accepted != 20:
            continue
        for group, pair in accepted_pairs:
            rows.extend(pair)
            used_groups.add(group)
        accepted_generators.append(generator)
        if len(accepted_generators) == 10:
            break
    if len(accepted_generators) != 10:
        raise SourceIntegrityError(f"only {len(accepted_generators)} speech generators yielded 20 valid pairs")
    for name in ("LICENSE", "original/LICENSE", "README.md"):
        fetch(base + name, output / "licenses" / ("MLAAD_" + name.replace("/", "_")))
    return rows, accepted_generators


def _echoes_member(entry: dict[str, int | str], output: Path) -> Path:
    if output.exists():
        return output
    start = int(entry["header_offset"])
    end = start + 30 + len(str(entry["filename"]).encode()) + int(entry["compress_size"]) + 512
    url = f"https://huggingface.co/datasets/Octavian97/Echoes/resolve/{ECHOES_REV}/Echoes.zip?member={stable_key(str(entry['filename']))[:12]}"
    result = subprocess.run(["curl", "-fLsS", "--retry", "3", "--max-time", "180", "--range", f"{start}-{end}", url], check=True, capture_output=True)
    return extract_local_member(result.stdout, entry, output)


def acquire_music(output: Path) -> list[SourceRow]:
    candidates = json.loads((ROOT / "_workspace/v2_discovery/echoes_fma_unique.json").read_text())
    metadata = list(csv.DictReader((ROOT / "_workspace/v2_discovery/archive_probes/echoes_manifest.csv").open()))
    entries = {item["filename"]: item for item in json.loads((ROOT / "_workspace/v2_discovery/archive_probes/echoes_entries.json").read_text())}
    generated = {(_normalized(row["original_audio"]), row["generator"]): row for row in metadata if row["type"] == "TTA"}
    ordered = sorted(candidates, key=lambda item: stable_key(item["group_id"] + "|" + item["echoes_original_audio"]))
    test_candidates = distinct_artists([item for item in ordered if all((_normalized(item["echoes_original_audio"]), p) in generated for p in TEST_MUSIC_PROVIDERS)], set(), 68)

    def load_reference(pair: tuple[dict[str, str], tuple[str, str, str]]) -> list[SourceRow] | None:
        item, providers = pair
        group, content = item["group_id"], item["echoes_original_audio"]
        try:
            fma_url = "https://files.freemusicarchive.org/storage-freemusicarchive-org/" + urllib.parse.quote(item["track_file"], safe="/")
            fma_path = fetch(fma_url, output / "originals" / "fma" / f"{item['track_id']}.mp3")
            retained = [_retain(output, fma_path, "fma_real", group, "FMA", content, fma_url, "FMA metadata snapshot", item["license_title"] + " " + item["license_url"], item["artist_name"], Labels("0", "", "0", "", "1"))]
            for provider in providers:
                meta = generated[(_normalized(content), provider)]
                archive_name = "Echoes/" + meta["path_in_dataset"]
                original = output / "originals" / "echoes" / meta["path_in_dataset"]
                _echoes_member(entries[archive_name], original)
                retained.append(_retain(output, original, "echoes_tta", group, provider, content, f"Echoes.zip::{archive_name}", ECHOES_REV, "CC-BY-SA-4.0", f"Echoes; {provider}", Labels("1", "", "1", "", "1")))
        except (subprocess.CalledProcessError, SourceIntegrityError, KeyError):
            return None
        return retained

    test_loaded = first_valid([(item, TEST_MUSIC_PROVIDERS) for item in test_candidates], load_reference, 20)
    if len(test_loaded) != 20:
        raise SourceIntegrityError("fewer than 20 test-provider FMA artists could be retained")
    held_artists = {rows[0].group_id for rows in test_loaded}
    dev_candidates = distinct_artists([item for item in ordered if all((_normalized(item["echoes_original_audio"]), p) in generated for p in DEV_MUSIC_PROVIDERS)], {group.removeprefix("fma_artist_") for group in held_artists}, 106)
    dev_loaded = first_valid([(item, DEV_MUSIC_PROVIDERS) for item in dev_candidates], load_reference, 40)
    if len(dev_loaded) != 40:
        raise SourceIntegrityError("fewer than 40 development-provider FMA artists could be retained")
    return [row for group_rows in dev_loaded + test_loaded for row in group_rows]


def acquire_nsynth(output: Path) -> list[SourceRow]:
    source = ROOT / "_workspace/v2_discovery/nsynth_test.parquet"
    preserved = output / "originals/nsynth/test.parquet"
    preserved.parent.mkdir(parents=True, exist_ok=True)
    if not preserved.exists():
        shutil.copyfile(source, preserved)
    parquet = pq.ParquetFile(source)
    metadata = []
    for batch in parquet.iter_batches(batch_size=256, columns=["id", "instrument", "instrument_str", "instrument_family_str", "instrument_source_str"]):
        metadata.extend(batch.to_pylist())
    eligible = [item for item in metadata if item["instrument_source_str"] == "acoustic" and item["instrument_family_str"] != "vocal"]
    by_instrument: dict[int, list[dict[str, str | int]]] = {}
    for item in eligible:
        by_instrument.setdefault(int(item["instrument"]), []).append(item)
    eligible_ids = {str(item["id"]) for item in eligible}
    chosen: dict[int, dict[str, tuple[str, dict[str, str | int | dict[str, list[float] | str | int]], np.ndarray]]] = {}
    for batch in parquet.iter_batches(batch_size=64, columns=["id", "instrument", "instrument_str", "audio"]):
        for item in batch.to_pylist():
            if item["id"] not in eligible_ids:
                continue
            instrument = int(item["instrument"])
            audio = np.asarray(item["audio"]["array"], dtype=np.float32)
            digest = pcm_sha256(audio)
            options = chosen.setdefault(instrument, {})
            candidate = (stable_key(str(item["id"])), item, audio)
            if digest not in options or candidate[0] < options[digest][0]:
                options[digest] = candidate
            if len(options) > 4:
                del options[max(options, key=lambda key: options[key][0])]
    if len(chosen) != 25 or any(len(items) != 4 for items in chosen.values()):
        raise SourceIntegrityError("NSynth acoustic non-vocal selection must be 100 notes from 25 instruments")
    rows: list[SourceRow] = []
    for instrument in sorted(chosen):
        for _, item, audio in sorted(chosen[instrument].values(), key=lambda value: value[0]):
            identifier = stable_key("nsynth|" + item["id"])[:20]
            original = output / "originals/nsynth/selected" / f"{identifier}.wav"
            prepared = output / "prepared" / f"{identifier}.wav"
            write_wav(original, audio)
            write_wav(prepared, audio)
            rows.append(SourceRow(identifier, str(prepared.relative_to(output)), str(original.relative_to(output)), "nsynth", f"nsynth_instrument_{item['instrument']}", "NSynth", item["id"], "NSynth test.parquet row", NSYNTH_REV, sha256_file(original), pcm_sha256(decode_audio(prepared)), "CC-BY-4.0", item["instrument_str"], Labels("0", "", "0", "0", "1")))
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=ROOT / "_workspace/v2_sources")
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    output: Path = args.output.resolve()
    old_rows = list(csv.DictReader((ROOT / "validation/manifest.csv").open()))
    old_groups = {part for row in old_rows for part in row["group_id"].split("|")}
    old_pcm = {pcm_sha256(decode_audio(ROOT / "validation" / row["path"])) for row in old_rows}
    speech_rows, accepted_generators = acquire_speech(output, old_groups)
    rows = speech_rows + acquire_music(output) + acquire_nsynth(output)
    validate_inventory(rows, old_groups, old_pcm)
    artifacts = [ROOT / "_workspace/v2_discovery/nsynth_test.parquet", ROOT / "_workspace/v2_discovery/archive_probes/echoes_manifest.csv", ROOT / "_workspace/v2_discovery/archive_probes/echoes_entries.json", ROOT / "_workspace/v2_discovery/echoes_fma_unique.json"]
    inventory = {"schema_version": 1, "sources": [row.json_record() for row in rows], "source_artifacts": [{"path": str(path.relative_to(ROOT)), "sha256": sha256_file(path), "bytes": path.stat().st_size} for path in artifacts], "acquisition": {"mlaad_revision": MLAAD_REV, "accepted_voice_generators": accepted_generators, "test_voice_generators": accepted_generators[-2:], "echoes_revision": ECHOES_REV, "nsynth_revision": NSYNTH_REV, "aime_rows": 0}}
    output.mkdir(parents=True, exist_ok=True)
    inventory_path = output / "source_inventory.json"
    inventory_path.write_text(json.dumps(inventory, indent=2, ensure_ascii=False) + "\n")
    report = {"sources": len(rows), "speech": sum(row.source_kind.startswith("mlaad") for row in rows), "music": sum(row.source_kind in {"fma_real", "echoes_tta"} for row in rows), "nsynth": sum(row.source_kind == "nsynth" for row in rows), "groups": len({row.group_id for row in rows}), "inventory_sha256": sha256_file(inventory_path)}
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report))


if __name__ == "__main__":
    main()
