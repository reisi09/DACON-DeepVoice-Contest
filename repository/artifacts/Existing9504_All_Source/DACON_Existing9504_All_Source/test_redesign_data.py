"""Behavioral checks for provenance, lineage, unknown masks, and paired audio truth."""
from __future__ import annotations

import csv
import hashlib
import json
import shutil
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from redesign_data import DataContractError, SOURCE_FIELDS, prepare_data, read_rows, verify_lineage, write_source_template


def _sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _source_row(identifier: str, path: Path, category: int) -> dict[str, str]:
    kind, file_fake, voice_fake, music_fake, voice_present, music_present = (
        ("speech", "0", "0", "", "1", "0"),
        ("speech", "1", "1", "", "1", "0"),
        ("instrumental", "0", "", "0", "0", "1"),
        ("instrumental", "1", "", "1", "0", "1"),
    )[category]
    return {
        "ID": identifier, "path": path.name, "audio_sha256": _sha256(path),
        "source": f"fixture/provider-{category}", "license": "fixture-test-only",
        "license_url": "https://example.invalid/fixture-license", "source_url": f"https://example.invalid/{identifier}",
        "parent_ids": json.dumps([f"parent-{identifier}"]), "artist_id": "", "speaker_id": "",
        "generator": "fixture-generator" if file_fake == "1" else "", "kind": kind, "domain": "fixture",
        "FILE_FAKE": file_fake, "VOICE_FAKE": voice_fake, "MUSIC_FAKE": music_fake,
        "VOICE_PRESENT": voice_present, "MUSIC_PRESENT": music_present,
    }


def _fixture_manifest(root: Path) -> Path:
    identifiers = [f"source-{index:03d}" for index in range(48)]
    ordered = sorted(identifiers, key=lambda identifier: hashlib.sha256(f"41|{identifier}".encode()).hexdigest())
    categories = {identifier: position % 4 for position, identifier in enumerate(ordered)}
    rows: list[dict[str, str]] = []
    for index, identifier in enumerate(identifiers):
        path = root / f"{identifier}.wav"
        signal = np.full(4800, (index + 1) / 100.0, dtype=np.float32)
        sf.write(path, signal, 8000, subtype="FLOAT")
        rows.append(_source_row(identifier, path, categories[identifier]))
    exact_path = root / "exact-duplicate.wav"
    exact_path.write_bytes((root / "source-006.wav").read_bytes())
    rows.append(_source_row("exact-duplicate", exact_path, categories["source-006"]))
    pcm_path = root / "pcm-duplicate.aiff"
    sf.write(pcm_path, np.full(4800, 0.09, dtype=np.float32), 8000, format="AIFF", subtype="FLOAT")
    rows.append(_source_row("pcm-duplicate", pcm_path, categories["source-008"]))
    rows[1]["parent_ids"] = json.dumps([rows[0]["ID"]])
    rows[2]["artist_id"] = rows[3]["artist_id"] = "shared-artist"
    rows[4]["speaker_id"] = rows[5]["speaker_id"] = "shared-speaker"
    singing_path = root / "singing-unknown.wav"
    sf.write(singing_path, np.linspace(-0.2, 0.2, 4800, dtype=np.float32), 8000, subtype="FLOAT")
    singing = _source_row("singing-unknown", singing_path, 0)
    singing.update(kind="singing", FILE_FAKE="", VOICE_FAKE="", MUSIC_FAKE="", VOICE_PRESENT="1", MUSIC_PRESENT="0")
    rows.append(singing)
    corrupt = root / "corrupt.wav"
    corrupt.write_bytes(b"not audio")
    bad_audio = _source_row("corrupt", corrupt, 0)
    false_claim_path = root / "false-claim.wav"
    sf.write(false_claim_path, np.zeros(80, dtype=np.float32), 8000, subtype="FLOAT")
    false_claim = _source_row("false-claim", false_claim_path, 0)
    false_claim.update(kind="instrumental", VOICE_FAKE="1", VOICE_PRESENT="0", MUSIC_PRESENT="1", MUSIC_FAKE="0")
    rows.extend((bad_audio, false_claim))
    manifest = root / "sources.csv"
    with manifest.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=SOURCE_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    return manifest


def test_prepare_preserves_unknowns_and_builds_deterministic_paired_truth(tmp_path: Path) -> None:
    manifest = _fixture_manifest(tmp_path)
    first, second = tmp_path / "prepared-a", tmp_path / "prepared-b"

    audit = prepare_data(manifest, first)
    prepare_data(manifest, second)

    first_rows = [row for split in ("train", "validation", "sealed") for row in read_rows(first / f"{split}.csv")]
    second_rows = [row for split in ("train", "validation", "sealed") for row in read_rows(second / f"{split}.csv")]
    by_id = {row["ID"]: row for row in first_rows}
    assert by_id["source-000"]["group_id"] == by_id["source-001"]["group_id"]
    assert by_id["source-002"]["group_id"] == by_id["source-003"]["group_id"]
    assert by_id["source-004"]["group_id"] == by_id["source-005"]["group_id"]
    assert by_id["source-006"]["group_id"] == by_id["exact-duplicate"]["group_id"]
    assert by_id["source-008"]["audio_sha256"] != by_id["pcm-duplicate"]["audio_sha256"]
    assert by_id["source-008"]["group_id"] == by_id["pcm-duplicate"]["group_id"]
    assert by_id["singing-unknown"]["VOICE_FAKE"] == by_id["singing-unknown"]["MUSIC_FAKE"] == ""
    assert {row["ID"] for row in first_rows}.isdisjoint({"corrupt", "false-claim"})
    assert {item["ID"] for item in audit["quarantined"]} == {"corrupt", "false-claim"}
    assert audit["lineage_overlap"] == {}
    assert audit["verified_pairs"] >= 12
    assert all(Path(row["path"]).is_file() for row in first_rows)
    generated = [row for row in first_rows if row["pair_id"]]
    states = {(row["VOICE_FAKE"], row["MUSIC_FAKE"]) for row in generated if row["view"] == "mixed"}
    assert states == {("0", "0"), ("1", "0"), ("0", "1"), ("1", "1")}
    assert {row["domain"].rsplit("/", 1)[-1] for row in generated} == {"simultaneous", "sequential", "sparse"}
    assert any(
        event["label"] == 1 and event["end"] - event["start"] == pytest.approx(0.5)
        for row in generated if row["view"] == "mixed" and row["domain"].endswith("/sparse")
        for event in json.loads(row["event_intervals"])
    )
    assert verify_lineage(first_rows)["pairs"] == audit["verified_pairs"]
    pairs: dict[str, list[dict[str, str]]] = {}
    for row in generated:
        pairs.setdefault(row["pair_id"], []).append(row)
    for pair_rows in pairs.values():
        pure = next(row for row in pair_rows if row["view"] == "pure")
        mixed = next(row for row in pair_rows if row["view"] == "mixed")
        pure_audio, rate = sf.read(pure["path"], dtype="float32")
        mixed_audio, mixed_rate = sf.read(mixed["path"], dtype="float32")
        assert rate == mixed_rate and len(pure_audio) == len(mixed_audio)
        assert json.loads(pure["event_intervals"])[0] == json.loads(mixed["event_intervals"])[1]
        voice_id = json.loads(mixed["parent_ids"])[0]
        voice_audio, voice_rate = sf.read(by_id[voice_id]["path"], dtype="float32")
        voice_event = json.loads(mixed["event_intervals"])[0]
        expected_voice = np.zeros_like(mixed_audio)
        start, end = round(voice_event["start"] * voice_rate), round(voice_event["end"] * voice_rate)
        expected_voice[start:end] = voice_audio[start:end] * 0.5
        assert np.allclose(mixed_audio - pure_audio, expected_voice, atol=4e-5)
    assert any(row["view"] == "mixed" and row["FILE_FAKE"] == row["VOICE_FAKE"] == row["MUSIC_FAKE"] == "0" for row in generated)
    stable = lambda rows: sorted((row["ID"], row["split"], row["pair_id"], row["audio_sha256"], row["event_intervals"]) for row in rows)
    assert stable(first_rows) == stable(second_rows)


def test_existing_output_and_cross_split_parent_fail_closed(tmp_path: Path) -> None:
    manifest = _fixture_manifest(tmp_path)
    output = tmp_path / "prepared"
    first = prepare_data(manifest, output)
    assert prepare_data(manifest, output) == first
    rows = [row for split in ("train", "validation", "sealed") for row in read_rows(output / f"{split}.csv")]
    mixed = next(row for row in rows if row["view"] == "mixed")
    parent_id = json.loads(mixed["parent_ids"])[0]
    tampered = [dict(row) for row in rows]
    parent = next(row for row in tampered if row["ID"] == parent_id)
    parent["split"] = "sealed" if mixed["split"] != "sealed" else "train"
    with pytest.raises(DataContractError, match="cross-split parent"):
        verify_lineage(tampered)
    Path(mixed["path"]).write_bytes(b"changed")
    with pytest.raises(DataContractError, match="existing output audio differs"):
        prepare_data(manifest, output)


def test_template_is_header_only(tmp_path: Path) -> None:
    template = tmp_path / "sources.csv"
    write_source_template(template)
    assert template.read_text().strip().split(",") == list(SOURCE_FIELDS)
    assert read_rows(template) == []


def test_sealed_panel_identity_survives_source_relocation_and_manifest_reordering(tmp_path: Path) -> None:
    original_root, relocated_root = tmp_path / "original", tmp_path / "relocated"
    original_root.mkdir()
    manifest = _fixture_manifest(original_root)
    original_audit = prepare_data(manifest, tmp_path / "first")
    relocated_root.mkdir()
    for source in original_root.iterdir():
        if source != manifest:
            shutil.copy2(source, relocated_root / source.name)
    rows = read_rows(manifest)
    relocated_manifest = relocated_root / "sources.csv"
    with relocated_manifest.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=SOURCE_FIELDS)
        writer.writeheader()
        writer.writerows(reversed(rows))

    relocated_audit = prepare_data(relocated_manifest, tmp_path / "second")

    assert len(original_audit["sealed_panel_identity"]) == 64
    assert original_audit["sealed_panel_identity"] == relocated_audit["sealed_panel_identity"]
