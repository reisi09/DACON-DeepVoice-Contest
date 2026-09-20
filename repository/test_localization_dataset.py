"""Protect construction boundaries, matched processing, and contributor isolation."""
from pathlib import Path
import csv
import json

import numpy as np
import pytest

from eight_audio import SR, load, manifest_row, replace_event, write
from eight_contract import write_manifest
from localization_dataset import DatasetError, audit_dataset, construct_training_partials, validate_partition


def test_partial_and_sham_preserve_the_same_bed_outside_the_true_interval(tmp_path: Path) -> None:
    # Given: genuine 8-second fixture beds and inserts, plus a held-out donor that must never enter train.
    source, output = tmp_path / "source", tmp_path / "output"
    time = np.arange(8 * SR, dtype=np.float32) / SR
    rows = []
    for name, label, split, frequency in (("real_a", 0, "train", 91), ("real_b", 0, "train", 173),
                                           ("fake", 1, "train", 279), ("held_fake", 1, "test", 431)):
        path = source / "audio" / f"{name}.wav"
        write(path, (0.2 * np.sin(2 * np.pi * frequency * time)).astype(np.float32))
        rows.append(manifest_row(name, path, [name + "_source"], {name + "_atom"}, split,
                                 "FMA" if not label else "suno", "" if not label else "suno",
                                 "music", "base", "core", "", (str(label), "", str(label), "", "1"), [], source))

    # When: the real construction path generates a balanced duration/position set.
    built = construct_training_partials(rows, source, output, pairs=12, seed=41)
    receipt = json.loads((output / "construction_receipt.json").read_text())

    # Then: both arms use identical processing and timing; only the declared event can differ.
    by_id = {row["ID"]: row for row in rows}
    for item in receipt:
        pair = [row for row in built if row["pair_id"] == item["pair_id"]]
        bed = load(source / by_id[item["bed_id"]]["path"]) * item["common_gain"]
        start, end = item["start_sample"], item["end_sample"]
        assert len(pair) == 2
        for row in pair:
            actual = load(output / row["path"])
            donor_id = item["fake_id"] if row["MUSIC_FAKE"] == "1" else item["sham_id"]
            donor = load(source / by_id[donor_id]["path"])
            offset = item["insert_start_sample"]
            expected = replace_event(bed, donor[offset:offset + end - start] * item["common_gain"], start)
            np.testing.assert_array_equal(actual, expected)
            np.testing.assert_array_equal(actual[:start], bed[:start])
            np.testing.assert_array_equal(actual[end:], bed[end:])
            assert np.any(actual[start:end] != bed[start:end])
            assert json.loads(row["parent_ids"]) == [item["bed_id"], donor_id]
            assert row["split"] == "train" and "held_fake" not in row["parent_ids"]
            assert row["VOICE_FAKE"] == row["VOICE_PRESENT"] == ""
            assert json.loads(row["event_intervals"])[0] == {
                "component": "music", "start": start / SR, "end": end / SR, "label": int(row["MUSIC_FAKE"])}
        first, second = [load(output / row["path"]) for row in pair]
        assert np.any(first[start:end] != second[start:end])
    assert {item["event_seconds"] for item in receipt} == {.25, .5, 1, 2}
    assert {item["position"] for item in receipt} == {"start", "mid", "end"}


def test_partial_builder_refuses_to_repeat_short_or_cross_split_donors(tmp_path: Path) -> None:
    # Given: source provenance cannot supply a real and fake 8-second training cohort.
    path = tmp_path / "short.wav"
    write(path, np.full(SR, .1, dtype=np.float32))
    row = manifest_row("short", path, ["s"], {"a"}, "train", "FMA", "", "music", "base", "core", "",
                       ("0", "", "0", "", "1"), [], tmp_path)

    # When/Then: refuse the missing known source material rather than pad, repeat, or borrow test audio.
    with pytest.raises(DatasetError):
        construct_training_partials([row], tmp_path, tmp_path / "out", pairs=4, seed=41)


@pytest.mark.parametrize("extra_split", ["confirmation", "train"])
def test_audit_rejects_original_source_aliases_even_when_crop_hashes_and_ids_differ(tmp_path: Path, extra_split: str) -> None:
    # Given: differently named/cropped new files from the same original source bytes.
    rows = []
    with (tmp_path / "sources.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=("ID", "group_id", "source_sha256"))
        writer.writeheader()
        for index, split in enumerate(("train", extra_split)):
            path = tmp_path / f"{index}.wav"
            write(path, np.full(SR, .1 + index * .1, dtype=np.float32))
            parent, atom = f"parent{index}", f"alias{index}"
            writer.writerow({"ID": parent, "group_id": atom, "source_sha256": "a" * 64})
            rows.append(manifest_row(f"view{index}", path, [parent], {atom}, split, "FMA", "", "music", "base",
                                     "core", "", ("0", "", "0", "", "1"), [], tmp_path))
    write_manifest(tmp_path / "manifest.csv", rows)
    (tmp_path / "original_music_ids.json").write_text(json.dumps({"train": ["view0"]}))

    # When/Then: original-source identity prevents renamed/cropped source leakage.
    with pytest.raises(DatasetError, match="overlap"):
        audit_dataset(tmp_path)
    report = json.loads((tmp_path / "dataset_audit.json").read_text())
    if extra_split == "confirmation":
        assert report["cross_split_original_source_hashes"] == {"a" * 64: ["confirmation", "train"]}
    else:
        assert report["additional_train_original_source_hash_overlap"] == ["a" * 64]


@pytest.mark.parametrize("field,value", [("parent_ids", "[]"), ("parent_id", "missing"), ("MUSIC_FAKE", "7"),
                                         ("duration_seconds", "nan")])
def test_audit_rejects_missing_contributors_or_invalid_truth(tmp_path: Path, field: str, value: str) -> None:
    # Given: an imported row whose contributor or target contract is malformed.
    path = tmp_path / "audio.wav"
    write(path, np.full(SR, .1, dtype=np.float32))
    row = manifest_row("row", path, ["source"], {"atom"}, "confirmation", "FMA", "", "music", "base",
                       "core", "", ("0", "", "0", "", "1"), [], tmp_path)
    row[field] = value
    write_manifest(tmp_path / "manifest.csv", [row])
    with (tmp_path / "sources.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=("ID", "group_id", "source_sha256"))
        writer.writeheader()
        writer.writerow({"ID": "source", "group_id": "atom", "source_sha256": "a" * 64})

    # When/Then: malformed labels/lineage cannot produce a successful audit receipt.
    with pytest.raises(DatasetError):
        audit_dataset(tmp_path)


def test_partition_rejects_same_count_external_parent_swaps(tmp_path: Path) -> None:
    # Given: an imported external row has the right source/count but a different parent from the pre-score lock.
    locked = tmp_path / "partition.json"
    locked.write_text(json.dumps({"expected_new_train": {}, "expected_confirmation_external": {"JamendoMaxCaps": 1},
                                  "jamendo_external_source_ids": ["locked_parent"], "selection_uses_model_outputs": False}))
    row = {"source": "JamendoMaxCaps", "split": "confirmation_external", "parent_ids": '["swapped_parent"]'}

    # When/Then: identical row counts cannot hide a changed held-out source assignment.
    with pytest.raises(DatasetError, match="parent assignment"):
        validate_partition([row], locked)
    row["parent_ids"] = '["locked_parent"]'
    validate_partition([row], locked)
