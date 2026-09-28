#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy", "soundfile", "imageio-ffmpeg"]
# ///
# How to run: .venv/bin/python new_panel_data_check.py
"""Replay actual confirmation audio to detect leakage, mask and splice regressions."""
from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path

import numpy as np

from eight_audio import core, load, replace_event
from new_panel_data_inventory import OUT, ROOT
from v2_data import DataIntegrityError, sha256_file


def main() -> None:
    """Validate original-history separation and every generated view from real parents."""
    source_manifest = OUT / "data_preparation_music_sources.csv"
    view_manifest = OUT / "data_preparation_confirmation_manifest.csv"
    with source_manifest.open() as stream:
        sources = {row["ID"]: row for row in csv.DictReader(stream)}
    with view_manifest.open() as stream:
        views = list(csv.DictReader(stream))
    history_path = OUT / "data_preparation_history.json"
    history = json.loads(history_path.read_text())
    broad_path = OUT / "data_preparation_speech_prior_hashes.json"
    old_hashes = set(history["recorded_hashes"]) | set(json.loads(broad_path.read_text()))
    if len(sources) != 64 or Counter(row["MUSIC_FAKE"] for row in sources.values()) != {"0": 32, "1": 32}:
        raise DataIntegrityError("Independent music parent quota changed")
    audio = {}
    original_starts = {}
    for source in sources.values():
        if source["source_sha256"] in old_hashes or source["pcm_sha256"] in old_hashes:
            raise DataIntegrityError("Recorded historical hash overlap")
        if source["artist_id"] in history["excluded_artists"]:
            raise DataIntegrityError("Historical FMA artist overlap")
        if source["reference_id"].removeprefix("sonics:") in history["excluded_sonics_ids"]:
            raise DataIntegrityError("Historical SONICS parent overlap")
        path = ROOT / source["path"]
        if sha256_file(path) != source["source_sha256"]:
            raise DataIntegrityError("Downloaded parent bytes changed")
        full = load(path)
        original_starts[source["ID"]] = (len(full) - 60 * 16000) // 2
        audio[source["ID"]] = core(path, 60)
    pairs = {}
    for row in views:
        parents = json.loads(row["parent_ids"])
        if set(parents) - sources.keys() or row["split"] != "confirmation":
            raise DataIntegrityError("View source lineage or split changed")
        if row["VOICE_FAKE"] or row["VOICE_PRESENT"]:
            raise DataIntegrityError("Unverified voice labels became ground truth")
        expected_groups = {sources[parent]["group_id"] for parent in parents}
        if set(row["group_id"].split("|")) != expected_groups:
            raise DataIntegrityError("A shared audio parent is missing from the split group")
        offsets = json.loads(row["source_offsets"])
        segments = []
        for offset in offsets:
            parent = offset["parent_id"]
            start = round(offset["start"] * 16000) - original_starts[parent]
            end = round(offset["end"] * 16000) - original_starts[parent]
            if not 0 <= start < end <= len(audio[parent]):
                raise DataIntegrityError("View extends beyond natural parent audio")
            segments.append(audio[parent][start:end])
        events = json.loads(row["event_intervals"])
        reference = segments[0]
        if row["probe_family"] == "partial":
            event = events[0]
            reference = replace_event(segments[0], segments[1], round(event["start"] * 16000), fade_samples=160)
            if event["end"] - event["start"] != 2 or str(event["label"]) != row["MUSIC_FAKE"]:
                raise DataIntegrityError("Partial label disagrees with event boundaries")
            pairs.setdefault(row["pair_id"], []).append(row)
        elif int(row["MUSIC_FAKE"]) != events[0]["label"]:
            raise DataIntegrityError("Natural source label disagrees with full interval")
        path = ROOT / row["path"]
        if sha256_file(path) != row["audio_sha256"] or not np.array_equal(load(path), reference):
            raise DataIntegrityError(f"Actual audio does not replay from its recorded parents: {row['ID']}")
        if row["FILE_FAKE"] != row["MUSIC_FAKE"] or row["MUSIC_PRESENT"] != "1":
            raise DataIntegrityError("Music/file construction truth became inconsistent")
    for pair in pairs.values():
        if len(pair) != 2 or {row["MUSIC_FAKE"] for row in pair} != {"0", "1"}:
            raise DataIntegrityError("Missing matched real-real control")
        if len({row["parent_ids"] for row in pair}) != 1 or len({row["group_id"] for row in pair}) != 1:
            raise DataIntegrityError("Matched edit controls have different lineage groups")
    receipt = {"status": "passed", "sources": len(sources), "views_replayed_exactly": len(views),
               "matched_partial_sham_pairs": len(pairs), "broader_historical_hashes": len(old_hashes),
               "recorded_history_overlap": 0, "voice_labels_remain_unknown": True,
               "source_manifest_sha256": sha256_file(source_manifest), "view_manifest_sha256": sha256_file(view_manifest),
               "historical_hash_manifest_sha256": sha256_file(broad_path), "check_script_sha256": sha256_file(Path(__file__)),
               "decision": "Inputs support new-parent music confirmation and edit sensitivity; no model quality claim"}
    (OUT / "data_preparation_checks.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt))


if __name__ == "__main__":
    main()
