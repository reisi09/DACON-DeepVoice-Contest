#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy", "soundfile", "imageio-ffmpeg"]
# ///
# How to run: .venv/bin/python new_panel_data_merge.py
"""Merge acquired audio under the pre-score whole-artist/prompt partition."""
from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path

import numpy as np

from eight_audio import core, load
from new_panel_data_inventory import DATA, OUT, ROOT
from v2_data import DataIntegrityError, pcm_sha256, sha256_file


def main() -> None:
    """Verify every real source/view before publishing final train and external rows."""
    partition_path = OUT / "data_preparation_additional_partition.json"
    partition = json.loads(partition_path.read_text())
    selection_path = DATA / "additional_training/jamendo/selection.json"
    if sha256_file(selection_path) != partition["jamendo_selection_sha256"]:
        raise DataIntegrityError("Frozen Jamendo selection changed before merge")
    frozen_jamendo = {("new_jamendo_" + str(row["id"]), str(row["artist_id"]))
                      for row in json.loads(selection_path.read_text())}
    external_jamendo = set(partition["jamendo_external_source_ids"])
    sources, views, input_hashes = [], [], {}
    for prefix in ("data_preparation_jamendo", "data_preparation_haim", "sunocaps"):
        source_path = OUT / (prefix + "_sources.csv")
        manifest_path = OUT / (prefix + "_manifest.csv")
        for path in (source_path, manifest_path):
            input_hashes[str(path.relative_to(ROOT))] = sha256_file(path)
        with source_path.open() as stream:
            component_sources = list(csv.DictReader(stream))
        with manifest_path.open() as stream:
            component_views = list(csv.DictReader(stream))
        if prefix == "data_preparation_jamendo" and {(row["ID"], row["artist_id"]) for row in component_sources} != frozen_jamendo:
            raise DataIntegrityError("Jamendo component sources differ from the frozen track/artist selection")
        for source in component_sources:
            external = source["ID"] in external_jamendo or prefix == "sunocaps"
            source["split"] = "confirmation_external" if external else "train"
        splits = {source["ID"]: source["split"] for source in component_sources}
        for view in component_views:
            parents = json.loads(view["parent_ids"])
            if len(parents) != 1 or parents[0] not in splits:
                raise DataIntegrityError("Additional base view must have one acquired source parent")
            view["split"] = splits[parents[0]]
        sources.extend(component_sources)
        views.extend(component_views)
    if len(sources) != 640 or len(views) != 640 or len({row["ID"] for row in sources}) != 640:
        raise DataIntegrityError("Final acquired-source quota changed")
    expected = {("train", "0"): 254, ("train", "1"): 256,
                ("confirmation_external", "0"): 66, ("confirmation_external", "1"): 64}
    if Counter((row["split"], row["MUSIC_FAKE"]) for row in views) != expected:
        raise DataIntegrityError("Frozen train/external class counts changed")
    source_by_id = {row["ID"]: row for row in sources}
    atoms, hashes, pcm_seen, source_pcm_seen = {}, {}, {}, {}
    with (OUT / "data_preparation_music_sources.csv").open() as stream:
        fixed_hashes = {value for row in csv.DictReader(stream) for key, value in row.items() if "sha256" in key}
    with (OUT / "data_preparation_confirmation_manifest.csv").open() as stream:
        fixed_hashes.update(pcm_sha256(load(ROOT / row["path"])) for row in csv.DictReader(stream))
    old_pcm = set(json.loads((OUT / "data_preparation_speech_prior_hashes.json").read_text()))
    for row in sources:
        path = ROOT / row["path"]
        digest = sha256_file(path)
        if digest != row["source_sha256"] or digest in fixed_hashes | old_pcm or digest in hashes:
            raise DataIntegrityError("Additional original changed or duplicates prior/confirmation audio")
        recorded_pcm = row.get("pcm_sha256", row.get("source_pcm_sha256", ""))
        if recorded_pcm and recorded_pcm in fixed_hashes | old_pcm | source_pcm_seen.keys():
            raise DataIntegrityError("Additional decoded original duplicates prior/confirmation audio")
        source_pcm_seen[recorded_pcm] = row["ID"]
        hashes[digest] = row["ID"]
        for atom in row["group_id"].split("|"):
            if atom in atoms and atoms[atom] != row["split"]:
                raise DataIntegrityError("Artist or prompt group crosses the frozen partition")
            atoms[atom] = row["split"]
    for row in views:
        parent = source_by_id[json.loads(row["parent_ids"])[0]]
        reference = core(ROOT / parent["path"], 8)
        audio = load(ROOT / row["path"])
        digest = pcm_sha256(audio)
        if not np.array_equal(reference, audio) or digest in old_pcm | fixed_hashes or digest in pcm_seen:
            raise DataIntegrityError(f"New natural crop did not replay or duplicates recorded audio: {row['ID']}")
        pcm_seen[digest] = row["ID"]
        if sha256_file(ROOT / row["path"]) != row["audio_sha256"]:
            raise DataIntegrityError("Additional view bytes changed")
        if row["VOICE_PRESENT"] or row["VOICE_FAKE"] or row["MUSIC_PRESENT"] != "1":
            raise DataIntegrityError("Unknown voice labels became ground truth")
        events = json.loads(row["event_intervals"])
        if len(events) != 1 or events[0]["start"] != 0 or abs(events[0]["end"] - len(audio) / 16000) > 1e-6:
            raise DataIntegrityError("Full-source interval does not match actual crop length")
        if str(events[0]["label"]) != row["MUSIC_FAKE"] or row["FILE_FAKE"] != row["MUSIC_FAKE"]:
            raise DataIntegrityError("Source authenticity disagrees with file/window labels")
    for name, records in (("sources", sources), ("manifest", views)):
        path = OUT / f"data_preparation_additional_training_{name}.csv"
        fields = list(dict.fromkeys(key for row in records for key in row))
        with path.open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            writer.writerows(records)
    receipt = {"status": "complete", "source_parents": 640, "train": 510, "confirmation_external": 130,
               "class_counts": {f"{split}:{label}": count for (split, label), count in expected.items()},
               "jamendo_artists": 135, "jamendo_external_artists": 27, "sunocaps_external_prompt_groups": 64,
               "exact_parent_view_replays": 640, "cross_split_source_atom_overlap": 0,
               "prior_or_fixed_confirmation_exact_source_hash_overlap": 0,
               "partition_sha256": sha256_file(partition_path), "component_sha256": input_hashes,
               "manifest_sha256": sha256_file(OUT / "data_preparation_additional_training_manifest.csv"),
               "source_manifest_sha256": sha256_file(OUT / "data_preparation_additional_training_sources.csv"),
               "script_sha256": sha256_file(Path(__file__)), "evaluated": False,
               "limitations": partition["limitations"] + ["SunoCaps includes one natural5s source; no repeat/padding was added",
                                                            "No independent listening review; music voice labels remain unknown"]}
    (OUT / "data_preparation_additional_training.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt))


if __name__ == "__main__":
    main()
