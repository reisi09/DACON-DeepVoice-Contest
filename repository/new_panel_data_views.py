#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy", "soundfile", "imageio-ffmpeg"]
# ///
# How to run: .venv/bin/python new_panel_data_views.py
"""Create bounded natural-duration and matched edit confirmation views."""
from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Final

import numpy as np

from eight_audio import core, load, manifest_row, replace_event, write
from new_panel_data_inventory import DATA, OUT, ROOT
from v2_data import DataIntegrityError, decode_audio, sha256_file, stable_key

SR: Final = 16_000
DURATIONS: Final = (4, 8, 16, 30, 60)


def main() -> None:
    """Preserve provenance and exact edit boundaries without extending recordings."""
    source_path = OUT / "data_preparation_music_sources.csv"
    with source_path.open() as stream:
        sources = list(csv.DictReader(stream))
    if len(sources) != 64 or any(float(row["duration_seconds"]) < 60 for row in sources):
        raise DataIntegrityError("Expected all 64 frozen parents with natural 60s intervals")
    rows = []
    audio_by_id = {}
    offset_by_id = {}
    gain_by_id = {}
    provenance = []
    for source in sources:
        path = ROOT / source["path"]
        if sha256_file(path) != source["source_sha256"]:
            raise DataIntegrityError(f"Original source changed: {path}")
        full = load(path)
        start = (len(full) - 60 * SR) // 2
        segment = full[start:start + 60 * SR]
        gain = min(1.0, .6 / max(float(np.max(np.abs(segment))), 1e-8))
        audio = core(path, 60)
        if not np.array_equal(audio, segment * gain):
            raise DataIntegrityError("Existing core decoder changed the source interval")
        audio_by_id[source["ID"]] = audio
        offset_by_id[source["ID"]] = start / SR
        gain_by_id[source["ID"]] = gain
        for seconds in DURATIONS:
            offset = (60 * SR - seconds * SR) // 2
            identifier = source["ID"] + f"_{seconds}s"
            target = DATA / "confirmation_audio" / (identifier + ".wav")
            write(target, audio[offset:offset + seconds * SR])
            label = source["MUSIC_FAKE"]
            events = [{"component": "music", "start": 0.0, "end": seconds, "label": int(label)}]
            row = manifest_row(identifier, target, [source["ID"]], {source["group_id"]}, "confirmation",
                               source["source"], source["generator"], "music", f"{seconds}s", "core",
                               "duration:" + source["ID"], (label, "", label, "", "1"), events, ROOT)
            row["source_offsets"] = json.dumps([{"parent_id": source["ID"], "start": (start + offset) / SR,
                                                 "end": (start + offset + seconds * SR) / SR, "gain": gain}])
            row["label_status"] = source["label_status"]
            rows.append(row)
    real = sorted((row for row in sources if row["MUSIC_FAKE"] == "0"), key=lambda r: stable_key(r["ID"]))
    fake = sorted((row for row in sources if row["MUSIC_FAKE"] == "1"), key=lambda r: stable_key(r["ID"]))
    replay_checks = 0
    for pair_index in range(4):
        bed_source, sham_source, fake_source = real[pair_index], real[pair_index + 4], fake[pair_index]
        parents = [bed_source["ID"], sham_source["ID"], fake_source["ID"]]
        atoms = {row["group_id"] for row in (bed_source, sham_source, fake_source)}
        for seconds in (8, 16, 30, 60):
            length = seconds * SR
            offset = (60 * SR - length) // 2
            bed = audio_by_id[bed_source["ID"]][offset:offset + length]
            event_length = 2 * SR
            starts = (0, (length - event_length) // 2, length - event_length)
            start = starts[pair_index % 3]
            pair = f"confirmation_edit:{pair_index}:{seconds}"
            for label, donor in ((0, sham_source), (1, fake_source)):
                donor_offset = 29 * SR
                event = audio_by_id[donor["ID"]][donor_offset:donor_offset + event_length]
                value = replace_event(bed, event, start, fade_samples=160)
                identifier = f"new_edit_{pair_index}_{seconds}s_{label}"
                target = DATA / "confirmation_audio" / (identifier + ".wav")
                write(target, value)
                replay = replace_event(bed, event, start, fade_samples=160)
                if not np.array_equal(decode_audio(target, limit_seconds=None), replay):
                    raise DataIntegrityError("Written construction differs from parent replay")
                replay_checks += 1
                events = [{"component": "music", "start": start / SR, "end": (start + event_length) / SR, "label": label}]
                row = manifest_row(identifier, target, parents, atoms, "confirmation", "constructed:matched_music",
                                   donor["generator"], "music", "partial_fake" if label else "sham_real", "partial",
                                   pair, (str(label), "", str(label), "", "1"), events, ROOT)
                offsets = [{"parent_id": bed_source["ID"], "start": offset_by_id[bed_source["ID"]] + offset / SR,
                            "end": offset_by_id[bed_source["ID"]] + (offset + length) / SR,
                            "gain": gain_by_id[bed_source["ID"]]},
                           {"parent_id": donor["ID"], "start": offset_by_id[donor["ID"]] + donor_offset / SR,
                            "end": offset_by_id[donor["ID"]] + (donor_offset + event_length) / SR,
                            "gain": gain_by_id[donor["ID"]]}]
                row["source_offsets"] = json.dumps(offsets)
                row["label_status"] = "source_documented_parent_authenticity_and_exact_construction; voice_unknown; no_independent_listening"
                rows.append(row)
                provenance.append({"ID": identifier, "pair_id": pair, "actual_audio_parents": [bed_source["ID"], donor["ID"]],
                                   "split_group_parents": parents, "fade_samples": 160,
                                   "rms_matching": "existing eight_audio.replace_event", "event_intervals": events,
                                   "sample_support_note": "Nominal edit interval; the first fade sample has zero donor weight"})
    if len(rows) != 352 or len({row["ID"] for row in rows}) != 352:
        raise DataIntegrityError("Confirmation construction quota or identity violation")
    for row in rows:
        if row["VOICE_PRESENT"] or row["VOICE_FAKE"]:
            raise DataIntegrityError("Unverified music-source voice labels must remain masked")
        for event in json.loads(row["event_intervals"]):
            if not 0 <= event["start"] < event["end"] <= float(row["duration_seconds"]):
                raise DataIntegrityError("Interval crosses the real audio boundary")
    manifest = OUT / "data_preparation_confirmation_manifest.csv"
    with manifest.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    receipt = {"status": "complete", "manifest_sha256": sha256_file(manifest),
               "source_manifest_sha256": sha256_file(source_path), "script_sha256": sha256_file(Path(__file__)),
               "rows": len(rows), "natural_duration_rows": 320, "paired_partial_sham_rows": 32,
               "independent_source_parents": 64, "construction_replays_identical": replay_checks,
               "source_repetition_or_padding": False, "crossfade_ms": 10, "source_gain": "one peak ceiling per central60s parent",
               "source_decoder": "eight_audio.load (same canonical source decoder as the existing training builder)",
               "selection_used_predictions": False, "evaluated": False, "independent_listening": False,
               "limitations": ["Same known FMA/SONICS corpora and generators", "The 352 views are dependent observations from64 parents",
                               "Mixed music voice labels are unknown", "Constructed splices do not represent all naturally edited music"],
               "constructions": provenance}
    (OUT / "data_preparation_confirmation.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps({"rows": len(rows), "parent_sources": len(sources), "replayed_constructions": replay_checks}))


if __name__ == "__main__":
    main()
