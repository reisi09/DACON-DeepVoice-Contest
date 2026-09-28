#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy", "imageio-ffmpeg"]
# ///
# How to run: .venv/bin/python new_panel_data_partition.py
"""Freeze whole-artist external holdouts before merging newly acquired sources."""
from __future__ import annotations

import json
from datetime import datetime, timezone

from new_panel_data_inventory import DATA, OUT
from v2_data import DataIntegrityError, sha256_file, stable_key


def main() -> None:
    """Keep HAIM in training and entire SunoCaps prompt groups in confirmation."""
    selection_path = DATA / "additional_training/jamendo/selection.json"
    selected = json.loads(selection_path.read_text())
    artists = sorted({row["artist_id"] for row in selected},
                     key=lambda artist: stable_key("20260918|external|jamendo|" + artist))
    external_artists = set(artists[:round(len(artists) * .2)])
    external_ids = sorted("new_jamendo_" + str(row["id"]) for row in selected if row["artist_id"] in external_artists)
    frozen = {"seed": 20260918, "jamendo_selection_sha256": sha256_file(selection_path),
              "jamendo_external_artist_ids": sorted(external_artists), "jamendo_external_source_ids": external_ids,
              "expected_new_train": {"JamendoMaxCaps": 254, "HAIM": 256},
              "expected_confirmation_external": {"JamendoMaxCaps": 66, "SunoCaps": 64},
              "HAIM_rule": "all train; missing MusicGen prompt/reference binding prevents a defensible internal group holdout",
              "SunoCaps_rule": "all 64 acquired unique prompt groups in confirmation_external, none in training",
              "original_64_confirmation": "unchanged", "selection_uses_model_outputs": False,
              "limitations": ["Cross-catalogue artist aliases unresolved beyond exact normalized-name exclusion",
                              "Unknown HAIM conditioning references are not claimed independent of every external source",
                              "Suno, ACE-Step and MusicGen are known generator families in this project"]}
    path = OUT / "data_preparation_additional_partition.json"
    if path.exists():
        existing = json.loads(path.read_text())
        if {key: value for key, value in existing.items() if key != "frozen_utc"} != frozen:
            raise DataIntegrityError("Previously frozen external partition differs")
    else:
        path.write_text(json.dumps(frozen | {"frozen_utc": datetime.now(timezone.utc).isoformat()}, indent=2) + "\n")
    if len(external_artists) != 27 or len(external_ids) != 66:
        raise DataIntegrityError("Approved whole-artist external partition quota changed")
    print(json.dumps({"partition_sha256": sha256_file(path), "heldout_real_artists": len(external_artists),
                      "heldout_real_tracks": len(external_ids), "new_real_train": len(selected) - len(external_ids)}))


if __name__ == "__main__":
    main()
