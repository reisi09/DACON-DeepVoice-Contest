#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy"]
# ///
# How to run: uv run audit_redesign_slices.py
"""Replay historical source slices without fitting or selecting a model."""

from __future__ import annotations

import csv
import json
from collections import Counter
from copy import deepcopy
from datetime import datetime, timezone
from math import isclose, isfinite
from pathlib import Path
from typing import Final

from graph_scoring import official_metrics
from redesign_evaluation_common import artifact_rows, components, file_sha256

ROOT: Final = Path(__file__).resolve().parent
SOURCE: Final = ROOT / "results/redesign_h100_20260917"
OUTPUT: Final = ROOT / "results/leaderboard_audit_20260917"
CANDIDATES: Final = (
    ("A_ensemble_max", "comparison_A_vs_B.json", "control"),
    ("B_ensemble_max", "comparison_A_vs_B.json", "candidate_metrics"),
    ("B_ensemble_noisy_or", "comparison_file_max_vs_noisy_or.json", "candidate_metrics"),
    ("B_ensemble_oof_relation", "comparison_file_max_vs_oof_relation.json", "candidate_metrics"),
    ("B_aligned_voice_noisy_or", "comparison_file_max_vs_noisy_or.json", "candidate_metrics"),
)


def aligned_voice(artifact: dict) -> dict:
    """Change only File fusion; retain the trained music bag and four other outputs."""
    result = deepcopy(artifact)
    for identifier, prediction in result["predictions"].items():
        windows = result["window_scores"][identifier]
        pairs = list(zip(windows["voice_present"], windows["voice_fake"], strict=True))
        assert pairs and all(isfinite(p) and 0 <= p <= 1 for pair in pairs for p in pair)
        # Serialized windows must reproduce the locked scalar voice inputs exactly.
        assert max(windows["voice_present"]) == prediction["VOICE_PRESENT_PROB"]
        assert max(windows["voice_fake"]) == prediction["VOICE_FAKE_PROB"]
        voice = prediction["VOICE_PRESENT_PROB"] * prediction["VOICE_FAKE_PROB"]
        music = prediction["MUSIC_PRESENT_PROB"] * prediction["MUSIC_FAKE_PROB"]
        assert isclose(1 - (1 - voice) * (1 - music), prediction["FILE_FAKE_PROB"], abs_tol=1e-12)
        aligned = max(presence * fake for presence, fake in pairs)
        probability = 1 - (1 - aligned) * (1 - music)
        assert probability <= prediction["FILE_FAKE_PROB"] + 1e-12
        prediction["FILE_FAKE_PROB"] = probability
        assert all(prediction[key] == value for key, value in artifact["predictions"][identifier].items()
                   if key != "FILE_FAKE_PROB")
    return result


def main() -> None:
    """Record exact cohort identities, metric replay, and real-song evidence."""
    reports = {}
    references = {}
    real_rows = []
    cohort_identity = None
    for candidate, reference_name, reference_key in CANDIDATES:
        derived = candidate == "B_aligned_voice_noisy_or"
        parent = "B_ensemble_noisy_or" if derived else candidate
        path = SOURCE / "downloaded/unpacked/output/predictions" / f"{parent}.json"
        artifact = json.loads(path.read_text())
        if derived:
            artifact = aligned_voice(artifact)
        labels, predictions, identifiers = artifact_rows(artifact)
        identity = (identifiers, artifact["labels"], artifact["strata"], artifact["groups"])
        if cohort_identity is not None and identity != cohort_identity:
            message = f"Candidate cohorts differ: {candidate}"
            raise ValueError(message)
        cohort_identity = identity
        connected = components(artifact)
        sources = artifact["strata"]
        masks = {
            "all": identifiers,
            "non_nsynth": [key for key in identifiers if sources[key]["source"] != "nsynth"],
        }
        panels = {}
        for name, selected in masks.items():
            subset = {key: {identifier: artifact[key][identifier] for identifier in selected}
                      for key in ("labels", "predictions")}
            truth, values, _ = artifact_rows(subset)
            panels[name] = {
                "ids": selected,
                "rows": len(selected),
                "independent_groups": len({connected[key] for key in selected}),
                "file_classes": dict(Counter(str(row["FILE_FAKE"]) for row in truth)),
                "metrics": official_metrics(truth, values),
            }
        reference_path = SOURCE / reference_name
        reference = json.loads(reference_path.read_text())[reference_key]
        # Protect against ID/label joins or scoring drift before interpreting a slice.
        if not derived:
            assert panels["all"]["metrics"] == reference, candidate
        else:
            assert all(panels["all"]["metrics"]["metrics"][key] == value
                       for key, value in reference["metrics"].items() if key != "file_eer")
        assert panels["all"]["metrics"]["score"] is None, candidate
        reports[candidate] = panels
        references[candidate] = {
            "path": str(path.relative_to(ROOT)), "sha256": file_sha256(path),
            "report_sha256": file_sha256(reference_path), "aggregate_replay_exact": not derived,
            "transformation": "max_t(vp_t * vf_t), retain music bag, noisy OR" if derived else None,
        }
        for label, prediction in zip(labels, predictions, strict=True):
            identifier = label["ID"]
            if sources[identifier]["source"].startswith("FMA"):
                voice = prediction["VOICE_PRESENT_PROB"] * prediction["VOICE_FAKE_PROB"]
                music = prediction["MUSIC_PRESENT_PROB"] * prediction["MUSIC_FAKE_PROB"]
                real_rows.append({
                    "candidate": candidate, "ID": identifier, "group": connected[identifier],
                    "voice_evidence_product_of_maxima": voice,
                    "voice_evidence_aligned": max(vp * vf for vp, vf in zip(
                        artifact["window_scores"][identifier]["voice_present"],
                        artifact["window_scores"][identifier]["voice_fake"], strict=True)),
                    "music_evidence": music,
                    "file_probability": prediction["FILE_FAKE_PROB"],
                })
    OUTPUT.mkdir(parents=True, exist_ok=True)
    report = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "script_sha256": file_sha256(Path(__file__)),
        "scoring_source_sha256": {name: file_sha256(ROOT / name) for name in
                                  ("graph_scoring.py", "redesign_evaluation_common.py")},
        "post_hoc": True, "fresh_holdout": False, "trained": False,
        "model_selected": None, "official_score": None,
        "limitations": [
            "Historical model-selection validation, not independent test evidence.",
            "Only seven real songs remain after excluding NSynth.",
            "Source slices diagnose sensitivity; their rankings do not justify promotion.",
            "Aligned voice is an unfitted post-hoc hypothesis, not a selected or deployed fix.",
            "Aligned voice only lowers scores; fresh fake-voice recall must be checked alongside real false positives.",
            "Archived extractor emits positive-duration aligned windows; serialized masks are absent, so this replay is scoped to that extraction contract.",
        ],
        "sources": references, "panels": reports,
    }
    (OUTPUT / "source_slice_replay.json").write_text(json.dumps(report, indent=2) + "\n")
    with (OUTPUT / "real_song_evidence.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(real_rows[0]))
        writer.writeheader()
        writer.writerows(real_rows)
    print(json.dumps({name: {key: value["metrics"]["metrics"]["file_eer"]
                           for key, value in panels.items()} for name, panels in reports.items()}))


if __name__ == "__main__":
    main()
