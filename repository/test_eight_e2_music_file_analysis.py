from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import torch

from eight_contract import MANIFEST_FIELDS
from eight_e2_music_file_analysis import AnalysisError, run
from eight_heads import FrozenHead


def _fixture(tmp_path: Path) -> tuple[Path, Path]:
    dataset, output = tmp_path / "dataset", tmp_path / "output"
    features, checkpoints = output / "features", output / "checkpoints"
    dataset.mkdir(); features.mkdir(parents=True); checkpoints.mkdir()
    rows: list[dict[str, str]] = []

    def add(identifier: str, split: str, kind: str, variant: str, family: str, pair: str,
            file_label: str, voice_label: str, music_label: str, parents: list[str]) -> None:
        row = {name: "" for name in MANIFEST_FIELDS}
        row.update(ID=identifier, path=f"audio/{identifier}.wav", parent_id=parents[0],
                   parent_ids=json.dumps(parents), group_id="|".join(sorted(parents)), split=split,
                   source="constructed:E2" if family == "swap" else "fixture",
                   generator="sonics" if "sonics" in variant else "", kind=kind, variant=variant,
                   probe_family=family, pair_id=pair, FILE_FAKE=file_label,
                   VOICE_FAKE=voice_label, MUSIC_FAKE=music_label, VOICE_PRESENT="1",
                   MUSIC_PRESENT="0" if "noise" in variant else "1", audio_sha256=identifier.ljust(64, "0")[:64],
                   duration_seconds="4", event_intervals="[]")
        rows.append(row)
        label = int(music_label or voice_label or file_label or 0)
        value = np.full((2, 4), 0.2 + 0.6 * label, dtype=np.float32)
        identity = json.dumps({"audio_sha256": row["audio_sha256"]})
        np.savez_compressed(features / f"{identifier}.npz", valid_samples=np.array([64000, 64000]),
                            mert_l24=value, anti_l24=value, presence=np.full((2, 2), 0.9, dtype=np.float32),
                            identity_json=np.array(identity))

    for split in ("train", "calibration", "dev", "test"):
        for kind, label_name in (("music", "music"), ("speech", "voice")):
            for label in (0, 1):
                add(f"{split}_{kind}_{label}", split, kind, "base", "core", f"core:{split}:{kind}:{label}",
                    str(label), str(label) if label_name == "voice" else "", str(label) if label_name == "music" else "",
                    [f"{split}:{kind}:{label}"])
        for vocal_fake in (0, 1):
            parent = f"{split}:vocal:{vocal_fake}"
            for background, music_fake in (("fma_a", "0"), ("fma_b", "0"), ("sonics_fake", "1"), ("noise", "")):
                file_fake = "1" if vocal_fake or music_fake == "1" else "0"
                add(f"{split}_{vocal_fake}_{background}", split, "mixed", f"swap_{background}_minus6dB", "swap",
                    f"e2:{split}:{vocal_fake}", file_fake, "" if background != "noise" else str(vocal_fake), music_fake,
                    [parent, f"{split}:backing:{background}"] if background != "noise" else [parent])
    with (dataset / "manifest.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=MANIFEST_FIELDS); writer.writeheader(); writer.writerows(rows)
    for branch in ("music", "voice"):
        for arm, mode in (("parent", "parent"), ("balanced", "source_generator_parent")):
            for seed in (41, 42, 43):
                model = FrozenHead(4, "mlp")
                torch.save({"model": model.state_dict(), "input_width": 4, "steps": 500, "batch_size": 32,
                            "seed": seed, "mode": mode, "consistency_weight": 0.0, "architecture": "mlp",
                            "draw_counts": {"class_0": 8000, "class_1": 8000}, "draw_sha256": str(seed) * 32},
                           checkpoints / f"E1_{branch}_{arm}_seed{seed}.pt")
    return dataset, output


def test_postrun_analysis_locks_calibration_before_heldout_scoring(tmp_path: Path) -> None:
    dataset, output = _fixture(tmp_path)
    analysis = tmp_path / "analysis"

    report = run(dataset, output, analysis)

    lock = analysis / "selection_E2_music_file_postrun_locked.json"
    predictions = analysis / "predictions_E2_music_file.csv"
    assert set(report["arms"]) == {"parent", "balanced"}
    assert lock.stat().st_mtime_ns <= predictions.stat().st_mtime_ns
    locked = json.loads(lock.read_text())
    assert all(identifier.startswith("calibration_") for identifier in locked["calibration_ids"])
    assert all("noise" not in identifier for arm in locked["threshold_calibration_ids"].values()
               for identifiers in arm.values() for identifier in identifiers)
    assert report["arms"]["parent"]["test"]["truth_changing_file"]["pairs"] == 1
    assert report["arms"]["parent"]["test"]["already_fake_file"]["pairs"] == 1
    assert report["noise_rows_excluded_from_truth_contrasts"] > 0
    assert len(json.loads((analysis / "normalizer_receipt.json").read_text())["normalizers"]) == 2


def test_postrun_analysis_rejects_checkpoint_metadata_drift(tmp_path: Path) -> None:
    dataset, output = _fixture(tmp_path)
    path = output / "checkpoints/E1_music_parent_seed41.pt"
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    checkpoint["steps"] = 499
    torch.save(checkpoint, path)

    try:
        run(dataset, output, tmp_path / "analysis")
    except AnalysisError as error:
        assert "metadata" in str(error)
    else:
        raise AssertionError("drifted checkpoint metadata was accepted")
