"""Bounded CPU entrypoint integration over a small, synthetic fixture.

This deliberately uses reduced-width finite features so the real report and
selection paths can be exercised without loading H100-only encoders.  It is
integration evidence for control flow, masks, and report contracts, never
experiment or performance evidence.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import soundfile as sf

from eight_contract import MANIFEST_FIELDS


def _row(identifier: str, split: str, kind: str, label: int, variant: str, family: str,
         parent: str, *, source: str = "fixture", generator: str = "fixture") -> dict[str, str]:
    event = [] if label == 0 else [{"component": "voice", "start": 1.0, "end": 1.5, "label": 1}]
    values = {name: "" for name in MANIFEST_FIELDS}
    values.update(
        ID=identifier, path=f"audio/{identifier}.wav", parent_id=parent,
        parent_ids=json.dumps([parent]), group_id=f"{parent}|{kind}:{split}:{label}", split=split,
        source=source, generator=generator if label else "", kind=kind, variant=variant,
        probe_family=family, pair_id=f"pair:{parent}", FILE_FAKE=str(label),
        VOICE_FAKE=str(label) if kind in {"speech", "mixed"} else "",
        MUSIC_FAKE=str(label) if kind == "music" else "", VOICE_PRESENT="1" if kind != "music" else "0",
        MUSIC_PRESENT="1" if kind != "speech" else "0", audio_sha256="0" * 64,
        duration_seconds="30.00000000", event_intervals=json.dumps(event),
    )
    return values


def _features(path: Path, identity: str, label: int) -> None:
    count = 4
    rng = np.random.default_rng(abs(hash(identity)) % (2**32))
    values: dict[str, np.ndarray] = {
        "starts": np.arange(count, dtype=np.int64) * 100,
        "valid_samples": np.full(count, 64600, dtype=np.int64),
        "padding_samples": np.zeros(count, dtype=np.int64),
        "raw_mask": np.ones(count, dtype=np.uint8), "stem_mask": np.ones(count, dtype=np.uint8),
        "presence": np.full((count, 2), 0.5, dtype=np.float32),
        "spectrum_raw": rng.normal(size=(count, 448)).astype(np.float32),
        "spectrum_stem": rng.normal(size=(count, 448)).astype(np.float32),
    }
    for prefix in ("mert_l6", "mert_l12", "mert_l24", "anti_l12", "anti_l24", "anti_l48"):
        values[prefix] = rng.normal(size=(count, 8)).astype(np.float32)
    # Make class signal finite and deterministic while retaining nontrivial rows.
    base = np.float32(label * 0.75)
    values["mert_l24"] += base
    values["anti_l24"] += base
    probs = np.asarray([0.1, 0.1, 0.9, 0.9] if label else [0.1, 0.1, 0.1, 0.1], dtype=np.float32)
    for key in ("df_raw", "df_vocals", "anti_raw", "anti_vocals"):
        values[key] = probs.copy()
    values["identity_json"] = np.asarray(json.dumps({"audio_sha256": "0" * 64}))
    np.savez_compressed(path, **values)


def _fixture(root: Path) -> None:
    dataset, feature_dir = root / "source/dataset", root / "output/features"
    dataset.joinpath("audio").mkdir(parents=True)
    feature_dir.mkdir(parents=True)
    rows: list[dict[str, str]] = []
    # Every branch has both classes in every role; processed views are needed
    # for E3 and an unseen chain is present only in holdout roles.
    for kind in ("music", "speech"):
        for split in ("train", "calibration", "dev", "test"):
            for label in (0, 1):
                repeats = (0, 1) if kind == "music" and split == "train" and label == 0 else (0,)
                for repeat in repeats:
                    parent = f"{kind}_{split}_{label}_{repeat}"
                    variants = ("base", "telephone", "mp3_32k") if split == "train" else (
                        "base", "telephone", "mp3_32k", "opus_telephone")
                    for variant in variants:
                        identifier = f"{parent}_{variant}"
                        rows.append(_row(identifier, split, kind, label, variant, "core", parent,
                                         source="FMA" if kind == "music" else "MLAAD",
                                         generator="g1" if label else ""))
    # E2 swap rows bridge a speech parent to a background parent in the same role.
    for split in ("train", "calibration", "dev", "test"):
        for label in (0, 1):
            parent = f"speech_{split}_{label}"
            identifier = f"swap_{split}_{label}"
            row = _row(identifier, split, "mixed", label, "swap_fma_0dB", "swap", parent,
                       source="FMA", generator="")
            row["parent_ids"] = json.dumps([parent, f"background_{split}"])
            row["group_id"] = f"background_{split}|{parent}"
            rows.append(row)
    # E5 has matched real controls and fake events in calibration/dev/test.
    for split in ("calibration", "dev", "test"):
        for label in (0, 1):
            parent = f"partial_{split}_{label}"
            rows.append(_row(f"partial_{split}_{label}", split, "speech", label,
                             f"partial_{'fake' if label else 'real'}_30s_0.5s_middle", "partial", parent,
                             source="constructed:E5", generator=""))
    with (dataset / "manifest.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=MANIFEST_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    # Entrypoints do not decode fixture audio; valid mono WAVs keep the fixture
    # realistic and make accidental decoder coupling visible.
    audio = np.sin(np.arange(480000, dtype=np.float32) * (2 * np.pi * 220 / 16000)) * 0.1
    for row in rows:
        sf.write(dataset / row["path"], audio, 16000, subtype="FLOAT")
        _features(feature_dir / f"{row['ID']}.npz", row["ID"], int(row["FILE_FAKE"]))


def test_all_cpu_entrypoints_emit_locked_reports(tmp_path: Path) -> None:
    _fixture(tmp_path)
    config = tmp_path / "config.json"
    config.write_text(json.dumps({
        "root": str(tmp_path), "shared": str(tmp_path / "shared"), "seeds": [41, 42, 43],
        "head_steps": 2, "head_batch_size": 4, "e5_target_fpr": 0.05,
    }), encoding="utf-8")

    from eight_ablation import run as run_ablation
    from eight_probes import run as run_probes

    fixed = run_ablation(config)
    probes = run_probes(config)
    assert set(fixed) == {"1", "3", "4", "6", "8"}
    assert set(probes) == {"E2", "E5"}
    for number in (1, 3, 4, 6, 8):
        report = json.loads((tmp_path / "output" / f"experiment_{number}.json").read_text())
        assert report["status"] == "complete"
        assert report["official_score"] is None
        assert report["selection_locked"]
        assert "test" in json.dumps(report)
    for number in (2, 5):
        report = json.loads((tmp_path / "output" / f"experiment_{number}.json").read_text())
        assert report["status"] == "complete"
        assert report["selection_locked"]
    lock = json.loads((tmp_path / "output/selection_E5_locked.json").read_text())
    assert lock["test_used"] is False
