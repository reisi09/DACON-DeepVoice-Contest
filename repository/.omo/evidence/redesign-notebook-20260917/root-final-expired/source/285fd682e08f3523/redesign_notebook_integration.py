#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy", "scipy", "scikit-learn", "torch"]
# ///

# ─── How to run ───
# 1. Install uv (if not installed):
#      curl -LsSf https://astral.sh/uv/install.sh | sh
# 2. Run directly (no venv, no pip install needed):
#      uv run redesign_notebook_integration.py .omo/evidence/redesign-notebook-20260917/cpu-integration
# 3. Or make executable and run:
#      chmod +x redesign_notebook_integration.py && ./redesign_notebook_integration.py .omo/evidence/redesign-notebook-20260917/cpu-integration
# ──────────────────
"""Deterministic CPU algorithm-integration QA for the redesign notebook."""

from __future__ import annotations

import csv
import hashlib
import json
import shutil
import sys
from pathlib import Path
from typing import Final

import numpy as np
import torch

from redesign_data import OUTPUT_FIELDS
from redesign_evaluate import compare, lock_candidate
from redesign_features_cache import (
    FeatureRecord,
    config_sha256,
    extractor_sha256,
    install_receipt,
    save_record,
)
from redesign_file import apply_relation, fit_relation
from redesign_notebook_runtime import export_results, write_file_control
from redesign_train import predict_ensemble_artifact, run_experiments

TRAIN_GROUPS: Final = 40
VALIDATION_GROUPS: Final = 12


class IntegrationError(RuntimeError):
    """Raised when the algorithm QA violates a required integration property."""


def _sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _record(index: int, target: int) -> FeatureRecord:
    rng = np.random.default_rng(10_000 + index)
    stem = rng.normal(0, 0.08, (3, 1024)).astype(np.float32)
    stem[:, :32] += np.float32(target * 1.5 - 0.75)
    raw = stem.copy()
    raw[:, 32:64] += np.float32(target * 0.8 - 0.4)
    aux = np.zeros((3, 6), dtype=np.float32)
    aux[:, :3] = np.asarray([-1.0, -1.1, -1.2], dtype=np.float32)
    aux[:, 3:6] = 1.0
    voice_probability = np.full(3, 0.75 if target else 0.25, dtype=np.float32)
    voice_logits = np.log(voice_probability / (1 - voice_probability)).astype(np.float32)
    presence = np.full((3, 2), 0.85, dtype=np.float32)
    return FeatureRecord(
        stem=stem,
        raw=raw,
        aux=aux,
        starts=np.asarray([0, 64_600, 129_200], dtype=np.int64),
        valid=np.ones(3, dtype=np.bool_),
        stem_audible=np.ones(3, dtype=np.bool_),
        voice_embedding=rng.normal(0, 0.05, (3, 1280)).astype(np.float32),
        voice_logits=voice_logits,
        voice_probability=voice_probability,
        presence=presence,
        evidence=np.asarray([voice_probability.max(), 0.85, 0.85], dtype=np.float32),
    )


def _row(root: Path, index: int, split: str, *, view: str, pair_id: str) -> dict[str, str]:
    target = index % 2
    singing = index % 4 < 2
    identifier = f"{split}-{index:03d}-{view}"
    audio = root / "audio" / f"{identifier}.bin"
    audio.parent.mkdir(parents=True, exist_ok=True)
    audio.write_bytes(f"fixture:{identifier}".encode())
    return {
        "ID": identifier,
        "path": str(audio.resolve()),
        "audio_sha256": _sha256(audio),
        "source": "deterministic-algorithm-fixture",
        "license": "fixture-only",
        "license_url": "fixture://local",
        "source_url": "fixture://local",
        "parent_ids": "[]",
        "artist_id": "",
        "speaker_id": "",
        "generator": "fixture",
        "kind": "singing" if singing else "speech",
        "domain": "natural" if singing else "telephone",
        "FILE_FAKE": str(target),
        "VOICE_FAKE": str(target),
        "MUSIC_FAKE": str(target),
        "VOICE_PRESENT": "1",
        "MUSIC_PRESENT": "1",
        "group_id": f"group-{split}-{index:03d}",
        "split": split,
        "pair_id": pair_id,
        "view": view,
        "event_intervals": json.dumps([
            {"component": "music", "start": 0.0, "end": 12.2, "label": target},
            {"component": "voice", "start": 0.0, "end": 12.2, "label": target},
        ]),
        "duration_seconds": "12.2",
    }


def _fixture(root: Path) -> tuple[Path, Path, list[dict[str, str]], list[dict[str, str]]]:
    data, cache = root / "data", root / "cache"
    data.mkdir(parents=True)
    train: list[dict[str, str]] = []
    for index in range(TRAIN_GROUPS):
        pair_id = f"pair-{index:03d}"
        train.extend((_row(root, index, "train", view="pure", pair_id=pair_id), _row(root, index, "train", view="mixed", pair_id=pair_id)))
    validation = [_row(root, index, "validation", view="observed", pair_id="") for index in range(VALIDATION_GROUPS)]
    for name, rows in (("train.csv", train), ("validation.csv", validation), ("sealed.csv", [])):
        with (data / name).open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=OUTPUT_FIELDS)
            writer.writeheader()
            writer.writerows(rows)
    receipt = install_receipt(cache, {
        "extractor_sha256": extractor_sha256(),
        "model_sha256": "algorithm-fixture-no-model-assets",
        "source_sha256": "algorithm-fixture-no-baseline-script",
        "config_sha256": config_sha256(),
    })
    entries: dict[str, dict[str, str]] = {}
    for index, row in enumerate(train + validation):
        target, digest = save_record(cache, row, receipt, _record(index, int(row["MUSIC_FAKE"])))
        entries[row["ID"]] = {
            "audio_sha256": row["audio_sha256"],
            "record": str(target.relative_to(cache)),
            "record_sha256": digest,
        }
    (cache / "manifest.json").write_text(json.dumps({"receipt_sha256": receipt, "records": entries}, indent=2), encoding="utf-8")
    return data, cache, train, validation


def run_cpu_integration(root: Path) -> dict[str, str | int | float | bool | None]:
    """Run real heads, OOF, File fitting, grouped evaluation, reload, and ZIP readback."""
    if root.exists():
        shutil.rmtree(root)
    root.mkdir(parents=True)
    torch.set_num_threads(1)
    data, cache, _train, validation = _fixture(root)
    experiments = root / "experiments"
    summary = run_experiments(
        data, cache, experiments, seeds=(41, 42), epochs=2, patience=2,
        batch_size=8, device="cpu", arms=("A", "B", "C", "D"), folds=2,
    )
    if summary["voice"]["status"] != "complete":
        raise IntegrationError("voice eligibility fixture did not train the optional candidate")
    candidate = str(summary["voice"]["ensemble_candidate"])
    prediction = experiments / "predictions" / f"{candidate}.json"
    oof = experiments / "oof" / f"{candidate}.json"
    maximum, noisy = root / "max.json", root / "noisy_or.json"
    write_file_control(prediction, maximum, noisy_or=False)
    write_file_control(prediction, noisy, noisy_or=True)
    relation = root / "relation.json"
    related = root / "related.json"
    fit_relation(oof, relation)
    apply_relation(prediction, relation, related)
    compare(maximum, noisy, root / "noisy_report.json", bootstrap=40, seed=41)
    report = compare(maximum, related, root / "relation_report.json", bootstrap=40, seed=41)
    panel_identity = hashlib.sha256(b"[]").hexdigest()
    lock_rejected = False
    try:
        lock_candidate(
            related,
            root / "relation_report.json",
            root / "locked_candidate.json",
            sealed_panel_identity=panel_identity,
        )
    except ValueError:
        lock_rejected = True
    voice_members = summary["voice"]["members"]
    checkpoints = tuple(Path(item["music_checkpoint"]) for item in voice_members)
    voice_checkpoints = tuple(Path(item["checkpoint"]) for item in voice_members)
    from redesign_features import load_records

    records = load_records(validation, cache)
    training_manifest_sha256 = _sha256(data / "train.csv")
    records_identity = json.loads((cache / "manifest.json").read_text(encoding="utf-8"))["receipt_sha256"]
    reloaded_artifact = predict_ensemble_artifact(
        checkpoints, validation, records,
        training_manifest_sha256=training_manifest_sha256,
        records_identity=records_identity,
        candidate=candidate,
        device="cpu",
        voice_checkpoints=voice_checkpoints,
    )
    saved_artifact = json.loads(prediction.read_text(encoding="utf-8"))
    reloaded = reloaded_artifact["predictions"]
    saved = saved_artifact["predictions"]
    max_difference = max(
        abs(float(reloaded[identifier][column]) - float(saved[identifier][column]))
        for identifier in saved
        for column in ("FILE_FAKE_PROB", "VOICE_FAKE_PROB", "MUSIC_FAKE_PROB", "VOICE_PRESENT_PROB", "MUSIC_PRESENT_PROB")
    )
    if max_difference > 1e-7:
        raise IntegrationError(f"saved/reloaded prediction mismatch: {max_difference}")
    if reloaded_artifact["upstream_identity"] != saved_artifact["upstream_identity"]:
        raise IntegrationError("saved/reloaded upstream identity mismatch")
    voice_weight_hashes = saved_artifact["upstream_identity"].get("seed_voice_checkpoint_sha256")
    if not isinstance(voice_weight_hashes, list) or len(voice_weight_hashes) != 2:
        raise IntegrationError("ensemble identity did not bind both ordered voice weights")
    oof_artifact = json.loads(oof.read_text(encoding="utf-8"))
    if len(oof_artifact.get("oof_members", ())) != 2:
        raise IntegrationError("two-seed ensemble did not preserve both OOF members")
    proof_names = [
        sorted(proof["name"] for proof in member.get("oof_proofs", ()))
        for member in oof_artifact["oof_members"]
    ]
    if proof_names != [["music", "voice"], ["music", "voice"]]:
        raise IntegrationError(f"ensemble OOF proofs are incomplete: {proof_names}")
    base_artifact = json.loads((experiments / "predictions" / "D_ensemble.json").read_text(encoding="utf-8"))
    voice_delta = max(
        abs(float(saved[identifier]["VOICE_FAKE_PROB"])
            - float(base_artifact["predictions"][identifier]["VOICE_FAKE_PROB"]))
        for identifier in saved
    )
    if voice_delta <= 1e-7:
        raise IntegrationError("trained voice candidate did not affect published predictions")
    if report["status"] == "accepted" or not lock_rejected:
        raise IntegrationError("algorithm fixture must never become an accepted performance candidate")
    result = {
        "status": "algorithm_qa_only",
        "real_model_fit": True,
        "oof_relation_fit": True,
        "grouped_evaluation": True,
        "save_load_prediction_max_difference": max_difference,
        "save_load_upstream_identity_equal": True,
        "ordered_voice_weight_hash_count": len(voice_weight_hashes),
        "seed_count": len(checkpoints),
        "oof_member_count": len(oof_artifact["oof_members"]),
        "oof_proof_names": proof_names,
        "trained_voice_prediction_max_delta": voice_delta,
        "candidate_status": report["status"],
        "ineligible_lock_rejected": lock_rejected,
        "actual_data_performance_verified": False,
        "h100_verified": False,
        "l4_verified": False,
        "final_submission_zip": False,
    }
    (root / "integration.json").write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    export = export_results((experiments, relation, related, root / "integration.json"), root / "algorithm_qa_results.zip")
    result.update(export_sha256=str(export["archive_sha256"]), export_verified=bool(export["verified"]))
    (root / "integration.json").write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    return result


def main() -> None:
    """Run the integration scenario at the requested output path."""
    output = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(".omo/evidence/redesign-notebook-20260917/cpu-integration")
    print(json.dumps(run_cpu_integration(output), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
