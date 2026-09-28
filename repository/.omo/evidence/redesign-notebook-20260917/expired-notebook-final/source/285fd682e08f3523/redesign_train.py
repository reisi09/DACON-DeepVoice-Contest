"""Portable entry point for matched redesign experiments and honest OOF."""

import json
from pathlib import Path
from typing import Sequence

from redesign_data import read_rows
from redesign_features import load_records
from redesign_fit import Arm, Rows, fit_music
from redesign_oof import build_oof, connected_groups, internal_earlystop, row_groups
from redesign_predictions import (
    _artifact, _ensemble, _manifest_sha, _overlay_voice, _predict_model, _write_json,
    predict_artifact, predict_checkpoint, predict_ensemble_artifact,
)
from redesign_voice import build_voice_oof, fit_voice, predict_voice, voice_gate

def arm_rows(rows: Rows, arm: Arm) -> tuple[Rows, dict[str, int | float]]:
    """Replace matched pair slots without changing exposure count or duration."""
    pairs: dict[str, dict[str, dict[str, str]]] = {}
    shared: Rows = []
    for row in rows:
        pair_id = row.get("pair_id", "")
        if pair_id:
            if row.get("MUSIC_PRESENT") != "1" or row.get("MUSIC_FAKE") not in ("0", "1"):
                continue
            view = row.get("view", "")
            if view not in ("pure", "mixed") or view in pairs.setdefault(pair_id, {}):
                raise ValueError(f"Invalid duplicate pair slot: {pair_id}/{view}")
            pairs[pair_id][view] = row
        elif row.get("view") in ("pure", "observed") and (
            row.get("MUSIC_FAKE") in ("0", "1") or row.get("VOICE_FAKE") in ("0", "1")
        ):
            shared.append(row)
    selected_view = "pure" if arm in ("A", "B") else "mixed"
    selected = list(shared)
    seconds = 0.0
    for pair_id, views in sorted(pairs.items()):
        if set(views) != {"pure", "mixed"}:
            raise ValueError(f"Pair shortage: {pair_id} lacks pure or mixed counterpart")
        pure, mixed = views["pure"], views["mixed"]
        if (pure["group_id"] != mixed["group_id"]
                or pure["MUSIC_FAKE"] != mixed["MUSIC_FAKE"]
                or abs(float(pure["duration_seconds"]) - float(mixed["duration_seconds"])) > 1e-6):
            raise ValueError(f"Pair changes group, target, or accompaniment seconds: {pair_id}")
        selected.append(views[selected_view])
        seconds += float(views[selected_view]["duration_seconds"])
    if not selected:
        raise ValueError(f"Arm {arm} has no eligible known music rows")
    return selected, {"paired_slots": len(pairs), "shared_unpaired": len(shared),
                      "draws_per_epoch": len(selected), "paired_target_seconds": seconds}


def run_experiments(
    data: Path,
    cache: Path,
    output: Path,
    *,
    seeds: Sequence[int] = (41, 42, 43),
    epochs: int = 300,
    patience: int = 20,
    batch_size: int = 64,
    device: str = "cuda",
    deadline: float | None = None,
    arms: Sequence[Arm] = ("A", "B", "C", "D"),
    folds: int = 5,
) -> dict[str, object]:
    """Run matched heads; global validation is prediction-only after fitting."""
    train_path, validation_path = data / "train.csv", data / "validation.csv"
    train_rows, validation_rows = read_rows(train_path), read_rows(validation_path)
    records = load_records(train_rows + validation_rows, cache)
    manifest_sha = _manifest_sha(train_path)
    records_identity = json.loads((cache / "manifest.json").read_text())["receipt_sha256"]
    summary: dict[str, object] = {"status": "complete", "arms": {}, "limitations": [
        "CPU fixtures validate execution only; no performance, H100, L4, sealed, or leaderboard claim"
    ]}
    prediction_sets: dict[str, list[dict[str, object]]] = {}
    oof_sets: dict[str, list[dict[str, object]]] = {}
    arm_training_rows: dict[str, Rows] = {}
    for arm in arms:
        selected, exposure = arm_rows(train_rows, arm)
        arm_training_rows[arm] = selected
        arm_summary: dict[str, object] = {"exposure": exposure, "seeds": []}
        seed_artifacts = []
        seed_oof_artifacts = []
        for seed in seeds:
            fitting, early = internal_earlystop(selected, seed)
            result = fit_music(fitting, records, early, output / "checkpoints" / arm / str(seed),
                               arm=arm, seed=seed, epochs=epochs, patience=patience,
                               batch_size=batch_size, device=device, deadline=deadline)
            values = _predict_model(result.model, validation_rows, records, device=device)
            artifact = _artifact(f"{arm}_seed{seed}", validation_rows, values, manifest_sha,
                                 result.checkpoint, records_identity)
            path = output / "predictions" / f"{arm}_seed{seed}.json"
            _write_json(path, artifact)
            seed_artifacts.append(artifact)
            oof_values, receipts, held, trained = build_oof(
                selected, records, output / "oof_checkpoints" / arm / str(seed), arm=arm,
                seed=seed, epochs=epochs, patience=patience, batch_size=batch_size,
                device=device, deadline=deadline, folds=folds, predict=_predict_model,
            )
            oof_artifact = _artifact(f"{arm}_seed{seed}", selected, oof_values, manifest_sha,
                                     Path(receipts[0]["checkpoint"]), records_identity)
            oof_artifact.update(held_out_group=held, trained_groups=trained, folds=receipts,
                                checkpoint_hashes=[receipt["checkpoint_sha256"] for receipt in receipts])
            oof_artifact["oof_proofs"] = [{
                "name": "music", "held_out_group": held, "trained_groups": trained,
                "folds": receipts,
                "checkpoint_hashes": [receipt["checkpoint_sha256"] for receipt in receipts],
            }]
            _write_json(output / "oof" / f"{arm}_seed{seed}.json", oof_artifact)
            seed_oof_artifacts.append(oof_artifact)
            arm_summary["seeds"].append({"seed": seed, "checkpoint": str(result.checkpoint),
                                          "selected_epoch": result.report["selected_epoch"],
                                          "optimizer_updates": result.report["optimizer_updates"]})
        ensemble = _ensemble(f"{arm}_ensemble", seed_artifacts)
        _write_json(output / "predictions" / f"{arm}_ensemble.json", ensemble)
        oof_ensemble = _ensemble(f"{arm}_ensemble", seed_oof_artifacts)
        oof_ensemble["oof_members"] = seed_oof_artifacts
        oof_ensemble["seed_folds"] = [artifact["folds"] for artifact in seed_oof_artifacts]
        oof_ensemble["checkpoint_hashes"] = [digest for artifact in seed_oof_artifacts
                                                for digest in artifact["checkpoint_hashes"]]
        _write_json(output / "oof" / f"{arm}_ensemble.json", oof_ensemble)
        prediction_sets[arm] = seed_artifacts
        oof_sets[arm] = seed_oof_artifacts
        if arm == "D" and seeds:
            seed = int(seeds[0])
            fitting, early = internal_earlystop(selected, seed)
            capacity = fit_music(
                fitting, records, early, output / "checkpoints" / "D_raw_equals_stem" / str(seed),
                arm="D", seed=seed, epochs=epochs, patience=patience, batch_size=batch_size,
                device=device, deadline=deadline, raw_equal_stem=True,
            )
            capacity_values = _predict_model(
                capacity.model, validation_rows, records, device=device, raw_equal_stem=True
            )
            capacity_artifact = _artifact(
                f"D_raw_equals_stem_seed{seed}", validation_rows, capacity_values, manifest_sha,
                capacity.checkpoint, records_identity,
            )
            _write_json(output / "predictions" / f"D_raw_equals_stem_seed{seed}.json", capacity_artifact)
            arm_summary["raw_equals_stem_capacity_control"] = {
                "seed": seed, "checkpoint": str(capacity.checkpoint),
                "selected_epoch": capacity.report["selected_epoch"],
                "optimizer_updates": capacity.report["optimizer_updates"],
            }
        summary["arms"][arm] = arm_summary
    base_arm = "D" if "D" in arm_training_rows else str(tuple(arms)[-1])
    voice_allowed, voice_reason = voice_gate(train_rows)
    if not voice_allowed:
        summary["voice"] = {"status": "provisional_skip", "reason": voice_reason,
                            "eligible_ids": []}
    else:
        voice_prediction_artifacts = []
        voice_oof_artifacts = []
        voice_members = []
        for index, seed in enumerate(seeds):
            voice = fit_voice(train_rows, records, output / "voice" / str(seed),
                              seed=int(seed), epochs=epochs, patience=patience, device=device,
                              deadline=deadline)
            if voice.checkpoint is None:
                raise RuntimeError("Eligible voice gate did not produce a checkpoint")
            values = predict_voice(voice.checkpoint, validation_rows, records, device=device)
            artifact = _overlay_voice(prediction_sets[base_arm][index], values,
                                      f"{base_arm}_voice_seed{seed}", voice.checkpoint)
            _write_json(output / "predictions" / f"{base_arm}_voice_seed{seed}.json", artifact)
            voice_prediction_artifacts.append(artifact)
            oof_values, voice_folds, held, trained = build_voice_oof(
                arm_training_rows[base_arm], records, output / "voice_oof" / str(seed),
                seed=int(seed), epochs=epochs, patience=patience, device=device,
                deadline=deadline, folds=folds, training_rows=train_rows,
            )
            oof_artifact = _overlay_voice(oof_sets[base_arm][index], oof_values,
                                          f"{base_arm}_voice_seed{seed}", voice.checkpoint)
            oof_artifact["music_folds"] = oof_artifact["folds"]
            oof_artifact["voice_folds"] = voice_folds
            oof_artifact["oof_proofs"].append({
                "name": "voice", "held_out_group": held, "trained_groups": trained,
                "folds": voice_folds,
                "checkpoint_hashes": [fold["checkpoint_sha256"] for fold in voice_folds],
            })
            oof_artifact["checkpoint_hashes"] += [fold["checkpoint_sha256"] for fold in voice_folds]
            _write_json(output / "oof" / f"{base_arm}_voice_seed{seed}.json", oof_artifact)
            voice_oof_artifacts.append(oof_artifact)
            voice_members.append({"seed": int(seed), "checkpoint": str(voice.checkpoint),
                                  "music_checkpoint": summary["arms"][base_arm]["seeds"][index]["checkpoint"],
                                  "eligible_ids": list(voice.eligible_ids)})
        voice_ensemble = _ensemble(f"{base_arm}_voice_ensemble", voice_prediction_artifacts)
        voice_oof_ensemble = _ensemble(f"{base_arm}_voice_ensemble", voice_oof_artifacts)
        voice_oof_ensemble["oof_members"] = voice_oof_artifacts
        _write_json(output / "predictions" / f"{base_arm}_voice_ensemble.json", voice_ensemble)
        _write_json(output / "oof" / f"{base_arm}_voice_ensemble.json", voice_oof_ensemble)
        summary["voice"] = {"status": "complete", "reason": "eligible",
                            "base_arm": base_arm, "members": voice_members,
                            "single_candidate": f"{base_arm}_voice_seed{seeds[0]}",
                            "ensemble_candidate": f"{base_arm}_voice_ensemble"}
    _write_json(output / "summary.json", summary)
    return summary
