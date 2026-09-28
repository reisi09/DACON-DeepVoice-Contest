"""Portable entry point for matched redesign experiments and honest OOF."""

import hashlib
import json
from pathlib import Path
from typing import Sequence

import numpy as np
import torch

from redesign_data import read_rows
from redesign_features import load_records
from redesign_fit import Arm, Records, Rows, fit_music, load_checkpoint, sha256
from redesign_models import MusicResidual, StemMusicWrapper
from redesign_oof import build_oof, connected_groups, internal_earlystop, row_groups
from redesign_voice import build_voice_oof, fit_voice, predict_voice, voice_gate

LABELS = ("FILE_FAKE", "VOICE_FAKE", "MUSIC_FAKE", "VOICE_PRESENT", "MUSIC_PRESENT")
PROBABILITIES = tuple(f"{name}_PROB" for name in LABELS)


def _write_json(path: Path, value: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".partial")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def _manifest_sha(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def arm_rows(rows: Rows, arm: Arm) -> tuple[Rows, dict[str, int | float]]:
    """Replace matched pair slots without changing exposure count or duration."""
    pairs: dict[str, dict[str, dict[str, str]]] = {}
    shared: Rows = []
    for row in rows:
        if row.get("MUSIC_PRESENT") != "1" or row.get("MUSIC_FAKE") not in ("0", "1"):
            continue
        pair_id = row.get("pair_id", "")
        if pair_id:
            view = row.get("view", "")
            if view not in ("pure", "mixed") or view in pairs.setdefault(pair_id, {}):
                raise ValueError(f"Invalid duplicate pair slot: {pair_id}/{view}")
            pairs[pair_id][view] = row
        elif row.get("view") in ("pure", "observed"):
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


def _predict_model(
    model: StemMusicWrapper | MusicResidual, rows: Rows, records: Records, *, device: str,
    raw_equal_stem: bool = False,
) -> dict[str, dict[str, float]]:
    model.eval()
    result: dict[str, dict[str, float]] = {}
    with torch.inference_mode():
        for row in rows:
            record = records[row["ID"]]
            stem = torch.as_tensor(record["stem"], device=device).float()[None]
            raw_values = record["stem"] if raw_equal_stem else record["raw"]
            raw = torch.as_tensor(raw_values, device=device).float()[None]
            aux_values = np.asarray(record["aux"], dtype=np.float32).copy()
            if raw_equal_stem:
                aux_values[:, 0] = aux_values[:, 1]
                aux_values[:, 3] = aux_values[:, 4]
            aux = torch.as_tensor(aux_values, device=device).float()[None]
            valid = torch.as_tensor(
                np.asarray(record["valid"], dtype=bool) & np.asarray(record["stem_audible"], dtype=bool),
                device=device,
            )[None]
            output = (model(stem, raw, aux, valid)
                      if isinstance(model, MusicResidual) else model(stem, valid))
            music_windows = output.window_logits.softmax(-1)[0, :, 1].cpu().numpy()
            music_windows[~valid[0].cpu().numpy()] = 0.0
            music = float(output.bag_logits.softmax(-1)[0, 1])
            voice_windows = np.asarray(record["voice_probability"], dtype=float)
            presence = np.asarray(record["presence"], dtype=float)
            usable = np.asarray(record["valid"], dtype=bool)
            voice = float(voice_windows[usable].max()) if usable.any() else 0.0
            vp = float(presence[usable, 0].max()) if usable.any() else 0.0
            mp = float(presence[usable, 1].max()) if usable.any() else 0.0
            result[row["ID"]] = {
                "FILE_FAKE_PROB": max(vp * voice, mp * music),
                "VOICE_FAKE_PROB": voice, "MUSIC_FAKE_PROB": music,
                "VOICE_PRESENT_PROB": vp, "MUSIC_PRESENT_PROB": mp,
                "voice_fake": voice_windows.tolist(), "music_fake": music_windows.tolist(),
                "voice_present": presence[:, 0].tolist(), "music_present": presence[:, 1].tolist(),
            }
    return result


def _artifact(
    candidate: str,
    rows: Rows,
    values: dict[str, dict[str, float]],
    manifest_sha: str,
    checkpoint: Path,
    records_identity: str,
) -> dict[str, object]:
    checkpoint_payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    checkpoint_report = checkpoint_payload["report"]
    recipe = {name: checkpoint_report[name] for name in
              ("arm", "seed", "epochs", "patience", "batch_size", "raw_equal_stem")}
    recipe_identity = hashlib.sha256(
        json.dumps(recipe, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    source_identity = {name: _manifest_sha(Path(__file__).with_name(name)) for name in
                       ("redesign_models.py", "redesign_fit.py", "redesign_oof.py", "redesign_train.py")}
    family = {"training_manifest_sha256": manifest_sha, "records_identity": records_identity,
              "model_recipe_identity": recipe_identity, "source_sha256": source_identity}
    family_identity = hashlib.sha256(
        json.dumps(family, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return {
        "schema_version": 1, "candidate": candidate,
        "labels": {row["ID"]: {name: int(row[name]) if row.get(name) in ("0", "1") else None
                                  for name in LABELS} for row in rows},
        "predictions": {row["ID"]: {name: values[row["ID"]][name] for name in PROBABILITIES}
                        for row in rows},
        "groups": {row["ID"]: list(row_groups(row)) for row in rows},
        "strata": {row["ID"]: {name: row.get(name, "") for name in
                                  ("source", "kind", "domain", "view", "pair_id")} for row in rows},
        "window_scores": {row["ID"]: {name: values[row["ID"]][name] for name in
                                        ("voice_fake", "music_fake", "voice_present", "music_present")}
                          for row in rows},
        "upstream_identity": {
            "training_manifest_sha256": manifest_sha,
            "records_identity": records_identity,
            "checkpoint_sha256": sha256(checkpoint),
            "model_recipe_identity": recipe_identity,
            "upstream_family_identity": family_identity,
            "upstream_family_payload": family,
            "source_sha256": source_identity,
        },
    }


def predict_checkpoint(
    checkpoint: Path, rows: Rows, records: Records, *, device: str = "cpu"
) -> dict[str, dict[str, float]]:
    """Load a locked head and return official probabilities plus aligned windows."""
    model, report = load_checkpoint(checkpoint, device=device)
    return _predict_model(model, rows, records, device=device,
                          raw_equal_stem=report.get("raw_equal_stem") == "True")


def predict_artifact(
    checkpoint: Path,
    rows: Rows,
    records: Records,
    *,
    training_manifest_sha256: str,
    records_identity: str,
    candidate: str = "locked",
    device: str = "cpu",
) -> dict[str, object]:
    """Build the complete prediction artifact needed by locked/sealed evaluation."""
    values = predict_checkpoint(checkpoint, rows, records, device=device)
    return _artifact(candidate, rows, values, training_manifest_sha256, checkpoint, records_identity)


def predict_ensemble_artifact(
    checkpoints: Sequence[Path],
    rows: Rows,
    records: Records,
    *,
    training_manifest_sha256: str,
    records_identity: str,
    candidate: str = "D_ensemble",
    device: str = "cpu",
) -> dict[str, object]:
    """Reload seed checkpoints and average probabilities for locked/sealed inference."""
    artifacts = [predict_artifact(
        checkpoint, rows, records, training_manifest_sha256=training_manifest_sha256,
        records_identity=records_identity, candidate=f"{candidate}_member{index}", device=device,
    ) for index, checkpoint in enumerate(checkpoints)]
    return _ensemble(candidate, artifacts)


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
            _write_json(output / "oof" / f"{arm}_seed{seed}.json", oof_artifact)
            seed_oof_artifacts.append(oof_artifact)
            arm_summary["seeds"].append({"seed": seed, "checkpoint": str(result.checkpoint),
                                          "selected_epoch": result.report["selected_epoch"],
                                          "optimizer_updates": result.report["optimizer_updates"]})
        ensemble = _ensemble(f"{arm}_ensemble", seed_artifacts)
        _write_json(output / "predictions" / f"{arm}_ensemble.json", ensemble)
        oof_ensemble = _ensemble(f"{arm}_ensemble", seed_oof_artifacts)
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
            oof_artifact["folds"] = voice_folds
            oof_artifact["held_out_group"] = held
            oof_artifact["trained_groups"] = trained
            oof_artifact["checkpoint_hashes"] += [fold["checkpoint_sha256"] for fold in voice_folds]
            _write_json(output / "oof" / f"{base_arm}_voice_seed{seed}.json", oof_artifact)
            voice_oof_artifacts.append(oof_artifact)
            voice_members.append({"seed": int(seed), "checkpoint": str(voice.checkpoint),
                                  "eligible_ids": list(voice.eligible_ids)})
        voice_ensemble = _ensemble(f"{base_arm}_voice_ensemble", voice_prediction_artifacts)
        voice_oof_ensemble = _ensemble(f"{base_arm}_voice_ensemble", voice_oof_artifacts)
        _write_json(output / "predictions" / f"{base_arm}_voice_ensemble.json", voice_ensemble)
        _write_json(output / "oof" / f"{base_arm}_voice_ensemble.json", voice_oof_ensemble)
        summary["voice"] = {"status": "complete", "reason": "eligible",
                            "base_arm": base_arm, "members": voice_members,
                            "single_candidate": f"{base_arm}_voice_seed{seeds[0]}",
                            "ensemble_candidate": f"{base_arm}_voice_ensemble"}
    _write_json(output / "summary.json", summary)
    return summary


def _ensemble(candidate: str, artifacts: list[dict[str, object]]) -> dict[str, object]:
    if not artifacts:
        raise ValueError("An ensemble needs at least one seed")
    result = json.loads(json.dumps(artifacts[0]))
    result["candidate"] = candidate
    for section in ("predictions", "window_scores"):
        for identifier, values in result[section].items():
            for name in values:
                stacked = [artifact[section][identifier][name] for artifact in artifacts]
                values[name] = np.asarray(stacked, dtype=float).mean(axis=0).tolist()
                if not isinstance(stacked[0], list):
                    values[name] = float(values[name])
    result["upstream_identity"]["seed_checkpoint_sha256"] = [
        artifact["upstream_identity"]["checkpoint_sha256"] for artifact in artifacts
    ]
    seed_families = [artifact["upstream_identity"]["upstream_family_identity"] for artifact in artifacts]
    result["upstream_identity"]["seed_model_recipe_identity"] = [
        artifact["upstream_identity"]["model_recipe_identity"] for artifact in artifacts
    ]
    result["upstream_identity"]["seed_upstream_family_identity"] = seed_families
    family_payload = {"seed_upstream_family_identity": seed_families}
    result["upstream_identity"]["upstream_family_payload"] = family_payload
    result["upstream_identity"]["upstream_family_identity"] = hashlib.sha256(
        json.dumps(family_payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return result


def _overlay_voice(
    base: dict[str, object], values: dict[str, dict[str, float | list[float]]],
    candidate: str, checkpoint: Path,
) -> dict[str, object]:
    result = json.loads(json.dumps(base))
    result["candidate"] = candidate
    for identifier, prediction in result["predictions"].items():
        voice = float(values[identifier]["voice"])
        prediction["VOICE_FAKE_PROB"] = voice
        prediction["FILE_FAKE_PROB"] = max(
            prediction["VOICE_PRESENT_PROB"] * voice,
            prediction["MUSIC_PRESENT_PROB"] * prediction["MUSIC_FAKE_PROB"],
        )
        result["window_scores"][identifier]["voice_fake"] = values[identifier]["windows"]
    result["upstream_identity"]["voice_checkpoint_sha256"] = sha256(checkpoint)
    voice_payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    voice_recipe = {name: voice_payload[name] for name in
                    ("seed", "epochs", "patience", "source_sha256")}
    voice_recipe_identity = hashlib.sha256(
        json.dumps(voice_recipe, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    result["upstream_identity"]["voice_recipe_identity"] = voice_recipe_identity
    base_family = result["upstream_identity"]["upstream_family_identity"]
    voice_family = {"base_upstream_family_identity": base_family,
                    "voice_recipe_identity": voice_recipe_identity}
    result["upstream_identity"]["base_upstream_family_identity"] = base_family
    result["upstream_identity"]["upstream_family_payload"] = voice_family
    result["upstream_identity"]["upstream_family_identity"] = hashlib.sha256(
        json.dumps(voice_family, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return result
