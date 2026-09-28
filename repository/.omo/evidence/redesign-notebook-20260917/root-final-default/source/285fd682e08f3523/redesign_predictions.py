"""Prediction, provenance, and JSON artifact helpers for redesign experiments."""

import hashlib
import json
from pathlib import Path
from typing import Sequence

import numpy as np
import torch

from redesign_checkpoint import Records, Rows, load_checkpoint, sha256
from redesign_models import MusicResidual, StemMusicWrapper
from redesign_oof import row_groups

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
    candidate: str, rows: Rows, values: dict[str, dict[str, float]], manifest_sha: str,
    checkpoint: Path, records_identity: str,
) -> dict[str, object]:
    checkpoint_payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    checkpoint_report = checkpoint_payload["report"]
    recipe = {name: checkpoint_report[name] for name in
              ("arm", "seed", "epochs", "patience", "batch_size", "raw_equal_stem")}
    recipe_identity = hashlib.sha256(
        json.dumps(recipe, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    source_names = ("redesign_models.py", "redesign_fit.py", "redesign_checkpoint.py",
                    "redesign_oof.py", "redesign_predictions.py", "redesign_train.py")
    source_identity = {name: _manifest_sha(Path(__file__).with_name(name)) for name in source_names}
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
            "training_manifest_sha256": manifest_sha, "records_identity": records_identity,
            "checkpoint_sha256": sha256(checkpoint), "model_recipe_identity": recipe_identity,
            "upstream_family_identity": family_identity, "upstream_family_payload": family,
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
    checkpoint: Path, rows: Rows, records: Records, *, training_manifest_sha256: str,
    records_identity: str, candidate: str = "locked", device: str = "cpu",
) -> dict[str, object]:
    """Build the complete prediction artifact needed by locked/sealed evaluation."""
    values = predict_checkpoint(checkpoint, rows, records, device=device)
    return _artifact(candidate, rows, values, training_manifest_sha256, checkpoint, records_identity)


def predict_ensemble_artifact(
    checkpoints: Sequence[Path], rows: Rows, records: Records, *, training_manifest_sha256: str,
    records_identity: str, candidate: str = "D_ensemble", device: str = "cpu",
    voice_checkpoints: Sequence[Path] | None = None,
) -> dict[str, object]:
    """Reload seed checkpoints and average probabilities for locked/sealed inference."""
    if voice_checkpoints is not None and len(voice_checkpoints) != len(checkpoints):
        raise ValueError("Music and voice checkpoint lists must align by seed")
    artifacts = []
    for index, checkpoint in enumerate(checkpoints):
        artifact = predict_artifact(
            checkpoint, rows, records, training_manifest_sha256=training_manifest_sha256,
            records_identity=records_identity, candidate=f"{candidate}_member{index}", device=device)
        if voice_checkpoints is not None:
            from redesign_voice import predict_voice

            voice_checkpoint = voice_checkpoints[index]
            artifact = _overlay_voice(artifact, predict_voice(voice_checkpoint, rows, records, device=device),
                                      f"{candidate}_member{index}", voice_checkpoint)
        artifacts.append(artifact)
    return _ensemble(candidate, artifacts)


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
        artifact["upstream_identity"]["checkpoint_sha256"] for artifact in artifacts]
    if all("voice_checkpoint_sha256" in artifact["upstream_identity"] for artifact in artifacts):
        result["upstream_identity"]["seed_voice_checkpoint_sha256"] = [
            artifact["upstream_identity"]["voice_checkpoint_sha256"] for artifact in artifacts]
    seed_families = [artifact["upstream_identity"]["upstream_family_identity"] for artifact in artifacts]
    result["upstream_identity"]["seed_model_recipe_identity"] = [
        artifact["upstream_identity"]["model_recipe_identity"] for artifact in artifacts]
    result["upstream_identity"]["seed_upstream_family_identity"] = seed_families
    family_payload = {"seed_upstream_family_identity": seed_families}
    result["upstream_identity"]["upstream_family_payload"] = family_payload
    result["upstream_identity"]["upstream_family_identity"] = hashlib.sha256(
        json.dumps(family_payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
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
            prediction["MUSIC_PRESENT_PROB"] * prediction["MUSIC_FAKE_PROB"])
        result["window_scores"][identifier]["voice_fake"] = values[identifier]["windows"]
    result["upstream_identity"]["voice_checkpoint_sha256"] = sha256(checkpoint)
    voice_payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    voice_recipe = {name: voice_payload[name] for name in
                    ("seed", "epochs", "patience", "source_sha256")}
    voice_recipe_identity = hashlib.sha256(
        json.dumps(voice_recipe, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    result["upstream_identity"]["voice_recipe_identity"] = voice_recipe_identity
    base_family = result["upstream_identity"]["upstream_family_identity"]
    voice_family = {"base_upstream_family_identity": base_family,
                    "voice_recipe_identity": voice_recipe_identity}
    result["upstream_identity"]["base_upstream_family_identity"] = base_family
    result["upstream_identity"]["upstream_family_payload"] = voice_family
    result["upstream_identity"]["upstream_family_identity"] = hashlib.sha256(
        json.dumps(voice_family, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    return result
