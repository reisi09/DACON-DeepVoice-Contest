import hashlib
import json
from pathlib import Path

import pytest

from redesign_evaluate import compare
from redesign_file import apply_relation, fit_relation


LABELS = ("FILE_FAKE", "VOICE_FAKE", "MUSIC_FAKE", "VOICE_PRESENT", "MUSIC_PRESENT")
PROBS = tuple(f"{name}_PROB" for name in LABELS)


def _artifact(rows: int = 24, *, candidate: str = "control") -> dict:
    labels = {}
    predictions = {}
    groups = {}
    strata = {}
    windows = {}
    held_out = {}
    trained = {}
    for index in range(rows):
        identifier = f"id-{index:03d}"
        target = index % 2
        group = f"g-{index:03d}"
        labels[identifier] = {name: target for name in LABELS}
        score = 0.15 + 0.7 * target
        predictions[identifier] = {name: score for name in PROBS}
        groups[identifier] = [group]
        strata[identifier] = {
            "source": "fixture", "kind": "singing", "domain": "telephone",
            "view": "pure", "pair_id": f"p-{index:03d}",
        }
        windows[identifier] = {
            "voice_fake": [score, score], "music_fake": [score, score],
            "voice_present": [0.9, 0.9], "music_present": [0.9, 0.9],
        }
        held_out[identifier] = [group]
        trained[identifier] = [
            f"g-{other:03d}" for other in range(rows)
            if other not in (index, (index + 1) % rows)
        ]
    family_payload = {
        "training_manifest_sha256": "1" * 64,
        "records_identity": "fixture-records",
        "model_recipe_identity": "4" * 64,
        "source_sha256": {"fixture": "3" * 64},
    }
    family_identity = hashlib.sha256(
        json.dumps(family_payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return {
        "schema_version": 1, "candidate": candidate, "labels": labels,
        "predictions": predictions, "groups": groups, "strata": strata,
        "window_scores": windows,
        "upstream_identity": {
            "training_manifest_sha256": "1" * 64,
            "records_identity": "fixture-records", "checkpoint_sha256": "2" * 64,
            "model_recipe_identity": "4" * 64,
            "upstream_family_payload": family_payload,
            "upstream_family_identity": family_identity,
            "source_sha256": {"fixture": "3" * 64},
        },
        "held_out_group": held_out, "trained_groups": trained,
        "folds": [{
            "fold": index, "held_out_groups": [f"g-{index:03d}"],
            "trained_groups": [
                f"g-{other:03d}" for other in range(rows)
                if other not in (index, (index + 1) % rows)
            ],
            "earlystop_groups": [f"g-{(index + 1) % rows:03d}"],
            "held_out_ids": [f"id-{index:03d}"],
            "train_ids": [
                f"id-{other:03d}" for other in range(rows)
                if other not in (index, (index + 1) % rows)
            ],
            "earlystop_ids": [f"id-{(index + 1) % rows:03d}"],
            "checkpoint": f"fold-{index}.pt", "checkpoint_sha256": f"{index + 10:064x}",
            "selected_epoch": 1,
        } for index in range(rows)],
        "checkpoint_hashes": [f"{index + 10:064x}" for index in range(rows)],
    }


def _write(path: Path, value: dict) -> Path:
    path.write_text(json.dumps(value), encoding="utf-8")
    return path


def test_fit_relation_preserves_endpoints_and_reduces_actual_offset_loss(tmp_path: Path) -> None:
    artifact = _artifact(40)
    for index, identifier in enumerate(artifact["predictions"]):
        target = artifact["labels"][identifier]["FILE_FAKE"]
        misleading = 0.8 if target == 0 else 0.2
        row = artifact["predictions"][identifier]
        row["FILE_FAKE_PROB"] = 0.0 if index == 0 else 1.0 if index == 1 else 0.5
        row["VOICE_FAKE_PROB"] = misleading
        artifact["window_scores"][identifier]["voice_fake"] = [misleading, misleading]
        if index in (0, 1):
            endpoint = float(index)
            for name in ("VOICE_FAKE_PROB", "MUSIC_FAKE_PROB", "VOICE_PRESENT_PROB", "MUSIC_PRESENT_PROB"):
                row[name] = endpoint
            for name in ("voice_fake", "music_fake", "voice_present", "music_present"):
                artifact["window_scores"][identifier][name] = [endpoint, endpoint]
    relation = fit_relation(_write(tmp_path / "oof.json", artifact), tmp_path / "relation.json")
    assert relation["fit"]["final_loss"] < relation["fit"]["initial_loss"] - 0.05

    prediction = json.loads(json.dumps(artifact))
    prediction.pop("held_out_group")
    prediction.pop("trained_groups")
    prediction.pop("folds")
    prediction.pop("checkpoint_hashes")
    prediction["upstream_identity"]["checkpoint_sha256"] = "9" * 64
    applied = apply_relation(
        _write(tmp_path / "prediction.json", prediction), tmp_path / "relation.json",
        tmp_path / "applied.json",
    )
    assert applied["predictions"]["id-000"]["FILE_FAKE_PROB"] == 0.0
    assert applied["predictions"]["id-001"]["FILE_FAKE_PROB"] == 1.0


@pytest.mark.parametrize("changed", ["model_recipe_identity", "records_identity", "source_sha256"])
def test_relation_rejects_changed_upstream_family(tmp_path: Path, changed: str) -> None:
    artifact = _artifact(24)
    relation_path = tmp_path / "relation.json"
    fit_relation(_write(tmp_path / "oof.json", artifact), relation_path)
    prediction = json.loads(json.dumps(artifact))
    for key in ("held_out_group", "trained_groups", "folds", "checkpoint_hashes"):
        prediction.pop(key)
    payload = prediction["upstream_identity"]["upstream_family_payload"]
    payload[changed] = {"changed": "f" * 64} if changed == "source_sha256" else "f" * 64
    prediction["upstream_identity"][changed] = payload[changed]
    prediction["upstream_identity"]["upstream_family_identity"] = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    with pytest.raises(ValueError, match="upstream family"):
        apply_relation(_write(tmp_path / "prediction.json", prediction), relation_path,
                       tmp_path / "applied.json")


def test_fit_relation_rejects_upstream_group_leakage(tmp_path: Path) -> None:
    artifact = _artifact()
    artifact["trained_groups"]["id-000"].append("g-000")
    with pytest.raises(ValueError, match="OOF leakage"):
        fit_relation(_write(tmp_path / "leaky.json", artifact), tmp_path / "relation.json")


def test_compare_rejects_changed_unknown_voice_instead_of_inventing_delta(tmp_path: Path) -> None:
    control = _artifact()
    candidate = json.loads(json.dumps(control))
    candidate["candidate"] = "changed-voice"
    control["labels"]["id-001"]["VOICE_FAKE"] = None
    candidate["labels"]["id-001"]["VOICE_FAKE"] = None
    candidate["predictions"]["id-001"]["VOICE_FAKE_PROB"] = 0.99
    report = compare(
        _write(tmp_path / "control.json", control),
        _write(tmp_path / "candidate.json", candidate),
        tmp_path / "report.json", bootstrap=20,
    )
    assert report["status"] == "rejected"
    assert report["reason"] == "changed output has incomplete or invalid labels: VOICE_FAKE_PROB"
    assert report["comparison"] is None


def test_complete_local_gain_stays_provisional_without_counts_and_actual_l4(tmp_path: Path) -> None:
    control = _artifact(40)
    candidate = json.loads(json.dumps(control))
    candidate["candidate"] = "better-file"
    for index, identifier in enumerate(control["predictions"]):
        if index < 20:
            control["strata"][identifier]["domain"] = "natural"
            candidate["strata"][identifier]["domain"] = "natural"
        target = control["labels"][identifier]["FILE_FAKE"]
        control["predictions"][identifier]["FILE_FAKE_PROB"] = 0.6 if target == 0 else 0.4
        candidate["predictions"][identifier]["FILE_FAKE_PROB"] = 0.1 if target == 0 else 0.9
    report = compare(
        _write(tmp_path / "control.json", control),
        _write(tmp_path / "candidate.json", candidate),
        tmp_path / "report.json", seed=41,
    )
    assert report["comparison"]["delta"] > 0.005
    assert report["comparison"]["bootstrap"]["resamples"] == 2000
    assert report["comparison"]["bootstrap"]["ci95"][0] > 0
    assert report["status"] == "provisional"
    assert report["comparison"]["actual_l4_verified"] is False
