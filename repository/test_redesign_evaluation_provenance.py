import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from redesign_file import _group_class_weights, fit_relation
from test_redesign_evaluation import _artifact, _write


def _oof_ensemble(rows: int = 24) -> dict:
    members = [_artifact(rows, candidate=f"D_seed{seed}") for seed in (41, 42)]
    ensemble = json.loads(json.dumps(members[0]))
    ensemble["candidate"] = "D_ensemble"
    ensemble["oof_members"] = members
    seed_families = [member["upstream_identity"]["upstream_family_identity"] for member in members]
    family_payload = {"seed_upstream_family_identity": seed_families}
    ensemble["upstream_identity"]["seed_upstream_family_identity"] = seed_families
    ensemble["upstream_identity"]["upstream_family_payload"] = family_payload
    ensemble["upstream_identity"]["upstream_family_identity"] = hashlib.sha256(
        json.dumps(family_payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return ensemble


def _voice_oof_member(rows: int = 24) -> dict:
    artifact = _artifact(rows, candidate="D_voice_seed41")
    proof = {name: json.loads(json.dumps(artifact[name])) for name in
             ("held_out_group", "trained_groups", "folds", "checkpoint_hashes")}
    artifact["oof_proofs"] = [
        {"name": "music", **json.loads(json.dumps(proof))},
        {"name": "voice", **json.loads(json.dumps(proof))},
    ]
    base_family = artifact["upstream_identity"]["upstream_family_identity"]
    voice_recipe = "a" * 64
    payload = {"base_upstream_family_identity": base_family,
               "voice_recipe_identity": voice_recipe}
    artifact["upstream_identity"].update(
        base_upstream_family_identity=base_family,
        voice_recipe_identity=voice_recipe,
        upstream_family_payload=payload,
        upstream_family_identity=hashlib.sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest(),
    )
    return artifact


def test_duplicate_rows_within_one_group_do_not_increase_group_training_mass() -> None:
    weights = _group_class_weights(
        np.asarray([0.0, 0.0, 0.0, 1.0]), ["duplicate", "duplicate", "peer", "positive"],
    )
    assert weights[0] + weights[1] == pytest.approx(weights[2])
    assert weights[:3].sum() == pytest.approx(0.5)
    assert weights[3] == pytest.approx(0.5)


def test_multiseed_oof_rejects_leak_present_only_in_second_seed(tmp_path: Path) -> None:
    ensemble = _oof_ensemble()
    ensemble["oof_members"][1]["trained_groups"]["id-000"].append("g-000")
    with pytest.raises(ValueError, match="OOF leakage"):
        fit_relation(_write(tmp_path / "second-seed-leak.json", ensemble),
                     tmp_path / "relation.json")


def test_multiseed_oof_rejects_nonmean_ensemble_prediction(tmp_path: Path) -> None:
    ensemble = _oof_ensemble()
    ensemble["predictions"]["id-000"]["FILE_FAKE_PROB"] += 0.01
    with pytest.raises(ValueError, match="not the numeric member mean"):
        fit_relation(_write(tmp_path / "wrong-mean.json", ensemble), tmp_path / "relation.json")


def test_voice_oof_validates_music_and_voice_training_proofs(tmp_path: Path) -> None:
    artifact = _voice_oof_member()
    artifact["oof_proofs"][1]["trained_groups"]["id-000"].append("g-000")
    with pytest.raises(ValueError, match="OOF leakage"):
        fit_relation(_write(tmp_path / "voice-proof-leak.json", artifact),
                     tmp_path / "relation.json")
