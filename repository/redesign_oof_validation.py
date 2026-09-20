"""Recursive OOF provenance and ensemble-mean validation."""

from __future__ import annotations

import numpy as np


def _validate_proof(artifact: dict, proof: dict) -> None:
    identifiers = list(artifact["predictions"])
    required = {"held_out_group", "trained_groups", "folds", "checkpoint_hashes"}
    missing = required - proof.keys()
    if missing:
        raise ValueError(f"incomplete OOF provenance: missing {sorted(missing)}")
    for key in ("held_out_group", "trained_groups"):
        if set(proof[key]) != set(identifiers):
            raise ValueError(f"incomplete OOF provenance for {key}")
    fold_hashes = {fold.get("checkpoint_sha256") for fold in proof["folds"]}
    if not fold_hashes or None in fold_hashes or fold_hashes != set(proof["checkpoint_hashes"]):
        raise ValueError("incomplete OOF fold/checkpoint hashes")
    all_held_out: set[str] = set()
    for fold in proof["folds"]:
        held_out = set(fold.get("held_out_groups", []))
        trained = set(fold.get("trained_groups", []))
        earlystop = set(fold.get("earlystop_groups", []))
        held_out_ids = set(fold.get("held_out_ids", []))
        train_ids = set(fold.get("train_ids", []))
        earlystop_ids = set(fold.get("earlystop_ids", []))
        if not held_out or not trained or not held_out_ids or not train_ids:
            raise ValueError("incomplete OOF fold group/row provenance")
        if held_out & trained or held_out & earlystop or trained & earlystop:
            raise ValueError("invalid OOF fold group partition")
        if held_out_ids & train_ids or held_out_ids & earlystop_ids or train_ids & earlystop_ids:
            raise ValueError("invalid OOF fold row partition")
        if not held_out_ids | train_ids | earlystop_ids <= set(identifiers):
            raise ValueError("OOF fold references unknown row IDs")
        for row_ids, group_ids in (
            (held_out_ids, held_out), (train_ids, trained), (earlystop_ids, earlystop),
        ):
            if any(not set(artifact["groups"][row_id]) <= group_ids for row_id in row_ids):
                raise ValueError("OOF fold row/group membership mismatch")
        if all_held_out & held_out:
            raise ValueError("OOF group appears in multiple held-out folds")
        all_held_out.update(held_out)
    for identifier in identifiers:
        groups = set(artifact["groups"][identifier])
        held_out = set(proof["held_out_group"][identifier])
        trained = set(proof["trained_groups"][identifier])
        if not groups or held_out != groups:
            raise ValueError(f"OOF held-out group mismatch: {identifier}")
        if held_out & trained:
            raise ValueError(f"OOF leakage: {identifier}")
        matches = [fold for fold in proof["folds"]
                   if identifier in fold["held_out_ids"]
                   and groups <= set(fold["held_out_groups"])
                   and trained == set(fold["trained_groups"])]
        if len(matches) != 1:
            raise ValueError(f"OOF row does not match exactly one fold receipt: {identifier}")


def _validate_mean(artifact: dict, members: list[dict], section: str) -> None:
    if any(set(member[section]) != set(artifact[section]) for member in members):
        raise ValueError(f"OOF ensemble/member IDs differ in {section}")
    for identifier, values in artifact[section].items():
        if any(set(member[section][identifier]) != set(values) for member in members):
            raise ValueError(f"OOF ensemble/member fields differ: {section}/{identifier}")
        for name, value in values.items():
            member_values = [np.asarray(member[section][identifier][name], dtype=np.float64)
                             for member in members]
            if any(item.shape != member_values[0].shape for item in member_values):
                raise ValueError(f"OOF ensemble/member shapes differ: {section}/{identifier}/{name}")
            expected = np.mean(np.stack(member_values), axis=0)
            actual = np.asarray(value, dtype=np.float64)
            if actual.shape != expected.shape or not np.allclose(actual, expected, rtol=0, atol=1e-12):
                raise ValueError(f"OOF ensemble is not the numeric member mean: {section}/{identifier}/{name}")


def validate_oof(artifact: dict) -> list[str]:
    """Validate all seed/head proofs and exact ensemble arithmetic."""
    base_required = {"labels", "predictions", "groups", "window_scores", "upstream_identity"}
    missing = base_required - artifact.keys()
    if missing:
        raise ValueError(f"incomplete OOF artifact: missing {sorted(missing)}")
    identifiers = list(artifact["predictions"])
    for key in ("labels", "groups", "window_scores"):
        if set(artifact[key]) != set(identifiers):
            raise ValueError(f"incomplete OOF artifact for {key}")
    members = artifact.get("oof_members")
    if members is not None:
        if not isinstance(members, list) or len(members) < 2:
            raise ValueError("multi-seed OOF requires at least two oof_members")
        for member in members:
            validate_oof(member)
            if member["labels"] != artifact["labels"] or member["groups"] != artifact["groups"]:
                raise ValueError("OOF ensemble/member labels or groups differ")
        _validate_mean(artifact, members, "predictions")
        _validate_mean(artifact, members, "window_scores")
        return identifiers
    proofs = artifact.get("oof_proofs")
    if proofs is None:
        _validate_proof(artifact, artifact)
    else:
        if not isinstance(proofs, list) or not proofs:
            raise ValueError("oof_proofs must be a non-empty list")
        names = [proof.get("name") for proof in proofs]
        if len(set(names)) != len(names) or None in names:
            raise ValueError("OOF proof names must be unique")
        if "voice_recipe_identity" in artifact["upstream_identity"] and not {"music", "voice"} <= set(names):
            raise ValueError("voice candidate requires both music and voice OOF proofs")
        for proof in proofs:
            _validate_proof(artifact, proof)
    return identifiers
