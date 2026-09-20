"""High-signal integrity checks for the V4 data contract."""
from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from prepare_v4_data import BuiltRow, V4IntegrityError, validate_rows
from v2_data import decode_audio, pcm_sha256
from v4_data import load_audio as load_runtime_audio
from v4_data import load_manifest
from v4_data_acquisition import DEV_HELDOUT, SEALED_HELDOUT, assign_music_roles, load_echoes, prior_identities

ROOT = Path(__file__).resolve().parent


def _built_rows() -> list[BuiltRow]:
    sources = json.loads((ROOT / "v4_validation/provenance.json").read_text())["sources"]
    return [BuiltRow(**row) for row in sources]


def test_full_decode_preserves_content_after_ten_seconds() -> None:
    inventory = json.loads((ROOT / "_workspace/v2_sources/source_inventory.json").read_text())["sources"]
    source = next(ROOT / "_workspace/v2_sources" / row["original_path"] for row in inventory if row["source_kind"] == "fma_real")
    centered = decode_audio(source)
    full = decode_audio(source, limit_seconds=None)
    assert len(centered) == 160000
    assert len(full) > 320000
    assert pcm_sha256(full[160000:320000]) != pcm_sha256(centered)
    assert float(np.sqrt(np.mean(np.square(full[160000:320000]), dtype=np.float64))) > 1e-4


def test_exclusive_first_artist_assignment_matches_audited_pool() -> None:
    pairs, generated, _ = load_echoes(ROOT)
    _, old_artists, _ = prior_identities(ROOT)
    dev, sealed = assign_music_roles(pairs, generated, old_artists)
    assert len(dev) == len(set(dev)) == 10
    assert len(sealed) == len(set(sealed)) == 10
    assert not set(dev) & set(sealed)
    assert {"966", "214", "233", "22222", "20348", "15212"} <= set(dev)
    assert {"17198", "21958"} <= set(sealed)


def test_manifest_lineage_labels_and_provider_holds() -> None:
    rows = _built_rows()
    by_id = {row.ID: row for row in rows}
    protocol = json.loads((ROOT / "v4_validation/protocol.json").read_text())
    old_groups, _, _ = prior_identities(ROOT)
    providers = tuple(protocol["music_roles"]["training"]["providers"])
    validate_rows(rows, old_groups, set(protocol["music_roles"]["dev_artists"]), set(protocol["music_roles"]["sealed_artists"]), providers)
    assert all(row.original_duration_seconds >= 4.0 for row in rows if row.panel not in {"mixture", "splice"})
    for row in rows:
        if row.panel not in {"mixture", "splice"}:
            continue
        parent_groups = {part for parent in row.parent_ids for part in by_id[parent].group_id.split("|")}
        assert set(row.group_id.split("|")) == parent_groups
        if row.MUSIC_FAKE == "1" and row.VOICE_FAKE != "1":
            assert row.VOICE_FAKE == ""
    assert not any(row.split == "train" and row.provider in {DEV_HELDOUT, SEALED_HELDOUT} for row in rows)


def test_injected_parent_and_provider_leaks_are_rejected() -> None:
    rows = _built_rows()
    protocol = json.loads((ROOT / "v4_validation/protocol.json").read_text())
    old_groups, _, _ = prior_identities(ROOT)
    providers = tuple(protocol["music_roles"]["training"]["providers"])
    dev_index = next(index for index, row in enumerate(rows) if row.split == "dev" and row.panel == "music_real")
    leaked = rows.copy()
    leaked[dev_index] = replace(leaked[dev_index], group_id=next(iter(old_groups)))
    with pytest.raises(V4IntegrityError, match="overlap|leaked"):
        validate_rows(leaked, old_groups, set(protocol["music_roles"]["dev_artists"]), set(protocol["music_roles"]["sealed_artists"]), providers)
    train_index = next(index for index, row in enumerate(rows) if row.split == "train")
    leaked = rows.copy()
    leaked[train_index] = replace(leaked[train_index], provider=DEV_HELDOUT)
    with pytest.raises(V4IntegrityError, match="leaked"):
        validate_rows(leaked, old_groups, set(protocol["music_roles"]["dev_artists"]), set(protocol["music_roles"]["sealed_artists"]), providers)


def test_declared_eval_durations_and_real_fake_splice_pairing() -> None:
    rows = _built_rows()
    for role in ("dev", "sealed"):
        music = [row for row in rows if row.split == role and row.panel.startswith("music_")]
        by_group = {group: [row for row in music if row.group_id == group] for group in {row.group_id for row in music}}
        assert len(by_group) == 10
        assert all(len(group_rows) == 5 and len({round(row.evaluation_duration_seconds, 3) for row in group_rows}) == 1 for group_rows in by_group.values())
        splices = [row for row in rows if row.split == role and row.panel == "splice"]
        assert sum(row.FILE_FAKE == row.MUSIC_FAKE == "0" and row.VOICE_FAKE == "" for row in splices) == 20
        assert sum(row.FILE_FAKE == row.MUSIC_FAKE == "1" and row.VOICE_FAKE == "" for row in splices) == 20
        assert all(abs(row.evaluation_duration_seconds - 30.0) < 1e-6 for row in splices)
    for split in ("train", "dev", "sealed"):
        manifest = load_manifest(ROOT / "v4_validation", split)
        assert all(row["split"] == split for row in manifest)
        sample = manifest[0]
        assert len(load_runtime_audio(ROOT / "v4_validation" / sample["path"])) > 0
