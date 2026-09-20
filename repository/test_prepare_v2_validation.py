from pathlib import Path

import pytest

import prepare_v2_validation as validation


def _row(identifier: str, group: str, role: str, voice_present: str = "") -> validation.ManifestRow:
    return validation.ManifestRow(identifier, f"audio/{identifier}.wav", group, role, "fixture", "0", "", "0", voice_present, "1", identifier)


def test_audit_rejects_atomic_group_crossing_roles() -> None:
    # Given a parent group reused across a development and locked role.
    rows = [_row("a", "voice:a|music:z", "dev_train"), _row("b", "music:z", "test_source")]
    # When the sealed split is audited, then transitive lineage leakage is rejected.
    with pytest.raises(validation.SplitIntegrityError, match="crosses roles"):
        validation.audit_rows(rows)


def test_unknown_label_survives_manifest_round_trip(tmp_path: Path) -> None:
    # Given a music item with no evidence about voice presence.
    rows = [_row("a", "music:a", "dev_train", voice_present="")]
    path = tmp_path / "manifest.csv"
    # When the manifest is serialized and parsed, then unknown stays unknown.
    validation.write_manifest(path, rows)
    parsed = validation.read_manifest(path)
    assert parsed[0].VOICE_PRESENT == ""


def test_mixture_parents_use_pure_voice_groups() -> None:
    # Given a pure speech row and an earlier mixed speech/music row.
    rows = [
        validation.ManifestRow("v", "audio/v.wav", "voice:v", "dev_train", "speech", "0", "0", "", "1", "0", "v"),
        validation.ManifestRow("m", "audio/m.wav", "voice:x|music:x", "dev_train", "mixture", "1", "1", "1", "1", "1", "m"),
    ]
    # When parents are selected, then a prior mixture cannot become a voice parent.
    assert validation.pure_voice_groups(rows, "dev_train") == ["voice:v"]


def test_mixture_music_parent_requires_both_authenticity_classes() -> None:
    # Given one unpaired legacy music group and one paired-content group.
    rows = [
        validation.ManifestRow("u", "audio/u.wav", "music:u", "dev_train", "legacy", "0", "", "0", "", "1", "u"),
        validation.ManifestRow("r", "audio/r.wav", "music:p", "dev_train", "FMA", "0", "", "0", "", "1", "r"),
        validation.ManifestRow("f", "audio/f.wav", "music:p", "dev_train", "Echoes", "1", "", "1", "", "1", "f"),
    ]
    # When parent groups are selected, then only a group with real and fake music qualifies.
    assert validation.paired_music_groups(rows, "dev_train") == ["music:p"]


def test_generated_music_does_not_turn_unknown_background_voice_real() -> None:
    # Given a real foreground voice over generated music that may itself contain vocals.
    # When the mixture label is derived, then voice authenticity remains unknown.
    assert validation.mixture_voice_label("0", "1") == ""
    assert validation.mixture_voice_label("1", "1") == "1"
    assert validation.mixture_voice_label("0", "0") == "0"
