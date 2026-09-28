from dataclasses import replace
from pathlib import Path

import pytest

import prepare_recovery_data as recovery


def _row(*, group: str = "new:g", content: str = "new.wav", pcm: str = "new-pcm", voice: str = "", source: str = "Echoes/suno") -> recovery.FreshRow:
    return recovery.FreshRow(
        "id",
        "audio/id.wav",
        group,
        "fresh_challenge",
        source,
        "1",
        voice,
        "1",
        voice,
        "1",
        content,
        "https://example.test/audio",
        "revision",
        "source-hash",
        pcm,
        61.0,
        60.0,
        "CC-BY-4.0",
    )


def test_audit_rejects_prior_lineage_pcm_and_unverified_music_voice() -> None:
    # Given a candidate that repeats an old content ID and makes an unsupported vocal claim.
    row = _row(content="old.wav", voice="1")
    # When fresh-bank integrity is checked, then lineage/PCM separation and blank music voice labels are required.
    with pytest.raises(recovery.RecoveryIntegrityError):
        recovery.audit_rows([row], {"old:g"}, {"old.wav"}, {"old-pcm"})


def test_audit_requires_balanced_real_fake_for_each_music_generator() -> None:
    # Given a real reference and only one fake provider for a selected artist.
    real = replace(_row(group="fma_artist_1", content="track", pcm="real", source="FMA/FMA"), FILE_FAKE="0", MUSIC_FAKE="0")
    fake = _row(group="fma_artist_1", content="track", pcm="fake")
    # When the music subset is checked, then every declared provider must have a matching fake row per artist.
    with pytest.raises(recovery.RecoveryIntegrityError, match="providers"):
        recovery.audit_rows([real, fake], set(), set(), set())


def test_audit_accepts_a_complete_source_disjoint_music_quartet() -> None:
    # Given one new artist with a licensed real track and every required fake provider.
    real = replace(_row(group="fma_artist_1", content="track", pcm="real", source="FMA/FMA"), FILE_FAKE="0", MUSIC_FAKE="0")
    fakes = [replace(_row(group="fma_artist_1", content="track", pcm=provider), ID=provider, source=f"Echoes/{provider}") for provider in recovery.MUSIC_PROVIDERS]
    # When the quartet is audited, then it is accepted as a usable real/fake music comparison.
    recovery.audit_rows([real, *fakes], set(), set(), set())


def test_mlaad_url_encodes_provider_names_for_curl() -> None:
    # Given an MLAAD provider name containing a space and parentheses.
    path = "fake/en/Cartesia.ai (Sonic-3)/sample.wav"
    # When its pinned source URL is built, then it is accepted by curl as one URL argument.
    assert recovery._mlaad_url("https://example.test/", path).endswith("Cartesia.ai%20%28Sonic-3%29/sample.wav")


def test_prune_audio_removes_unmanifested_failed_candidate(tmp_path: Path) -> None:
    # Given a failed quartet that left one FMA WAV beside a retained row.
    row = _row()
    for root in (tmp_path / "fresh", tmp_path / "recovery"):
        (root / "audio").mkdir(parents=True)
        (root / "audio" / "id.wav").write_bytes(b"kept")
        (root / "audio" / "failed.wav").write_bytes(b"stale")
    # When final artifacts are pruned, then only manifest-addressable audio remains.
    recovery._prune_audio(tmp_path / "fresh", tmp_path / "recovery", [row])
    assert not (tmp_path / "fresh/audio/failed.wav").exists()
    assert (tmp_path / "recovery/audio/id.wav").exists()
