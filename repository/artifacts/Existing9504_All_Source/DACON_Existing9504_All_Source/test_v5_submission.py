"""Behavioral checks for the four offline V5 submission archives."""

from __future__ import annotations

import csv
import hashlib
import json
import zipfile
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf
import torch
from torch import nn

from package_v5_submission import PackagingError, build_v5_submissions
from v2_data import decode_audio
from v5_submission import (
    InferenceError,
    encode_temporal,
    ordered_audio_files,
    predict_raw,
    read_audio,
    read_sample_ids,
)


class TinyExpert(nn.Module):
    """Expose deterministic logits and embeddings for inference-contract checks."""

    def __init__(self, width: int, offset: float, heads: int = 1) -> None:
        super().__init__()
        self.window_samples = width
        self.offset = nn.Parameter(torch.tensor(float(offset)))
        self.heads = heads

    def forward(self, waveforms: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        means = waveforms.mean(dim=1, keepdim=True) + self.offset
        logits = torch.cat(tuple(means + index for index in range(self.heads)), dim=1)
        return logits, torch.cat((means, means + 10), dim=1)


def _models() -> dict[str, nn.Module]:
    return {
        "voice": TinyExpert(4, 0),
        "music": TinyExpert(5, 1),
        "presence": TinyExpert(4, 2, heads=2),
    }


def test_raw_prediction_uses_tail_max_and_presence_weighted_file_max() -> None:
    # Given: a file whose strongest evidence exists only in the tail-anchored window.
    audio = np.array([0, 0, 0, 0, 0, 0, 4], dtype=np.float32)

    # When: all raw experts evaluate the complete file.
    actual = predict_raw(_models(), audio)

    # Then: component maxima and the presence-weighted FILE maximum match the V5 rule.
    sigmoid = torch.sigmoid
    voice = float(sigmoid(torch.tensor(1.0)))
    music = float(sigmoid(torch.tensor(1.8)))
    voice_present = float(sigmoid(torch.tensor(3.0)))
    music_present = float(sigmoid(torch.tensor(4.0)))
    expected = np.array(
        [max(voice_present * voice, music_present * music), voice, music,
         voice_present, music_present], dtype=np.float32,
    )
    np.testing.assert_allclose(actual, expected, rtol=0, atol=1e-7)


def test_temporal_encoding_preserves_feature_order_timestamps_and_file_independence() -> None:
    # Given: aligned raw and stem views with two tail-covering windows.
    raw = np.arange(7, dtype=np.float32)
    voice = raw + 10
    music = raw + 20
    models = _models()

    # When: the second file is encoded alone and after an unrelated first file.
    expected_features, expected_times = encode_temporal(
        models, raw, voice, music, timeline_samples=4,
    )
    encode_temporal(models, np.ones(9, dtype=np.float32), np.ones(9, dtype=np.float32),
                    np.ones(9, dtype=np.float32))
    actual_features, actual_times = encode_temporal(
        models, raw, voice, music, timeline_samples=4,
    )

    # Then: state does not cross files and the first expert block has the exact trained order.
    np.testing.assert_array_equal(actual_features, expected_features)
    np.testing.assert_array_equal(actual_times, expected_times)
    np.testing.assert_array_equal(actual_times, np.array([0, 3 / 16_000], dtype=np.float32))
    # voice: raw embedding(2), stem embedding(2), raw logit, stem logit, difference
    np.testing.assert_allclose(actual_features[0, :7], [1.5, 11.5, 11.5, 21.5, 1.5, 11.5, -10])


def test_input_boundary_rejects_malformed_schema_and_id_mismatch(tmp_path: Path) -> None:
    # Given: a malformed sample header and later a sample with no matching audio.
    sample = tmp_path / "sample_submission.csv"
    sample.write_text("ID,FILE_FAKE_PROB\na,0\n", encoding="utf-8")

    # When/Then: malformed evaluator metadata is rejected before model loading.
    with pytest.raises(InferenceError, match="six-column"):
        read_sample_ids(sample)

    sample.write_text(
        "ID,FILE_FAKE_PROB,VOICE_FAKE_PROB,MUSIC_FAKE_PROB,VOICE_PRESENT_PROB,MUSIC_PRESENT_PROB\n"
        "a,0,0,0,0,0\n", encoding="utf-8",
    )
    test_dir = tmp_path / "test"
    test_dir.mkdir()
    (test_dir / "b.wav").write_bytes(b"not audio")
    with pytest.raises(InferenceError, match="mismatch"):
        ordered_audio_files(test_dir, read_sample_ids(sample))
    with pytest.raises(InferenceError, match="Cannot decode"):
        read_audio(test_dir / "b.wav")


def test_runtime_decode_matches_training_decoder_for_stereo_flac(tmp_path: Path) -> None:
    # Given: a stereo compressed input, which is sensitive to decoder and downmix choices.
    path = tmp_path / "probe.flac"
    time = np.arange(22_050, dtype=np.float32) / 22_050
    sf.write(path, np.stack((np.sin(time * 77), np.cos(time * 53)), axis=1), 22_050)

    # When: training and packaged inference decode the same source.
    expected = decode_audio(path, limit_seconds=None)
    actual = read_audio(path)

    # Then: every PCM sample is identical, including resampling and channel order.
    np.testing.assert_array_equal(actual, expected)


def _write_complete_job(root: Path, name: str) -> None:
    job = root / ("jobs" if name.startswith(("B1_", "B2_")) else "") / name
    job.mkdir(parents=True, exist_ok=True)
    checkpoint = job / "final.pt"
    torch.save({"weight": torch.tensor([1.0])}, checkpoint)
    digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    identity = ({"arm": name[:2], "kind": name[3:], "augmented": name.startswith("B2_")}
                if "_" in name else {"name": name, "joint": name == "B4"})
    (job / "training.json").write_text(json.dumps({
        "status": "complete", "epochs_requested": 2, "epochs_completed": 2,
        "exposures": {"source-a": 2}, "state_sha256": digest,
        "full_data_refit": {"status": "complete"}, **identity,
    }), encoding="utf-8")


def _asset_trees(root: Path) -> tuple[Path, Path]:
    baseline = root / "baseline" / "model"
    for directory in (baseline / "df_arena_1b", baseline / "panns", baseline / "htdemucs"):
        directory.mkdir(parents=True, exist_ok=True)
    for name in ("config.json", "backbone.py", "modeling_antispoofing.py"):
        (baseline / "df_arena_1b" / name).write_text("{}" if name.endswith("json") else "# source\n")
    (baseline / "df_arena_1b" / "pytorch_model.bin").write_bytes(b"original giant weight")
    (baseline / "panns" / "class_labels_indices.csv").write_text("display_name\nSpeech\n")
    (baseline / "panns" / "component_labels.json").write_text('{"voice": [], "music": []}')
    (baseline / "panns" / "Cnn14_mAP=0.431.pth").write_bytes(b"original giant weight")
    (baseline / "htdemucs" / "htdemucs.yaml").write_text("models: []\n")
    (baseline / "htdemucs" / "weights.th").write_bytes(b"demucs")
    (baseline / "MODEL_INFO.txt").write_text("baseline provenance\n")
    (baseline / "SHA256SUMS.txt").write_text("baseline checksums\n")
    sonics = root / "sonics_assets"
    sonics.mkdir()
    (sonics / "config.json").write_text("{}")
    (sonics / "pytorch_model.bin").write_bytes(b"original weight")
    (sonics / "LICENSE").write_text("SONICS license\n")
    (sonics / "provenance.json").write_text('{"source": "SONICS"}')
    vendor = root / "sonics_vendor" / "sonics"
    vendor.mkdir(parents=True)
    (vendor / "__init__.py").write_text("")
    return baseline.parent, sonics


@pytest.mark.parametrize("arms", [("B1", "B2", "B3", "B4"), ("B1", "B2")])
def test_builder_packages_selected_completed_arms_without_original_expert_weights(tmp_path: Path, arms: tuple[str, ...]) -> None:
    # Given: complete digest-pinned refits and local architecture assets.
    baseline, sonics = _asset_trees(tmp_path)
    refits = tmp_path / "refits"
    (refits / "data_audit.json").parent.mkdir(parents=True, exist_ok=True)
    (refits / "data_audit.json").write_text('{"scope": "full local data"}')
    provenance = tmp_path / "v4_validation"
    provenance.mkdir()
    (provenance / "provenance.json").write_text('{"sources": []}')
    (provenance / "licenses").mkdir()
    (provenance / "licenses" / "SOURCE_LICENSE.txt").write_text("source license\n")
    for arm in ("B1", "B2"):
        for kind in ("voice", "music", "presence"):
            _write_complete_job(refits, f"{arm}_{kind}")
    for arm in arms:
        if arm not in {"B3", "B4"}:
            continue
        _write_complete_job(refits, arm)

    # When: all submission variants are packaged.
    outputs = build_v5_submissions(baseline, sonics, refits, tmp_path / "out", arms=arms)

    # Then: every archive has the official root and no redundant original expert tensors.
    assert tuple(path.name for path in outputs) == tuple(f"V5_{arm}.zip" for arm in arms)
    for path in outputs:
        with zipfile.ZipFile(path) as archive:
            names = set(archive.namelist())
            assert {name.split("/", 1)[0] for name in names} == {"script.py", "requirements.txt", "model"}
            assert "model/runtime/v5_submission.py" in names
            assert "model/MODEL_INFO.txt" in names
            assert "model/provenance/final_data_audit.json" in names
            assert "model/provenance/source_provenance.json" in names
            assert "model/sonics/LICENSE" in names
            assert "model/weights/voice.pt" in names
            assert not any(name.endswith(("pytorch_model.bin", "Cnn14_mAP=0.431.pth")) for name in names)
            arm = path.stem.removeprefix("V5_")
            assert ("model/weights/fusion.pt" in names) == (arm in {"B3", "B4"})
            assert ("model/htdemucs/weights.th" in names) == (arm in {"B3", "B4"})


def test_builder_rejects_incomplete_or_tampered_refit(tmp_path: Path) -> None:
    # Given: one metadata record claims an incomplete run.
    baseline, sonics = _asset_trees(tmp_path)
    refits = tmp_path / "refits"
    (refits / "data_audit.json").parent.mkdir(parents=True, exist_ok=True)
    (refits / "data_audit.json").write_text('{"scope": "full local data"}')
    provenance = tmp_path / "v4_validation"
    provenance.mkdir()
    (provenance / "provenance.json").write_text('{"sources": []}')
    for arm in ("B1", "B2"):
        for kind in ("voice", "music", "presence"):
            _write_complete_job(refits, f"{arm}_{kind}")
    for arm in ("B3", "B4"):
        _write_complete_job(refits, arm)
    metadata = refits / "jobs" / "B1_voice" / "training.json"
    value = json.loads(metadata.read_text())
    value["status"] = "budget_stop"
    metadata.write_text(json.dumps(value))

    # When/Then: no partial weights can be labeled as a final submission.
    with pytest.raises(PackagingError, match="not complete"):
        build_v5_submissions(baseline, sonics, refits, tmp_path / "out")
