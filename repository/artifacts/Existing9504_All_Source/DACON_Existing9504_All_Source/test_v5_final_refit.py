"""High-signal checks for resumable V5 full-data final refits."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import numpy as np
import torch
from torch import nn

import v5_final_refit
from v5_dataset import Example
from v5_features import EncodedFile
from v5_final_refit import (
    FinalExpertConfig,
    FinalFusionConfig,
    train_final_expert,
    train_final_fusion,
)


class TinyExpert(nn.Module):
    window_samples = 4

    def __init__(self) -> None:
        super().__init__()
        self.encoder = nn.Linear(4, 3)
        self.head = nn.Linear(3, 1)
        self.register_buffer("batches_seen", torch.tensor(0, dtype=torch.long))

    def forward(self, audio: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        embedding = torch.tanh(self.encoder(audio))
        return self.head(embedding), embedding


class StepClock:
    def __init__(self, expire_after: int) -> None:
        self.calls = 0
        self.expire_after = expire_after

    def __call__(self) -> float:
        self.calls += 1
        return 0.0 if self.calls <= self.expire_after else 100.0


def _voice_rows() -> list[Example]:
    return [
        Example(f"row-{index}", Path(f"{index}.wav"), (f"group-{index}",),
                (float(index % 2), float(index % 2), math.nan, 1.0, 0.0), "speech")
        for index in range(4)
    ]


def _crop(row: Example, _epoch: int, width: int, _augmented: bool,
          _seed: int) -> tuple[np.ndarray, tuple[float, ...]]:
    return np.linspace(-1.0, 1.0, width, dtype=np.float32) + row.labels[1], row.labels


def test_expert_refit_covers_every_eligible_id_and_updates_encoder_and_head(
    tmp_path: Path, monkeypatch, capsys,
) -> None:
    # Given: partial-label rows and a connected expert initialized once by the caller.
    monkeypatch.setattr(v5_final_refit, "training_crop", _crop)
    monkeypatch.setitem(v5_final_refit.EXPERT_EPOCHS, ("B1", "voice"), 2)
    model = TinyExpert()
    before = {name: value.detach().clone() for name, value in model.named_parameters()}

    # When: the fixed two-epoch full-data refit completes without a selection split.
    rows = [*_voice_rows(), Example("sealed-splice", Path("sealed.wav"), ("sealed",),
            (1.0, 1.0, math.nan, 1.0, 0.0), "sealed_splice")]
    metadata = train_final_expert(model, rows, FinalExpertConfig(
        arm="B1", kind="voice", output=tmp_path, deadline=float("inf"),
        device="cpu", microbatch=2, accumulation=1,
    ))

    # Then: every eligible ID was used twice and both intended parameter groups changed.
    final = tmp_path / "jobs/B1_voice/final.pt"
    assert metadata["status"] == "complete"
    assert metadata["epochs_completed"] == 2
    assert metadata["exposures"] == {row.id: 2 for row in _voice_rows()}
    assert metadata["excluded_ids"] == {"sealed-splice": "unknown_crop_target"}
    assert metadata["gradient_audit"]["encoder"] > 0
    assert metadata["gradient_audit"]["head"] > 0
    assert metadata["update_audit"]["encoder"] > 0
    assert metadata["update_audit"]["head"] > 0
    assert metadata["cosine_horizon_epochs"] == 60
    assert metadata["dev_selection"] is False
    assert hashlib.sha256(final.read_bytes()).hexdigest() == metadata["state_sha256"]
    saved = torch.load(final, map_location="cpu", weights_only=True)
    assert all(not torch.equal(saved[name], before[name]) for name in before)
    assert saved["batches_seen"].dtype == torch.long
    events = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    assert [(event["epoch"], event["epochs"]) for event in events] == [(1, 2), (2, 2)]


def test_deadline_is_fail_closed_and_checkpoint_resume_matches_uninterrupted(
    tmp_path: Path, monkeypatch,
) -> None:
    # Given: identical seeds and an interruption after the first complete epoch.
    monkeypatch.setattr(v5_final_refit, "training_crop", _crop)
    monkeypatch.setitem(v5_final_refit.EXPERT_EPOCHS, ("B2", "voice"), 2)
    torch.manual_seed(17)
    initial = TinyExpert().state_dict()
    interrupted = TinyExpert()
    interrupted.load_state_dict(initial)
    output = tmp_path / "resume"

    # When: the first invocation expires, then the same job resumes to epoch two.
    stopped = train_final_expert(interrupted, _voice_rows(), FinalExpertConfig(
        arm="B2", kind="voice", output=output, deadline=50.0, device="cpu",
        microbatch=4, accumulation=1, clock=StepClock(expire_after=3),
    ))
    assert stopped["status"] == "incomplete_deadline"
    assert not (output / "jobs/B2_voice/final.pt").exists()
    assert (output / "jobs/B2_voice/checkpoint.pt").is_file()
    resumed_model = TinyExpert()
    resumed = train_final_expert(resumed_model, _voice_rows(), FinalExpertConfig(
        arm="B2", kind="voice", output=output, deadline=float("inf"),
        device="cpu", microbatch=4, accumulation=1,
    ))

    torch.manual_seed(999)
    uninterrupted_model = TinyExpert()
    uninterrupted_model.load_state_dict(initial)
    reference = tmp_path / "reference"
    train_final_expert(uninterrupted_model, _voice_rows(), FinalExpertConfig(
        arm="B2", kind="voice", output=reference, deadline=float("inf"),
        device="cpu", microbatch=4, accumulation=1,
    ))

    # Then: optimizer/scheduler/model restoration yields the exact uninterrupted state.
    resumed_state = torch.load(output / "jobs/B2_voice/final.pt", map_location="cpu", weights_only=True)
    reference_state = torch.load(reference / "jobs/B2_voice/final.pt", map_location="cpu", weights_only=True)
    assert resumed["status"] == "complete"
    assert resumed["resumed_from_epoch"] == 1
    assert all(torch.equal(resumed_state[name], reference_state[name]) for name in reference_state)


def _encoded(identifier: str, label: tuple[float, ...]) -> EncodedFile:
    rng = np.random.default_rng(sum(identifier.encode()))
    return EncodedFile(identifier, rng.normal(size=(2, 4)).astype(np.float32),
                       np.array([0.0, 4.0375], dtype=np.float32), label)


def test_temporal_refits_preserve_b3_b4_architecture_and_full_pool_coverage(
    tmp_path: Path, monkeypatch,
) -> None:
    # Given: cached B2 raw/stem features with partial known labels.
    nan = math.nan
    features = [
        _encoded("speech-real", (0.0, 0.0, nan, 1.0, 0.0)),
        _encoded("speech-fake", (1.0, 1.0, nan, 1.0, 0.0)),
        _encoded("music-real", (0.0, nan, 0.0, 0.0, 1.0)),
        _encoded("music-fake", (1.0, nan, 1.0, 0.0, 1.0)),
    ]
    monkeypatch.setattr(v5_final_refit, "FUSION_EPOCHS", 2)

    # When: original non-joint B3 and nine-state B4 each train for two fixed epochs.
    b3 = train_final_fusion(features, FinalFusionConfig(
        name="B3", joint=False, output=tmp_path, deadline=float("inf"), device="cpu"
    ))
    b4 = train_final_fusion(features, FinalFusionConfig(
        name="B4", joint=True, output=tmp_path, deadline=float("inf"), device="cpu"
    ))

    # Then: both consume every file per epoch with the original input width and no dev selection.
    for name, metadata, joint in (("B3", b3, False), ("B4", b4, True)):
        final = tmp_path / name / "final.pt"
        assert metadata["status"] == "complete"
        assert metadata["joint"] is joint
        assert metadata["input_dim"] == 4
        assert metadata["exposures"] == {item.identifier: 2 for item in features}
        assert metadata["all_model_gradients_connected"] is True
        assert metadata["dev_selection"] is False
        assert hashlib.sha256(final.read_bytes()).hexdigest() == metadata["state_sha256"]


def test_temporal_deadline_resume_restores_optimizer_and_scheduler(
    tmp_path: Path, monkeypatch,
) -> None:
    # Given: a full feature pool and a deadline after one completed temporal epoch.
    nan = math.nan
    features = [
        _encoded("real", (0.0, 0.0, nan, 1.0, 0.0)),
        _encoded("fake", (1.0, 1.0, nan, 1.0, 0.0)),
    ]
    monkeypatch.setattr(v5_final_refit, "FUSION_EPOCHS", 2)
    output = tmp_path / "resume"

    # When: B3 is interrupted and resumed from its optimizer/scheduler checkpoint.
    stopped = train_final_fusion(features, FinalFusionConfig(
        name="B3", joint=False, output=output, deadline=50.0,
        device="cpu", clock=StepClock(expire_after=3), seed=23,
    ))
    assert stopped["status"] == "incomplete_deadline"
    assert not (output / "B3/final.pt").exists()
    resumed = train_final_fusion(features, FinalFusionConfig(
        name="B3", joint=False, output=output, deadline=float("inf"),
        device="cpu", seed=23,
    ))
    reference_root = tmp_path / "reference"
    train_final_fusion(features, FinalFusionConfig(
        name="B3", joint=False, output=reference_root, deadline=float("inf"),
        device="cpu", seed=23,
    ))

    # Then: resumed and uninterrupted final model tensors are bit-identical.
    resumed_state = torch.load(output / "B3/final.pt", map_location="cpu", weights_only=True)
    reference_state = torch.load(reference_root / "B3/final.pt", map_location="cpu", weights_only=True)
    assert resumed["resumed_from_epoch"] == 1
    assert all(torch.equal(resumed_state[name], reference_state[name]) for name in reference_state)
