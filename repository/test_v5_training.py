"""High-signal numerical and optimizer checks for the V5 expert trainer."""

from __future__ import annotations

import json
import math
import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch
from torch import nn

import v5_training
from v5_dataset import Example
from v5_training import (
    EvaluationResult,
    HeadMetric,
    TrainConfig,
    evaluate_expert,
    masked_balanced_bce,
    train_expert,
)


class ManualClock:
    def __init__(self) -> None:
        self.value = 0.0

    def __call__(self) -> float:
        return self.value

    def advance(self, seconds: float) -> None:
        self.value += seconds


class SurrogateVoiceExpert(nn.Module):
    """Small real autograd graph; it is not evidence about production encoders."""

    embedding_dim = 3
    window_samples = 4

    def __init__(self) -> None:
        super().__init__()
        self.encoder = nn.Linear(4, 3)
        self.batch_norm = nn.BatchNorm1d(3)
        self.head = nn.Linear(3, 1)
        self.forward_modes: list[bool] = []

    def forward(self, audio: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        self.forward_modes.append(self.training)
        embeddings = torch.tanh(self.batch_norm(self.encoder(audio)))
        return self.head(embeddings), embeddings


def test_balanced_loss_gives_each_known_class_equal_mass() -> None:
    # Given: one negative/positive microbatch from a 3:1 imbalanced population.
    logits = torch.tensor([2.0, -2.0])
    targets = torch.tensor([0.0, 1.0])
    population = torch.tensor([0.0, 0.0, 0.0, 1.0, math.nan])

    # When: class weights are derived from the complete known target population.
    loss = masked_balanced_bce(logits, targets, population)

    # Then: negative and positive class means contribute exactly one half each.
    losses = torch.nn.functional.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    assert torch.allclose(loss, (losses[0] * (2 / 3) + losses[1] * 2) / 2)
    per_example = [masked_balanced_bce(logits[index:index + 1], targets[index:index + 1], population)
                   for index in range(2)]
    assert torch.allclose(per_example[0] * 3, per_example[1])


def test_evaluation_uses_all_windows_and_tail_anchor() -> None:
    # Given: a nine-sample file whose largest score is visible only in the tail window.
    row = Example("tail", Path("tail.wav"), ("source",), (0, 0, math.nan, 1, 0), "dev")
    model = SurrogateVoiceExpert()
    with torch.no_grad():
        model.encoder.weight.fill_(0.25)
        model.encoder.bias.zero_()
        model.head.weight.fill_(1.0)
        model.head.bias.zero_()
    original = v5_training.read_audio
    v5_training.read_audio = lambda _path: np.arange(9, dtype=np.float32)
    try:
        # When: full-file evaluation runs with native non-overlapping windows.
        result = evaluate_expert(model, [row], "voice", device="cpu")
    finally:
        v5_training.read_audio = original

    # Then: starts 0, 4, and tail-anchored 5 are all pooled by maximum probability.
    windows = torch.tensor([[0, 1, 2, 3], [4, 5, 6, 7], [5, 6, 7, 8]], dtype=torch.float32)
    expected = torch.sigmoid(model(windows)[0]).max().item()
    assert result.predictions == (("tail", (expected,)),)
    assert result.window_starts == (("tail", (0, 4, 5)),)


def test_train_expert_updates_encoder_and_counts_only_complete_epochs(
    capsys: pytest.CaptureFixture[str],
) -> None:
    # Given: balanced known voice labels and a fully connected surrogate expert.
    rows = [
        Example(f"row-{index}", Path(f"{index}.wav"), (f"source-{index % 2}",),
                (float(index % 2), float(index % 2), math.nan, 1.0, 0.0), "train")
        for index in range(4)
    ]
    model = SurrogateVoiceExpert()
    initial = {name: value.detach().clone() for name, value in model.named_parameters()}
    original_crop, original_audio = v5_training.training_crop, v5_training.read_audio
    v5_training.training_crop = lambda row, epoch, window_samples, augmented, seed: (
        np.linspace(-1, 1, window_samples, dtype=np.float32) + row.labels[1], row.labels
    )
    v5_training.read_audio = lambda path: np.linspace(-1, 1, 9, dtype=np.float32) + int(path.stem)
    try:
        with tempfile.TemporaryDirectory() as directory:
            # When: two complete epochs run with accumulation and development evaluation.
            result = train_expert(model, rows, rows, TrainConfig(
                kind="voice", output=Path(directory), deadline=float("inf"), epochs=2,
                evaluation_interval=1, microbatch=1, accumulation=2, device="cpu",
            ))

            # Then: both encoder groups changed, each source was exposed twice, and weights reload.
            assert result.status == "completed"
            assert result.epochs_completed == 2
            assert result.updates == 4
            assert result.first_step_audit is not None
            assert result.first_step_audit.gradient_norms["encoder"] > 0
            assert result.first_step_audit.parameter_deltas["encoder"] > 0
            assert result.epoch_metrics[-1].source_exposures == {"source-0": 2, "source-1": 2}
            assert result.best_path is not None and result.best_path.is_file()
            assert result.final_path is not None and result.final_path.is_file()
            saved = torch.load(result.final_path, map_location="cpu", weights_only=True)
            assert any(not torch.equal(saved[name], initial[name]) for name in initial)
            first_evaluation = model.forward_modes.index(False)
            assert any(model.forward_modes[index] for index in range(first_evaluation + 1,
                                                                      len(model.forward_modes)))
            events = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
            assert [event["event"] for event in events].count("epoch_complete") == 2
            assert events[0]["event"] == "first_step_audit"
            assert all(event.get("batch_norm_stats_frozen") == 1 for event in events[1:])
    finally:
        v5_training.training_crop, v5_training.read_audio = original_crop, original_audio


def test_budget_stop_evaluates_and_selects_last_completed_epoch() -> None:
    # Given: each training epoch and full evaluation take two seconds under a nine-second budget.
    rows = [
        Example(f"budget-{index}", Path(f"{index}.wav"), (f"source-{index}",),
                (float(index), float(index), math.nan, 1.0, 0.0), "train")
        for index in range(2)
    ]
    model = SurrogateVoiceExpert()
    clock = ManualClock()
    evaluation_states: list[dict[str, torch.Tensor]] = []
    original_crop, original_evaluate = v5_training.training_crop, v5_training.evaluate_expert

    def timed_crop(row: Example, _epoch: int, window_samples: int, _augmented: bool,
                   _seed: int) -> tuple[np.ndarray, tuple[float, ...]]:
        clock.advance(1.0)
        return np.full(window_samples, row.labels[1], dtype=np.float32), row.labels

    def improving_evaluation(_model: nn.Module, _rows: list[Example], _kind: str, **_kwargs) -> EvaluationResult:
        clock.advance(2.0)
        evaluation_states.append({name: value.detach().clone()
                                  for name, value in _model.state_dict().items()})
        score = float(len(evaluation_states))
        return EvaluationResult((), (), (HeadMetric("voice_fake", 2, 1.0, 0.5, 0.5),),
                                1.0, score)

    v5_training.training_crop = timed_crop
    v5_training.evaluate_expert = improving_evaluation
    try:
        with tempfile.TemporaryDirectory() as directory:
            # When: interval five would normally leave epoch two unevaluated before budget stop.
            result = train_expert(model, rows, rows, TrainConfig(
                kind="voice", output=Path(directory), deadline=9.0, epochs=6,
                evaluation_interval=5, microbatch=2, accumulation=1, device="cpu", clock=clock,
            ))

            # Then: epoch two is evaluated adaptively and its improving weights become best.
            assert result.status == "budget_stop"
            assert result.epochs_completed == 2
            assert [report.dev is not None for report in result.epoch_metrics] == [True, True]
            assert len(evaluation_states) == 2
            assert result.best_path is not None
            best = torch.load(result.best_path, map_location="cpu", weights_only=True)
            assert all(torch.equal(best[name], evaluation_states[-1][name]) for name in best)
    finally:
        v5_training.training_crop, v5_training.evaluate_expert = original_crop, original_evaluate
