"""High-signal checks for V4 optimizer resume and deadline behavior."""

from __future__ import annotations

import math
import hashlib
import json
import argparse
import shutil
import time
from pathlib import Path

import torch
import numpy as np
import csv
from safetensors.torch import save_file

from v4_batches import epoch_crop, p4_splice, p4_starts
from v4_training import Batch, RunState, build_optimizer, load_checkpoint, run_batches
from v4_training_trials import (TrialSpec, _early_stop, _weighted_music_loss,
                                model_fingerprint, reset_training_seed, train_trial)
from train_v4 import _budget_schedule, _effective_deadline
from recheck_v4_matched import run as run_matched_recheck


def test_real_sonics_batch_checkpoint_resume_and_weight_reload(tmp_path: Path) -> None:
    # Given: pinned SONICS and two real full-source rows with opposite music labels.
    from v4_batches import music_supervision_rows
    from v4_data import load_audio
    from v4_model import PreparedWindows, load_model, parameter_groups, predict_files, window_logits
    from v4_training import build_scheduler
    from v4_training_data import EpochBatches

    root = Path(__file__).resolve().parent
    assets = root / "_workspace/recovery_v3/sonics_assets"
    data = root / "v4_validation"
    with (data / "train.csv").open(newline="", encoding="utf-8") as stream:
        rows = music_supervision_rows(list(csv.DictReader(stream)))
    selected = [next(row for row in rows if row["MUSIC_FAKE"] == label) for label in ("0", "1")]
    loaded = load_model("sonics", assets, "cpu")
    groups = parameter_groups(loaded)
    optimizer = build_optimizer(groups.encoder, groups.head, encoder_lr=1e-5, head_lr=3e-5)
    scheduler = build_scheduler(optimizer, steps_per_epoch=1)
    checkpoint = tmp_path / "real-latest.pt"

    def loss_fn(_model: torch.nn.Module, batch: Batch) -> torch.Tensor:
        assert batch.file_indices is not None and batch.window_targets is not None
        prepared = PreparedWindows(batch.inputs, batch.file_indices, batch.starts, batch.window_targets)
        return torch.nn.functional.binary_cross_entropy_with_logits(
            window_logits(loaded, prepared).float(), batch.targets)

    # When: one actual update is saved, loaded into a fresh pinned model, and continued.
    first_batches = EpochBatches(loaded, selected, data, 0, microbatch=2, p3=False, p4=False,
                                 codec_cache=tmp_path, deadline=float("inf"))
    first = run_batches(loaded.model, optimizer, scheduler, first_batches, RunState(), checkpoint,
                        float("inf"), contract="real-sonics", loss_fn=loss_fn)
    restored = load_model("sonics", assets, "cpu")
    restored_groups = parameter_groups(restored)
    restored_optimizer = build_optimizer(restored_groups.encoder, restored_groups.head,
                                         encoder_lr=1e-5, head_lr=3e-5)
    restored_scheduler = build_scheduler(restored_optimizer, steps_per_epoch=1)
    state = load_checkpoint(checkpoint, restored.model, restored_optimizer, restored_scheduler,
                            "real-sonics")
    loaded = restored
    before = {name: value.detach().clone() for name, value in loaded.model.named_parameters()}
    second_batches = EpochBatches(loaded, selected, data, 1, microbatch=2, p3=False, p4=False,
                                  codec_cache=tmp_path, deadline=float("inf"))
    resumed_state = RunState(1, 0, state.optimizer_steps, state.exposures,
                             state.window_exposures, state.exposure_records)
    second = run_batches(loaded.model, restored_optimizer, restored_scheduler, second_batches,
                         resumed_state, checkpoint, float("inf"), contract="real-sonics",
                         loss_fn=loss_fn)
    weights = tmp_path / "best.safetensors"
    save_file({name: value.detach() for name, value in loaded.model.state_dict().items()}, weights)
    reloaded = load_model("sonics", assets, "cpu", state_path=weights)
    audio = load_audio(data / selected[0]["path"])
    loaded.model.eval()
    reloaded.model.eval()

    # Then: optimizer position/exposure continue and the inference weight artifact replays exactly.
    assert first.status == second.status == "complete"
    assert second.state.optimizer_steps == 2
    assert second.state.exposures == {selected[0]["ID"]: 2, selected[1]["ID"]: 2}
    assert any(not torch.equal(before[name], value) for name, value in loaded.model.named_parameters())
    torch.testing.assert_close(predict_files(loaded, [audio], p4=False).probabilities,
                               predict_files(reloaded, [audio], p4=False).probabilities)


def test_epoch_crop_uses_deterministic_strata_across_full_source() -> None:
    # Given: a 45-second ramp whose crop offset is directly observable.
    audio = np.arange(45 * 16_000, dtype=np.float32)

    # When: eight epoch crops are drawn for one five-second native window.
    crops = [epoch_crop(audio, "track-a", epoch, 5 * 16_000) for epoch in range(8)]
    starts = [int(crop[0]) for crop in crops]

    # Then: every temporal stratum is visited once and replay is exact.
    span = len(audio) - 5 * 16_000 + 1
    assert {min(7, start * 8 // span) for start in starts} == set(range(8))
    np.testing.assert_array_equal(crops[3], epoch_crop(audio, "track-a", 3, 5 * 16_000))


def test_short_sources_use_every_distinct_legal_start_without_empty_strata() -> None:
    # Given: only three legal starts for a native window.
    audio = np.arange(80_002, dtype=np.float32)

    # When: crop and P4 schedules cycle more epochs/slots than legal starts.
    crops = [epoch_crop(audio, "short", epoch, 80_000) for epoch in range(8)]
    starts = p4_starts(len(audio), 80_000, "short", 0)

    # Then: both schedules remain valid and P4 keeps each available start unique.
    assert all(len(crop) == 80_000 for crop in crops)
    assert set(starts) == {0, 1, 2}


def test_partial_fake_overlap_sets_bag_truth_and_smooths_both_boundaries() -> None:
    # Given: one real ten-second window and a constant fake donor.
    real = np.zeros(160_000, dtype=np.float32)
    fake = np.ones(160_000, dtype=np.float32)

    # When: a five-to-twenty-second fake edit intersects the selected AST window.
    edited, labels, bag_target = p4_splice(
        real, fake, (0,), 160_000, "partial", 0, fake_donor=True
    )

    # Then: the bag is fake despite a boundary-masked auxiliary label, with smooth entry/exit.
    fractional = edited[(edited > 0) & (edited < 1)]
    midpoint = len(fractional) // 2
    assert math.isnan(labels[0])
    assert bag_target == 1.0
    assert np.all(np.diff(fractional[:midpoint]) >= 0)
    assert np.all(np.diff(fractional[midpoint:]) <= 0)


def test_p4_auxiliary_is_invariant_to_repeated_known_windows() -> None:
    # Given: one bag whose identical known window is represented once or four times.
    from v4_model import PreparedWindows

    targets = torch.tensor([1.0])
    weights = torch.tensor([1.0])
    one = PreparedWindows(torch.zeros(1, 1), torch.tensor([0]), ((0,),), torch.tensor([1.0]))
    four = PreparedWindows(torch.zeros(4, 1), torch.zeros(4, dtype=torch.long),
                           ((0, 1, 2, 3),), torch.ones(4))

    # When: the P4 bag plus known-window objective is evaluated at identical logits.
    loss_one = _weighted_music_loss(torch.full((1,), 0.3), one, targets, weights,
                                    p4=True, p4_control=False, file_count=1)
    loss_four = _weighted_music_loss(torch.full((4,), 0.3), four, targets, weights,
                                     p4=True, p4_control=False, file_count=1)

    # Then: K=4 cannot quadruple the auxiliary mass of the same bag.
    torch.testing.assert_close(loss_one, loss_four, rtol=0, atol=1e-7)


def test_zero_patience_disables_early_stopping_for_fixed_refit() -> None:
    # A fixed refit must reach its locked epoch count even after many flat epochs.
    assert not _early_stop(epoch=30, patience_epoch=1, patience=0)
    assert _early_stop(epoch=7, patience_epoch=1, patience=5)


def test_delayed_setup_rebudgets_stages_without_resetting_allocation() -> None:
    # Given: the actual allocation, first-cell start, and Python runner event timestamps.
    allocation_start = 1_789_283_305.0
    first_cell_start = 1_789_283_350.5691943
    runner_start = first_cell_start + 636.5252071
    original_safe_deadline = allocation_start + 7_080
    budget = {"safe_shutdown_seconds": 7_080,
              "safe_deadline_unix": original_safe_deadline}

    # When: a retry offers a fresh session-relative deadline that would exceed the allocation.
    hard = _effective_deadline(runner_start + 7_080, budget, allocation_start)
    stages = _budget_schedule(runner_start, hard)

    # Then: the old +8 minute profile cutoff is expired, but every new stage is usable and reserved.
    assert allocation_start + 8 * 60 < runner_start
    assert hard == original_safe_deadline
    assert runner_start < stages["profile"] < stages["g0"] < stages["probes"]
    assert stages["probes"] < stages["main"] < stages["temporal"] < stages["refit"]
    assert stages["refit"] < stages["sealed"] < stages["export"] < stages["shutdown"]
    assert stages["shutdown"] == hard


def test_matched_recheck_rejects_changed_real_train_manifest_before_training(tmp_path: Path) -> None:
    # Given: the real audited metadata except for one changed byte in train.csv.
    source = Path(__file__).resolve().parent / "v4_validation"
    data = tmp_path / "data"
    data.mkdir()
    shutil.copy2(source / "dev.csv", data / "dev.csv")
    (data / "train.csv").write_bytes((source / "train.csv").read_bytes() + b"\n")
    now = time.time()
    args = argparse.Namespace(data=data, sonics=tmp_path / "unused",
        g0=Path(__file__).resolve().parent / ".omo/evidence/v4-retry-G0_predictions.raw.csv",
        g0_sha256="bd1c27227829f36e9ae6191db7600920bf6649810f0896a791730b10555f0854",
        output=tmp_path / "output", allocation_start=now, deadline=now + 60)

    # When/Then: the wrapper refuses the drift before loading a model or starting either arm.
    import pytest
    with pytest.raises(ValueError, match="train.csv SHA-256"):
        run_matched_recheck(args)
    assert not (args.output / "remedy_contract.json").exists()


def test_train_trial_real_update_best_binding_and_terminal_resume(
    tmp_path: Path, monkeypatch,
) -> None:
    # Given: pinned SONICS, opposite-label full-source files, and the real 210-row dev contract.
    from v4_batches import music_supervision_rows
    from v4_data import load_manifest
    from v4_model import load_model, parameter_groups

    root = Path(__file__).resolve().parent
    assets, data = root / "_workspace/recovery_v3/sonics_assets", root / "v4_validation"
    train_rows = music_supervision_rows(load_manifest(data, "train"))
    selected = [next(row for row in train_rows if row["MUSIC_FAKE"] == label)
                for label in ("0", "1")]
    dev_rows = load_manifest(data, "dev")
    baseline = [{"ID": row["ID"], "FILE_FAKE_PROB": .25, "VOICE_FAKE_PROB": .5,
                 "MUSIC_FAKE_PROB": .5, "VOICE_PRESENT_PROB": .5,
                 "MUSIC_PRESENT_PROB": .5} for row in dev_rows]
    monkeypatch.setattr("v4_training_trials.predict_rows",
                        lambda loaded, rows, root, baseline, **kwargs: list(baseline))
    loaded = load_model("sonics", assets, "cpu")
    groups = parameter_groups(loaded)
    spec = TrialSpec("real-trial", "sonics", 1e-5, 3e-5, epochs=1,
                     patience=0, contract_hash="integration-v1")

    # When: train_trial performs one real full-model update, evaluates, and is invoked again.
    result = train_trial(loaded, (groups.encoder, groups.head), spec, selected, dev_rows,
                         baseline, data, tmp_path, float("inf"))
    best = Path(str(result["weights"]))
    first_hash = hashlib.sha256(best.read_bytes()).hexdigest()
    replay = load_model("sonics", assets, "cpu")
    replay_groups = parameter_groups(replay)
    resumed = train_trial(replay, (replay_groups.encoder, replay_groups.head), spec, selected,
                          dev_rows, baseline, data, tmp_path, float("inf"))

    # Then: best metadata binds the selected checkpoint and terminal replay runs no extra update.
    metadata = json.loads((tmp_path / "best.json").read_text())
    assert result == resumed
    assert result["status"] == "completed" and result["run_total_updates"] == 1
    assert result["best_epoch"] == metadata["epoch"] == 1
    assert metadata["checkpoint_sha256"] == first_hash
    assert hashlib.sha256(best.read_bytes()).hexdigest() == first_hash
    assert (tmp_path / "terminal_result.json").is_file()


def test_p4_and_control_have_exact_real_window_exposure(tmp_path: Path) -> None:
    # Given: actual long real/fake sources and the native SONICS window schedule.
    from v4_batches import music_supervision_rows
    from v4_data import load_manifest
    from v4_model import load_model
    from v4_training_data import EpochBatches

    root = Path(__file__).resolve().parent
    data = root / "v4_validation"
    rows = music_supervision_rows(load_manifest(data, "train"))
    selected = [next(row for row in rows if row["MUSIC_FAKE"] == label
                     and (data / row["path"]).stat().st_size > 200_000)
                for label in ("0", "1")]
    loaded = load_model("sonics", root / "_workspace/recovery_v3/sonics_assets", "cpu")

    # When: treatment and matched control materialize the same epoch positions.
    treatment = EpochBatches(loaded, selected, data, 0, microbatch=2, p3=False, p4=True,
                             p4_control=False, codec_cache=tmp_path, deadline=float("inf"))
    control = EpochBatches(loaded, selected, data, 0, microbatch=2, p3=False, p4=True,
                           p4_control=True, codec_cache=tmp_path, deadline=float("inf"))
    treatment_windows = [treatment[index].inputs.shape[0] for index in range(len(treatment))]
    control_windows = [control[index].inputs.shape[0] for index in range(len(control))]

    # Then: equal optimizer steps also mean equal actual prepared-window exposure.
    assert len(treatment) == len(control)
    assert treatment_windows == control_windows


def test_train_trial_wires_distinct_p4_treatment_and_parent_control(
    tmp_path: Path, monkeypatch,
) -> None:
    # Given: identical multiwindow bags whose auxiliary labels oppose the bag target.
    from v4_model import LoadedModel

    class OneBag:
        def __init__(self, *args, **kwargs) -> None:
            self.batch = Batch(ids=("music",), inputs=torch.tensor([[-1.0], [0.0], [1.0], [2.0]]),
                targets=torch.tensor([1.0]), sample_weights=torch.tensor([1.0]),
                file_indices=torch.zeros(4, dtype=torch.long), window_targets=torch.zeros(4),
                starts=((0, 1, 2, 3),))

        def __len__(self) -> int:
            return 1

        def __getitem__(self, index: int) -> Batch:
            if index != 0:
                raise IndexError(index)
            return self.batch

    prediction_modes = []
    baseline = [{"ID": "dev", "FILE_FAKE_PROB": .25, "VOICE_FAKE_PROB": .5,
                 "MUSIC_FAKE_PROB": .5, "VOICE_PRESENT_PROB": .5,
                 "MUSIC_PRESENT_PROB": .5}]
    monkeypatch.setattr("v4_training_trials.EpochBatches", OneBag)

    def fake_predict(*args, **kwargs):
        prediction_modes.append(bool(kwargs["p4"]))
        return list(baseline)

    monkeypatch.setattr("v4_training_trials.predict_rows", fake_predict)
    monkeypatch.setattr("v4_training_trials.evaluate_development",
                        lambda *args: {"rank_score": 0.0})
    monkeypatch.setattr("v4_training_trials.build_optimizer",
                        lambda encoder, head, **kwargs: torch.optim.SGD(
                            [parameter for _, parameter in (*encoder, *head)], lr=.1))
    trained = []
    for name, control in (("treatment", False), ("control", True)):
        model = torch.nn.Linear(1, 1, bias=False)
        model.weight.data.fill_(0.2)
        loaded = LoadedModel("sonics", model, None, 16_000, 80_000)
        spec = TrialSpec(name, "sonics", .01, .01, epochs=1, p4=True,
                         p4_control=control, patience=0, contract_hash=name)
        result = train_trial(loaded, ((('weight', model.weight),), ()), spec,
            [{"ID": "music", "path": "unused", "MUSIC_PRESENT": "1", "MUSIC_FAKE": "1"}],
            [{"ID": "dev"}], baseline, tmp_path, tmp_path / name, float("inf"))
        trained.append(model.weight.detach().clone())
        assert result["run_total_updates"] == 1 and result["run_total_window_exposures"] == 4

    # Then: control uses mean parent loss/no auxiliary and parent-style prediction pooling.
    assert prediction_modes == [True, False]
    assert not torch.equal(trained[0], trained[1])


def _model() -> torch.nn.Linear:
    torch.manual_seed(20260913)
    return torch.nn.Linear(2, 1)


def _batches() -> tuple[Batch, ...]:
    return tuple(
        Batch(
            ids=(f"file-{index}",),
            inputs=torch.tensor([[float(index), float(index + 1)]]),
            targets=torch.tensor([float(index % 2)]),
        )
        for index in range(4)
    )


def _optimizer(model: torch.nn.Module) -> torch.optim.AdamW:
    return torch.optim.AdamW(model.parameters(), lr=0.01)


def _scheduler(optimizer: torch.optim.Optimizer) -> torch.optim.lr_scheduler.LambdaLR:
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: 1.0 - 0.1 * step)


def test_optimizer_keeps_every_layer_trainable_with_declared_decay_and_lrs() -> None:
    # Given: an encoder and head with matrix and bias parameters.
    encoder = torch.nn.Linear(2, 2)
    head = torch.nn.Linear(2, 1)

    # When: the fixed SONICS optimizer groups are built.
    optimizer = build_optimizer(
        tuple(encoder.named_parameters(prefix="encoder")),
        tuple(head.named_parameters(prefix="classifier")),
        encoder_lr=1e-5,
        head_lr=3e-5,
    )

    # Then: no learned parameter is omitted/frozen and decay follows tensor role.
    configured = {
        id(parameter): (float(group["lr"]), float(group["weight_decay"]))
        for group in optimizer.param_groups
        for parameter in group["params"]
    }
    expected = {id(parameter) for parameter in (*encoder.parameters(), *head.parameters())}
    assert set(configured) == expected
    assert all(parameter.requires_grad for parameter in (*encoder.parameters(), *head.parameters()))
    assert configured[id(encoder.weight)] == (1e-5, 0.01)
    assert configured[id(encoder.bias)] == (1e-5, 0.0)
    assert configured[id(head.weight)] == (3e-5, 0.01)
    assert configured[id(head.bias)] == (3e-5, 0.0)


def test_real_optimizer_resume_matches_uninterrupted_next_updates(tmp_path: Path) -> None:
    # Given: one deterministic four-update schedule and identical initialized models.
    uninterrupted = _model()
    uninterrupted_optimizer = _optimizer(uninterrupted)
    uninterrupted_scheduler = _scheduler(uninterrupted_optimizer)
    resumed_source = _model()
    resumed_optimizer = _optimizer(resumed_source)
    resumed_scheduler = _scheduler(resumed_optimizer)
    checkpoint = tmp_path / "latest.pt"

    # When: one run is interrupted after two real AdamW updates and resumed from disk.
    complete = run_batches(
        uninterrupted,
        uninterrupted_optimizer,
        uninterrupted_scheduler,
        _batches(),
        RunState(),
        tmp_path / "complete.pt",
        deadline=100.0,
        now=lambda: 0.0,
        max_updates=4,
    )
    interrupted = run_batches(
        resumed_source,
        resumed_optimizer,
        resumed_scheduler,
        _batches(),
        RunState(),
        checkpoint,
        deadline=100.0,
        now=lambda: 0.0,
        max_updates=2,
    )
    restored = _model()
    restored_optimizer = _optimizer(restored)
    restored_scheduler = _scheduler(restored_optimizer)
    restored_state = load_checkpoint(
        checkpoint, restored, restored_optimizer, restored_scheduler, "test-contract"
    )
    resumed = run_batches(
        restored,
        restored_optimizer,
        restored_scheduler,
        _batches(),
        restored_state,
        checkpoint,
        deadline=100.0,
        now=lambda: 0.0,
    )

    # Then: the resumed trajectory, scheduler position and exact file exposure match.
    assert interrupted.status == "interrupted"
    assert resumed.status == complete.status == "complete"
    assert resumed.state.optimizer_steps == complete.state.optimizer_steps == 4
    assert resumed.state.exposures == complete.state.exposures == {
        "file-0": 1,
        "file-1": 1,
        "file-2": 1,
        "file-3": 1,
    }
    for resumed_parameter, complete_parameter in zip(
        restored.parameters(), uninterrupted.parameters(), strict=True
    ):
        torch.testing.assert_close(resumed_parameter, complete_parameter, rtol=0, atol=1e-6)


def test_seed_receipt_reproduces_dropout_across_checkpoint_resume(tmp_path: Path) -> None:
    # Given: a stochastic dropout model and the same recorded seed/initial weights for both arms.
    class DropoutModel(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.input = torch.nn.Linear(2, 4)
            self.dropout = torch.nn.Dropout(.5)
            self.output = torch.nn.Linear(4, 1)

        def forward(self, values: torch.Tensor) -> torch.Tensor:
            return self.output(self.dropout(torch.relu(self.input(values))))

    def initialized() -> DropoutModel:
        reset_training_seed(20_260_913)
        return DropoutModel()

    uninterrupted, interrupted = initialized(), initialized()
    assert model_fingerprint(uninterrupted) == model_fingerprint(interrupted)
    first_rng = reset_training_seed(20_260_913)
    uninterrupted_optimizer = _optimizer(uninterrupted)
    uninterrupted_scheduler = _scheduler(uninterrupted_optimizer)
    complete = run_batches(uninterrupted, uninterrupted_optimizer, uninterrupted_scheduler,
        _batches(), RunState(), tmp_path / "dropout-complete.pt", 100.0, now=lambda: 0.0)

    # When: the matched arm uses the same RNG receipt, stops after two updates, and restores RNG.
    assert reset_training_seed(20_260_913) == first_rng
    interrupted_optimizer = _optimizer(interrupted)
    interrupted_scheduler = _scheduler(interrupted_optimizer)
    partial = run_batches(interrupted, interrupted_optimizer, interrupted_scheduler, _batches(),
        RunState(), tmp_path / "dropout-resume.pt", 100.0, now=lambda: 0.0, max_updates=2)
    resumed = initialized()
    resumed_optimizer = _optimizer(resumed)
    resumed_scheduler = _scheduler(resumed_optimizer)
    state = load_checkpoint(tmp_path / "dropout-resume.pt", resumed, resumed_optimizer,
                            resumed_scheduler, "test-contract")
    restored = run_batches(resumed, resumed_optimizer, resumed_scheduler, _batches(), state,
                           tmp_path / "dropout-resume.pt", 100.0, now=lambda: 0.0)

    # Then: stochastic masks continue on the exact uninterrupted optimization trajectory.
    assert partial.status == "interrupted" and restored.status == complete.status == "complete"
    for resumed_parameter, complete_parameter in zip(
            resumed.parameters(), uninterrupted.parameters(), strict=True):
        torch.testing.assert_close(resumed_parameter, complete_parameter, rtol=0, atol=1e-7)


def test_deadline_saves_partial_state_without_success_claim(tmp_path: Path) -> None:
    # Given: a clock that crosses the hard boundary before the second update.
    model = _model()
    optimizer = _optimizer(model)
    scheduler = _scheduler(optimizer)
    calls = 0

    def clock() -> float:
        nonlocal calls
        calls += 1
        return 0.0 if calls <= 3 else 2.0
    checkpoint = tmp_path / "latest.pt"

    # When: the real training loop reaches its deadline.
    result = run_batches(
        model,
        optimizer,
        scheduler,
        _batches(),
        RunState(),
        checkpoint,
        deadline=1.0,
        now=clock,
    )

    # Then: exactly one update is durable and the epoch is not reported complete.
    assert result.status == "stopped_deadline"
    assert result.state.optimizer_steps == 1
    assert result.state.batch_index == 1
    assert result.state.exposures == {"file-0": 1}
    assert checkpoint.is_file()
