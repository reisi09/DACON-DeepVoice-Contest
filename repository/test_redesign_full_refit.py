"""Protect full-data optimization, honest reports, and early-stop compatibility."""

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

import redesign_fit
from redesign_checkpoint import Records, Rows, load_checkpoint
from redesign_fit import _forward, _pad, fit_music
from redesign_models import MusicResidual, make_control


def _samples() -> tuple[Rows, Records]:
    rng = np.random.default_rng(71)
    rows: Rows = []
    records: Records = {}
    for index in range(6):
        identifier = f"source-{index}"
        rows.append({"ID": identifier, "group_id": identifier,
                     "MUSIC_PRESENT": "0" if index == 4 else "1",
                     "MUSIC_FAKE": "" if index == 5 else str(index % 2),
                     "event_intervals": json.dumps([
                         {"component": "music", "start": 0, "end": 9, "label": index % 2}
                     ])})
        records[identifier] = {
            "stem": rng.normal(size=(2, 1024)).astype(np.float32),
            "raw": rng.normal(size=(2, 1024)).astype(np.float32),
            "aux": np.ones((2, 6), dtype=np.float32),
            "starts": np.array([0, 64600]), "valid": np.ones(2, dtype=bool),
            "stem_audible": np.ones(2, dtype=bool),
        }
    return rows, records


def test_empty_validation_requires_explicit_fixed_epochs(tmp_path: Path) -> None:
    # Given: full training data with no held-out rows.
    rows, records = _samples()
    # When/Then: the legacy API must not silently disable model selection.
    with pytest.raises(ValueError, match="early-stop"):
        fit_music(rows, records, [], tmp_path, arm="B", seed=41, epochs=2,
                  patience=1, batch_size=3, device="cpu", deadline=None)


def test_fixed_epochs_optimizes_all_inputs_and_reloads_final_checkpoint(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Given: known and unknown music rows and a seeded untrained B head.
    rows, records = _samples()
    batch = _pad(rows, records, "cpu")
    initial = MusicResidual(make_control(41)).eval()
    with torch.inference_mode():
        initial_logits = _forward(initial, batch).bag_logits
    final_parameters: list[torch.Tensor] = []
    optimizer_step = torch.optim.AdamW.step

    def observe_step(optimizer: torch.optim.AdamW) -> None:
        optimizer_step(optimizer)
        final_parameters[:] = [parameter.detach().clone()
                               for group in optimizer.param_groups for parameter in group["params"]]

    monkeypatch.setattr(torch.optim.AdamW, "step", observe_step)
    # When: two fixed epochs train without a validation set.
    result = fit_music(rows, records, [], tmp_path, arm="B", seed=41, epochs=2,
                       patience=1, batch_size=3, device="cpu", deadline=None,
                       fixed_epochs=True)
    # Then: the final epoch is selected, no validation score is fabricated, and reload is exact.
    report = result.report
    assert report["mode"] == "fixed_epochs"
    assert report["fixed_epochs"] is True
    assert report["train_ids"] == sorted(row["ID"] for row in rows)
    assert report["earlystop_ids"] == []
    assert report["selected_epoch"] == report["epochs_completed"] == 2
    assert report["optimizer_updates"] == 4
    assert report["validation_loss"] == []
    assert "best_validation_loss" not in report
    loaded, loaded_report = load_checkpoint(result.checkpoint,
                                             expected_identity=str(report["identity"]))
    with torch.inference_mode():
        actual = _forward(result.model, batch)
        reloaded = _forward(loaded, batch)
    assert not torch.allclose(actual.bag_logits, initial_logits, atol=1e-7, rtol=0)
    assert torch.equal(actual.bag_logits, reloaded.bag_logits)
    assert torch.equal(actual.window_logits, reloaded.window_logits)
    assert loaded_report == report
    assert result.checkpoint.name == "best.pt"
    for actual_parameter, final_parameter in zip(result.model.parameters(), final_parameters, strict=True):
        assert torch.equal(actual_parameter, final_parameter)


def test_ineligible_labels_cannot_change_full_refit_weights(tmp_path: Path) -> None:
    # Given: identical inputs with contradictory labels only on ineligible rows.
    rows, records = _samples()
    changed = [dict(row) for row in rows]
    changed[4]["MUSIC_FAKE"] = "1"
    changed[5]["event_intervals"] = json.dumps([
        {"component": "music", "start": 0, "end": 9, "label": 0}
    ])
    # When: both datasets undergo identical seeded optimization.
    results = [fit_music(candidate, records, [], tmp_path / str(index), arm="B",
                         seed=41, epochs=2, patience=1, batch_size=3, device="cpu",
                         deadline=None, fixed_epochs=True)
               for index, candidate in enumerate((rows, changed))]
    # Then: neither bag nor event labels excluded by the mask can alter gradients or weights.
    for name, tensor in results[0].model.state_dict().items():
        assert torch.equal(tensor, results[1].model.state_dict()[name]), name
    assert results[0].report["identity"] != results[1].report["identity"]


def test_fixed_epochs_rejects_validation_rows(tmp_path: Path) -> None:
    # Given: a caller requests fixed epochs while still passing a validation split.
    rows, records = _samples()
    # When/Then: ambiguous selection intent is rejected before training.
    with pytest.raises(ValueError, match="empty validation"):
        fit_music(rows[:-1], records, rows[-1:], tmp_path, arm="B", seed=41, epochs=2,
                  patience=1, batch_size=3, device="cpu", deadline=None, fixed_epochs=True)


def test_deadline_during_fixed_epoch_never_publishes_complete_checkpoint(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Given: the deadline expires after the first of two minibatches.
    rows, records = _samples()
    clock = iter((0.0, 0.0, 2.0))
    monkeypatch.setattr(redesign_fit, "time", SimpleNamespace(time=lambda: next(clock)))
    # When: a partially completed fixed epoch is interrupted.
    with pytest.raises(TimeoutError, match="inside epoch"):
        fit_music(rows, records, [], tmp_path, arm="B", seed=41, epochs=2,
                  patience=1, batch_size=3, device="cpu", deadline=1.0, fixed_epochs=True)
    # Then: the partial state cannot masquerade as a completed full-data model.
    assert not (tmp_path / "best.pt").exists()
    _, report = load_checkpoint(tmp_path / "last.pt")
    assert report["status"] == "stopped_deadline_restart_from_seed"
    assert report["optimizer_updates"] == 1
    assert report["epochs_completed"] == 0
    assert report["validation_loss"] == []
    assert report["exact_resume"] == "false"


def test_early_stop_still_restores_selected_epoch(tmp_path: Path) -> None:
    # Given: held-out labels oppose the training labels for the same feature pattern.
    rows, records = _samples()
    rows = rows[:4]
    validation = [dict(row, ID=f"early-{row['ID']}", group_id=f"early-{row['ID']}",
                       MUSIC_FAKE=str(1 - int(row["MUSIC_FAKE"])), event_intervals="[]")
                  for row in rows]
    records.update({row["ID"]: records[original["ID"]]
                    for row, original in zip(validation, rows, strict=True)})
    # When: validation-based selection runs with patience one.
    selected = fit_music(rows, records, validation, tmp_path / "early", arm="B", seed=41,
                         epochs=8, patience=1, batch_size=4, device="cpu", deadline=None)
    # Then: selected weights equal a fresh fixed refit ending at the selected epoch.
    assert selected.report["mode"] == "early_stopping"
    assert selected.report["epochs_completed"] < 8
    history = selected.report["validation_loss"]
    selected_epoch = int(selected.report["selected_epoch"])
    assert selected_epoch == int(np.argmin(history)) + 1
    reference = fit_music(rows, records, [], tmp_path / "reference", arm="B", seed=41,
                          epochs=selected_epoch, patience=1, batch_size=4, device="cpu",
                          deadline=None, fixed_epochs=True)
    for name, tensor in selected.model.state_dict().items():
        assert torch.equal(tensor, reference.model.state_dict()[name]), name
