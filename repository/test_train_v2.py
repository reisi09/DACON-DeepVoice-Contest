"""High-signal boundaries for the v2 training state."""

from pathlib import Path
import time

import numpy as np
import pytest
import soundfile as sf
import torch
from transformers import ASTConfig, ASTFeatureExtractor, ASTModel

import train_v2
from ast_multitask import ASTMultiTask
from v2_inference import predict_audio_file
from train_v2 import (FrozenEmbeddingCache, load_training_checkpoint,
                      _trainable_state, read_development_manifest,
                      save_training_checkpoint, seed_everything)


class TinyHeads(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.heads = torch.nn.ModuleList([torch.nn.Linear(3, 1)])

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.heads[0](values)


def _step(model: TinyHeads, optimizer: torch.optim.Optimizer, values: torch.Tensor,
          targets: torch.Tensor) -> None:
    optimizer.zero_grad(set_to_none=True)
    torch.nn.functional.binary_cross_entropy_with_logits(model(values), targets).backward()
    optimizer.step()


def test_rejects_locked_test_before_audio_access(tmp_path: Path) -> None:
    locked = tmp_path / "locked_test"
    locked.mkdir()
    manifest = locked / "manifest.csv"
    manifest.write_text("ID,path,role\nx,missing.wav,test_source\n", encoding="utf-8")
    with pytest.raises(ValueError, match="training accepts development manifest only"):
        read_development_manifest(manifest, tmp_path)


def test_resume_parity_and_wrong_config_rejection(tmp_path: Path) -> None:
    seed_everything(7)
    values = torch.tensor([[1.0, -1.0, .5], [-.5, .25, 1.0]])
    targets = torch.tensor([[1.0], [0.0]])
    initial = TinyHeads().state_dict()

    uninterrupted = TinyHeads()
    uninterrupted.load_state_dict(initial)
    optimizer = torch.optim.AdamW(uninterrupted.parameters(), lr=.01)
    _step(uninterrupted, optimizer, values, targets)
    checkpoint = tmp_path / "last.pt"
    save_training_checkpoint(checkpoint, uninterrupted, optimizer, 1, "same", [{"epoch": 1}])
    _step(uninterrupted, optimizer, values, targets)

    resumed = TinyHeads()
    resumed.load_state_dict(initial)
    resumed_optimizer = torch.optim.AdamW(resumed.parameters(), lr=.01)
    epoch, reports = load_training_checkpoint(checkpoint, resumed, resumed_optimizer, "same")
    assert (epoch, reports) == (1, [{"epoch": 1}])
    _step(resumed, resumed_optimizer, values, targets)
    for expected, actual in zip(uninterrupted.parameters(), resumed.parameters(), strict=True):
        torch.testing.assert_close(actual, expected, rtol=0, atol=1e-7)

    with pytest.raises(ValueError, match="configuration mismatch"):
        load_training_checkpoint(checkpoint, TinyHeads(),
                                 torch.optim.AdamW(TinyHeads().parameters()), "changed")


def test_best_epoch_snapshot_survives_later_optimizer_steps() -> None:
    model = TinyHeads()
    optimizer = torch.optim.SGD(model.parameters(), lr=.1)
    values = torch.tensor([[1.0, -1.0, .5]])
    targets = torch.tensor([[1.0]])
    best = _trainable_state(model)
    saved = {name: value.clone() for name, value in best.items()}
    _step(model, optimizer, values, targets)
    assert all(torch.equal(best[name], saved[name]) for name in best)
    assert any(not torch.equal(best[name], model.state_dict()[name]) for name in best)


def test_epoch_extension_preserves_adam_step_and_rejects_learning_rate_drift(tmp_path: Path) -> None:
    """Increasing epochs must not reset Adam or silently change the experiment."""
    seed_everything(7)
    model = TinyHeads()
    optimizer = torch.optim.AdamW(model.parameters(), lr=.01)
    values, targets = torch.tensor([[1., -1., .5]]), torch.tensor([[1.]])
    _step(model, optimizer, values, targets)
    old = {"epochs": 1, "lr": .01, "manifest": "same"}
    new = {**old, "epochs": 3, "patience": 2}
    source, destination = tmp_path / "old.pt", tmp_path / "extended.pt"
    save_training_checkpoint(source, model, optimizer, 1, train_v2._json_hash(old), [])
    train_v2.extend_checkpoint(source, destination, old, new)
    _step(model, optimizer, values, targets)
    resumed = TinyHeads()
    resumed_optimizer = torch.optim.AdamW(resumed.parameters(), lr=.01)
    load_training_checkpoint(destination, resumed, resumed_optimizer, train_v2._json_hash(new))
    _step(resumed, resumed_optimizer, values, targets)
    for actual, expected in zip(resumed.parameters(), model.parameters(), strict=True):
        torch.testing.assert_close(actual, expected, rtol=0, atol=1e-7)
    with pytest.raises(ValueError, match="only epochs and patience"):
        train_v2.extend_checkpoint(source, tmp_path / "wrong.pt", old, {**new, "lr": .02})


def test_non_16k_training_prediction_matches_deployment_resampling(tmp_path: Path) -> None:
    path = tmp_path / "clip.wav"
    sf.write(path, np.sin(np.arange(22050, dtype=np.float32) * .013), 44100)
    config = ASTConfig(hidden_size=16, num_hidden_layers=1, num_attention_heads=2,
                       intermediate_size=32, max_length=1024, num_mel_bins=128)
    model = ASTMultiTask(ASTModel(config), "frozen").eval()
    processor = ASTFeatureExtractor()
    training = train_v2.predict_rows(
        model, processor, [{"ID": "clip", "path": "clip.wav"}], tmp_path,
        torch.device("cpu"))[0]
    deployed = predict_audio_file(model, processor, path)
    torch.testing.assert_close(
        torch.tensor([training[column] for column in train_v2.PREDICTION_COLUMNS]),
        torch.tensor(deployed), rtol=0, atol=1e-5)


def test_frozen_cache_key_changes_with_audio_model_or_processor(tmp_path: Path) -> None:
    audio = tmp_path / "audio.wav"
    audio.write_bytes(b"same audio bytes")
    base = FrozenEmbeddingCache(tmp_path / "cache", "model-a", "processor-a")
    base.put(audio, "clip", torch.tensor([1.0, 2.0]))
    torch.testing.assert_close(base.get(audio, "clip"), torch.tensor([1.0, 2.0]))
    assert FrozenEmbeddingCache(tmp_path / "cache", "model-b", "processor-a").get(audio, "clip") is None
    assert FrozenEmbeddingCache(tmp_path / "cache", "model-a", "processor-b").get(audio, "clip") is None
    audio.write_bytes(b"changed audio bytes")
    assert base.get(audio, "clip") is None


@pytest.mark.parametrize("epochs,patience,completed", [(1, 0, 1), (8, 2, 4)])
def test_tiny_ast_training_records_real_gradient_and_weight_update(tmp_path: Path,
                                                                  monkeypatch, epochs, patience, completed) -> None:
    rows = []
    for index in range(4):
        identifier = f"clip{index}"
        sf.write(tmp_path / f"{identifier}.wav",
                 np.sin(np.arange(8000, dtype=np.float32) * (.01 + index * .001)), 16000)
        label = str(index % 2)
        rows.append({"ID": identifier, "path": f"{identifier}.wav",
                     "split": "dev_train" if index < 2 else "dev_select",
                     **{column: label for column in train_v2.LABEL_COLUMNS}})
    config = ASTConfig(hidden_size=16, num_hidden_layers=1, num_attention_heads=2,
                       intermediate_size=32, max_length=1024, num_mel_bins=128)
    model = ASTMultiTask(ASTModel(config), "frozen")
    monkeypatch.setitem(train_v2.CONFIGS, "AST_FROZEN",
                        {**train_v2.CONFIGS["AST_FROZEN"], "epochs": epochs,
                         "batch_size": 2, "patience": patience})
    if patience:
        scores = iter([.5, .6, .59, .58, .57, .56, .55, .54])
        monkeypatch.setattr(train_v2, "_proxy", lambda report: next(scores))
    _, report = train_v2.train_candidate(
        "AST_FROZEN", model, ASTFeatureExtractor(), rows, tmp_path, tmp_path / "run",
        time.time() + train_v2.RESERVE_SECONDS + 120, "manifest", "tiny-model")
    assert report["trainable_absolute_update"] > 0
    assert report["reports"][0]["max_gradient_norm"] > 0
    assert report["reports"][0]["metric_report"]["counts"]["file_eer"] == 2
    assert report["epochs_completed"] == completed
    if patience:
        assert report["best_epoch"] == 2
        saved = torch.load(tmp_path / "run/AST_FROZEN/best.pt", weights_only=True)
        for name, value in saved.items():
            torch.testing.assert_close(model.state_dict()[name], value, rtol=0, atol=0)
