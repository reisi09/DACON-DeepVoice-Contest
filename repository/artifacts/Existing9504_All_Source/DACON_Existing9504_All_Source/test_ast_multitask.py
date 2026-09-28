from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch
from transformers import ASTConfig, ASTFeatureExtractor, ASTModel

import ast_multitask as ast


def tiny_model(mode: str) -> torch.nn.Module:
    torch.manual_seed(236749)
    config = ASTConfig(
        hidden_size=32,
        num_hidden_layers=2,
        num_attention_heads=4,
        intermediate_size=64,
        max_length=32,
        num_mel_bins=128,
    )
    return ast.ASTMultiTask(ASTModel(config), mode)


@pytest.mark.parametrize(
    ("mode", "ast_trainable"),
    [("frozen", False), ("last_block", True)],
)
def test_gradient_and_update_reach_only_the_selected_parameters(
    mode: str,
    ast_trainable: bool,
) -> None:
    # Given: a real, small AST and one eligible label for every head.
    model = tiny_model(mode)
    inputs = torch.randn(2, 32, 128)
    targets = torch.tensor([[0, 0, 1, 1, 1], [1, 1, 0, 1, 1]], dtype=torch.float32)
    before = {name: value.detach().clone() for name, value in model.named_parameters()}
    optimizer = torch.optim.SGD((p for p in model.parameters() if p.requires_grad), lr=0.05)

    # When: one real optimization step runs through the selected graph.
    loss = ast.masked_multitask_loss(model(inputs), targets, ast.compute_pos_weights(targets))
    loss.backward()
    trainable = {name for name, value in model.named_parameters() if value.requires_grad}
    expected_ast = {
        name
        for name, _ in model.named_parameters()
        if name.startswith("backbone.encoder.layer.1.") or name.startswith("backbone.layernorm.")
    }
    assert trainable == ({name for name in before if name.startswith("heads.")} | (expected_ast if ast_trainable else set()))
    for name, value in model.named_parameters():
        assert (value.grad is not None) == (name in trainable)
        if value.grad is not None:
            assert torch.isfinite(value.grad).all()
            assert value.grad.abs().sum() > 0
    optimizer.step()

    # Then: frozen weights remain byte-identical and each selected group updates.
    changed = {name for name, value in model.named_parameters() if not torch.equal(before[name], value)}
    assert changed
    assert changed <= trainable
    assert any(name.startswith("heads.") for name in changed)
    assert any(name.startswith("backbone.") for name in changed) is ast_trainable


def test_masked_loss_ignores_unknown_and_absent_component_truth() -> None:
    # Given: one row with absent components and one row with known present components.
    logits = torch.tensor(
        [[0.7, -1.1, 1.4, -0.2, -0.4], [-0.8, 0.9, -0.7, 0.6, 0.5]],
        requires_grad=True,
    )
    targets = torch.tensor([[0, 0, 1, 0, 0], [1, 1, 0, 1, 1]], dtype=torch.float32)
    weights = ast.compute_pos_weights(targets)
    baseline = ast.masked_multitask_loss(logits, targets, weights)

    # When: only unknown/absent truth changes, then one eligible truth changes.
    ignored = targets.clone()
    ignored[0, 1:3] = 1 - ignored[0, 1:3]
    ignored[0, 0] = torch.nan
    original_unknown = targets.clone()
    original_unknown[0, 0] = torch.nan
    eligible = targets.clone()
    eligible[1, 1] = 0

    # Then: excluded values are bit-identical, while eligible truth changes the loss.
    assert torch.equal(
        ast.masked_multitask_loss(logits, ignored, weights),
        ast.masked_multitask_loss(logits, original_unknown, weights),
    )
    assert not torch.equal(baseline, ast.masked_multitask_loss(logits, eligible, weights))
    baseline.backward()
    assert logits.grad is not None
    assert logits.grad[0, 1] == 0
    assert logits.grad[0, 2] == 0
    assert torch.count_nonzero(logits.grad[1]) == 5


def test_windows_cover_the_whole_file_once_and_aggregate_by_mean() -> None:
    # Given: a file slightly longer than six ten-second windows.
    audio = np.arange(60 * 16_000 + 317, dtype=np.float32)

    # When: the deterministic file-local splitter and aggregator run.
    windows = ast.split_audio(audio)
    probabilities = ast.aggregate_window_logits(
        torch.tensor([[0.0] * 5, [2.0] * 5, [-2.0] * 5]),
    )

    # Then: no sample is skipped/duplicated and all windows contribute equally.
    assert len(windows) == 7
    assert len(ast.split_audio(audio[:60 * 16_000])) == 6
    assert np.array_equal(np.concatenate(windows), audio)
    assert all(len(window) == 160_000 for window in windows[:-1])
    assert len(windows[-1]) == 317
    assert torch.allclose(probabilities, torch.sigmoid(torch.tensor([0.0, 2.0, -2.0])).mean().repeat(5))


def test_export_reload_replays_and_rejects_contract_drift(tmp_path: Path) -> None:
    # Given: a trained tiny AST and the exact official processor contract.
    model = tiny_model("last_block").eval()
    processor = ASTFeatureExtractor()
    inputs = torch.randn(2, 32, 128)
    expected = model(inputs).detach()
    ast.export_checkpoint(model, processor, tmp_path)

    # When: the self-contained directory is loaded without a hub checkpoint.
    restored, restored_processor = ast.load_checkpoint(tmp_path, "last_block")

    # Then: weights replay exactly and processor/head metadata fail closed on drift.
    assert torch.equal(expected, restored.eval()(inputs).detach())
    assert restored_processor.sampling_rate == 16_000
    processor_path = tmp_path / "preprocessor_config.json"
    metadata_path = tmp_path / "multitask_config.json"
    original_processor = processor_path.read_text(encoding="utf-8")
    payload = json.loads(original_processor)
    payload["sampling_rate"] = 8_000
    processor_path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="AST processor contract mismatch"):
        ast.load_checkpoint(tmp_path, "last_block")
    processor_path.write_text(original_processor, encoding="utf-8")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["columns"] = list(reversed(metadata["columns"]))
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
    with pytest.raises(ValueError, match="AST checkpoint contract mismatch"):
        ast.load_checkpoint(tmp_path, "last_block")


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason='MPS-specific partial LayerNorm backward')
def test_mps_last_block_gradient_matches_cpu_with_frozen_prefix():
    # Protect against MPS returning zero/garbage LayerNorm bias gradients for a frozen input.
    from copy import deepcopy
    cpu = tiny_model('last_block').train()
    mps = deepcopy(cpu).to('mps').train()
    for _ in range(2):
        inputs = torch.randn(2, 32, 128)
        for model, device in ((cpu, 'cpu'), (mps, 'mps')):
            model.zero_grad(set_to_none=True)
            model(inputs.to(device)).square().mean().backward()
        for (name, left), (_, right) in zip(cpu.named_parameters(), mps.named_parameters(), strict=True):
            if left.requires_grad:
                torch.testing.assert_close(right.grad.cpu(), left.grad, atol=2e-4, rtol=2e-3, msg=name)
            else:
                assert right.grad is None
