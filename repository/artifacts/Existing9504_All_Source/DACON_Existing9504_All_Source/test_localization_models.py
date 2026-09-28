"""Numerical, masking and optimization checks for the localization pilot."""

from __future__ import annotations

import pytest
import torch
from transformers import ASTConfig, ASTModel

import localization_models as models


def test_mean_preserves_raw_feature_pooling_and_ignores_padding() -> None:
    # Given: the existing MERT12 head, fitted scaler, and corrupted padding.
    torch.manual_seed(41)
    mean, scale = torch.arange(4).float(), torch.tensor([1., 2., 3., 4.])
    model = models.FrozenMeanMLP(mean, scale)
    valid = torch.tensor([[True, True, False]])
    features = torch.tensor([[[2., 4., 8., 16.], [3., 6., 9., 12.],
                              [torch.nan, torch.inf, -torch.inf, 999.]]], requires_grad=True)
    expected, _ = model.head((features[:, :2].mean(1) - mean) / scale)

    # When: the file prediction and backward pass traverse the actual mean head.
    output = model(features, valid)
    output.file_logits.sum().backward()

    # Then: raw-mean baseline is reproduced and padding cannot affect gradients.
    torch.testing.assert_close(output.file_logits, expected)
    assert features.grad is not None
    assert torch.isfinite(features.grad).all()
    assert torch.count_nonzero(features.grad[:, 2]) == 0
    assert features.grad[:, :2].abs().sum() > 0


def test_mil_uses_matched_scaler_and_is_invariant_to_duplicate_bags() -> None:
    # Given: a matched MLP and one valid bag with arbitrary padded values.
    torch.manual_seed(42)
    mean, scale = torch.randn(4), torch.tensor([1., 2., 3., 4.])
    model = models.WindowMIL(mean, scale)
    features = torch.randn(2, 3, 4)
    valid = torch.tensor([[True, False, True], [True, True, True]])
    features[~valid] = torch.nan
    expected_windows, _ = model.head((features.masked_fill(~valid[..., None], 0) - mean) / scale)
    expected = torch.stack([torch.logsumexp(row[mask], 0) - mask.sum().log()
                            for row, mask in zip(expected_windows, valid, strict=True)])

    # When: MIL sees the bag, a duplicate of every window, and extra padding.
    result = model(features, valid)
    duplicated = model(features.repeat(1, 2, 1), valid.repeat(1, 2))
    padded = model(torch.cat((features, torch.full((2, 2, 4), torch.nan)), 1),
                   torch.cat((valid, torch.zeros(2, 2, dtype=torch.bool)), 1))

    # Then: the fixed normalized log-sum-exp has no window-count or padding bias.
    torch.testing.assert_close(result.file_logits, expected)
    torch.testing.assert_close(result.file_logits, duplicated.file_logits)
    torch.testing.assert_close(result.file_logits, padded.file_logits)


def test_patch_targets_preserve_fake_fraction_and_exclude_unknown_truth() -> None:
    # Given: known real/fake frame truth with a partial final patch and unknown row.
    targets = torch.zeros(2, 32)
    targets[:, 10:18] = 1
    targets[:, 22:] = -1
    targets[1, 20] = torch.nan
    valid = torch.arange(32)[None] < torch.tensor([[22], [22]])

    # When: frame truth is mapped using the actual 16-frame, stride-10 kernel.
    patches = models.frame_targets_to_patches(targets, valid, patch_size=16, time_stride=10)

    # Then: mixed patch targets use known fake occupancy, never invented real truth.
    torch.testing.assert_close(patches[0], torch.tensor([6 / 16, 8 / 12]))
    torch.testing.assert_close(patches[1, 0], torch.tensor(6 / 16))
    assert torch.isnan(patches[1, 1])


def test_known_loss_blocks_unknown_and_padded_patch_gradients() -> None:
    # Given: observed soft targets, unknown truth, and a padded patch.
    logits = torch.tensor([[0.4, -0.8, 1.1, -0.3]], requires_grad=True)
    targets = torch.tensor([[0.25, torch.nan, -1., 1.]])
    valid = torch.tensor([[True, True, True, False]])

    # When: known-interval BCE runs backward.
    loss = models.masked_localization_loss(logits, targets, valid)
    loss.backward()

    # Then: only observed, valid interval truth contributes any optimization signal.
    expected = torch.nn.functional.binary_cross_entropy_with_logits(logits[0, :1], targets[0, :1])
    torch.testing.assert_close(loss, expected)
    assert logits.grad is not None
    assert logits.grad[0, 0].abs() > 0
    assert torch.count_nonzero(logits.grad[0, 1:]) == 0


def test_ast_time_mapping_matches_native_patch_convolution_order() -> None:
    # Given: a native AST projection encoding the top-left value of every patch.
    config = ASTConfig(hidden_size=8, num_hidden_layers=2, num_attention_heads=2,
                       intermediate_size=16, max_length=12, num_mel_bins=8,
                       patch_size=4, frequency_stride=4, time_stride=4)
    model = models.ASTLocalizer(ASTModel(config))
    projection = model.backbone.embeddings.patch_embeddings.projection
    with torch.no_grad():
        projection.weight.zero_()
        projection.bias.zero_()
        projection.weight[:, 0, 0, 0] = 1
    values = torch.arange(12)[:, None] + 100 * torch.arange(8)[None]
    patches = model.backbone.embeddings.patch_embeddings(values.float()[None])
    sequence = torch.cat((torch.full((1, 2, 8), -999.), patches), 1)

    # When: the localization path removes both special tokens and pools frequency.
    temporal = model.temporal_embeddings(sequence)

    # Then: time is the inner Conv2d dimension, not an accidental frequency sequence.
    torch.testing.assert_close(temporal[0, :, 0], torch.tensor([200., 204., 208.]))


def test_ast_localization_gradients_update_only_last_block_norm_and_head() -> None:
    # Given: a real offline AST with a frozen prefix and known local supervision.
    torch.manual_seed(43)
    config = ASTConfig(hidden_size=16, num_hidden_layers=2, num_attention_heads=2,
                       intermediate_size=32, max_length=32, num_mel_bins=16,
                       patch_size=4, frequency_stride=4, time_stride=4)
    model = models.ASTLocalizer(ASTModel(config)).train()
    inputs = torch.randn(2, 32, 16)
    valid = torch.arange(32)[None] < torch.tensor([[20], [32]])
    targets = torch.zeros(2, 32)
    targets[0, 8:16] = 1
    targets[1, 16:24] = 1
    before = {name: value.detach().clone() for name, value in model.named_parameters()}
    optimizer = torch.optim.SGD((p for p in model.parameters() if p.requires_grad), lr=.03)

    # When: file MIL and known patch BCE train the real suffix from a frozen cache.
    prefix = model.encode_prefix(inputs)
    assert not prefix.requires_grad
    output = model.forward_encoded(prefix, valid)
    assert output.patch_logits is not None and output.patch_mask is not None
    truth = models.frame_targets_to_patches(targets, valid, patch_size=4, time_stride=4)
    loss = torch.nn.functional.binary_cross_entropy_with_logits(output.file_logits, torch.ones(2))
    loss = loss + models.masked_localization_loss(output.patch_logits, truth, output.patch_mask)
    loss.backward()
    optimizer.step()

    # Then: all selected groups receive finite gradients; frozen weights stay exact.
    allowed = ("backbone.encoder.layer.1.", "backbone.layernorm.", "head.")
    for name, parameter in model.named_parameters():
        if name.startswith(allowed):
            assert parameter.requires_grad
            assert parameter.grad is not None and torch.isfinite(parameter.grad).all()
        else:
            assert not parameter.requires_grad and parameter.grad is None
            assert torch.equal(before[name], parameter)
    for prefix in allowed:
        assert any(name.startswith(prefix) and not torch.equal(before[name], value)
                   for name, value in model.named_parameters())
    assert not model.backbone.encoder.layer[0].training
    assert model.backbone.encoder.layer[-1].training
    assert not output.patch_mask[0, 5:].any()


def test_cached_prefix_replays_native_ast_and_direct_inference_exactly() -> None:
    # Given: a native tiny AST, two different real lengths, and unchanged cache dtype.
    torch.manual_seed(44)
    config = ASTConfig(hidden_size=16, num_hidden_layers=3, num_attention_heads=2,
                       intermediate_size=32, max_length=32, num_mel_bins=16,
                       patch_size=4, frequency_stride=4, time_stride=4)
    model = models.ASTLocalizer(ASTModel(config)).eval()
    inputs = torch.randn(2, 32, 16)
    valid = torch.arange(32)[None] < torch.tensor([[18], [32]])

    # When: native whole-backbone, direct localizer, and cached suffix paths execute.
    with torch.no_grad():
        native = model.backbone(inputs).last_hidden_state
        native_logits = model.head(model.temporal_embeddings(native)).squeeze(-1)
        prefix = model.encode_prefix(inputs)
        direct = model(inputs, valid)
        cached = model.forward_encoded(prefix, valid)

    # Then: splitting at the last block changes neither native logits nor predictions.
    assert direct.patch_mask is not None
    assert torch.equal(cached.file_logits, direct.file_logits)
    assert torch.equal(cached.patch_logits, direct.patch_logits)
    assert torch.equal(direct.patch_logits, native_logits.masked_fill(~direct.patch_mask, 0))


def test_stale_prefix_is_rejected_before_running_the_trainable_suffix() -> None:
    # Given: a cache from a different AST width but aligned window/frame counts.
    config = ASTConfig(hidden_size=16, num_hidden_layers=2, num_attention_heads=2,
                       intermediate_size=32, max_length=32, num_mel_bins=16,
                       patch_size=4, frequency_stride=4, time_stride=4)
    model = models.ASTLocalizer(ASTModel(config)).eval()
    stale = torch.zeros(2, 34, 24)
    valid = torch.ones(2, 32, dtype=torch.bool)

    # When/Then: replay fails at the cache boundary instead of inside an AST kernel.
    with pytest.raises(models.LocalizationContractError):
        model.forward_encoded(stale, valid)
