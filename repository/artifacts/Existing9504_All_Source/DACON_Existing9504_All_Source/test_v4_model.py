from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest
import torch

import v4_model as v4

ROOT = Path(__file__).resolve().parent
SONICS_ASSETS = ROOT / "_workspace" / "recovery_v3" / "sonics_assets"
AST_ASSETS = ROOT / "_workspace" / "v2_models" / "ast"


def _audio_pair(samples: int) -> tuple[np.ndarray, np.ndarray]:
    generator = np.random.default_rng(236749)
    quiet = generator.normal(0.0, 1e-4, samples).astype(np.float32)
    loud = generator.normal(0.0, 0.2, samples).clip(-1.0, 1.0).astype(np.float32)
    return quiet, loud


@pytest.mark.parametrize(
    ("kind", "assets", "samples"),
    [("sonics", SONICS_ASSETS, 80_000), ("ast", AST_ASSETS, 160_000)],
)
def test_real_weight_batch_matches_serial_for_quiet_and_loud_audio(
    kind: v4.ModelKind,
    assets: Path,
    samples: int,
) -> None:
    # Given: the pinned real model and audio whose levels expose batch-wide dB clipping.
    torch.set_num_threads(2)
    torch.manual_seed(236749)
    loaded = v4.load_model(kind, assets, "cpu")
    loaded.model.eval()
    quiet, loud = _audio_pair(samples)

    # When: the files run together and separately through the actual feature/model path.
    with torch.inference_mode():
        together = v4.window_logits(loaded, v4.prepare_windows(loaded, [quiet, loud]))
        serial = torch.cat(
            [
                v4.window_logits(loaded, v4.prepare_windows(loaded, [audio]))
                for audio in (quiet, loud)
            ]
        )
        unrelated = v4.window_logits(loaded, v4.prepare_windows(loaded, [loud, quiet]))

    # Then: another file's amplitude or batch position cannot alter either prediction.
    torch.testing.assert_close(together, serial, atol=1e-5, rtol=0.0)
    torch.testing.assert_close(unrelated.flip(0), serial, atol=1e-5, rtol=0.0)
    if kind == "sonics":
        safe_db = loaded.model.ft_extractor.amplitude_to_db
        loaded.model.ft_extractor.amplitude_to_db = safe_db.transform
        coupled = v4.window_logits(loaded, v4.prepare_windows(loaded, [quiet, loud]))
        assert torch.max(torch.abs(coupled - serial)) > 1e-5


@pytest.mark.parametrize(
    ("kind", "assets", "samples", "groups", "expected_parameters"),
    [
        (
            "sonics",
            SONICS_ASSETS,
            80_000,
            (
                "encoder.transformer.blocks.0.",
                "encoder.transformer.blocks.6.",
                "encoder.transformer.blocks.11.",
                "classifier.",
            ),
            17_172_493,
        ),
        (
            "ast",
            AST_ASSETS,
            160_000,
            (
                "backbone.encoder.layer.0.",
                "backbone.encoder.layer.6.",
                "backbone.encoder.layer.11.",
                "head.",
            ),
            86_188_033,
        ),
    ],
)
def test_real_weight_step_updates_early_middle_late_and_classifier(
    kind: v4.ModelKind,
    assets: Path,
    samples: int,
    groups: tuple[str, ...],
    expected_parameters: int,
) -> None:
    # Given: every pinned encoder parameter and the one music head are optimizer-owned.
    torch.set_num_threads(2)
    torch.manual_seed(236749)
    loaded = v4.load_model(kind, assets, "cpu")
    loaded.model.train()
    if kind == "sonics":
        probe = torch.randn(2, 128, 128)
        probe_targets = torch.tensor([0, 1])
        augmented, augmented_targets = loaded.model.augment(probe, probe_targets)
        assert torch.equal(augmented, probe)
        assert torch.equal(augmented_targets, probe_targets)
    quiet, loud = _audio_pair(samples)
    prepared = v4.prepare_windows(loaded, [quiet, loud])
    parameters = dict(loaded.model.named_parameters())
    split = v4.parameter_groups(loaded)
    trainable = [parameter for _, parameter in split.encoder + split.head]
    optimizer = torch.optim.SGD(trainable, lr=0.01)
    optimizer_ids = {
        id(parameter)
        for group in optimizer.param_groups
        for parameter in group["params"]
    }
    representatives = {
        prefix: next(
            name
            for name in parameters
            if name.startswith(prefix) and parameters[name].ndim > 1
        )
        for prefix in groups
    }
    before = {
        name: parameters[name].detach().clone() for name in representatives.values()
    }

    # When: one genuine full-fine-tuning update runs through real weights.
    logits = v4.window_logits(loaded, prepared)
    loss = v4.music_loss(logits, prepared, torch.tensor([0.0, 1.0]), p4=False)
    loss.backward()

    # Then: a detached encoder, partial freeze, or optimizer omission fails explicitly.
    assert sum(parameter.numel() for parameter in trainable) == expected_parameters
    assert {name for name, _ in split.encoder + split.head} == set(parameters)
    assert optimizer_ids == {id(parameter) for parameter in trainable}
    assert all(parameter.requires_grad for parameter in parameters.values())
    for name in representatives.values():
        gradient = parameters[name].grad
        assert gradient is not None, name
        assert torch.isfinite(gradient).all(), name
        assert torch.count_nonzero(gradient) > 0, name
    optimizer.step()
    for name, original in before.items():
        assert not torch.equal(parameters[name], original), name


def test_temporal_pooling_and_masked_segment_loss_follow_p4_contract() -> None:
    # Given: two bags, including a rare late fake and partially unknown segment truth.
    logits = torch.tensor([-2.0, -2.0, -2.0, 3.0, -1.0, -1.0], requires_grad=True)
    indices = torch.tensor([0, 0, 0, 0, 1, 1])
    prepared = v4.PreparedWindows(
        values=torch.empty(6, 1),
        file_indices=indices,
        starts=((0, 1, 2, 3), (0, 1)),
        window_targets=torch.tensor([0.0, 0.0, math.nan, 1.0, math.nan, math.nan]),
    )

    # When: P4 pooling and its bag-plus-known-segment objective are evaluated.
    pooled = v4.pool_bag_logits(logits, indices, 2)
    duplicated = v4.pool_bag_logits(
        logits.repeat_interleave(2), indices.repeat_interleave(2), 2
    )
    targets = torch.tensor([1.0, 0.0])
    loss = v4.music_loss(logits, prepared, targets, p4=True)
    loss.backward()
    bag_logits = logits.detach().requires_grad_()
    torch.nn.functional.binary_cross_entropy_with_logits(
        v4.pool_bag_logits(bag_logits, indices, 2),
        targets,
    ).backward()

    # Then: normalized pooling ignores duplicate multiplicity and unknown segments add no gradient.
    torch.testing.assert_close(pooled, duplicated, atol=1e-6, rtol=0.0)
    assert (
        pooled[0]
        > v4.pool_bag_logits(
            torch.full((4,), -2.0), torch.zeros(4, dtype=torch.long), 1
        )[0]
    )
    assert logits.grad is not None
    assert bag_logits.grad is not None
    torch.testing.assert_close(logits.grad[2], bag_logits.grad[2])
    torch.testing.assert_close(logits.grad[4:], bag_logits.grad[4:])
    assert not torch.equal(logits.grad[3], bag_logits.grad[3])


def test_explicit_p4_starts_select_distinct_late_file_content() -> None:
    # Given: a 30-second file whose samples identify their exact source position.
    loaded = v4.LoadedModel("sonics", torch.nn.Identity(), None, 16_000, 80_000)
    audio = np.arange(480_000, dtype=np.float32)
    starts = ((0, 160_000, 320_000, 400_000),)

    # When: the runner supplies its four stratified P4 starts and segment labels.
    prepared = v4.prepare_windows(
        loaded,
        [audio],
        starts=starts,
        window_targets=((0.0, 0.0, 1.0, 1.0),),
    )

    # Then: each native window comes from that exact stratum, including the late tail.
    assert prepared.starts == starts
    assert prepared.values[:, 0].tolist() == [0.0, 160_000.0, 320_000.0, 400_000.0]
    assert prepared.values[:, -1].tolist() == [
        79_999.0,
        239_999.0,
        399_999.0,
        479_999.0,
    ]


def test_real_ast_inference_scores_a_sub_kaldi_window_tail() -> None:
    # Given: one valid file with a one-sample AST remainder below Kaldi's 400-sample floor.
    torch.set_num_threads(2)
    torch.manual_seed(236749)
    loaded = v4.load_model("ast", AST_ASSETS, "cpu")
    audio = np.zeros(160_001, dtype=np.float32)

    # When: full-file inference follows the native nonoverlapping AST grid.
    prediction = v4.predict_files(loaded, [audio], p4=False)

    # Then: the actual tail start remains recorded and produces a finite file probability.
    assert prediction.starts == ((0, 160_000),)
    assert torch.isfinite(prediction.probabilities).all()
