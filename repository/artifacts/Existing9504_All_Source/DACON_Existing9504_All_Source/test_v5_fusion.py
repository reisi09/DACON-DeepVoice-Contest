"""Behavioral checks for V5 temporal fusion and partial-label likelihood."""

import math
import time
from pathlib import Path

import numpy as np
import torch

from v5_features import EncodedFile
from v5_fusion import TemporalFusion, joint_partial_label_loss, joint_probabilities
from v5_fusion_training import train_fusion


def test_joint_probabilities_are_exact_state_marginals() -> None:
    # Given: an explicit distribution over voice x music absent/real/fake states.
    states = torch.tensor(
        [[0.02, 0.03, 0.05], [0.10, 0.15, 0.05], [0.20, 0.10, 0.30]]
    )

    # When: the nine logits are converted to the five submission probabilities.
    actual = joint_probabilities(states.log().reshape(1, 9))

    # Then: fake heads are conditional on presence and file fake is the fake union.
    expected = torch.tensor([[0.70, 2 / 3, 0.40 / 0.68, 0.90, 0.68]])
    torch.testing.assert_close(actual, expected, atol=1e-6, rtol=0.0)


def test_partial_labels_marginalize_unknown_states_and_backpropagate() -> None:
    # Given: one music-fake label with unknown voice truth and one real-file label.
    logits = torch.tensor(
        [
            [0.2, -0.1, 0.8, -0.4, 0.3, 0.5, 0.7, -0.2, 1.0],
            [-0.3, 0.4, 0.9, 0.1, -0.2, 0.8, 0.5, 0.6, -0.7],
        ],
        requires_grad=True,
    )
    labels = torch.tensor(
        [
            [math.nan, math.nan, 1.0, math.nan, 1.0],
            [0.0, math.nan, math.nan, math.nan, math.nan],
        ]
    )

    # When: likelihood sums every state compatible with each known label.
    loss = joint_partial_label_loss(logits, labels)
    expected = torch.stack(
        [
            torch.logsumexp(logits[0], 0) - torch.logsumexp(logits[0, [2, 5, 8]], 0),
            torch.logsumexp(logits[1], 0) - torch.logsumexp(logits[1, [0, 1, 3, 4]], 0),
        ]
    ).mean()
    loss.backward()

    # Then: unknown voice fake is unconstrained, FILE_FAKE=0 excludes both fake states,
    # and the marginal objective remains connected to every joint logit.
    torch.testing.assert_close(loss, expected)
    assert logits.grad is not None
    assert torch.isfinite(logits.grad).all()
    assert torch.count_nonzero(logits.grad) == logits.numel()


def test_padding_cannot_change_a_file_prediction_in_eval_mode() -> None:
    # Given: one file and the same file extended by arbitrary padded windows.
    torch.manual_seed(236749)
    model = TemporalFusion(input_dim=6, joint=True).eval()
    features = torch.randn(1, 3, 6)
    times = torch.tensor([[0.0, 5.0, 10.0]])
    base_mask = torch.zeros(1, 3, dtype=torch.bool)
    extended_features = torch.cat((features, torch.randn(1, 2, 6) * 100), dim=1)
    extended_times = torch.cat((times, torch.tensor([[1_000.0, -1_000.0]])), dim=1)
    extended_mask = torch.tensor([[False, False, False, True, True]])

    # When: both forms pass through the real two-layer temporal model.
    with torch.inference_mode():
        base = model(features, base_mask, times).probabilities
        extended = model(extended_features, extended_mask, extended_times).probabilities

    # Then: masked content contributes neither attention nor learned pooling mass.
    torch.testing.assert_close(base, extended, atol=1e-6, rtol=1e-6)


def test_batch_members_never_fuse_across_files() -> None:
    # Given: a target file and an unrelated second file in the same batch.
    torch.manual_seed(41)
    model = TemporalFusion(input_dim=4, joint=False).eval()
    target = torch.randn(1, 4, 4)
    other = torch.randn(1, 4, 4) * 20
    mask = torch.zeros(1, 4, dtype=torch.bool)
    times = torch.tensor([[0.0, 2.0, 4.0, 6.0]])

    # When: the target is inferred alone and alongside the unrelated file.
    with torch.inference_mode():
        alone = model(target, mask, times).probabilities
        batched = model(
            torch.cat((target, other)), mask.expand(2, -1), times.expand(2, -1)
        ).probabilities[0:1]

    # Then: transformer attention and pooling stay inside each file.
    torch.testing.assert_close(alone, batched, atol=1e-6, rtol=1e-6)


def test_nonjoint_file_fake_keeps_original_max_identity() -> None:
    # Given: a non-joint baseline whose four logits define conditional fake and presence.
    torch.manual_seed(73)
    model = TemporalFusion(input_dim=2, joint=False).eval()
    features = torch.randn(2, 3, 2)
    mask = torch.zeros(2, 3, dtype=torch.bool)
    times = torch.tensor([[0.0, 1.0, 2.0], [0.0, 4.0, 9.0]])

    # When: file and component predictions are produced together.
    with torch.inference_mode():
        result = model(features, mask, times)
    voice_fake, music_fake, voice_present, music_present = result.logits.sigmoid().unbind(1)

    # Then: FILE_FAKE retains the identifiable legacy max formula exactly.
    expected = torch.maximum(voice_present * voice_fake, music_present * music_fake)
    torch.testing.assert_close(result.probabilities[:, 0], expected)


def test_window_order_and_timestamps_can_change_learned_fusion() -> None:
    # Given: distinct ordered evidence and a fixed initialized temporal model.
    torch.manual_seed(7)
    model = TemporalFusion(input_dim=3, joint=True).eval()
    features = torch.tensor([[[2.0, 0.0, -1.0], [0.0, 3.0, 1.0], [-2.0, 1.0, 4.0]]])
    mask = torch.zeros(1, 3, dtype=torch.bool)
    times = torch.tensor([[0.0, 1.5, 9.0]])

    # When: feature order and timestamp placement are independently perturbed.
    with torch.inference_mode():
        base = model(features, mask, times).probabilities
        reordered = model(features.flip(1), mask, times).probabilities
        retimed = model(features, mask, times.flip(1)).probabilities

    # Then: learned temporal fusion is not a symmetric window average.
    assert not torch.allclose(base, reordered, atol=1e-7, rtol=1e-7)
    assert not torch.allclose(base, retimed, atol=1e-7, rtol=1e-7)


def _encoded(identifier: str, windows: int, label: tuple[float, ...]) -> EncodedFile:
    rng = np.random.default_rng(sum(map(ord, identifier)))
    return EncodedFile(
        identifier,
        rng.normal(size=(windows, 4)).astype(np.float32),
        np.arange(windows, dtype=np.float32) * 4.0375,
        label,
    )


def test_training_consumes_partial_examples_and_packages_best_state(tmp_path: Path) -> None:
    # Given: mixed-length files whose labels include unknown component truth.
    nan = math.nan
    train = [
        _encoded("speech-real", 2, (0.0, 0.0, nan, 1.0, 0.0)),
        _encoded("speech-fake", 3, (1.0, 1.0, nan, 1.0, 0.0)),
        _encoded("music-real", 1, (0.0, nan, 0.0, nan, 1.0)),
        _encoded("music-fake", 4, (1.0, nan, 1.0, nan, 1.0)),
    ]
    dev = [train[0], train[3]]

    # When: one complete joint epoch trains from the cached ordered features.
    model, probabilities, metadata = train_fusion(
        train, dev, True, tmp_path, time.time() + 30, epochs=1, seed=9
    )
    packaged = torch.load(tmp_path / "best.pt", map_location="cpu", weights_only=True)

    # Then: every partial example contributes once and the returned model is the saved best.
    assert metadata["status"] == "complete"
    assert metadata["epochs_completed"] == 1
    assert metadata["exposures"] == {item.identifier: 1 for item in train}
    assert metadata["all_model_gradients_connected"] is True
    assert probabilities.shape == (len(dev), 5)
    assert np.isfinite(probabilities).all()
    for name, value in model.state_dict().items():
        torch.testing.assert_close(value.cpu(), packaged[name])


def test_expired_budget_returns_an_explicit_zero_epoch_result(tmp_path: Path) -> None:
    # Given: valid cached features but a stage deadline that has already elapsed.
    nan = math.nan
    train = [_encoded("partial", 2, (nan, nan, 1.0, nan, 1.0))]
    dev = [_encoded("dev", 1, (0.0, nan, 0.0, 0.0, 1.0))]

    # When: training is asked to preserve enough budget for final evaluation.
    _, probabilities, metadata = train_fusion(
        train, dev, False, tmp_path, time.time() - 1, epochs=3, seed=11
    )

    # Then: it reports actual work rather than claiming requested epochs ran.
    assert metadata["status"] == "censored_deadline"
    assert metadata["epochs_completed"] == 0
    assert metadata["updates"] == 0
    assert metadata["exposures"] == {"partial": 0}
    assert probabilities.shape == (1, 5)
