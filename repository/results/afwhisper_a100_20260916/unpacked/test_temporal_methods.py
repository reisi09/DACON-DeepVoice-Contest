"""Check padding isolation, classifier/trunk gradients, and saved inference for new heads."""
from pathlib import Path

import pytest
import torch
from music_temporal import TemporalMusic, load_temporal_head


@pytest.mark.parametrize('head', ['prototype', 'attention', 'conv_mean', 'conv_lme'])
def test_alternative_heads_learn_and_reload_without_padding_leakage(tmp_path: Path, head: str):
    torch.manual_seed(41)
    model = TemporalMusic(0., simplified=True, compact=True, input_dim=1024,
                          softmax_output=True, mlp_output=True, alternative_output=head).eval()
    x = torch.randn(4, 7, 1024)
    mask = torch.zeros(4, 7, dtype=torch.bool)
    mask[:, -2:] = True
    expected = model(x, mask)
    padded = torch.cat([x, torch.full((4, 3, 1024), float('nan'))], 1)
    extended = torch.cat([mask, torch.ones(4, 3, dtype=torch.bool)], 1)
    torch.testing.assert_close(model(padded, extended), expected, atol=1e-6, rtol=1e-6)
    torch.nn.functional.cross_entropy(expected, torch.tensor([0, 1, 0, 1])).backward()
    for name, parameter in model.named_parameters():
        assert parameter.grad is not None, name
        assert torch.isfinite(parameter.grad).all() and parameter.grad.abs().sum() > 0, name
    empty = model(x, torch.ones_like(mask)).softmax(-1)[:, 1]
    assert torch.isfinite(empty).all() and empty.max() < 1e-6
    path = tmp_path / 'temporal_music.pt'
    torch.save(dict(architecture=model.architecture, dropout=0., state_dict=model.state_dict()), path)
    torch.testing.assert_close(load_temporal_head(path)(x, mask), expected, atol=0, rtol=0)
