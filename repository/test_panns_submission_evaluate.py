import numpy as np
import pytest
import torch

from panns_cached_fake import compare_gate, head_scores
from panns_submission_evaluate import choose_model


def test_frozen_voice_pooling_and_silence_match_preserved_runtime():
    voice = [torch.nn.Linear(2, 2, bias=False), torch.nn.Linear(2, 2, bias=False)]
    with torch.no_grad():
        voice[0].weight.copy_(torch.tensor([[2., 0.], [0., 0.]]))
        voice[1].weight.copy_(torch.tensor([[0., 1.], [0., 0.]]))
    features = np.array([[1., 0.], [0., 1.]], np.float32)
    data = dict(voice_embeddings=features, voice_silent=np.array(False), mert_windows=features)
    class Music(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.mean, self.scale = torch.zeros(2), torch.ones(2)
        def head(self, values):
            return values[:, :1], None
    result = head_scores(([Music()], voice, 0), data, 'cpu')
    expected = (torch.sigmoid(torch.tensor(2.)) + torch.sigmoid(torch.tensor(1.))).item()/2
    assert result['VOICE_FAKE_PROB'] == pytest.approx(expected, abs=1e-7)
    data['voice_silent'] = np.array(True)
    assert head_scores(([Music()], voice, 0), data, 'cpu')['VOICE_FAKE_PROB'] == 0.
    fresh = {'a': dict(ID='a', **result)}
    assert compare_gate(fresh, fresh)['passed']
    corrupt = {'a': dict(fresh['a'], MUSIC_FAKE_PROB=result['MUSIC_FAKE_PROB']+.001)}
    assert not compare_gate(fresh, corrupt)['passed']


def test_selection_ads_improvement_cannot_bypass_cps_or_use_development():
    reports = {name: dict(ads=ads, cps=cps) for name, ads, cps in
               [('R0', .8, .99), ('R1', .9, .98), ('R3', .82, .995)]}
    assert choose_model(reports) == 'R3'
    reports['R3']['ads'] = .79
    assert choose_model(reports) == 'R0'
    reports['R1']['cps'] = None
    assert choose_model(reports) == 'R0'
