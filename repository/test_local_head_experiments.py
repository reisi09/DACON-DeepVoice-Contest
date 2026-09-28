"""Real-cache checks for baseline preservation, trainable corrections and local replay.

These protect the proposed experiment, not a leaderboard-quality claim. Set
DACON_LOCAL_DATA_ROOT to the existing project containing its preserved archive.
"""
import copy
import os
from pathlib import Path

import numpy as np
import pytest
import torch


@pytest.fixture(scope='module')
def real_inputs(tmp_path_factory):
    root = os.environ.get('DACON_LOCAL_DATA_ROOT')
    if not root:
        pytest.skip('Real preserved archive/cache required; no synthetic quality substitute')
    from local_head_experiments import load_reference, read_records, padded_batch
    root = Path(root)
    run = root / '_workspace/lb_a100_run_20260924_v1/final_readback/unpacked'
    heads = load_reference(root / 'submissions/DataV4_DurationBalanced/submit.zip',
                           tmp_path_factory.mktemp('real_reference'))
    records, _ = read_records(run / 'runs/partial_training/available_rows.csv', run / 'cache')
    music = [r for r in records if r['row']['role'] == 'train'
             and r['row']['MUSIC_PRESENT'] == '1']
    voice = [r for r in records if r['row']['role'] == 'train' and not r['silent']]
    selected = []
    for label in ('0', '1'):
        selected += sorted([r for r in music if r['row']['MUSIC_FAKE'] == label],
                           key=lambda r: len(r['music']))[:3]
    selected += [max(music, key=lambda r: len(r['music']))]
    voice = [next(r for r in voice if r['row']['VOICE_FAKE'] == label) for label in ('0', '1')]
    return heads, selected, voice, padded_batch


def test_real_baseline_parity_padding_and_actual_optimizer_connectivity(real_inputs):
    from local_head_experiments import MusicResidual, VoiceResidual
    heads, music, voice, batch = real_inputs
    for route, records in [('music', music), ('voice', voice)]:
        x, valid, starts = batch(records, route, 'cpu')
        labels = torch.tensor([float(r['row'][route.upper() + '_FAKE']) for r in records])
        variants = [MusicResidual(copy.deepcopy(heads[0][0]), kernel=k) for k in (1, 3)] if route == 'music' else [VoiceResidual(copy.deepcopy(heads[1][0]), heads[2])]
        for model in variants:
            model.eval()
            if route == 'music':
                expected = model.base(x, valid).file_logits
            else:
                odds = model.base(x)
                expected = (odds[..., heads[2]] - odds[..., 1-heads[2]]).masked_fill(~valid, -torch.inf).max(1).values
            torch.testing.assert_close(model(x, valid, starts), expected, atol=2e-6, rtol=1e-6)
            before = {k: v.clone() for k, v in model.base.state_dict().items()}
            parameters = [p for p in model.parameters() if p.requires_grad]
            original = [p.detach().clone() for p in parameters]
            optimizer = torch.optim.AdamW(parameters, lr=1e-3)
            for _ in range(3):
                optimizer.zero_grad(set_to_none=True)
                loss = torch.nn.functional.binary_cross_entropy_with_logits(model(x, valid, starts), labels)
                loss.backward()
                assert torch.isfinite(loss)
                optimizer.step()
            assert all(p.grad is not None and torch.isfinite(p.grad).all() and p.grad.abs().sum() > 0 for p in parameters)
            assert all(not torch.equal(p.detach(), old) for p, old in zip(parameters, original))
            assert all(p.grad is None for p in model.base.parameters())
            assert all(torch.equal(v, model.base.state_dict()[k]) for k, v in before.items())
            model.eval()
            assert (model(x, valid, starts).detach() - expected.detach()).abs().max() > 1e-5
            first = batch(records[:1], route, 'cpu')
            torch.testing.assert_close(model(*first)[0], model(x, valid, starts)[0], atol=2e-5, rtol=2e-5)
            polluted = x.clone(); polluted[~valid] = float('nan')
            torch.testing.assert_close(model(polluted, valid, starts), model(x, valid, starts), atol=0, rtol=0)


def test_real_mps_cpu_forward_update_and_checkpoint_replay(real_inputs, tmp_path):
    from local_head_experiments import MusicResidual, VoiceResidual
    if not torch.backends.mps.is_available():
        pytest.skip('MPS unavailable')
    heads, music, voice, batch = real_inputs
    for route, records in [('music', music), ('voice', voice)]:
        torch.manual_seed(41)
        cpu = MusicResidual(copy.deepcopy(heads[0][0]), kernel=3) if route == 'music' else VoiceResidual(copy.deepcopy(heads[1][0]), heads[2])
        mps = copy.deepcopy(cpu).to('mps')
        models = [cpu, mps]
        for model, device in zip(models, ('cpu', 'mps')):
            optimizer = torch.optim.SGD([p for p in model.parameters() if p.requires_grad], lr=1e-3)
            for _ in range(3):
                optimizer.zero_grad(set_to_none=True)
                logits = model(*batch(records, route, device))
                labels = torch.tensor([float(r['row'][route.upper()+'_FAKE']) for r in records], device=device)
                torch.nn.functional.binary_cross_entropy_with_logits(logits, labels).backward()
                optimizer.step()
            model.eval()
        expected = cpu(*batch(records, route, 'cpu')).sigmoid().detach()
        actual = mps(*batch(records, route, 'mps')).sigmoid().detach().cpu()
        torch.testing.assert_close(actual, expected, atol=2e-4, rtol=2e-4)
        path = tmp_path / (route + '.pt')
        torch.save(mps.cpu().state_dict(), path)
        cpu.load_state_dict(torch.load(path, weights_only=True))
        torch.testing.assert_close(cpu(*batch(records, route, 'cpu')).sigmoid().detach(), actual, atol=2e-4, rtol=2e-4)
