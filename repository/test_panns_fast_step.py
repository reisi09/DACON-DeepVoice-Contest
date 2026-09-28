import gc
import json
import os
from pathlib import Path

import numpy as np
import pytest
import torch

from panns_adaptation_cache import readout_assets
from panns_adaptation_data import TARGETS
from panns_adaptation_run import frozen_digest, model_for_arm, optimization_step, trainable_state


@pytest.mark.parametrize('arm', ['R1', 'R3'])
def test_reduced_gradient_checks_preserve_real_masked_updates(arm: str) -> None:
    root = os.environ.get('DACON_LOCAL_DATA_ROOT')
    if not root:
        pytest.skip('Real checkpoint/cache not configured; not a throughput readiness pass')
    from panns_fast_step import optimization_step as fast_step

    work = Path(root) / '_workspace/panns_cnn_v1'
    source = work / 'notebook_execution_20260925/presence_fix_20260926/saved_probe/cuda'
    cached = json.loads((source / 'index.json').read_text())['entries']
    eligible = sorted((entry for entry in cached if entry['role'] == 'train'),
                      key=lambda entry: entry['segments'])
    chosen = eligible[:3] + [eligible[-1]]
    entries = [dict(entry, path=str(source / Path(entry['path']).name)) for entry in chosen]
    entries[0] = dict(entries[0], targets=[None, entries[0]['targets'][1]])
    package = Path(json.loads((work / 'reference_lock.json').read_text())['package'])
    heads, labels, groups = readout_assets(package)
    config = json.loads((work / 'notebook_execution_20260925/cloud_v11/preflight_work/run.json').read_text())
    rows = [json.loads(line) for line in Path(config['rows']).read_text().splitlines()]
    train = [row for row in rows if row['role'] == 'train']
    fraction = [sum(row[key] is not None for row in train) / len(train) for key in TARGETS]
    torch.set_num_threads(4)
    results = []
    for step_fn, microbatch in ((optimization_step, 4), (fast_step, 4)):
        model = model_for_arm(package, arm, 41, 'cpu')
        optimizer = torch.optim.AdamW(model.optimizer_groups(), weight_decay=1e-4, foreach=False)
        frozen = frozen_digest(model)
        initial = trainable_state(model)
        rng = torch.get_rng_state()
        batch_sizes = []
        hook = model.base.conv_block6.register_forward_pre_hook(
            lambda module, args: batch_sizes.append(len(args[0])))
        for _ in range(2):
            record, gradients = step_fn(model, optimizer, entries, heads, labels, groups,
                                        fraction, 'cpu', microbatch, capture_gradients=True)
        hook.remove()
        updated = trainable_state(model)
        assert frozen_digest(model) == frozen
        assert torch.equal(rng, torch.get_rng_state())
        assert any(not torch.equal(initial[key], value) for key, value in updated.items())
        results.append((record, gradients, updated, batch_sizes))
        if step_fn is fast_step:
            parameter = next(value for value in model.parameters() if value.requires_grad)
            corrupt = parameter.register_hook(lambda gradient: torch.full_like(gradient, float('nan')))
            with pytest.raises(RuntimeError, match='non-finite'):
                fast_step(model, optimizer, entries, heads, labels, groups,
                          fraction, 'cpu', microbatch)
            corrupt.remove()
            assert all(torch.equal(value, updated[key])
                       for key, value in trainable_state(model).items())
        del model, optimizer, initial
        gc.collect()
    reference, actual = results
    assert actual[0]['IDs'] == reference[0]['IDs']
    assert abs(actual[0]['loss'] - reference[0]['loss']) <= 1e-6
    np.testing.assert_allclose(actual[0]['probabilities'], reference[0]['probabilities'],
                               rtol=0, atol=1e-6)
    for key, gradient in reference[1].items():
        denominator = max(float(gradient.double().norm()), 1e-12)
        assert float((actual[1][key] - gradient).double().norm()) / denominator <= 1e-3, key
        torch.testing.assert_close(actual[2][key], reference[2][key], rtol=0, atol=1e-6)
    assert actual[3] == reference[3]
    print(json.dumps(dict(arm=arm, probability_max_abs=float(np.abs(
        np.asarray(actual[0]['probabilities']) - reference[0]['probabilities']).max()),
        gradient_relative_l2_max=max(float((actual[1][key] - value).double().norm()) /
            max(float(value.double().norm()), 1e-12) for key, value in reference[1].items()),
        update_max_abs=max(float((actual[2][key] - value).abs().max())
            for key, value in reference[2].items()), forward_batches=[reference[3], actual[3]],
        scope='CPU numerical parity only; no GPU throughput or model quality claim')))
