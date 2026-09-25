import os
from pathlib import Path
import signal
import subprocess
import sys
import time

import pytest


def test_budget_covers_whole_passes_with_equal_steps_and_reserve():
    from panns_training_budget import budget_steps

    seconds = {'R1': 1., 'R2': 2., 'R3': 3., 'R5': 4.}
    steps = budget_steps(seconds, 5, 1.)
    assert steps == 96
    assert steps % 2 == 0
    assert steps * 3 * sum(seconds.values()) <= 3600 * .8
    assert budget_steps({arm: .001 for arm in seconds}, 9, 72.) == 99999
    with pytest.raises(ValueError, match='whole pass'):
        budget_steps(seconds, 5, .001)
    for invalid in (float('nan'), float('inf'), 0., -1.):
        with pytest.raises(ValueError):
            budget_steps(seconds, 5, invalid)
        with pytest.raises(ValueError):
            budget_steps({**seconds, 'R3': invalid}, 5, 1.)


def test_budget_refuses_runtime_drift_and_altered_fixed_steps(tmp_path):
    from panns_adaptation_data import identity, sha256, write_json
    from panns_training_budget import load_budget, profile_binding

    config = {'identity': 'locked-run', 'output': str(tmp_path), 'checkpoint_every': 100}
    (tmp_path / 'cache').mkdir()
    (tmp_path / 'cache/index.json').write_text('{"real": "index bytes"}')
    binding = profile_binding(config, 'cpu')
    profile = {'binding': binding, 'train_files': 5,
               'checkpoint_seconds': dict.fromkeys(('R1', 'R2', 'R3', 'R5'), .01),
               'checkpoint_hash_seconds': dict.fromkeys(('R1', 'R2', 'R3', 'R5'), .01),
               'seconds_per_step': {'R1': 1., 'R2': 2., 'R3': 3., 'R5': 4.}}
    write_json(tmp_path / 'throughput_profile.json', profile)
    budget = {'binding': binding, 'profile_sha256': sha256(tmp_path / 'throughput_profile.json'),
              'training_hours': 1., 'training_steps': 94, 'steps_per_pass': 2}
    budget['identity'] = identity(budget)
    write_json(tmp_path / 'training_budget.json', budget)
    assert load_budget(config, 'cpu')[0]['training_steps'] == 94
    with pytest.raises(ValueError, match='device'):
        load_budget(config, 'cuda')
    (tmp_path / 'cache/index.json').write_text('{"changed": "index bytes"}')
    with pytest.raises(ValueError, match='binding'):
        load_budget(config, 'cpu')
    (tmp_path / 'cache/index.json').write_text('{"real": "index bytes"}')
    budget['training_steps'] = 98
    budget['identity'] = identity({k: v for k, v in budget.items() if k != 'identity'})
    write_json(tmp_path / 'training_budget.json', budget)
    with pytest.raises(ValueError, match='measured'):
        load_budget(config, 'cpu')


def test_checkpoint_io_is_included_without_spending_the_timing_reserve():
    from panns_training_budget import budget_steps

    # Given: expensive checkpoint writes every 100 updates, plus one final hash.
    seconds = dict(R1=.2, R2=.2, R3=.2, R5=.2)
    saves = dict(R1=15., R2=15., R3=15., R5=15.)
    hashes = dict(R1=2., R2=2., R3=2., R5=2.)
    # When: the budget is frozen for 1,000 files and all 12 conditions.
    steps = budget_steps(seconds, 1000, 1., checkpoint_seconds=saves,
                         checkpoint_hash_seconds=hashes, checkpoint_every=100)
    # Then: exactly two full passes fit; a third would exceed the reserved limit.
    assert steps == 500
    assert 12 * (steps*.2 + 5*15 + 2) <= 3600*.8
    assert 12 * (750*.2 + 8*15 + 2) > 3600*.8


def test_offline_budget_read_preserves_identity_without_authorizing_cuda(tmp_path, monkeypatch):
    import torch
    from panns_adaptation_data import identity, sha256, write_json
    from panns_training_budget import load_budget, profile_binding

    config = dict(identity='archived-run', output=str(tmp_path), checkpoint_every=100)
    write_json(tmp_path/'cache/index.json', dict(archived=True))
    binding = profile_binding(config, 'cpu') | dict(device='cuda', hardware=dict(name='archived A100'),
                                                  torch='archived-torch', cuda='archived-cuda')
    profile = dict(binding=binding, train_files=5,
                   seconds_per_step=dict(R1=1., R2=2., R3=3., R5=4.),
                   checkpoint_seconds=dict.fromkeys(('R1', 'R2', 'R3', 'R5'), .01),
                   checkpoint_hash_seconds=dict.fromkeys(('R1', 'R2', 'R3', 'R5'), .01))
    write_json(tmp_path/'throughput_profile.json', profile)
    budget = dict(binding=binding, profile_sha256=sha256(tmp_path/'throughput_profile.json'),
                  training_hours=1., training_steps=94, steps_per_pass=2)
    budget['identity'] = identity(budget)
    write_json(tmp_path/'training_budget.json', budget)
    def unavailable(*args):
        raise RuntimeError('No CUDA in artifact reader')
    monkeypatch.setattr(torch.cuda, 'current_device', unavailable)
    assert load_budget(config, 'cuda', validate_runtime=False)[0]['training_steps'] == 94
    with pytest.raises(RuntimeError, match='No CUDA'):
        load_budget(config, 'cuda')
    with pytest.raises(ValueError, match='device'):
        load_budget(config, 'cpu', validate_runtime=False)
    write_json(tmp_path/'cache/index.json', dict(corrupted=True))
    with pytest.raises(ValueError, match='binding'):
        load_budget(config, 'cuda', validate_runtime=False)


def training_loop_config(tmp_path, monkeypatch, checkpoint_every=1):
    import torch
    import panns_cloud_run as runner
    from panns_adaptation_data import write_json

    class SmallSuffix(torch.nn.Linear):
        def optimizer_groups(self):
            return [dict(params=list(self.parameters()), lr=.001)]
    def update(model, optimizer, *args):
        optimizer.zero_grad(set_to_none=True)
        loss = model(torch.ones(1, 2)).square().mean()
        loss.backward()
        norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), 1.))
        optimizer.step()
        return dict(loss=float(loss.detach()), gradient_norms=dict(suffix=norm)), None
    config = dict(identity='loop-regression-only', output=str(tmp_path), arms=['R3'], seeds=[41],
                  microbatch=4, checkpoint_every=checkpoint_every)
    write_json(tmp_path/'cache/index.json', dict(loop_fixture=True))
    monkeypatch.setattr(runner, 'validate_config', lambda c: tmp_path)
    monkeypatch.setattr(runner, 'load_budget', lambda *args, **kwargs: (dict(training_steps=2), 'budget'))
    monkeypatch.setattr(runner, 'read_cache_index', lambda c: dict(entries=[dict(role='train', targets=[i%2, i%2]) for i in range(5)]))
    monkeypatch.setattr(runner, 'model_for_arm', lambda *args: SmallSuffix(2, 1))
    monkeypatch.setattr(runner, 'readout_assets', lambda p: (None, None, None))
    monkeypatch.setattr(runner, 'optimization_step', update)
    return config


@pytest.mark.parametrize('checkpoint_every,stop_after,expected_saves', [(1, None, 2), (1, 1, 1), (3, None, 1), (3, 1, 1)])
def test_training_persists_each_completed_step_at_most_once(tmp_path, monkeypatch, checkpoint_every, stop_after, expected_saves):
    import torch
    import panns_cloud_run as runner

    config = training_loop_config(tmp_path, monkeypatch, checkpoint_every)
    saved_steps, original = [], torch.save
    def record_save(record, path):
        saved_steps.append(record['step'])
        original(record, path)
    monkeypatch.setattr(torch, 'save', record_save)
    runner.train(config, 'R3', 41, 'cpu', stop_after=stop_after)
    saved = torch.load(tmp_path/'train_R3_41/checkpoint.pt', weights_only=True)
    assert len(saved_steps) == expected_saves
    assert saved['step'] == (stop_after or 2)
    assert saved['optimizer']['state']


@pytest.mark.parametrize('state', ['owned', 'no_receipt', 'foreign_receipt', 'progress', 'unexpected', 'symlink'])
def test_first_partial_checkpoint_requires_matching_receipt_and_no_committed_progress(tmp_path, monkeypatch, state):
    import io
    import torch
    import panns_cloud_run as runner
    from panns_adaptation_data import write_json

    config = training_loop_config(tmp_path, monkeypatch)
    folder = tmp_path/'train_R3_41'
    folder.mkdir()
    buffer = io.BytesIO()
    torch.save(dict(model=torch.ones(2)), buffer)
    partial = folder/'checkpoint.pt.tmp'
    partial.write_bytes(buffer.getvalue()[:100])
    if state != 'no_receipt':
        write_json(folder/'run_identity.json', dict(run_identity='foreign' if state == 'foreign_receipt'
                                                  else runner.training_identity(config, 'R3', 41, 'cpu')))
    if state == 'progress':
        write_json(folder/'progress.json', dict(step=1))
    if state == 'unexpected':
        (folder/'keep.txt').write_text('Foreign work must survive')
    if state == 'symlink':
        target = tmp_path/'foreign.pt'
        partial.rename(target)
        partial.symlink_to(target)
    before = {path.name: path.read_bytes() for path in folder.iterdir()}
    if state == 'owned':
        runner.train(config, 'R3', 41, 'cpu', resume=True)
        checkpoint = torch.load(folder/'checkpoint.pt', weights_only=True)
        assert checkpoint['step'] == 2 and checkpoint['sampler']['cursor'] == 5
        assert not partial.exists()
    else:
        with pytest.raises((ValueError, FileNotFoundError)):
            runner.train(config, 'R3', 41, 'cpu', resume=True)
        assert {path.name: path.read_bytes() for path in folder.iterdir()} == before


def test_parent_sigterm_reaches_child_and_grandchild_and_fails(tmp_path):
    root = Path(__file__).resolve().parent
    child = tmp_path / 'child.py'
    child.write_text("""import os, pathlib, signal, subprocess, sys, time
out=pathlib.Path(sys.argv[1]); label=sys.argv[2]
def stop(*args):
    (out/(label+'.term')).write_text('terminated')
    raise SystemExit(0)
signal.signal(signal.SIGTERM, stop)
if label=='child':
    subprocess.Popen([sys.executable,__file__,str(out),'grandchild'])
(out/(label+'.ready')).write_text(str(os.getpid()))
while True: time.sleep(.05)
""")
    driver = tmp_path / 'driver.py'
    driver.write_text("from panns_cloud_pipeline import run_command\n"
                      "import sys, time\n"
                      "run_command([sys.executable, sys.argv[1], sys.argv[2], 'child'], time.monotonic()+120)\n")
    environment = dict(os.environ, PYTHONPATH=str(root))
    parent = subprocess.Popen([sys.executable, str(driver), str(child), str(tmp_path)], env=environment,
                              stdout=subprocess.PIPE, stderr=subprocess.PIPE, start_new_session=True)
    try:
        until = time.monotonic() + 10
        while not (tmp_path / 'grandchild.ready').exists() and time.monotonic() < until:
            time.sleep(.05)
        assert (tmp_path / 'grandchild.ready').exists(), 'Child process never became ready'
        parent.send_signal(signal.SIGTERM)
        _, error = parent.communicate(timeout=10)
        assert parent.returncode != 0, error.decode()
        assert (tmp_path / 'child.term').read_text() == 'terminated'
        assert (tmp_path / 'grandchild.term').read_text() == 'terminated'
    finally:
        if parent.poll() is None:
            parent.terminate()
            try:
                parent.wait(timeout=65)
            except subprocess.TimeoutExpired:
                parent.kill()
                parent.wait()
