import os
from pathlib import Path
import signal
import subprocess
import sys
import time

import pytest


def test_epoch_cap_covers_every_file_without_a_throughput_or_step_cap():
    from panns_training_budget import budget_steps

    seconds = {'R1': 1., 'R2': 2., 'R3': 3., 'R5': 4.}
    assert budget_steps(seconds, 5, 200) == 400
    assert budget_steps(seconds, 2001, 200) == 100200
    for timing in (.001, 10000.):
        assert budget_steps(dict.fromkeys(seconds, timing), 2001, 200,
                            checkpoint_seconds=dict.fromkeys(seconds, timing),
                            checkpoint_hash_seconds=dict.fromkeys(seconds, timing),
                            checkpoint_every=1) == 100200


def test_epoch_budget_rejects_invalid_policy_and_timing_inputs():
    from panns_training_budget import budget_steps

    seconds = {'R1': 1., 'R2': 2., 'R3': 3., 'R5': 4.}
    for invalid in (float('nan'), float('inf'), 0, -1, 1.5, True):
        with pytest.raises(ValueError):
            budget_steps(seconds, 5, invalid)
    for invalid in (float('nan'), float('inf'), 0., -1.):
        with pytest.raises(ValueError):
            budget_steps({**seconds, 'R3': invalid}, 5, 200)
    for timings in ({'R1': .1}, {**seconds, 'R2': float('nan')}, {**seconds, 'R2': -1.}):
        with pytest.raises(ValueError):
            budget_steps(seconds, 5, 200, checkpoint_seconds=timings)
        with pytest.raises(ValueError):
            budget_steps(seconds, 5, 200, checkpoint_hash_seconds=timings)


def budget_artifact(tmp_path):
    from panns_adaptation_data import identity, sha256, write_json
    from panns_training_budget import profile_binding, training_estimate

    config = dict(identity='locked-run', output=str(tmp_path), checkpoint_every=100,
                  early_stopping=dict(max_epochs=200, patience=15, min_delta=1e-4, monitor='selection_presence_bce'))
    write_json(tmp_path/'cache/index.json', dict(archived=True))
    binding = profile_binding(config, 'cpu')
    profile = {'binding': binding, 'train_files': 5,
               'checkpoint_seconds': dict.fromkeys(('R1', 'R2', 'R3', 'R5'), .01),
               'checkpoint_hash_seconds': dict.fromkeys(('R1', 'R2', 'R3', 'R5'), .01),
               'seconds_per_step': {'R1': 1., 'R2': 2., 'R3': 3., 'R5': 4.}}
    write_json(tmp_path / 'throughput_profile.json', profile)
    budget = {'binding': binding, 'profile_sha256': sha256(tmp_path / 'throughput_profile.json'),
              'early_stopping': config['early_stopping'].copy(), 'training_steps': 400, 'steps_per_pass': 2,
              'estimated_training_seconds': training_estimate(profile, 400, 200, 100),
              'validation_time_included': False}
    budget['identity'] = identity(budget)
    write_json(tmp_path / 'training_budget.json', budget)
    return config, profile, budget


def test_budget_refuses_runtime_data_and_profile_drift(tmp_path):
    from panns_adaptation_data import identity, sha256, write_json
    from panns_training_budget import load_budget

    config, profile, budget = budget_artifact(tmp_path)
    assert load_budget(config, 'cpu')[0]['training_steps'] == 400
    with pytest.raises(ValueError, match='device'):
        load_budget(config, 'cuda')
    write_json(tmp_path/'cache/index.json', dict(corrupted=True))
    with pytest.raises(ValueError, match='binding'):
        load_budget(config, 'cpu')
    write_json(tmp_path/'cache/index.json', dict(archived=True))
    profile['seconds_per_step']['R1'] = 10.
    write_json(tmp_path/'throughput_profile.json', profile)
    with pytest.raises(ValueError, match='profile'):
        load_budget(config, 'cpu')
    profile['seconds_per_step']['R1'] = 1.
    profile['binding'] = profile['binding'] | dict(torch='different-runtime')
    budget['binding'] = profile['binding']
    write_json(tmp_path/'throughput_profile.json', profile)
    budget['profile_sha256'] = sha256(tmp_path/'throughput_profile.json')
    budget['identity'] = identity({k: v for k, v in budget.items() if k != 'identity'})
    write_json(tmp_path/'training_budget.json', budget)
    with pytest.raises(ValueError, match='binding'):
        load_budget(config, 'cpu')


@pytest.mark.parametrize('change', ['max_epochs', 'patience', 'min_delta', 'monitor', 'training_steps'])
def test_budget_refuses_altered_stopping_policy_or_step_limit(tmp_path, change):
    from panns_adaptation_data import identity, write_json
    from panns_training_budget import load_budget

    config, _, budget = budget_artifact(tmp_path)
    assert load_budget(config, 'cpu')[0]['training_steps'] == 400
    if change == 'training_steps':
        budget['training_steps'] = 398
        budget['identity'] = identity({k: v for k, v in budget.items() if k != 'identity'})
        write_json(tmp_path/'training_budget.json', budget)
    else:
        config['early_stopping'][change] = dict(max_epochs=201, patience=16, min_delta=2e-4,
                                                monitor='development_presence_bce')[change]
    with pytest.raises(ValueError):
        load_budget(config, 'cpu')


def test_offline_budget_read_preserves_identity_without_authorizing_cuda(tmp_path, monkeypatch):
    import torch
    from panns_adaptation_data import identity, sha256, write_json
    from panns_training_budget import load_budget

    config, profile, budget = budget_artifact(tmp_path)
    binding = profile['binding'] | dict(device='cuda', hardware=dict(name='archived A100'),
                                       torch='archived-torch', cuda='archived-cuda')
    profile['binding'] = binding
    write_json(tmp_path/'throughput_profile.json', profile)
    budget.update(binding=binding, profile_sha256=sha256(tmp_path/'throughput_profile.json'))
    budget['identity'] = identity({k: v for k, v in budget.items() if k != 'identity'})
    write_json(tmp_path/'training_budget.json', budget)
    def unavailable(*args):
        raise RuntimeError('No CUDA in artifact reader')
    monkeypatch.setattr(torch.cuda, 'current_device', unavailable)
    assert load_budget(config, 'cuda', validate_runtime=False)[0]['training_steps'] == 400
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
    config['early_stopping'] = dict(max_epochs=1, patience=15, min_delta=1e-4, monitor='selection_presence_bce')
    write_json(tmp_path/'cache/index.json', dict(loop_fixture=True))
    monkeypatch.setattr(runner, 'validate_config', lambda c: tmp_path)
    monkeypatch.setattr(runner, 'load_budget', lambda *args, **kwargs: (
        dict(training_steps=2, steps_per_pass=2, early_stopping=config['early_stopping']), 'budget'))
    monkeypatch.setattr(runner, 'read_cache_index', lambda c: dict(entries=[
        *[dict(role='train', targets=[i%2, i%2]) for i in range(5)], dict(role='selection', targets=[0, 0])]))
    monkeypatch.setattr(runner, 'model_for_arm', lambda *args: SmallSuffix(2, 1))
    monkeypatch.setattr(runner, 'readout_assets', lambda p: (None, None, None))
    monkeypatch.setattr(runner, 'optimization_step', update)
    monkeypatch.setattr(runner, 'selection_metrics', lambda *args: dict(
        selection_presence_bce=.5, voice_bce=.5, music_bce=.5, known_counts=[1, 1]))
    return config


@pytest.mark.parametrize('checkpoint_every,stop_after,expected_saves', [(1, None, 3), (1, 1, 2), (3, None, 2), (3, 1, 2)])
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
