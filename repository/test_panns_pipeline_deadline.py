import contextlib
import inspect
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

import pytest

import panns_cloud_pipeline as pipeline


def mixed_preflight_inputs(tmp_path):
    rows = [dict(ID=rid, role=role, VOICE_PRESENT=voice, MUSIC_PRESENT=music, verified_audio=dict(seconds=seconds))
            for rid, role, voice, music, seconds in (
                ('train', 'train', 1, 1, .1), ('dev-b', 'development', 1, 1, 2.),
                ('dev-a', 'development', 1, 1, 2.), ('sel-a', 'selection', 1, 1, 3.),
                ('sel-no-voice', 'selection', 0, 1, .1), ('outside', 'selection', 1, 1, .1))]
    path = tmp_path/'rows.jsonl'
    path.write_text(''.join(json.dumps(row)+'\n' for row in rows))
    return dict(rows=str(path), panel_ids=[row['ID'] for row in rows if row['ID'] != 'outside'])


def test_expired_session_cannot_start_or_modify_a_partial_checkpoint(tmp_path):
    checkpoint = tmp_path/'checkpoint.pt'
    checkpoint.write_bytes(b'Existing resumable state')
    marker = tmp_path/'must_not_start'
    cutoff = time.time()-1
    with pytest.raises(TimeoutError):
        pipeline.run_command([sys.executable, '-c', 'import pathlib,sys; pathlib.Path(sys.argv[1]).touch()', str(marker)],
                             time.monotonic()-1, deadline_epoch=cutoff)
    assert not marker.exists()
    assert checkpoint.read_bytes() == b'Existing resumable state'


def test_two_real_stages_share_session_cap_and_flush_checkpoint_before_cleanup(tmp_path, monkeypatch):
    monkeypatch.setattr(pipeline, 'CHILD_GRACE_SECONDS', .2)
    monkeypatch.setattr(pipeline, 'CHILD_REAP_SECONDS', .1)
    started = time.monotonic()
    epoch = time.time()+1.2
    deadline = started+1.2
    pipeline.run_command([sys.executable, '-c', 'import time; time.sleep(.3)'], deadline,
                         deadline_epoch=epoch)
    receipt = tmp_path/'checkpoint.json'
    program = ('import pathlib,signal,sys,time\n'
               'def stop(*args):\n'
               ' time.sleep(.05)\n'
               ' pathlib.Path(sys.argv[1]).write_text("checkpoint flushed")\n'
               ' raise SystemExit(0)\n'
               'signal.signal(signal.SIGTERM,stop)\n'
               'while True: time.sleep(.01)\n')
    with pytest.raises(TimeoutError):
        pipeline.run_command([sys.executable, '-c', program, str(receipt)], deadline,
                             deadline_epoch=epoch)
    assert receipt.read_text() == 'checkpoint flushed'
    assert time.monotonic()-started < 1.2


def test_legacy_phase_contract_cannot_be_relabelled_as_early_stopping(tmp_path, monkeypatch):
    config = dict(identity='legacy-partial-training', output=str(tmp_path))
    run = tmp_path/'run.json'
    run.write_text(json.dumps(config))
    deadline = tmp_path/'training_deadline.json'
    deadline.write_bytes(b'Original immutable fixed-duration receipt')
    checkpoint = tmp_path/'train_R3_41/checkpoint.pt'
    checkpoint.parent.mkdir()
    checkpoint.write_bytes(b'Existing committed checkpoint must be preserved')
    monkeypatch.setattr(sys, 'argv', ['pipeline', '--run', str(run), '--bundle', str(tmp_path),
                                    '--hours','8','--deadline-epoch',str(time.time()+8*3600)])
    with pytest.raises(ValueError, match='fresh experiment namespace'):
        pipeline.main()
    assert checkpoint.read_bytes() == b'Existing committed checkpoint must be preserved'
    assert deadline.read_bytes() == b'Original immutable fixed-duration receipt'


def test_outer_guard_cannot_orphan_stubborn_separate_session_child(tmp_path):
    root = Path(__file__).resolve().parent
    marker = tmp_path/'owned_child.pid'
    child = tmp_path/'stubborn.py'
    child.write_text('import os,pathlib,signal,sys,time\n'
                     'signal.signal(signal.SIGTERM,signal.SIG_IGN)\n'
                     'pathlib.Path(sys.argv[1]).write_text(str(os.getpid()))\n'
                     'os.close(1); os.close(2)\n'
                     'while True: time.sleep(.01)\n')
    runner = tmp_path/'pipeline.py'
    runner.write_text('import panns_cloud_pipeline as p\nimport sys,time\n'
                      'p.CHILD_GRACE_SECONDS=.15\np.CHILD_REAP_SECONDS=.05\n'
                      'p.run_command(sys.argv[1:],time.monotonic()+120)\n')
    command = ('from panns_launch_guard import supervise\nimport sys,time\n'
               'print(supervise(sys.argv[1:],time.time()+.8,1.))\n')
    try:
        result = subprocess.run([sys.executable, '-c', command, sys.executable, str(runner),
                                 sys.executable, str(child), str(marker)],
                                env=dict(os.environ, PYTHONPATH=str(root)),
                                capture_output=True, text=True, timeout=5)
        assert result.returncode == 0 and 'deadline_exhausted' in result.stdout
        pid = int(marker.read_text())
        with pytest.raises(ProcessLookupError):
            os.kill(pid, 0)
    finally:
        if marker.exists():
            pid = int(marker.read_text())
            with contextlib.suppress(ProcessLookupError):
                assert os.getpgid(pid) == pid
                os.killpg(pid, signal.SIGKILL)
    from panns_launch_guard import supervise
    assert pipeline.CHILD_GRACE_SECONDS + pipeline.CHILD_REAP_SECONDS < inspect.signature(supervise).parameters['grace_seconds'].default


def pin_gold_index(tmp_path, config, rid):
    from panns_adaptation_data import sha256
    (tmp_path/'gold').mkdir()
    (tmp_path/'gold/index.json').write_text(json.dumps(dict(entries=[dict(ID=rid)])))
    manifest = tmp_path/'bundle_manifest.json'
    manifest.write_text(json.dumps(dict(sha256={'gold/index.json': sha256(tmp_path/'gold/index.json')})))
    config['inputs'] = {str(manifest): sha256(manifest)}


def pipeline_run(tmp_path, monkeypatch):
    import panns_cloud_run
    import panns_training_budget
    policy = dict(max_epochs=200, patience=15, min_delta=1e-4, monitor='selection_presence_bce')
    config = dict(identity='wiring-only', output=str(tmp_path/'output'), cache_device='cuda', arms=['R1','R2','R3','R5'],
                  seeds=[41,42,43], early_stopping=policy, **mixed_preflight_inputs(tmp_path))
    pin_gold_index(tmp_path, config, 'probe')
    run = tmp_path/'run.json'
    run.write_text(json.dumps(config))
    session_epoch = time.time()+8*3600
    monkeypatch.setattr(sys, 'argv', ['pipeline', '--run', str(run), '--bundle', str(tmp_path),
                                    '--hours','8','--deadline-epoch',str(session_epoch)])
    budget = dict(training_steps=400, steps_per_pass=2, early_stopping=policy)
    monkeypatch.setattr(panns_training_budget, 'load_budget', lambda *args, **kwargs: (budget, 'budget-sha'))
    monkeypatch.setattr(panns_cloud_run, 'training_identity', lambda config, arm, seed, device: arm+'-'+str(seed))
    return config, run, session_epoch


def save_condition(config, command):
    from panns_adaptation_data import sha256
    arm, seed = command[command.index('--arm')+1], int(command[command.index('--seed')+1])
    folder = Path(config['output'])/f'train_{arm}_{seed}'
    folder.mkdir(parents=True, exist_ok=True)
    checkpoint = folder/'checkpoint.pt'
    checkpoint.write_bytes((arm+'-'+str(seed)).encode())
    epochs = 200 if arm == 'R1' else 16
    record = dict(complete=True, early_stopping=config['early_stopping'], monitor='selection_presence_bce',
        steps=epochs*2, epochs_completed=epochs, last_validated_epoch=epochs,
        best_epoch=200 if arm == 'R1' else 1, best_loss=.5, patience_anchor=.5,
        bad_epochs=0 if arm == 'R1' else 15, stop_reason='max_epochs' if arm == 'R1' else 'early_stopping',
        run_identity=arm+'-'+str(seed), checkpoint_sha256=sha256(checkpoint))
    (folder/'complete.json').write_text(json.dumps(record))


def test_pipeline_uses_one_session_cutoff_and_accepts_different_terminal_epochs(tmp_path, monkeypatch):
    config, run, session_epoch = pipeline_run(tmp_path, monkeypatch)
    calls = []
    def record(command, deadline, **kwargs):
        calls.append((command, deadline, kwargs))
        if 'train' in command[2:]: save_condition(config, command)
    monkeypatch.setattr(pipeline, 'run_command', record)
    pipeline.main()
    training = [call for call in calls if Path(call[0][1]).name == 'panns_training_budget.py'
                or 'train' in call[0][2:]]
    assert len(training) == 13 and len({call[1] for call in calls}) == 1
    assert all(call[2]['deadline_epoch'] == session_epoch for call in calls)
    assert all('--training-hours' not in call[0] for call in calls)
    phase = json.loads((Path(config['output'])/'training_phase_complete.json').read_text())
    assert len(phase['conditions']) == 12 and phase['training_budget_sha256'] == 'budget-sha'
    assert not (Path(config['output'])/'training_deadline.json').exists()
    calls.clear()
    pipeline.main()
    assert not any(Path(call[0][1]).name == 'panns_training_budget.py' or 'train' in call[0][2:] for call in calls)
    path = Path(config['output'])/'train_R3_41/complete.json'
    changed = json.loads(path.read_text()); changed['complete'] = False; path.write_text(json.dumps(changed))
    with pytest.raises(ValueError): pipeline.main()


def test_training_interruption_never_publishes_phase_or_pipeline_completion(tmp_path, monkeypatch):
    config, _, _ = pipeline_run(tmp_path, monkeypatch)
    checkpoint = Path(config['output'])/'train_R1_41/checkpoint.pt'
    def interrupt(command, deadline, **kwargs):
        if 'train' in command[2:]:
            checkpoint.parent.mkdir(parents=True)
            checkpoint.write_bytes(b'Latest checkpoint with persistent patience state')
            raise TimeoutError('Authorized session exhausted')
    monkeypatch.setattr(pipeline, 'run_command', interrupt)
    with pytest.raises(TimeoutError, match='session exhausted'): pipeline.main()
    assert checkpoint.read_bytes() == b'Latest checkpoint with persistent patience state'
    assert not (Path(config['output'])/'training_phase_complete.json').exists()
    assert not (Path(config['output'])/'pipeline_complete.json').exists()


def test_resume_does_not_rederive_policy_from_remaining_session_time(tmp_path, monkeypatch):
    _, run, session_epoch = pipeline_run(tmp_path, monkeypatch)
    original = run.read_bytes()
    now = [session_epoch-8*3600]
    monkeypatch.setattr(pipeline.time, 'time', lambda: now[0])
    profiles = []
    def stop_at_profile(command, deadline, **kwargs):
        if Path(command[1]).name == 'panns_training_budget.py':
            profiles.append((command, kwargs['deadline_epoch']))
            raise RuntimeError('Pause at profiling')
    monkeypatch.setattr(pipeline, 'run_command', stop_at_profile)
    for _ in range(2):
        with pytest.raises(RuntimeError, match='Pause at profiling'): pipeline.main()
        now[0] += 600
    assert profiles[0] == profiles[1]
    assert '--training-hours' not in profiles[0][0]
    assert run.read_bytes() == original


def test_fixed_engine_preflight_failure_stops_before_full_cache_and_training(tmp_path, monkeypatch):
    from panns_adaptation_data import identity
    config = dict(identity='engine-readiness', output=str(tmp_path/'output'), cache_device='cuda', arms=['R1','R2','R3','R5'], seeds=[41,42,43],
                  **mixed_preflight_inputs(tmp_path))
    pin_gold_index(tmp_path, config, 'train')
    run = tmp_path/'run.json'
    run.write_text(json.dumps(config))
    epoch = time.time()+8*3600
    monkeypatch.setattr(sys, 'argv', ['pipeline', '--run', str(run), '--bundle', str(tmp_path),
                                    '--hours','8','--deadline-epoch',str(epoch)])
    calls = []
    def fail_engine(command, deadline, **kwargs):
        calls.append((command, deadline, kwargs))
        if len(calls) < 4:
            return
        assert Path(command[1]).name == 'panns_fixed_predictions.py', 'Expensive full cache preceded fixed Engine preflight'
        probe = json.loads(Path(command[command.index('--run')+1]).read_text())
        assert probe['panel_ids'] == ['dev-a', 'sel-a']
        assert probe['rows'] == config['rows']
        assert probe['output'] == str(Path(config['output'])/'fixed_preflight')
        assert probe['fixed_predictions'] == str(Path(probe['output'])/'fixed_predictions.json')
        assert probe['identity'] == identity({key:value for key,value in probe.items() if key != 'identity'})
        raise RuntimeError('Fixed Engine preflight failed')
    monkeypatch.setattr(pipeline, 'run_command', fail_engine)
    with pytest.raises(RuntimeError, match='Fixed Engine preflight failed'):
        pipeline.main()
    assert 'cache' in calls[0][0] and 'cpu' in calls[0][0] and '--parity-only' in calls[2][0]
    assert len(calls) == 4 and len({call[1] for call in calls}) == 1
    assert all(call[2]['deadline_epoch'] == epoch for call in calls)
    assert not (Path(config['output'])/'training_deadline.json').exists()


def test_fixed_preflight_refuses_a_train_only_panel(tmp_path):
    config = dict(identity='train-only-gold', output=str(tmp_path/'output'), **mixed_preflight_inputs(tmp_path))
    config['panel_ids'] = ['train']
    with pytest.raises(ValueError, match='development'):
        pipeline.fixed_preflight_config(config)


@pytest.mark.parametrize('failed_stage', ['cpu_reference', 'parity'])
def test_reference_or_parity_failure_cannot_reach_full_cache_or_training(tmp_path, monkeypatch, failed_stage):
    config, _, epoch = pipeline_run(tmp_path, monkeypatch)
    calls = []
    def fail_preflight(command, deadline, **kwargs):
        calls.append((command, deadline, kwargs))
        is_cpu = '--device' in command and command[command.index('--device')+1] == 'cpu'
        if is_cpu:
            probe = json.loads(Path(command[command.index('--run')+1]).read_text())
            assert probe['output'] == str(Path(config['output'])/'cuda_preflight/cpu_reference')
            assert probe['cache_device'] == 'cpu' and probe['panel_ids'] == ['probe']
            if failed_stage == 'cpu_reference': raise RuntimeError('CPU reference failed')
            return
        if '--parity-only' in command:
            assert len(calls) == 3
            raise RuntimeError('Parity failed')
        if len(calls) == 2 and 'cache' in command and 'cuda' in command: return
        pytest.fail('Expensive work started before the required CPU reference and parity gate')
    monkeypatch.setattr(pipeline, 'run_command', fail_preflight)
    with pytest.raises(RuntimeError, match='CPU reference failed|Parity failed'):
        pipeline.main()
    assert len(calls) == (1 if failed_stage == 'cpu_reference' else 3)
    assert all(call[2]['deadline_epoch'] == epoch for call in calls)
    assert not (Path(config['output'])/'pipeline_complete.json').exists()


def test_resume_refuses_changed_probe_platform_before_any_child(tmp_path, monkeypatch):
    config, run, _ = pipeline_run(tmp_path, monkeypatch)
    probe_path = run.with_name('cuda_preflight.json')
    previous = dict(config, parity_runtime=dict(system='Linux', architecture='previous-architecture', python='3.11'))
    probe_path.write_text(json.dumps(previous))
    before = probe_path.read_bytes()
    def forbid_child(*args, **kwargs):
        pytest.fail('Changed producer runtime reached a cache child')
    monkeypatch.setattr(pipeline, 'run_command', forbid_child)
    with pytest.raises(ValueError, match='Existing parity probe config differs'):
        pipeline.main()
    assert probe_path.read_bytes() == before
