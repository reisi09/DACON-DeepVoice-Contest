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


def test_training_deadline_is_immutable_across_resume_and_rejects_changed_contract(tmp_path, monkeypatch):
    config = dict(identity='locked-run', output=str(tmp_path))
    monkeypatch.setattr(pipeline.time, 'time', lambda: 1000.)
    cutoff = pipeline.training_phase_deadline(config, 10000., 1.)
    path = tmp_path/'training_deadline.json'
    original = path.read_bytes()
    monkeypatch.setattr(pipeline.time, 'time', lambda: 2800.)
    assert pipeline.training_phase_deadline(config, 10000., 1.) == cutoff == 4600.
    assert path.read_bytes() == original
    for changed, session, hours in ((dict(config, identity='other'), 10000., 1.),
                                     (config, 10001., 1.), (config, 10000., 2.)):
        with pytest.raises(ValueError, match='contract'):
            pipeline.training_phase_deadline(changed, session, hours)
        assert path.read_bytes() == original
    monkeypatch.setattr(pipeline.time, 'time', lambda: 4601.)
    assert pipeline.training_phase_deadline(config, 10000., 1.) == cutoff
    with pytest.raises(TimeoutError):
        pipeline.run_command([sys.executable, '-c', 'raise AssertionError("must not start")'],
                             time.monotonic()+cutoff-time.time(), deadline_epoch=cutoff)
    assert path.read_bytes() == original


def test_two_real_stages_share_training_cap_and_cleanup_finishes_inside_it(tmp_path, monkeypatch):
    monkeypatch.setattr(pipeline, 'CHILD_GRACE_SECONDS', .2, raising=False)
    monkeypatch.setattr(pipeline, 'CHILD_REAP_SECONDS', .1, raising=False)
    config = dict(identity='shared-real-stages', output=str(tmp_path))
    session_epoch = time.time()+120
    started = time.monotonic()
    epoch = pipeline.training_phase_deadline(config, session_epoch, 1.2/3600)
    deadline = time.monotonic()+epoch-time.time()
    pipeline.run_command([sys.executable, '-c', 'import time; time.sleep(.3)'], deadline,
                         deadline_epoch=epoch)
    assert pipeline.training_phase_deadline(config, session_epoch, 1.2/3600) == epoch
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


def test_missing_phase_deadline_cannot_grant_new_time_to_partial_training(tmp_path):
    config = dict(identity='existing-partial-training', output=str(tmp_path))
    checkpoint = tmp_path/'train_R3_41/checkpoint.pt'
    checkpoint.parent.mkdir()
    checkpoint.write_bytes(b'Existing committed checkpoint must be preserved')
    before = checkpoint.read_bytes()
    with pytest.raises(FileNotFoundError, match='deadline'):
        pipeline.training_phase_deadline(config, time.time()+8*3600, 4.)
    assert checkpoint.read_bytes() == before
    assert not (tmp_path/'training_deadline.json').exists()


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


def test_pipeline_passes_one_training_cutoff_to_pilot_and_all_twelve_conditions(tmp_path, monkeypatch):
    config = dict(identity='wiring-only', output=str(tmp_path/'output'), arms=['R1','R2','R3','R5'], seeds=[41,42,43],
                  **mixed_preflight_inputs(tmp_path))
    run = tmp_path/'run.json'
    run.write_text(json.dumps(config))
    (tmp_path/'gold').mkdir()
    (tmp_path/'gold/index.json').write_text(json.dumps(dict(entries=[dict(ID='probe')])))
    session_epoch = time.time()+8*3600
    calls = []
    monkeypatch.setattr(sys, 'argv', ['pipeline', '--run', str(run), '--bundle', str(tmp_path),
                                    '--hours','8','--training-hours','4','--deadline-epoch',str(session_epoch)])
    def record(command, deadline, **kwargs):
        calls.append((command, deadline, kwargs))
        if 'train' in command[2:]:
            arm, seed = command[command.index('--arm')+1], command[command.index('--seed')+1]
            folder = Path(config['output'])/f'train_{arm}_{seed}'
            folder.mkdir(parents=True, exist_ok=True)
            (folder/'complete.json').write_text(json.dumps(dict(arm=arm, seed=seed)))
    monkeypatch.setattr(pipeline, 'run_command', record)
    pipeline.main()
    training = [call for call in calls if Path(call[0][1]).name == 'panns_training_budget.py'
                or 'train' in call[0][2:]]
    other = [call for call in calls if call not in training]
    assert len(training) == 13
    assert len({call[1] for call in training}) == 1
    phase = json.loads((tmp_path/'output/training_deadline.json').read_text())
    assert all(call[2]['deadline_epoch'] == phase['deadline_epoch'] for call in training)
    assert all(call[1] > training[0][1] for call in other)
    assert phase['deadline_epoch'] <= session_epoch
    calls.clear()
    monkeypatch.setattr(pipeline.time, 'time', lambda: phase['deadline_epoch']+1)
    pipeline.main()
    assert not any(Path(call[0][1]).name == 'panns_training_budget.py' or 'train' in call[0][2:] for call in calls)
    (Path(config['output'])/'train_R3_41/complete.json').write_text('changed condition')
    with pytest.raises(ValueError, match='condition receipts'):
        pipeline.main()


def test_profile_uses_fixed_remaining_session_allowance_on_resume(tmp_path, monkeypatch):
    config = dict(identity='late-cache-completion', output=str(tmp_path/'output'), arms=['R1','R2','R3','R5'], seeds=[41,42,43],
                  **mixed_preflight_inputs(tmp_path))
    run = tmp_path/'run.json'
    run.write_text(json.dumps(config))
    (tmp_path/'gold').mkdir()
    (tmp_path/'gold/index.json').write_text(json.dumps(dict(entries=[])))
    now = [1000.]
    monkeypatch.setattr(pipeline.time, 'time', lambda: now[0])
    monkeypatch.setattr(sys, 'argv', ['pipeline', '--run', str(run), '--bundle', str(tmp_path),
                                    '--hours','8','--training-hours','4','--deadline-epoch',str(now[0]+3*3600)])
    allowances = []
    def stop_at_profile(command, deadline, **kwargs):
        if Path(command[1]).name == 'panns_training_budget.py':
            allowances.append(float(command[command.index('--training-hours')+1]))
            raise RuntimeError('Pause after profile planning')
    monkeypatch.setattr(pipeline, 'run_command', stop_at_profile)
    with pytest.raises(RuntimeError, match='Pause after profile'):
        pipeline.main()
    path = tmp_path/'output/training_deadline.json'
    original = path.read_bytes()
    now[0] += 600
    with pytest.raises(RuntimeError, match='Pause after profile'):
        pipeline.main()
    expected = (3*3600-pipeline.CHILD_GRACE_SECONDS-pipeline.CHILD_REAP_SECONDS)/3600
    assert allowances == [expected, expected]
    assert path.read_bytes() == original
    assert json.loads(original)['training_hours'] == 4.


def test_fixed_engine_preflight_failure_stops_before_full_cache_and_training(tmp_path, monkeypatch):
    from panns_adaptation_data import identity
    config = dict(identity='engine-readiness', output=str(tmp_path/'output'), arms=['R1','R2','R3','R5'], seeds=[41,42,43],
                  **mixed_preflight_inputs(tmp_path))
    run = tmp_path/'run.json'
    run.write_text(json.dumps(config))
    (tmp_path/'gold').mkdir()
    (tmp_path/'gold/index.json').write_text(json.dumps(dict(entries=[dict(ID='train')])))
    epoch = time.time()+8*3600
    monkeypatch.setattr(sys, 'argv', ['pipeline', '--run', str(run), '--bundle', str(tmp_path),
                                    '--hours','8','--training-hours','4','--deadline-epoch',str(epoch)])
    calls = []
    def fail_engine(command, deadline, **kwargs):
        calls.append((command, deadline, kwargs))
        if len(calls) < 3:
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
    assert 'cache' in calls[0][0] and '--parity-only' in calls[1][0]
    assert len(calls) == 3 and len({call[1] for call in calls}) == 1
    assert all(call[2]['deadline_epoch'] == epoch for call in calls)
    assert not (Path(config['output'])/'training_deadline.json').exists()


def test_fixed_preflight_refuses_a_train_only_panel(tmp_path):
    config = dict(identity='train-only-gold', output=str(tmp_path/'output'), **mixed_preflight_inputs(tmp_path))
    config['panel_ids'] = ['train']
    with pytest.raises(ValueError, match='development'):
        pipeline.fixed_preflight_config(config)
