import json
import os
from pathlib import Path
import subprocess
import sys
import time

import pytest


def test_launcher_timeout_covers_blocked_archive_and_keeps_original_deadline(tmp_path):
    # Given: the actual launcher's first archive read blocks on a real FIFO.
    root = Path(__file__).resolve().parent
    inputs = tmp_path / 'shared/panns-cnn-v1-20260925/inputs-v9'
    inputs.mkdir(parents=True)
    (inputs / 'launch_lock.json').write_text('{"workspace":"wsp-nvxqw1jf9zdv","sha256":{"run_bundle.tar":"unused"}}')
    os.mkfifo(inputs / 'run_bundle.tar')
    state = tmp_path / 'launch_deadline.json'
    epoch = time.time() + 62
    environment = dict(os.environ, PANNS_LOCAL_QA='1', PANNS_PAUSE_ON_EXIT='0',
        PANNS_PYTHON=sys.executable, PANNS_SHARED=str(tmp_path / 'shared'),
        PANNS_WORK=str(tmp_path / 'work'), PANNS_DEADLINE_STATE=str(state),
        PANNS_DEADLINE_EPOCH=str(epoch))
    command = ['bash', str(root / 'panns_a100_launch.sh'), '--hours', '1']
    # When: execute the real launcher, never reaching restore or any GPU/API action.
    started = time.monotonic()
    result = subprocess.run(command, env=environment, capture_output=True, text=True, timeout=8)
    elapsed = time.monotonic() - started
    # Then: staging is stopped at the compute cutoff and its failure stays explicit.
    receipt = json.loads(state.with_name('launch_exit.json').read_text())
    assert result.returncode == 124, result.stderr
    assert elapsed < 6
    assert receipt['status'] == 'deadline_exhausted'
    assert receipt['pause_status'] == 'disabled_local_qa'
    assert receipt['deadline_epoch'] == epoch
    assert json.loads(state.read_text())['deadline_epoch'] == epoch
    assert not (tmp_path / 'work/run.json').exists()
    environment['PANNS_DEADLINE_EPOCH'] = str(epoch + 3600)
    retry = subprocess.run(command, env=environment, capture_output=True, text=True, timeout=5)
    assert retry.returncode != 0
    assert 'original deadline' in retry.stderr
    assert json.loads(state.read_text())['deadline_epoch'] == epoch


def test_guard_allows_real_sigterm_checkpoint_before_timeout_return(tmp_path):
    # Given: a child writes its checkpoint only after receiving TERM and a short flush.
    root = Path(__file__).resolve().parent
    child = tmp_path / 'child.py'
    checkpoint = tmp_path / 'checkpoint.json'
    child.write_text('''import json, pathlib, signal, sys, time
def stop(*args):
    time.sleep(.2)
    pathlib.Path(sys.argv[1]).write_text(json.dumps({'step': 7, 'signal': 'SIGTERM'}))
    raise SystemExit(0)
signal.signal(signal.SIGTERM, stop)
while True: time.sleep(.01)
''')
    # When: the production supervisor reaches its cutoff around an actual subprocess.
    pipeline = tmp_path / 'pipeline.py'
    pipeline.write_text('from panns_cloud_pipeline import run_command\nimport sys,time\n'
                        'run_command(sys.argv[1:], time.monotonic()+120)\n')
    driver = ('from panns_launch_guard import supervise\nimport sys,time\n'
              'result=supervise(sys.argv[1:],time.time()+1.,.5)\n'
              'print(result)\n')
    result = subprocess.run([sys.executable, '-c', driver, sys.executable, str(pipeline),
        sys.executable, str(child), str(checkpoint)],
        env=dict(os.environ, PYTHONPATH=str(root)), capture_output=True, text=True, timeout=5)
    # Then: checkpoint flush finishes, but a censored run is never reported successful.
    assert result.returncode == 0, result.stderr
    assert json.loads(checkpoint.read_text()) == {'step': 7, 'signal': 'SIGTERM'}
    assert "'deadline_exhausted'" in result.stdout
    assert '124' in result.stdout


@pytest.mark.parametrize('blocked_name', ['launch_deadline.json', 'launch_exit.json'])
def test_shared_metadata_fifo_cannot_prevent_pause_stage(tmp_path, blocked_name):
    # Given: real shared-state or receipt IO blocks, with cloud calls disabled on Mac.
    root = Path(__file__).resolve().parent
    state = tmp_path / 'launch_deadline.json'
    os.mkfifo(tmp_path / blocked_name)
    epoch = time.time() + 62
    environment = dict(os.environ, PANNS_LOCAL_QA='1', PANNS_PAUSE_ON_EXIT='0',
        PANNS_DEADLINE_STATE=str(state), PANNS_DEADLINE_EPOCH=str(epoch))
    command = [sys.executable, str(root / 'panns_launch_guard.py'), '-c', 'exit 7']
    # When: the real supervisor reaches blocked metadata IO under a finite deadline.
    process = subprocess.Popen(command, env=environment, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
                               start_new_session=True)
    try:
        output, error = process.communicate(timeout=9)
    finally:
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=3)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=1)
    # Then: pause handling is reached without claiming an actual cloud pause succeeded.
    events = [json.loads(line) for line in output.splitlines() if line.startswith('{')]
    (tmp_path / 'observed_launch.json').write_text(json.dumps(events, indent=2)+'\n')
    assert process.returncode != 0, error
    assert any(event.get('event') == 'pause_stage' for event in events), output
    final = events[-1]
    assert final['pause_status'] == 'disabled_local_qa'
    assert final['status'] != 'complete'
    if blocked_name == 'launch_deadline.json':
        assert final['phase'] == 'state'
        assert final['status'] == 'deadline_exhausted'
    else:
        assert final['child_exit_code'] == 7
        assert final['receipt_status'] == 'deadline_exhausted'


@pytest.mark.parametrize('deadline', [None, '1', 'not-a-number'])
def test_invalid_deadline_preserves_bounded_pause_child_allowance(tmp_path, deadline):
    # Given: a local pause-protocol child needs longer than the defective 0.1-second window.
    root = Path(__file__).resolve().parent
    binary = tmp_path / 'vesslctl'
    marker = tmp_path / 'pause_protocol_completed'
    binary.write_text(f'#!{sys.executable}\nimport pathlib,time\ntime.sleep(.3)\n'
                      f'pathlib.Path({str(marker)!r}).write_text("local CPU probe; no API")\n')
    binary.chmod(0o700)
    environment = dict(os.environ, PATH=f'{tmp_path}:/usr/bin:/bin', PANNS_PAUSE_ON_EXIT='1',
                       PANNS_DEADLINE_STATE=str(tmp_path / 'launch_deadline.json'))
    environment.pop('PANNS_DEADLINE_EPOCH', None)
    if deadline is not None:
        environment['PANNS_DEADLINE_EPOCH'] = deadline
    # When: the actual guard rejects missing, expired, or malformed startup deadlines.
    result = subprocess.run([sys.executable, str(root / 'panns_launch_guard.py'), '-c', 'exit 0'],
        env=environment, capture_output=True, text=True, timeout=5)
    # Then: no compute runs, but bounded cleanup still completes the real child protocol.
    events = [json.loads(line) for line in result.stdout.splitlines() if line.startswith('{')]
    assert result.returncode != 0
    assert marker.read_text() == 'local CPU probe; no API'
    assert events[-1]['status'] == 'launch_failed'
    assert events[-1]['pause_status'] == 'request_accepted_not_verified'
    assert 'child_exit_code' not in events[-1]


def test_hup_between_stages_and_term_during_cleanup_preserve_pause_protocol(tmp_path):
    # Given: a real HUP at the stage boundary and another TERM during a local pause child.
    root = Path(__file__).resolve().parent
    binary = tmp_path / 'vesslctl'
    marker = tmp_path / 'pause_protocol_completed'
    binary.write_text(f'#!{sys.executable}\nimport os,pathlib,signal,time\n'
        'time.sleep(.1)\nos.kill(os.getppid(), signal.SIGTERM)\ntime.sleep(.3)\n'
        f'pathlib.Path({str(marker)!r}).write_text("cleanup completed; no API")\n')
    binary.chmod(0o700)
    driver = '''import os,signal,sys
import panns_launch_guard as guard
original=guard.supervise
def signal_at_boundary(command,*args,**kwargs):
    result=original(command,*args,**kwargs)
    if '--check-state' in command:
        os.kill(os.getpid(),signal.SIGHUP)
    return result
guard.supervise=signal_at_boundary
sys.argv=[guard.__file__,'-c','exit 0']
raise SystemExit(guard.main())
'''
    environment = dict(os.environ, PATH=f'{tmp_path}:/usr/bin:/bin', PYTHONPATH=str(root),
        PANNS_PAUSE_ON_EXIT='1', PANNS_DEADLINE_EPOCH=str(time.time()+180),
        PANNS_DEADLINE_STATE=str(tmp_path / 'launch_deadline.json'))
    # When: real OS signals arrive after actual state validation and during actual child cleanup.
    result = subprocess.run([sys.executable, '-c', driver], env=environment,
                            capture_output=True, text=True, timeout=5)
    # Then: cancellation prevents compute while both signals preserve bounded pause cleanup.
    assert result.returncode == 130, result.stderr
    assert marker.read_text() == 'cleanup completed; no API'
    events = [json.loads(line) for line in result.stdout.splitlines() if line.startswith('{')]
    assert events[-1]['status'] == 'interrupted'
    assert events[-1]['pause_status'] == 'request_accepted_not_verified'


def test_external_controller_contract_runs_without_remote_cli_and_cannot_drift(tmp_path):
    # Given: remote-like PATH has no VESSL CLI and a target/source/deadline-bound stop owner.
    root = Path(__file__).resolve().parent
    epoch = time.time()+120
    contract = dict(org='qwdkw', workspace='wsp-nvxqw1jf9zdv', deadline_epoch=epoch,
        run_id='controlled-cpu-probe', bundle_sha256='a'*64, stop_owner='local_panns_workspace_watch')
    environment = dict(os.environ, PATH='/usr/bin:/bin', PANNS_PAUSE_ON_EXIT='1',
        PANNS_DEADLINE_EPOCH=str(epoch), PANNS_DEADLINE_STATE=str(tmp_path / 'launch_deadline.json'),
        PANNS_CONTROLLER_CONTRACT=json.dumps(contract))
    command = [sys.executable, str(root / 'panns_launch_guard.py'), '-c', 'exit 7']
    # When: execute the guard, then attempt to resume a different bundle under the same deadline.
    first = subprocess.run(command, env=environment, capture_output=True, text=True, timeout=5)
    environment['PANNS_CONTROLLER_CONTRACT'] = json.dumps({**contract, 'bundle_sha256': 'b'*64})
    second = subprocess.run(command, env=environment, capture_output=True, text=True, timeout=5)
    # Then: compute uses the declared controller, while resume source drift is fatal before compute.
    events = [json.loads(line) for line in first.stdout.splitlines() if line.startswith('{')]
    assert first.returncode == 7, first.stderr
    assert events[-1]['pause_status'] == 'external_controller_required'
    assert events[-1]['controller_contract'] == contract
    assert 'pause_exit_code' not in events[-1]
    assert second.returncode != 0
    assert 'original deadline and controller contract' in second.stderr
    environment.pop('PANNS_CONTROLLER_CONTRACT')
    third = subprocess.run(command, env=environment, capture_output=True, text=True, timeout=5)
    assert third.returncode != 0
    assert 'vesslctl required before starting paid compute' in third.stderr


def test_launcher_refuses_external_controller_bundle_mismatch(tmp_path):
    # Given: the actual launcher lock and controller declare different bundle identities.
    root = Path(__file__).resolve().parent
    inputs = tmp_path / 'shared/panns-cnn-v1-20260925/inputs-v9'
    inputs.mkdir(parents=True)
    (inputs / 'launch_lock.json').write_text(json.dumps(dict(workspace='wsp-nvxqw1jf9zdv',
        sha256={'run_bundle.tar': 'a'*64})))
    epoch = time.time()+120
    contract = dict(org='qwdkw', workspace='wsp-nvxqw1jf9zdv', deadline_epoch=epoch,
        run_id='wrong-bundle-probe', bundle_sha256='b'*64, stop_owner='local_panns_workspace_watch')
    environment = dict(os.environ, PATH='/usr/bin:/bin', PANNS_PAUSE_ON_EXIT='1',
        PANNS_DEADLINE_EPOCH=str(epoch), PANNS_DEADLINE_STATE=str(tmp_path / 'launch_deadline.json'),
        PANNS_CONTROLLER_CONTRACT=json.dumps(contract), PANNS_SHARED=str(tmp_path/'shared'),
        PANNS_WORK=str(tmp_path/'work'))
    environment.pop('PANNS_LOCAL_QA', None)
    environment.pop('PANNS_PYTHON', None)
    # When: invoke the production launcher path without any remote CLI or GPU operation.
    result = subprocess.run(['bash', str(root/'panns_a100_launch.sh')], env=environment,
                            capture_output=True, text=True, timeout=5)
    # Then: source mismatch is rejected before archive reads or environment restoration.
    assert result.returncode != 0
    assert 'Controller contract does not identify the uploaded bundle' in result.stderr
    assert not (tmp_path/'work/run.json').exists()
