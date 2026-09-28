"""Local child-process protocol checks; these do not prove VESSL API integration."""
import hashlib
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

import pytest

import panns_workspace_watch as watch


CHILD = '''#!/usr/bin/env python3
import json, pathlib, sys, time
p = pathlib.Path(__file__).parent
c = json.loads((p/'fixture.json').read_text())
with (p/'calls').open('a') as f: f.write(json.dumps(sys.argv[1:])+'\\n')
state = p/'state'
if 'workspace' in sys.argv:
    action = sys.argv[sys.argv.index('workspace')+1]
    if action == 'start':
        state.write_text('running')
    elif action == 'pause':
        state.write_text(c.get('pause_state', 'stopped'))
        c['show_failures'] = c.get('show_failures_after_pause', 0)
        (p/'fixture.json').write_text(json.dumps(c))
    else:
        if c.get('show_payloads'):
            payload = c['show_payloads'].pop(0)
            (p/'fixture.json').write_text(json.dumps(c))
            print(payload)
            sys.exit(0)
        if c.get('show_failures', 0):
            c['show_failures'] -= 1
            (p/'fixture.json').write_text(json.dumps(c))
            sys.exit(6)
        w = dict(slug=c['workspace'], name='fixture', resourceSpecSlug='fixture-a100',
                 state=state.read_text(), phase='ready')
        print(json.dumps(dict(workspace=w, endpoints=[dict(name='jupyter',
          httpUrl='https://jupyter-wsp-nvxqw1jf9zdv.betelgeuse.cloud.vessl.ai/?token=SECRET')])) )
else:
    action = pathlib.Path(sys.argv[-1]).name
    if action == 'start.py':
        (p/'started').write_text('yes')
        if c.get('ambiguous'): sys.exit(4)
    elif c.get('hang') and (p/'started').exists():
        time.sleep(60)
    elif c.get('bad_probe'):
        print('PANNS_STATUS {}')
    elif ((p/'started').exists() or c.get('fail_before_start')) and c.get('probe_failures_after_start', 0):
        c['probe_failures_after_start'] -= 1
        (p/'fixture.json').write_text(json.dumps(c))
        if c.get('probe_failure_kind') == 'transient':
            print('TypeError: fetch failed; https://fixture/?token=SECRET', file=sys.stderr)
        elif c.get('probe_failure_kind') == 'endpoint_unavailable':
            print('PANNS_TRANSPORT_ERROR '+json.dumps(dict(phase='workspace_identity', kind='endpoint_unavailable', code=None, http_status=None)), file=sys.stderr)
        elif c.get('probe_failure_kind') == 'domain':
            print(json.dumps(dict(error='ValueError', value='SECRET identity mismatch')))
            print('TypeError: fetch failed; token=SECRET', file=sys.stderr)
        elif c.get('probe_failure_kind') == 'output_limit':
            sys.stderr.write('SECRET'*12000)
        elif c.get('probe_failure_kind') == 'auth':
            print('Error: Kernel HTTP 403; token=SECRET', file=sys.stderr)
        sys.exit(6)
    else:
        started = (p/'started').exists()
        print('PANNS_STATUS '+json.dumps({**c['contract'],
            'state': 'finished' if started else 'not_started',
            'exit_code': c.get('exit_code', 0) if started else None}))
'''


def setup_request(tmp_path, **changes):
    child = tmp_path/'child.py'
    child.write_text(CHILD)
    child.chmod(0o700)
    start, probe = tmp_path/'start.py', tmp_path/'probe.py'
    start.write_text('# reviewed fixture start\n')
    probe.write_text('# reviewed fixture probe\n')
    request = dict(cli=[str(child), '--org', 'qwdkw'], workspace=watch.WORKSPACE,
        name='fixture', resource_spec_slug='fixture-a100', pause_deadline_epoch=time.time()+90,
        run_id='fixture-run', bundle_sha256='a'*64, transport=[sys.executable, str(child)],
        transport_sha256=hashlib.sha256(child.read_bytes()).hexdigest(),
        start_code=str(start), start_code_sha256=hashlib.sha256(start.read_bytes()).hexdigest(),
        probe_code=str(probe), probe_code_sha256=hashlib.sha256(probe.read_bytes()).hexdigest())
    contract = dict(org='qwdkw', workspace=watch.WORKSPACE, deadline_epoch=request['pause_deadline_epoch'],
        run_id=request['run_id'], bundle_sha256=request['bundle_sha256'], stop_owner='local_panns_workspace_watch')
    (tmp_path/'fixture.json').write_text(json.dumps(dict(workspace=watch.WORKSPACE, contract=contract, **changes)))
    (tmp_path/'state').write_text('stopped')
    path = tmp_path/'request.json'
    path.write_text(json.dumps(request))
    return path, tmp_path/'output'


def calls(tmp_path):
    return [json.loads(line) for line in (tmp_path/'calls').read_text().splitlines()]


@pytest.mark.parametrize('scenario,category', [('success', None), ('cleanup', None), ('cleanup_http', None), ('identity', 'remote_error'), ('cli', 'child_exit'),
    ('missing_endpoint', 'endpoint_unavailable'), ('missing_token', 'endpoint_unavailable'),
    ('wrong_slug_empty', 'remote_error'), ('wrong_host_no_token', 'remote_error'), ('duplicates', 'remote_error'),
    ('post', 'kernel_http_503'), ('execute', 'websocket_failed'), ('domain', 'remote_error'), ('foreign_reply', 'transport_timeout')])
def test_versioned_transport_classifies_failures_and_preserves_verified_reply(tmp_path, scenario, category):
    request_path, output = setup_request(tmp_path)
    request_data = json.loads(request_path.read_text())
    status = dict(watch.read_request(request_path).contract(), state='finished', exit_code=0)
    transport = Path(watch.__file__).with_name('panns_jupyter_transport.mjs')
    wrapper = tmp_path/'transport_fixture.mjs'
    wrapper.write_text('const scenario = '+json.dumps(scenario)+'; const status = '+json.dumps(status)+';\n'+'''
import child from 'node:child_process';
import {syncBuiltinESMExports} from 'node:module';
child.execFileSync = () => {
  if (scenario === 'cli') throw Object.assign(new Error('SECRET'), {code:'ENOENT'});
  const endpoint = {name:'jupyter', httpUrl:scenario === 'wrong_host_no_token' ? 'https://untrusted.invalid/' :
    'https://jupyter-wsp-nvxqw1jf9zdv.betelgeuse.cloud.vessl.ai/' + (scenario === 'missing_token' ? '' : '?token=SECRET')};
  return JSON.stringify({workspace:{slug:['identity','wrong_slug_empty'].includes(scenario) ? 'wsp-other' : 'wsp-nvxqw1jf9zdv'},
    endpoints:['missing_endpoint','wrong_slug_empty'].includes(scenario) ? [] : scenario === 'duplicates' ? [endpoint,endpoint] : [endpoint]});
};
syncBuiltinESMExports();
const realTimeout = setTimeout;
globalThis.setTimeout = (callback, delay) => realTimeout(callback, delay === 40000 ? 40 : delay);
globalThis.fetch = async (_, options) => {
  if (options.method === 'DELETE') {
    if (scenario === 'cleanup') throw Object.assign(new TypeError('fetch failed'), {cause:{code:'ECONNRESET',secret:'SECRET'}});
    if (scenario === 'cleanup_http') return {ok:false,status:503};
    return {ok:true,status:204};
  }
  if (scenario === 'post') return {ok:false,status:503};
  return {ok:true,json:async()=>({id:'fixture-kernel'})};
};
globalThis.WebSocket = class {
  constructor() { setImmediate(()=>this.onopen()); }
  send(payload) {
    if (scenario === 'execute') { this.onerror(); return; }
    const request = JSON.parse(payload);
    const parent_header = {msg_id:scenario === 'foreign_reply' ? 'foreign' : request.header.msg_id};
    const send = (channel,msg_type,content) => this.onmessage({data:JSON.stringify({channel,header:{msg_type},parent_header,content})});
    if (scenario === 'domain') send('iopub','error',{ename:'ValueError',evalue:'SECRET'});
    else send('iopub','stream',{text:'PANNS_STATUS '+JSON.stringify(status)+'\\n'});
    send('shell','execute_reply',{status:scenario === 'domain' ? 'error' : 'ok'});
    send('iopub','status',{execution_state:'idle'});
  }
  close() {}
};
await import('''+json.dumps(transport.as_uri())+');\n')
    request_data.update(transport=['/opt/homebrew/bin/node', str(wrapper)],
        transport_sha256=hashlib.sha256(wrapper.read_bytes()).hexdigest())
    request_path.write_text(json.dumps(request_data))
    output.mkdir()
    controller = watch.Watch(watch.read_request(request_path), output, [])
    if category is None:
        assert controller.probe(timeout=5) == status
    else:
        with pytest.raises(watch.ChildOperationError) as failure:
            controller.call(controller.request.transport+[controller.request.probe_code], timeout=5)
        assert failure.value.category == category
    diagnostics = ''.join(path.read_text() for path in output.glob('*.json'))
    assert 'SECRET' not in diagnostics and 'https://' not in diagnostics
    if scenario != 'success':
        record = json.loads((output/'transport_diagnostic.json').read_text())['records'][0]
        assert record['phase'] == {'cli':'workspace_lookup','post':'kernel_create',
            'execute':'execute','domain':'execute','foreign_reply':'execute','cleanup':'kernel_cleanup',
            'cleanup_http':'kernel_cleanup','identity':'workspace_identity',
            'wrong_slug_empty':'workspace_identity','wrong_host_no_token':'workspace_identity','duplicates':'workspace_identity',
            'missing_endpoint':'workspace_identity','missing_token':'workspace_identity'}[scenario]
        assert record['event'] == ('PANNS_TRANSPORT_WARNING' if scenario.startswith('cleanup') else 'PANNS_TRANSPORT_ERROR')


@pytest.mark.parametrize('changes', [{}, {'ambiguous': True}, {'exit_code': 9}, {'bad_probe': True}])
def test_compute_outcome_and_ambiguous_start_always_pause_owned_target(tmp_path, changes):
    # Given: the CLI and transport are real local children implementing only the protocol.
    request, output = setup_request(tmp_path, **changes)
    # When: start succeeds, becomes ambiguous, fails compute, or returns foreign status.
    result = watch.run(request, output)
    # Then: start never duplicates, protocol-reported state is stopped, failures stay failures.
    events = calls(tmp_path)
    assert sum(event[-1].endswith('/start.py') for event in events) <= 1
    assert sum('pause' in event for event in events) == 1
    receipt = json.loads((output/'local_receipt.json').read_text())
    assert receipt['pause_verified'] is True
    assert (result == 0) is not bool(changes.get('exit_code') or changes.get('bad_probe'))
    assert 'SECRET' not in ''.join(path.read_text() for path in output.glob('*.json'))


def test_running_unowned_target_never_mutates(tmp_path):
    # Given: a running target belongs to someone else.
    request, output = setup_request(tmp_path)
    (tmp_path/'state').write_text('running')
    # When: a fresh controller inspects it.
    assert watch.run(request, output) != 0
    # Then: the controller neither adopts nor pauses it.
    assert all('show' in event for event in calls(tmp_path))
    assert json.loads((output/'local_receipt.json').read_text())['owned_target'] is False


def test_repeated_signals_interrupt_child_but_allow_verified_pause(tmp_path):
    # Given: a transport blocks after start, and repeated signals arrive during cleanup.
    request, output = setup_request(tmp_path, hang=True)
    process = subprocess.Popen([sys.executable, str(Path(watch.__file__)), '--request', str(request),
        '--output', str(output)], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        end = time.monotonic()+5
        while not (tmp_path/'started').exists() and time.monotonic() < end:
            time.sleep(.02)
        assert (tmp_path/'started').exists()
        # When: the controller receives TERM then HUP.
        os.kill(process.pid, signal.SIGTERM)
        time.sleep(.02)
        if process.poll() is None:
            os.kill(process.pid, signal.SIGHUP)
        assert process.wait(timeout=5) != 0
        # Then: its blocking child is killed and the exact workspace is verified stopped.
        assert json.loads((output/'pause_receipt.json').read_text())['unverified'] is False
        assert (tmp_path/'state').read_text() == 'stopped'
    finally:
        if process.poll() is None:
            process.kill()
            process.wait(timeout=1)


@pytest.mark.parametrize('failure', ['payload_hash', 'past_deadline', 'resume_deadline', 'lock'])
def test_rejected_request_cannot_mutate_or_overwrite_another_controller(tmp_path, failure):
    request, output = setup_request(tmp_path)
    output.mkdir()
    original = '{"protected_receipt": true}\n'
    (output/'local_receipt.json').write_text(original)
    data = json.loads(request.read_text())
    if failure == 'payload_hash':
        (tmp_path/'start.py').write_text('changed code')
    elif failure == 'past_deadline':
        data['pause_deadline_epoch'] = time.time()-1
        request.write_text(json.dumps(data))
    elif failure == 'resume_deadline':
        (output/'controller_contract.json').write_text('{}')
    with (output/'controller.lock').open('a') as lock:
        if failure == 'lock':
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        assert watch.run(request, output) != 0
    assert not (tmp_path/'calls').exists()
    if failure == 'lock':
        assert (output/'local_receipt.json').read_text() == original


@pytest.mark.parametrize('show_failures', [1, 2, 3])
def test_owned_resume_retries_one_show_failure_then_cleans_up(tmp_path, show_failures):
    request, output = setup_request(tmp_path)
    output.mkdir()
    controller = watch.Watch(watch.read_request(request), output, [])
    assert controller.execute() == 0
    config = json.loads((tmp_path/'fixture.json').read_text())
    config['show_failures'] = show_failures
    (tmp_path/'fixture.json').write_text(json.dumps(config))
    assert (watch.run(request, output) == 0) is (show_failures == 1)
    assert sum('pause' in event for event in calls(tmp_path)) == 1
    assert sum('start' in event for event in calls(tmp_path)) == 1


@pytest.mark.parametrize('changes', [dict(probe_failures_after_start=1),
                                    dict(show_failures_after_pause=3)])
def test_transient_status_failure_preserves_single_start_and_verified_pause(tmp_path, changes):
    request, output = setup_request(tmp_path, **changes)
    original_request = request.read_bytes()
    assert watch.run(request, output) == 0
    events = calls(tmp_path)
    assert sum(event[-1].endswith('/start.py') for event in events) == 1
    assert sum('start' in event for event in events) == 1
    assert sum('pause' in event for event in events) == 1
    receipt = json.loads((output/'local_receipt.json').read_text())
    assert receipt['compute_exit_code'] == 0 and receipt['pause_verified'] is True
    assert request.read_bytes() == original_request


@pytest.mark.parametrize('kind,failures,success', [('transient', 3, True), ('transient', 100, False),
                                                ('endpoint_unavailable', 3, True), ('endpoint_unavailable', 100, False),
                                                ('domain', 3, False), ('output_limit', 3, False)])
def test_probe_outage_recovery_is_bounded_and_never_exposes_child_diagnostics(tmp_path, monkeypatch, kind, failures, success):
    request, output = setup_request(tmp_path, probe_failures_after_start=failures, probe_failure_kind=kind)
    original_request = request.read_bytes()
    monkeypatch.setattr(watch, 'PROBE_OUTAGE_SECONDS', .6, raising=False)
    monkeypatch.setattr(watch, 'PROBE_RETRY_SECONDS', .02, raising=False)
    began = time.monotonic()
    assert (watch.run(request, output) == 0) is success
    assert time.monotonic()-began < 3
    events = calls(tmp_path)
    assert sum(event[-1].endswith('/start.py') for event in events) == 1
    assert sum('pause' in event for event in events) == 1
    assert request.read_bytes() == original_request
    assert json.loads((output/'pause_receipt.json').read_text())['unverified'] is False
    if kind in ('domain', 'output_limit'):
        assert sum(event[-1].endswith('/probe.py') for event in events) == 2
    diagnostics = ''.join(path.read_text() for path in output.glob('*.json'))
    assert 'SECRET' not in diagnostics and 'https://' not in diagnostics
    if kind != 'output_limit':
        diagnostic = json.loads((output/'child_failure.json').read_text())
        assert diagnostic['category'] == {'transient':'fetch_failed', 'endpoint_unavailable':'endpoint_unavailable'}.get(kind, 'remote_error')
        assert diagnostic['exit_code'] == 6 and diagnostic['stderr_bytes'] > 0


@pytest.mark.parametrize('kind', ['domain', 'auth'])
def test_startup_rejects_remote_domain_or_authorization_failure_without_retrying(tmp_path, kind):
    request, output = setup_request(tmp_path, fail_before_start=True,
        probe_failures_after_start=3, probe_failure_kind=kind)
    assert watch.run(request, output) != 0
    events = calls(tmp_path)
    assert sum(event[-1].endswith('/probe.py') for event in events) == 1
    assert not any(event[-1].endswith('/start.py') for event in events)
    assert sum('pause' in event for event in events) == 1
    assert json.loads((output/'pause_receipt.json').read_text())['unverified'] is False


def test_deadline_kills_blocked_transport_and_stopping_is_unverified(tmp_path, monkeypatch):
    request, output = setup_request(tmp_path, hang=True, pause_state='stopping')
    data = json.loads(request.read_text())
    data['pause_deadline_epoch'] = time.time()+3.5
    request.write_text(json.dumps(data))
    config = json.loads((tmp_path/'fixture.json').read_text())
    config['contract']['deadline_epoch'] = data['pause_deadline_epoch']
    (tmp_path/'fixture.json').write_text(json.dumps(config))
    monkeypatch.setattr(watch, 'STOP_RESERVE', 1.5)
    start = time.monotonic()
    assert watch.run(request, output) != 0
    assert time.monotonic()-start < 4.5
    receipt = json.loads((output/'pause_receipt.json').read_text())
    assert receipt['state'] == 'stopping' and receipt['unverified'] is True
    assert sum('pause' in event for event in calls(tmp_path)) == 1


def test_expired_compute_deadline_still_pauses_owned_target_without_reauthorizing_work(tmp_path, monkeypatch):
    # Given: the real controller has acquired its pinned target before the wall clock jumps past expiry.
    request, output = setup_request(tmp_path)
    output.mkdir()
    controller = watch.Watch(watch.read_request(request), output, [])
    assert controller.execute() == 0
    original_deadline = controller.deadline
    monkeypatch.setattr(watch.time, 'time', lambda: original_deadline+1)
    assert controller.remaining() < 0
    # When: cleanup starts after the original compute deadline.
    assert controller.pause() is True
    # Then: a real local pause-protocol child runs, while start/probe remain forbidden in cleanup.
    assert sum('pause' in event for event in calls(tmp_path)) == 1
    assert controller.deadline == original_deadline
    assert controller.request.pause_deadline_epoch == original_deadline
    with pytest.raises(ValueError, match='Cleanup'):
        controller.call(controller.request.cli+['workspace', 'start', watch.WORKSPACE])
    with pytest.raises(ValueError, match='Cleanup'):
        controller.probe()


@pytest.mark.parametrize('payload', ['{', '{}', '{"workspace":null}', '[]'])
def test_malformed_cleanup_inspection_still_pauses_previously_owned_target(tmp_path, payload):
    # Given: prior target ownership is valid but the next inspection response is malformed.
    request, output = setup_request(tmp_path)
    output.mkdir()
    controller = watch.Watch(watch.read_request(request), output, [])
    assert controller.execute() == 0
    config = json.loads((tmp_path/'fixture.json').read_text())
    config['show_payloads'] = [payload]
    (tmp_path/'fixture.json').write_text(json.dumps(config))
    # When: the actual cleanup calls the local protocol child and cannot parse that response.
    assert controller.pause() is True
    # Then: the pinned pause request still executes and a subsequent valid response proves stopped.
    assert sum('pause' in event for event in calls(tmp_path)) == 1
    assert json.loads((output/'pause_receipt.json').read_text())['unverified'] is False


def test_actual_cleanup_identity_mismatch_still_refuses_pause(tmp_path):
    # Given: an inspection reports a different workspace after prior ownership was acquired.
    request, output = setup_request(tmp_path)
    output.mkdir()
    controller = watch.Watch(watch.read_request(request), output, [])
    assert controller.execute() == 0
    config = json.loads((tmp_path/'fixture.json').read_text())
    config['show_payloads'] = [json.dumps(dict(workspace=dict(slug='wsp-other',
        name='fixture', resourceSpecSlug='fixture-a100', state='running')))]
    (tmp_path/'fixture.json').write_text(json.dumps(config))
    # When: cleanup receives a parseable identity contradiction.
    assert controller.pause() is False
    # Then: malformed-response recovery must never bypass a real target mismatch.
    assert not any('pause' in event for event in calls(tmp_path))
    assert json.loads((output/'pause_receipt.json').read_text())['unverified'] is True


def test_monitor_sleep_rechecks_compute_deadline_after_wall_clock_jump(tmp_path, monkeypatch):
    # Given: the deadline expires on the first monitor sleep tick.
    request, output = setup_request(tmp_path)
    controller = watch.Watch(watch.read_request(request), output, [])
    real_sleep = time.sleep
    def jump_during_sleep(seconds):
        monkeypatch.setattr(watch.time, 'time', lambda: controller.deadline+1)
        real_sleep(min(seconds, .01))
    monkeypatch.setattr(watch.time, 'sleep', jump_during_sleep)
    # When: the monitor waits for a nominal two-second interval.
    started = time.monotonic()
    controller.sleep(2.)
    # Then: it notices expiry on the next tick rather than waiting the whole polling interval.
    assert time.monotonic()-started < .2


@pytest.mark.parametrize('failure', [None, 'payload_changed', 'request_changed', 'ownership_changed',
                                  'released', 'released_before_final_receipt'])
def test_expired_cli_restart_only_recovers_unreleased_pinned_ownership(tmp_path, monkeypatch, failure):
    # Given: an actual controller acquired this workspace before its original deadline.
    request, output = setup_request(tmp_path)
    data = json.loads(request.read_text())
    data['pause_deadline_epoch'] = time.time()+3.5
    request.write_text(json.dumps(data))
    config = json.loads((tmp_path/'fixture.json').read_text())
    config['contract']['deadline_epoch'] = data['pause_deadline_epoch']
    (tmp_path/'fixture.json').write_text(json.dumps(config))
    with monkeypatch.context() as scaled:
        scaled.setattr(watch, 'STOP_RESERVE', 1.5)
        if failure in ('released', 'released_before_final_receipt'):
            assert watch.run(request, output) == 0
            (tmp_path/'state').write_text('running')
            if failure == 'released_before_final_receipt':
                (output/'local_receipt.json').unlink()
        else:
            output.mkdir()
            controller = watch.Watch(watch.read_request(request), output, [])
            assert controller.execute() == 0
    if failure == 'payload_changed':
        (tmp_path/'start.py').write_text('changed after ownership was acquired')
    elif failure == 'request_changed':
        data['name'] = 'changed request'
        request.write_text(json.dumps(data))
    elif failure == 'ownership_changed':
        (output/'owned_target.json').write_text('{}')
    original_request = request.read_bytes()
    original_pin = (output/'controller_contract.json').read_bytes()
    before = len(calls(tmp_path))
    time.sleep(max(0., data['pause_deadline_epoch']-time.time())+.05)
    # When: a fresh CLI process loads the genuinely expired request, twice after rejection.
    for _ in range(2 if failure else 1):
        result = subprocess.run([sys.executable, str(Path(watch.__file__)), '--request', str(request),
            '--output', str(output)], capture_output=True, text=True, timeout=5)
        assert result.returncode == 2
    # Then: only the still-owned run can inspect/pause; no expired run starts or probes compute.
    events = calls(tmp_path)[before:]
    assert request.read_bytes() == original_request
    assert (output/'controller_contract.json').read_bytes() == original_pin
    if failure:
        assert events == []
        assert (tmp_path/'state').read_text() == 'running'
    else:
        assert sum('pause' in event for event in events) == 1
        assert all('show' in event or 'pause' in event for event in events)
        receipt = json.loads((output/'local_receipt.json').read_text())
        assert receipt['pause_verified'] is True
        assert receipt['recovery_only'] is True
        assert receipt['compute_exit_code'] is None
        assert (tmp_path/'state').read_text() == 'stopped'


@pytest.mark.parametrize('missing_marker', [False, True])
def test_released_controller_cannot_adopt_later_job_before_deadline(tmp_path, missing_marker):
    request, output = setup_request(tmp_path)
    assert watch.run(request, output) == 0
    state = 'stopped' if missing_marker else 'running'
    (tmp_path/'state').write_text(state)
    if missing_marker:
        (output/'owned_target.json').unlink()
    before = len(calls(tmp_path))
    original_receipt = (output/'local_receipt.json').read_bytes()
    for _ in range(2):
        assert watch.run(request, output) != 0
    assert calls(tmp_path)[before:] == []
    assert (tmp_path/'state').read_text() == state
    assert (output/'local_receipt.json').read_bytes() == original_receipt
