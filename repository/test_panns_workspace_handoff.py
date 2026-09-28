"""Authenticated handoff boundary with real local controller/child processes, not VESSL."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

import pytest

import panns_workspace_watch as watch
from test_panns_workspace_watch import calls, setup_request


@pytest.mark.parametrize('failure', [None, 'wrong_workspace', 'live_predecessor', 'changed_receipt', 'wrong_finished'])
def test_active_handoff_requires_authenticated_dead_predecessor(tmp_path, failure):
    prior = tmp_path/'prior'
    prior.mkdir()
    old_request, old_output = setup_request(prior)
    child = prior/'child.py'
    child.write_text(child.read_text().replace("'state': 'finished' if started", "'state': 'running' if started")
        .replace("c.get('exit_code', 0) if started else None", 'None'))
    old_data = json.loads(old_request.read_text())
    old_data['transport_sha256'] = hashlib.sha256(child.read_bytes()).hexdigest()
    old_request.write_text(json.dumps(old_data))
    old_script = prior/'panns_workspace_watch.py'
    old_script.write_bytes(Path(watch.__file__).read_bytes())
    command = [sys.executable, str(old_script), '--request', str(old_request), '--output', str(old_output)]
    start = time.ctime()
    process = subprocess.Popen(command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        until = time.monotonic()+5
        status_path = old_output/'remote_status.json'
        while time.monotonic() < until:
            if status_path.exists() and json.loads(status_path.read_text())['state'] == 'running':
                break
            time.sleep(.02)
        assert status_path.exists() and json.loads(status_path.read_text())['state'] == 'running'
        predecessor = dict(pid=process.pid, start=start, command=' '.join(command))
        if failure != 'live_predecessor':
            process.kill()
            process.wait(timeout=2)
        old = watch.read_request(old_request)
        assert old.pin()['request_sha256'] == hashlib.sha256(json.dumps(old_data, sort_keys=True).encode()).hexdigest()
        remote_path = prior/'remote_request.json'
        remote_path.write_text(json.dumps(dict(contract=old.contract(), sha256=dict(notebook='c'*64))))
        remote_sha = hashlib.sha256(remote_path.read_bytes()).hexdigest()
        finished = dict(contract=old.contract(), request_sha256=remote_sha, notebook_sha256='c'*64, exit_code=1)
        receipt = dict(version=1, predecessor_request=str(old_request),
            predecessor_request_sha256=hashlib.sha256(old_request.read_bytes()).hexdigest(),
            predecessor_output=str(old_output), predecessor_watch=predecessor,
            remote_request_path=str(remote_path), remote_request_sha256=remote_sha, remote_finished=finished)
        current = tmp_path/'current'
        current.mkdir()
        request, output = setup_request(current)
        data = json.loads(request.read_text())
        if failure == 'wrong_workspace':
            old_data['workspace'] = 'wsp-other'
            old_request.write_text(json.dumps(old_data))
            receipt['predecessor_request_sha256'] = hashlib.sha256(old_request.read_bytes()).hexdigest()
        if failure == 'wrong_finished':
            receipt['remote_finished']['contract'] = dict(old.contract(), run_id='foreign-job')
        handoff = current/'handoff.json'
        handoff.write_text(json.dumps(receipt))
        data.update(handoff_receipt=str(handoff), handoff_receipt_sha256=hashlib.sha256(handoff.read_bytes()).hexdigest())
        request.write_text(json.dumps(data))
        pinned_request = request.read_bytes()
        if failure == 'changed_receipt':
            handoff.write_text('{}')
        (current/'state').write_text('running')
        assert (watch.run(request, output) == 0) is (failure is None)
        assert request.read_bytes() == pinned_request
        events = calls(current) if (current/'calls').exists() else []
        if failure:
            assert all('show' in event for event in events)
            assert (current/'state').read_text() == 'running'
            assert json.loads((output/'local_receipt.json').read_text())['owned_target'] is False
        else:
            assert not any('start' in event for event in events)
            assert sum(event[-1].endswith('/start.py') for event in events) == 1
            assert sum('pause' in event for event in events) == 1
            assert json.loads((output/'local_receipt.json').read_text())['pause_verified'] is True
        assert json.loads((old_output/'controller_contract.json').read_text()) == old.pin()
        assert not (old_output/'local_receipt.json').exists()
    finally:
        if process.poll() is None:
            process.kill()
            process.wait(timeout=2)
