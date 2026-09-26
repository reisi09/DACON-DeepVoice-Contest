"""Own one existing workspace through a pinned run and verified deadline pause."""
from __future__ import annotations

import argparse
import contextlib
from dataclasses import dataclass
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import re
import signal
import subprocess
import time
from urllib.parse import urlparse

WORKSPACE = 'wsp-nvxqw1jf9zdv'
STOP_RESERVE = 30.
POLL_SECONDS = 300.


@dataclass(frozen=True)
class Request:
    cli: list[str]
    workspace: str
    name: str
    resource_spec_slug: str
    pause_deadline_epoch: float
    run_id: str
    bundle_sha256: str
    transport: list[str]
    transport_sha256: str
    start_code: str
    start_code_sha256: str
    probe_code: str
    probe_code_sha256: str

    def contract(self) -> dict:
        return dict(org='qwdkw', workspace=self.workspace, deadline_epoch=self.pause_deadline_epoch,
                    run_id=self.run_id, bundle_sha256=self.bundle_sha256, stop_owner='local_panns_workspace_watch')

    def pin(self) -> dict:
        return {**self.contract(), 'request_sha256': hashlib.sha256(
            json.dumps(self.__dict__, sort_keys=True).encode()).hexdigest()}


def read_request(path: Path, *, allow_expired: bool = False) -> Request:
    request = Request(**json.loads(path.read_text()))
    if (type(request.pause_deadline_epoch) not in (int, float)
            or not math.isfinite(request.pause_deadline_epoch)
            or request.pause_deadline_epoch <= 0
            or request.pause_deadline_epoch-time.time() > 72*3600
            or (not allow_expired and request.pause_deadline_epoch <= time.time())):
        raise ValueError('Invalid original deadline')
    if (request.workspace != WORKSPACE or not isinstance(request.cli, list)
            or len(request.cli) != 3 or request.cli[1:] != ['--org', 'qwdkw']
            or not isinstance(request.transport, list) or len(request.transport) != 2
            or not all(isinstance(value, str) and value for value in [request.name, request.resource_spec_slug, request.run_id])):
        raise ValueError('Invalid target or command contract')
    for value in [*request.cli[:1], *request.transport, request.start_code, request.probe_code]:
        if not isinstance(value, str) or not Path(value).is_absolute() or not Path(value).is_file():
            raise ValueError('Command and payload paths must be absolute files')
    for digest in (request.bundle_sha256, request.transport_sha256, request.start_code_sha256, request.probe_code_sha256):
        if not isinstance(digest, str) or not re.fullmatch('[0-9a-f]{64}', digest):
            raise ValueError('Invalid pinned hash')
    for filename, digest in ((request.transport[1], request.transport_sha256),
                             (request.start_code, request.start_code_sha256),
                             (request.probe_code, request.probe_code_sha256)):
        if hashlib.sha256(Path(filename).read_bytes()).hexdigest() != digest:
            raise ValueError('Pinned code changed')
    return request


class Watch:
    def __init__(self, request: Request, output: Path, stopped: list[int]):
        self.request, self.output, self.stopped = request, output, stopped
        self.deadline = request.pause_deadline_epoch
        self.monotonic_deadline = time.monotonic()+self.deadline-time.time()
        self.cleaning = False
        self.owned = False
        self.latest = {}

    def remaining(self) -> float:
        if self.cleaning:
            return self.cleanup_deadline-time.monotonic()
        return min(self.deadline-time.time(), self.monotonic_deadline-time.monotonic())-STOP_RESERVE

    def save(self, name: str, value: dict) -> None:
        temporary = self.output/(name+'.tmp')
        temporary.write_text(json.dumps(value, indent=2)+'\n')
        temporary.replace(self.output/name)

    def call(self, argv: list[str], timeout: float = 20.) -> str:
        if self.cleaning and argv not in (
                self.request.cli+['workspace', 'show', WORKSPACE, '-o', 'json'],
                self.request.cli+['workspace', 'pause', WORKSPACE]):
            raise ValueError('Cleanup permits only pinned workspace inspection and pause')
        end = time.monotonic()+min(timeout, self.remaining()-1.)
        if end <= time.monotonic() or (self.stopped and not self.cleaning):
            raise TimeoutError('Deadline or termination signal')
        process = subprocess.Popen(argv, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, start_new_session=True)
        try:
            while True:
                remaining = min(end-time.monotonic(), self.remaining())
                if remaining <= 0 or (self.stopped and not self.cleaning):
                    raise TimeoutError('Deadline or termination signal')
                try:
                    stdout, _ = process.communicate(timeout=min(.1, remaining))
                    if process.returncode or len(stdout) > 1024*1024:
                        raise RuntimeError('Child operation failed or exceeded output limit')
                    return stdout.decode('utf-8')
                except subprocess.TimeoutExpired as error:
                    if error.output and len(error.output) > 1024*1024:
                        raise RuntimeError('Child output limit exceeded') from None
        finally:
            if process.poll() is None:
                with contextlib.suppress(ProcessLookupError):
                    os.killpg(process.pid, signal.SIGKILL)
            try:
                process.wait(timeout=1.)
            except subprocess.TimeoutExpired:
                print('OWNED_CHILD_REAP_UNVERIFIED', flush=True)
            if process.stdout:
                process.stdout.close()

    def read_status(self, argv: list[str], timeout: float) -> str:
        if argv not in (self.request.cli+['workspace', 'show', WORKSPACE, '-o', 'json'],
                        self.request.transport+[self.request.probe_code]):
            raise ValueError('Status retries permit only pinned read operations')
        try:
            return self.call(argv, timeout)
        except (RuntimeError, TimeoutError, OSError):
            if self.remaining() <= 1. or (self.stopped and not self.cleaning):
                raise
            print('READ_ONLY_STATUS_RETRY', flush=True)
            self.sleep(.5)
        return self.call(argv, timeout)

    def show(self, timeout: float = 5.) -> dict:
        detail = json.loads(self.read_status(self.request.cli+['workspace', 'show', WORKSPACE, '-o', 'json'], timeout))
        workspace = detail['workspace']
        if (workspace['slug'] != WORKSPACE or workspace['name'] != self.request.name
                or workspace['resourceSpecSlug'] != self.request.resource_spec_slug):
            raise ValueError('Workspace identity changed')
        self.latest = {key: workspace.get(key) for key in ('slug', 'name', 'state', 'phase', 'resourceSpecSlug')}
        self.save('workspace_latest.json', self.latest)
        return detail

    def probe(self, timeout: float = 75.) -> dict:
        result = self.read_status(self.request.transport+[self.request.probe_code], timeout)
        lines = [line.removeprefix('PANNS_STATUS ') for line in result.splitlines() if line.startswith('PANNS_STATUS ')]
        if len(lines) != 1:
            raise ValueError('Expected one remote status record')
        status = json.loads(lines[0])
        expected = self.request.contract()
        if (set(status) != set(expected) | {'state', 'exit_code'}
                or any(status[key] != value for key, value in expected.items())
                or status['state'] not in ('not_started', 'running', 'finished')
                or (status['state'] == 'finished' and type(status['exit_code']) is not int)
                or (status['state'] != 'finished' and status['exit_code'] is not None)):
            raise ValueError('Remote status contract mismatch')
        self.save('remote_status.json', status)
        return status

    def sleep(self, seconds: float) -> None:
        end = time.monotonic()+min(seconds, max(0., self.remaining()))
        while time.monotonic() < end and self.remaining() > 0 and (self.cleaning or not self.stopped):
            time.sleep(min(.1, max(0., end-time.monotonic())))

    def execute(self) -> int:
        contract = self.request.contract()
        pin = self.request.pin()
        saved = self.output/'controller_contract.json'
        resume = saved.exists()
        if resume and json.loads(saved.read_text()) != pin:
            raise ValueError('Resume must retain the original pinned request')
        self.save('controller_contract.json', pin)
        marker = self.output/'owned_target.json'
        if resume and marker.exists():
            self.recover_owned()
        else:
            self.require_unreleased()
        idle = self.show()['workspace']['state'] in ('stopped', 'standby')
        if not idle and not self.owned:
            raise ValueError('Refuse to adopt an unowned active workspace')
        self.save('owned_target.json', contract)
        self.owned = True
        print(json.dumps(contract), flush=True)
        startup_end = time.monotonic()+min(600., self.remaining())
        if idle:
            self.call(self.request.cli+['workspace', 'start', WORKSPACE], min(20., startup_end-time.monotonic()))
        while True:
            detail = self.show(min(5., startup_end-time.monotonic()))
            endpoints = [item for item in detail.get('endpoints', []) if item.get('name') == 'jupyter']
            if endpoints:
                endpoint = urlparse(endpoints[0]['httpUrl'])
                if (len(endpoints) != 1 or endpoint.scheme != 'https'
                        or endpoint.hostname != 'jupyter-'+WORKSPACE+'.betelgeuse.cloud.vessl.ai'
                        or endpoint.username or endpoint.password or endpoint.port not in (None, 443)):
                    raise ValueError('Jupyter endpoint identity mismatch')
                if detail['workspace']['state'] == 'running':
                    try:
                        status = self.probe(min(75., startup_end-time.monotonic()))
                        break
                    except (RuntimeError, TimeoutError):
                        print('JUPYTER_NOT_READY', flush=True)
            if time.monotonic() >= startup_end:
                raise TimeoutError('Startup readiness expired')
            self.sleep(min(5., startup_end-time.monotonic()))
        attempted = self.output/'remote_start_attempted.json'
        if status['state'] == 'not_started':
            if attempted.exists():
                raise RuntimeError('Previous remote start outcome is unresolved')
            self.save('remote_start_attempted.json', contract)
            try:
                self.call(self.request.transport+[self.request.start_code], 75.)
            except (RuntimeError, TimeoutError, OSError):
                status = self.probe()
                if status['state'] == 'not_started':
                    raise RuntimeError('Ambiguous remote start was not adopted') from None
            else:
                status = self.probe()
        while status['state'] == 'running':
            self.sleep(POLL_SECONDS)
            status = self.probe()
        if status['state'] != 'finished':
            raise RuntimeError('Remote start did not produce a running or finished job')
        return status['exit_code']

    def recover_owned(self) -> None:
        if (json.loads((self.output/'controller_contract.json').read_text()) != self.request.pin()
                or json.loads((self.output/'owned_target.json').read_text()) != self.request.contract()):
            raise ValueError('Recovery requires the original pinned ownership')
        self.require_unreleased()
        self.owned = True

    def require_unreleased(self) -> None:
        for name in ('local_receipt.json', 'pause_receipt.json'):
            path = self.output/name
            if path.exists():
                receipt = json.loads(path.read_text())
                if (receipt.get('pause_verified') is True or (receipt.get('unverified') is False
                        and receipt.get('state') in ('stopped', 'standby'))):
                    raise ValueError('Verified stop already released this controller ownership')

    def pause(self) -> bool:
        if not self.owned:
            raise ValueError('Pause requires pinned workspace ownership')
        if not self.cleaning:
            self.cleanup_deadline = time.monotonic()+STOP_RESERVE
            self.cleaning = True
        verified = False
        try:
            try:
                self.show()
            except (json.JSONDecodeError, KeyError, TypeError, AttributeError, RuntimeError, TimeoutError, OSError):
                self.latest = {}
                print('FRESH_IDENTITY_UNAVAILABLE_USING_PINNED_OWNERSHIP', flush=True)
            if self.latest.get('state') not in ('stopped', 'standby'):
                try:
                    self.call(self.request.cli+['workspace', 'pause', WORKSPACE], 10.)
                except (RuntimeError, TimeoutError, OSError):
                    print('PAUSE_REQUEST_OUTCOME_UNVERIFIED', flush=True)
                while self.remaining() > 0:
                    try:
                        self.show()
                    except (json.JSONDecodeError, KeyError, TypeError, AttributeError, RuntimeError, TimeoutError, OSError):
                        print('PAUSE_VERIFICATION_RETRY', flush=True)
                    if self.latest.get('state') in ('stopped', 'standby'):
                        break
                    self.sleep(.5)
            verified = self.latest.get('state') in ('stopped', 'standby')
        except (KeyError, TypeError, ValueError, RuntimeError, TimeoutError, OSError):
            print('PAUSE_REQUIRES_IMMEDIATE_ATTENTION', flush=True)
        finally:
            self.save('pause_receipt.json', {**self.latest, 'state': self.latest.get('state', 'unknown'), 'unverified': not verified})
        return verified


def run(request_path: Path, output: Path) -> int:
    output.mkdir(parents=True, exist_ok=True)
    stopped: list[int] = []
    previous = {sig: signal.signal(sig, lambda number, _: stopped.append(number))
                for sig in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP)}
    controller = None
    code, verified, recovery_only = 2, False, False
    with (output/'controller.lock').open('a') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            for sig, handler in previous.items():
                signal.signal(sig, handler)
            print('CONTROLLER_ALREADY_OWNS_OUTPUT', flush=True)
            return 2
        try:
            controller = Watch(read_request(request_path, allow_expired=True), output, stopped)
            if controller.deadline <= time.time():
                recovery_only, code = True, None
                controller.recover_owned()
                print('EXPIRED_OWNED_RUN_CLEANUP_ONLY', flush=True)
            else:
                code = controller.execute()
        except (KeyError, AttributeError, TypeError, ValueError, RuntimeError, TimeoutError, OSError) as error:
            (output/'controller_error.json').write_text(json.dumps({'type': type(error).__name__,
                'message': str(error) if isinstance(error, (ValueError, RuntimeError, TimeoutError)) else 'Controller operation failed'})+'\n')
        finally:
            if controller and controller.owned:
                verified = controller.pause()
            receipt = {'owned_target': bool(controller and controller.owned), 'pause_verified': verified,
                       'unverified': not verified, 'compute_exit_code': code, 'termination_signals': stopped,
                       'workspace_state': controller.latest if controller else {}, 'recovery_only': recovery_only}
            if receipt['owned_target'] or not (output/'local_receipt.json').exists():
                (output/'local_receipt.json').write_text(json.dumps(receipt)+'\n')
            for sig, handler in previous.items():
                signal.signal(sig, handler)
    return 0 if code == 0 and verified and not stopped else 2


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for argument in ('--request', '--output'):
        parser.add_argument(argument, required=True, type=Path)
    args = parser.parse_args()
    raise SystemExit(run(args.request, args.output))
