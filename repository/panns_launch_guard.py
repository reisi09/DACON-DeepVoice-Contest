"""Bound staging and compute before the offline environment is available."""
from __future__ import annotations

import contextlib
import json
import math
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time


def supervise(command: list[str], cutoff_epoch: float, grace_seconds: float = 60.,
              stop_requested: list[int] | None = None) -> tuple[int, str]:
    """Own one process group and preserve TERM checkpoint grace on timeout."""
    stopped = [] if stop_requested is None else stop_requested
    previous = {sig: signal.signal(sig, lambda number, _: stopped.append(number))
                for sig in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP)} if stop_requested is None else {}
    process = None
    monotonic_cutoff = time.monotonic() + max(0., cutoff_epoch - time.time())
    try:
        if stopped:
            return 130, 'interrupted'
        if time.time() >= cutoff_epoch:
            return 124, 'deadline_exhausted'
        process = subprocess.Popen(command, start_new_session=True)
        while not stopped:
            remaining = min(cutoff_epoch-time.time(), monotonic_cutoff-time.monotonic())
            if remaining <= 0:
                return 124, 'deadline_exhausted'
            try:
                code = process.wait(timeout=min(remaining, .2))
            except subprocess.TimeoutExpired:
                continue
            if stopped:
                return 130, 'interrupted'
            return (code, 'complete' if code == 0 else 'child_failed')
        return 130, 'interrupted'
    finally:
        if process is not None and process.poll() is None:
            with contextlib.suppress(ProcessLookupError):
                os.killpg(process.pid, signal.SIGTERM if grace_seconds > 0 else signal.SIGKILL)
            try:
                process.wait(timeout=grace_seconds if grace_seconds > 0 else 1.)
            except subprocess.TimeoutExpired:
                if grace_seconds > 0:
                    with contextlib.suppress(ProcessLookupError):
                        os.killpg(process.pid, signal.SIGKILL)
                    try:
                        process.wait(timeout=1.)
                    except subprocess.TimeoutExpired:
                        print('Owned child did not reap after SIGKILL; requesting workspace pause', file=sys.stderr)
                else:
                    print('Owned child did not reap after SIGKILL; requesting workspace pause', file=sys.stderr)
        for sig, handler in previous.items():
            signal.signal(sig, handler)


def metadata_phase() -> None:
    """Run potentially blocking shared-volume IO only in a supervised child."""
    phase, filename, payload = sys.argv[1:]
    path = Path(filename)
    path.parent.mkdir(parents=True, exist_ok=True)
    if phase == '--check-state':
        deadline = float(payload)
        controller = json.loads(os.environ['PANNS_CONTROLLER_CONTRACT']) if 'PANNS_CONTROLLER_CONTRACT' in os.environ else None
        try:
            with path.open('x') as stream:
                json.dump({'deadline_epoch': deadline, 'created_epoch': time.time(),
                           'controller_contract': controller}, stream)
        except FileExistsError:
            previous = json.loads(path.read_text())
            if previous['deadline_epoch'] != deadline or previous.get('controller_contract') != controller:
                raise ValueError('Resume must retain the original deadline and controller contract') from None
    else:
        path.write_text(payload+'\n')


def launch(stopped: list[int]) -> int:
    external = 'PANNS_CONTROLLER_CONTRACT' in os.environ
    pause = os.environ.get('PANNS_PAUSE_ON_EXIT', '1') == '1' and not external
    shared = Path(os.environ.get('PANNS_SHARED', '/shared'))
    state = Path(os.environ.get('PANNS_DEADLINE_STATE',
        str(shared/'panns-cnn-v1-20260925/run-v4/launch_deadline.json')))
    receipt = {'status': 'launch_failed', 'pause_status': 'external_controller_unvalidated' if external
               else 'pending' if pause else 'disabled_local_qa'}
    code = 2
    deadline = 0.
    deadline_valid = False
    try:
        if not pause and not external and not (sys.platform == 'darwin' and os.environ.get('PANNS_LOCAL_QA') == '1'):
            raise ValueError('Disabling workspace pause is restricted to local Mac QA')
        if pause and shutil.which('vesslctl') is None:
            raise FileNotFoundError('vesslctl required before starting paid compute')
        deadline = float(os.environ['PANNS_DEADLINE_EPOCH'])
        if not math.isfinite(deadline) or not 0 < deadline-time.time() <= 72*3600:
            raise ValueError('Original absolute deadline must be future and within 72 hours')
        if external:
            controller = json.loads(os.environ['PANNS_CONTROLLER_CONTRACT'])
            expected = {'org': 'qwdkw', 'workspace': 'wsp-nvxqw1jf9zdv', 'deadline_epoch': deadline,
                        'stop_owner': 'local_panns_workspace_watch'}
            if (not isinstance(controller, dict) or any(controller.get(key) != value for key, value in expected.items())
                or not isinstance(controller.get('run_id'), str) or not 0 < len(controller['run_id']) <= 128
                or not isinstance(controller.get('bundle_sha256'), str) or len(controller['bundle_sha256']) != 64
                or set(controller['bundle_sha256']) - set('0123456789abcdef')
                or os.environ.get('PANNS_PAUSE_ON_EXIT', '1') != '1'):
                raise ValueError('Invalid existing-workspace controller contract')
            receipt.update(controller_contract=controller, pause_status='external_controller_required')
        deadline_valid = True
        cutoff = deadline - 60 - (30 if pause or external else 0) - 1
        receipt.update(deadline_epoch=deadline, compute_cutoff_epoch=cutoff)
        code, status = supervise([sys.executable, __file__, '--check-state', str(state), str(deadline)],
                                 cutoff, 0., stop_requested=stopped)
        receipt.update(phase='state', status=status, state_exit_code=code)
        if code == 0:
            os.environ['PANNS_GUARD_PID'] = str(os.getpid())
            os.environ['PANNS_COMPUTE_DEADLINE_EPOCH'] = str(cutoff)
            code, status = supervise(['/bin/bash', *sys.argv[1:]], cutoff, stop_requested=stopped)
            receipt.update(phase='compute', status=status, child_exit_code=code)
    except (KeyError, ValueError, OSError) as error:
        receipt.update(status='launch_failed', error=str(error))
        print(str(error), file=sys.stderr)
    finally:
        cleanup_deadline = deadline if deadline_valid else time.time()+37.
        receipt.update(cleanup_deadline_epoch=cleanup_deadline, startup_deadline_valid=deadline_valid)
        receipt['finished_epoch'] = time.time()
        target = state.with_name('launch_exit.json')
        print(json.dumps({'event': 'pause_stage', **receipt}), flush=True)
        if pause:
            executable = shutil.which('vesslctl')
            try:
                if executable is None:
                    raise FileNotFoundError('vesslctl unavailable for automatic pause')
                pause_cutoff = min(time.time()+30., cleanup_deadline-7.)
                pause_code, pause_status = supervise([executable, '--org', 'qwdkw', 'workspace',
                    'pause', 'wsp-nvxqw1jf9zdv'], pause_cutoff, 0., stop_requested=[])
                receipt.update(pause_exit_code=pause_code, pause_command_status=pause_status)
                receipt['pause_status'] = 'request_accepted_not_verified' if pause_code == 0 else 'request_failed'
                if pause_code != 0:
                    code = code or 2
            except OSError as error:
                receipt.update(pause_status='request_failed', pause_error=str(error))
                print(f'Workspace pause failed: {error}', file=sys.stderr)
                code = code or 2
        try:
            receipt_code, receipt_status = supervise([sys.executable, __file__, '--write-receipt',
                str(target), json.dumps(receipt, indent=2)], min(time.time()+5., cleanup_deadline-1.), 0.,
                stop_requested=[])
            receipt.update(receipt_status=receipt_status)
            if receipt_code != 0:
                code = code or 2
        except OSError as error:
            receipt.update(receipt_status='write_failed', receipt_error=str(error))
            code = code or 2
        if stopped:
            receipt['termination_signals'] = list(stopped)
            code = code or 130
        print(json.dumps({'event': 'launch_exit', **receipt}), flush=True)
    return code


def main() -> int:
    stopped = []
    previous = {sig: signal.signal(sig, lambda number, _: stopped.append(number))
                for sig in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP)}
    try:
        return launch(stopped)
    finally:
        for sig, handler in previous.items():
            signal.signal(sig, handler)


if __name__ == '__main__':
    if len(sys.argv) > 1 and sys.argv[1] in ('--check-state', '--write-receipt'):
        metadata_phase()
    else:
        raise SystemExit(main())
