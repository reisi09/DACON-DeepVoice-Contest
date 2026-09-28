# /// script
# requires-python = ">=3.11"
# dependencies = []
# ///
# Invoked only by the pinned remote_start.py payload; no package installation.
"""One detached, pinned notebook job with durable state for the local controller."""
from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import traceback
from typing import TypedDict


class Contract(TypedDict):
    org: str
    workspace: str
    deadline_epoch: float
    run_id: str
    bundle_sha256: str
    stop_owner: str


class Request(TypedDict):
    contract: Contract
    paths: dict[str, str]
    sha256: dict[str, str]
    state_dir: str


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def read_request(path: Path, expected_sha256: str) -> Request:
    raw = path.read_bytes()
    require(hashlib.sha256(raw).hexdigest() == expected_sha256, 'Request hash mismatch')
    request = json.loads(raw)
    require(set(request) == {'contract', 'paths', 'sha256', 'state_dir'}, 'Unexpected request fields')
    contract = request['contract']
    require(set(contract) == set(Contract.__annotations__), 'Unexpected controller contract fields')
    require(contract['org'] == 'qwdkw' and contract['workspace'] == 'wsp-nvxqw1jf9zdv'
            and contract['stop_owner'] == 'local_panns_workspace_watch', 'Wrong existing workspace controller')
    require(type(contract['deadline_epoch']) in (int, float) and math.isfinite(contract['deadline_epoch'])
            and contract['deadline_epoch'] > 0, 'Invalid original deadline')
    require(type(contract['run_id']) is str and 0 < len(contract['run_id']) <= 128, 'Invalid run ID')
    roles = {'notebook', 'notebook_config', 'launcher', 'guard', 'runner'}
    require(set(request['paths']) == roles and set(request['sha256']) == roles, 'Incomplete standalone release pins')
    for digest in [contract['bundle_sha256'], *request['sha256'].values()]:
        require(type(digest) is str and len(digest) == 64 and not set(digest)-set('0123456789abcdef'), 'Invalid SHA256')
    release = path.parent.resolve()
    for role, filename in request['paths'].items():
        member = Path(filename)
        require(member.is_absolute() and member.parent.resolve() == release and not member.is_symlink()
                and member.is_file(), 'Pinned file must be regular and inside release: '+role)
        require(hashlib.sha256(member.read_bytes()).hexdigest() == request['sha256'][role], 'Pinned file changed: '+role)
    state = Path(request['state_dir'])
    require(state.is_absolute() and state.resolve().is_relative_to(release.parent)
            and not state.resolve().is_relative_to(release), 'State directory must be outside inputs within project root')
    return request


def atomic_json(path: Path, value) -> None:
    temporary = path.with_name(path.name+'.tmp')
    with temporary.open('w') as stream:
        json.dump(value, stream, sort_keys=True)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def process_identity(pid: int) -> tuple[str, list[str]]:
    require(sys.platform.startswith('linux'), 'Production process identity requires Linux /proc')
    fields = Path(f'/proc/{pid}/stat').read_text().rsplit(')', 1)[1].split()
    require(fields[0] != 'Z', 'Task process is a zombie')
    command = Path(f'/proc/{pid}/cmdline').read_bytes().rstrip(b'\0').split(b'\0')
    return fields[19], [item.decode() for item in command]


def command_for(request: Request, request_path: Path, request_sha: str) -> list[str]:
    return [sys.executable, '-u', request['paths']['runner'], str(request_path), request_sha]


def status(request: Request, request_sha: str, request_path: Path) -> dict:
    root = Path(request['state_dir'])
    marker, state, finished = (root/name for name in ('launch.json', 'state.json', 'finished.json'))
    if not root.exists() or not any(root.iterdir()):
        return {**request['contract'], 'state': 'not_started', 'exit_code': None}
    require(marker.is_file() and not marker.is_symlink(), 'Partial remote launch; refusing a second job')
    expected = {'contract': request['contract'], 'request_sha256': request_sha}
    require(json.loads(marker.read_text()) == expected, 'Launch marker identity mismatch')
    def completed() -> dict:
        require(not finished.is_symlink(), 'Finished receipt must be regular')
        record = json.loads(finished.read_text())
        require(record.get('contract') == request['contract'] and record.get('request_sha256') == request_sha
                and type(record.get('exit_code')) is int and record.get('notebook_sha256') == request['sha256']['notebook'],
                'Finished receipt identity mismatch')
        return {**request['contract'], 'state': 'finished', 'exit_code': record['exit_code']}
    if finished.is_file():
        return completed()
    require(state.is_file() and not state.is_symlink(), 'Launch outcome unresolved; refusing a second job')
    record = json.loads(state.read_text())
    require(record.get('contract') == request['contract'] and record.get('request_sha256') == request_sha
            and type(record.get('pid')) is int and record['pid'] > 1, 'Running state identity mismatch')
    try:
        expected_command = command_for(request, request_path, request_sha)
        require(record['cmdline'] == expected_command,
                'State does not identify the pinned runner command')
        start, command = process_identity(record['pid'])
        require(start == record['start_ticks'] and command == record['cmdline'], 'Running PID was replaced')
    except (OSError, subprocess.SubprocessError, ValueError):
        if finished.is_file():
            return completed()
        raise
    return {**request['contract'], 'state': 'running', 'exit_code': None}


def start(request_path: Path, request_sha: str) -> None:
    request = read_request(request_path, request_sha)
    observed = status(request, request_sha, request_path)
    if observed['state'] != 'not_started':
        print('PANNS_STATUS '+json.dumps(observed), flush=True)
        return
    require(0 < request['contract']['deadline_epoch']-time.time() <= 72*3600, 'Original deadline expired')
    root = Path(request['state_dir'])
    root.mkdir(parents=True, exist_ok=True)
    with (root/'launch.json').open('x') as marker:
        json.dump({'contract': request['contract'], 'request_sha256': request_sha}, marker)
        marker.flush()
        os.fsync(marker.fileno())
    with (root/'notebook.log').open('x') as log:
        subprocess.Popen(command_for(request, request_path, request_sha), stdin=subprocess.DEVNULL,
                         stdout=log, stderr=subprocess.STDOUT, start_new_session=True,
                         close_fds=True, cwd=request_path.parent)
    until = time.monotonic()+5
    while time.monotonic() < until:
        if (root/'state.json').exists() or (root/'finished.json').exists():
            print('PANNS_STATUS '+json.dumps(status(request, request_sha, request_path)), flush=True)
            return
        time.sleep(.05)
    message = 'Detached start outcome unresolved; probe before any further action'
    raise RuntimeError(message)


def run(request_path: Path, request_sha: str) -> int:
    request = read_request(request_path, request_sha)
    root = Path(request['state_dir'])
    binding = {'contract': request['contract'], 'request_sha256': request_sha}
    require(json.loads((root/'launch.json').read_text()) == binding, 'Exclusive launch marker mismatch')
    require(not (root/'state.json').exists() and not (root/'finished.json').exists(), 'Job already started')
    start_ticks, command = process_identity(os.getpid())
    expected = command_for(request, request_path, request_sha)
    require(command == expected, 'Unexpected runner command line')
    atomic_json(root/'state.json', {**binding, 'pid': os.getpid(), 'start_ticks': start_ticks, 'cmdline': command})
    signal.signal(signal.SIGHUP, signal.SIG_IGN)
    def interrupted(*_) -> None:
        raise KeyboardInterrupt
    signal.signal(signal.SIGTERM, interrupted)
    code, cells = 1, []
    began = time.monotonic()
    try:
        require(time.time() < request['contract']['deadline_epoch'], 'Original deadline expired')
        os.environ['PANNS_CONTROLLER_CONTRACT'] = json.dumps(request['contract'])
        os.environ['DACON_EXPERIMENT_CONFIG'] = request['paths']['notebook_config']
        notebook = json.loads(Path(request['paths']['notebook']).read_text())
        code_cells = [(i, c) for i, c in enumerate(notebook['cells']) if c['cell_type'] == 'code']
        require(notebook['nbformat'] == 4 and len(code_cells) == 4, 'Expected the reviewed four-cell notebook')
        namespace = {'__name__': '__main__'}
        for index, cell in code_cells:
            source = ''.join(cell['source']) if isinstance(cell['source'], list) else cell['source']
            exec(compile(source, request['paths']['notebook']+':cell:'+str(index), 'exec'), namespace)
            cells.append(index)
            print('NOTEBOOK_CELL_COMPLETE '+str(index), flush=True)
        code = 0
    except (Exception, KeyboardInterrupt):  # noqa: BROAD_EXCEPT_OK -- log and report CLI boundary failure.
        traceback.print_exc()
    finally:
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
        signal.signal(signal.SIGINT, signal.SIG_IGN)
        atomic_json(root/'finished.json', {**binding, 'exit_code': code, 'completed_cells': cells,
            'notebook_sha256': request['sha256']['notebook'], 'seconds': time.monotonic()-began})
    return code


if __name__ == '__main__':
    raise SystemExit(run(Path(sys.argv[1]), sys.argv[2]))
