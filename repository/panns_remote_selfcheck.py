# /// script
# requires-python = ">=3.11"
# dependencies = []
# ///
# Reviewed payload for the existing Linux workspace, before any model computation.
"""Real detached-process control checks using four synthetic stdlib-only cells."""
from contextlib import redirect_stdout
import hashlib
from io import StringIO
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import types

def run_selfcheck(runner_path: Path, dedicated_qa_root: Path) -> dict:
    result = subprocess.run([sys.executable, __file__, str(runner_path), str(dedicated_qa_root)],
                            check=True, capture_output=True, text=True, timeout=25)
    lines = result.stdout.splitlines()
    if len(lines) != 1 or not lines[0].startswith('PANNS_OPS_SELF_CHECK '):
        message = 'Expected one self-check receipt'
        raise ValueError(message)
    print(lines[0], flush=True)
    return json.loads(lines[0].removeprefix('PANNS_OPS_SELF_CHECK '))


def selfcheck(runner_path: Path, base: Path) -> None:
    if not sys.platform.startswith('linux'):
        message = 'This integration check requires Linux /proc'
        raise RuntimeError(message)
    raw = runner_path.read_bytes()
    runner_sha = hashlib.sha256(raw).hexdigest()
    module = types.ModuleType('panns_ops_selfcheck')
    module.__file__ = str(runner_path)
    exec(compile(raw, str(runner_path), 'exec'), module.__dict__)
    base.mkdir(parents=True, exist_ok=False)
    release = base/'inputs'
    release.mkdir()
    runner = release/'panns_notebook_remote.py'
    runner.write_bytes(raw)
    contract = dict(org='qwdkw', workspace='wsp-nvxqw1jf9zdv', deadline_epoch=time.time()+30,
        run_id='ops-controlplane-selfcheck', bundle_sha256='a'*64, stop_owner='local_panns_workspace_watch')
    config = release/'config.json'
    config.write_text(json.dumps({'events': str(base/'events.json'), 'release_flag': str(base/'release.flag'), 'contract': contract}))
    for name in ('launcher.sh', 'guard.py'):
        (release/name).write_text('raise RuntimeError("Synthetic pin only; must not execute")\n')
    cells = [
        "import json,os,time\nfrom pathlib import Path\ncfg=json.loads(Path(os.environ['DACON_EXPERIMENT_CONFIG']).read_text())\nassert json.loads(os.environ['PANNS_CONTROLLER_CONTRACT'])==cfg['contract']\nevents=[0]\nPath(cfg['events']).write_text(json.dumps(events))",
        "events.append(1)\nPath(cfg['events']).write_text(json.dumps(events))\nwait_until=time.monotonic()+10\nwhile not Path(cfg['release_flag']).exists():\n    if time.monotonic()>=wait_until: raise TimeoutError('Parent did not release fixture')\n    time.sleep(.05)",
        "events.append(2)\nPath(cfg['events']).write_text(json.dumps(events))",
        "events.append(3)\nassert events==[0,1,2,3]\nPath(cfg['events']).write_text(json.dumps(events))",
    ]
    notebook = release/'fixture.ipynb'
    notebook.write_text(json.dumps({'nbformat': 4, 'cells': [{'cell_type': 'code', 'source': text.splitlines(keepends=True)} for text in cells]}))
    paths = dict(notebook=str(notebook), notebook_config=str(config), launcher=str(release/'launcher.sh'),
                 guard=str(release/'guard.py'), runner=str(runner))
    request = dict(contract=contract, paths=paths,
        sha256={key: hashlib.sha256(Path(value).read_bytes()).hexdigest() for key, value in paths.items()},
        state_dir=str(base/'state'))
    request_path = release/'remote_request.json'
    request_path.write_text(json.dumps(request))
    request_sha = hashlib.sha256(request_path.read_bytes()).hexdigest()
    assert module.status(request, request_sha, request_path)['state'] == 'not_started'
    with redirect_stdout(StringIO()):
        module.start(request_path, request_sha)
    state = json.loads((base/'state/state.json').read_text())
    with redirect_stdout(StringIO()):
        module.start(request_path, request_sha)
    assert json.loads((base/'state/state.json').read_text())['pid'] == state['pid']
    assert module.status(request, request_sha, request_path)['state'] == 'running'
    (base/'release.flag').write_text('release\n')
    until = time.monotonic()+10
    while time.monotonic() < until:
        observed = module.status(request, request_sha, request_path)
        if observed['state'] == 'finished':
            break
        time.sleep(.1)
    assert observed['state'] == 'finished' and observed['exit_code'] == 0
    assert json.loads((base/'events.json').read_text()) == [0, 1, 2, 3]
    with redirect_stdout(StringIO()):
        module.start(request_path, request_sha)
    assert json.loads((base/'events.json').read_text()) == [0, 1, 2, 3]
    finished = base/'state/finished.json'
    original = finished.read_bytes()
    finished.rename(base/'state/finished.saved')
    module.atomic_json(base/'state/state.json', {**state, 'pid': os.getpid()})
    try:
        module.status(request, request_sha, request_path)
    except (OSError, ValueError):
        stale_rejected = True
    else:
        raise AssertionError('Replaced PID was adopted')
    module.atomic_json(base/'state/state.json', state)
    (base/'state/finished.saved').rename(finished)
    corrupt = json.loads(original)
    corrupt['contract'] = {**contract, 'run_id': 'different-run'}
    module.atomic_json(finished, corrupt)
    try:
        module.status(request, request_sha, request_path)
    except ValueError:
        wrong_contract_rejected = True
    else:
        raise AssertionError('Wrong finished contract was adopted')
    finished.write_bytes(original)
    partial = {**request, 'state_dir': str(base/'partial')}
    Path(partial['state_dir']).mkdir()
    (Path(partial['state_dir'])/'launch.json').write_text('')
    try:
        module.status(partial, request_sha, request_path)
    except ValueError:
        partial_rejected = True
    else:
        raise AssertionError('Partial marker was treated as not_started')
    report = dict(runner_sha256=runner_sha, linux_proc_startticks_executed=True,
        four_synthetic_cells_executed=[0, 1, 2, 3], repeated_start_same_pid=True,
        repeat_after_completion_no_new_execution=True, stale_pid_rejected=stale_rejected,
        wrong_finished_contract_rejected=wrong_contract_rejected, partial_state_rejected=partial_rejected,
        actual_model_notebook_executed=False, gpu_used=False, exit_code=0)
    module.atomic_json(base/'selfcheck.json', report)
    print('PANNS_OPS_SELF_CHECK '+json.dumps(report), flush=True)


if __name__ == '__main__':
    selfcheck(Path(sys.argv[1]), Path(sys.argv[2]))
