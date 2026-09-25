import os
from pathlib import Path
import signal
import subprocess
import sys
import time

import pytest


def test_budget_covers_whole_passes_with_equal_steps_and_reserve():
    from panns_training_budget import budget_steps

    seconds = {'R1': 1., 'R2': 2., 'R3': 3., 'R5': 4.}
    steps = budget_steps(seconds, 5, 1.)
    assert steps == 96
    assert steps % 2 == 0
    assert steps * 3 * sum(seconds.values()) <= 3600 * .8
    assert budget_steps({arm: .001 for arm in seconds}, 9, 72.) == 99999
    with pytest.raises(ValueError, match='whole pass'):
        budget_steps(seconds, 5, .001)
    for invalid in (float('nan'), float('inf'), 0., -1.):
        with pytest.raises(ValueError):
            budget_steps(seconds, 5, invalid)
        with pytest.raises(ValueError):
            budget_steps({**seconds, 'R3': invalid}, 5, 1.)


def test_budget_refuses_runtime_drift_and_altered_fixed_steps(tmp_path):
    from panns_adaptation_data import identity, sha256, write_json
    from panns_training_budget import load_budget, profile_binding

    config = {'identity': 'locked-run', 'output': str(tmp_path)}
    (tmp_path / 'cache').mkdir()
    (tmp_path / 'cache/index.json').write_text('{"real": "index bytes"}')
    binding = profile_binding(config, 'cpu')
    profile = {'binding': binding, 'train_files': 5,
               'seconds_per_step': {'R1': 1., 'R2': 2., 'R3': 3., 'R5': 4.}}
    write_json(tmp_path / 'throughput_profile.json', profile)
    budget = {'binding': binding, 'profile_sha256': sha256(tmp_path / 'throughput_profile.json'),
              'training_hours': 1., 'training_steps': 96, 'steps_per_pass': 2}
    budget['identity'] = identity(budget)
    write_json(tmp_path / 'training_budget.json', budget)
    assert load_budget(config, 'cpu')[0]['training_steps'] == 96
    with pytest.raises(ValueError, match='device'):
        load_budget(config, 'cuda')
    (tmp_path / 'cache/index.json').write_text('{"changed": "index bytes"}')
    with pytest.raises(ValueError, match='binding'):
        load_budget(config, 'cpu')
    (tmp_path / 'cache/index.json').write_text('{"real": "index bytes"}')
    budget['training_steps'] = 98
    budget['identity'] = identity({k: v for k, v in budget.items() if k != 'identity'})
    write_json(tmp_path / 'training_budget.json', budget)
    with pytest.raises(ValueError, match='measured'):
        load_budget(config, 'cpu')


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
                      "run_command([sys.executable, sys.argv[1], sys.argv[2], 'child'], time.monotonic()+20)\n")
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
