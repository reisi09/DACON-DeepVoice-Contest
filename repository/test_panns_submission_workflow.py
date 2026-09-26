import json
import sys
import time

import pytest

from panns_submission_workflow import run_jobs


def test_deadline_preserves_checkpoint_and_reaps_owned_child(tmp_path):
    checkpoint = tmp_path / 'checkpoint.json'
    child = tmp_path / 'child.py'
    child.write_text(
        'import json, signal, sys, time\n'
        'from pathlib import Path\n'
        'def stop(*_):\n'
        '    Path(sys.argv[1]).write_text(json.dumps({"step": 7, "saved_on_term": True}))\n'
        '    raise SystemExit(0)\n'
        'signal.signal(signal.SIGTERM, stop)\n'
        'while True: time.sleep(.02)\n'
    )
    with pytest.raises(TimeoutError, match='deadline'):
        run_jobs([('training', [sys.executable, str(child), str(checkpoint)], time.time() + 1)], tmp_path)
    assert json.loads(checkpoint.read_text()) == {'step': 7, 'saved_on_term': True}
    receipt = json.loads((tmp_path / 'jobs.json').read_text())
    assert receipt['training']['reaped'] is True


def test_child_failure_is_not_reported_as_completed_training(tmp_path):
    with pytest.raises(RuntimeError, match='failed'):
        run_jobs([('training', [sys.executable, '-c', 'raise SystemExit(7)'], time.time() + 5)], tmp_path)
    receipt = json.loads((tmp_path / 'jobs.json').read_text())
    assert receipt['training']['exit_code'] == 7
    assert receipt['training']['reaped'] is True
