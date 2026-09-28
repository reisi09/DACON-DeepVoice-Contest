"""A deadline-censored initialized model must never become a comparison result."""

import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from v5_compare import record_temporal
from v5_features import EncodedFile
from v5_dataset import Example
from v5_fusion_training import train_fusion
import v5_compare


def test_expired_temporal_training_cannot_claim_five_evaluated(tmp_path: Path) -> None:
    sample = EncodedFile("sample", np.ones((2, 8), dtype=np.float32),
                         np.array([0., 4.], dtype=np.float32), (0., 0., 0., 1., 1.))
    _, probabilities, metadata = train_fusion([sample], [sample], True, tmp_path / "B4", time.time() - 1)
    result = record_temporal(tmp_path, "B4", [], probabilities, metadata)
    comparison = {name: {"status": "evaluated"} for name in ("B0", "B1", "B2", "B3")}
    comparison["B4"] = result
    assert result["status"] == "budget_censored"
    assert "metrics" not in result, "Untrained probabilities must not enter the performance comparison"
    assert not all(row["status"] == "evaluated" for row in comparison.values())


def test_slow_separation_can_borrow_unused_stage_time_without_extending_allocation(tmp_path, monkeypatch):
    # The real run aborted at its 35% preparation slice with global time still free.
    now = [1_000.0]
    deadline = now[0] + 2_600
    monkeypatch.setattr(v5_compare.time, 'time', lambda: now[0])
    def consume(limit, seconds):
        if now[0] + seconds > limit:
            raise TimeoutError('Phase stopped although the overall allocation can fit it')
        now[0] += seconds
    def prepare(rows, baseline, cache, limit, *, training):
        consume(limit, 1_170 if training else 260)
        return {}
    def encode(model, rows, views, kind, root, limit):
        consume(limit, 130)
    model = SimpleNamespace(load_state_dict=lambda *args, **kwargs: None)
    fusion_calls = []
    def fuse(train, dev, joint, output, limit, *, epochs):
        consume(limit, 130)
        fusion_calls.append(joint)
        return model, np.array([[.1, .1, .1, .9, .9], [.9, .9, .9, .9, .9]]), {
            'status': 'complete', 'epochs_completed': 20, 'updates': 20, 'best_epoch': 1}
    monkeypatch.setattr(v5_compare, 'add_compositions', lambda train, output: (train, []))
    monkeypatch.setattr(v5_compare, 'prepare_views', prepare)
    monkeypatch.setattr(v5_compare, 'encode_expert_views', encode)
    monkeypatch.setattr(v5_compare, 'assemble_features', lambda rows, roots: [])
    monkeypatch.setattr(v5_compare, 'load_expert', lambda *args: model)
    monkeypatch.setattr(v5_compare.torch, 'load', lambda *args, **kwargs: {})
    monkeypatch.setattr(v5_compare, 'release', lambda model: None)
    monkeypatch.setattr(v5_compare, 'train_fusion', fuse)
    weight = tmp_path / 'weight.pt'
    weight.write_bytes(b'isolated scheduling test; encoder behavior tested separately')
    paths = {('B2', kind): weight for kind in ('voice', 'music', 'presence')}
    comparison = {}
    dev = [Example(str(label), tmp_path / f'{label}.wav', (str(label),),
                   (label, label, label, 1., 1.), 'mixture') for label in (0., 1.)]
    v5_compare.run_temporal(tmp_path, tmp_path / 'results', [], dev, deadline, paths, comparison)
    assert fusion_calls == [False, True]
    assert comparison['B3']['status'] == comparison['B4']['status'] == 'evaluated'
    assert comparison['B4_joint']['status'] == 'not_run_budget'
    assert now[0] < deadline
