"""Lifecycle fixtures exercise real checkpoint, optimizer and sampler persistence."""
import json
from pathlib import Path
import signal
import sys

import pytest
import torch

import panns_submission_train as subject
from panns_adaptation_data import sha256
from panns_adaptation_run import frozen_digest, restore_checkpoint, trainable_state
from panns_cloud_run import save_checkpoint
from panns_training_state import EpochSampler, record_validation


class TinyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor([0.]))
        self.register_buffer('frozen', torch.tensor([7.]))

    def optimizer_groups(self):
        return [dict(params=[self.weight], lr=.1, name='block6')]


def fixture(tmp_path, monkeypatch):
    policy = dict(max_epochs=200, patience=15, min_delta=1e-4, monitor='selection_presence_bce')
    old = tmp_path/'old'
    (old/'cache').mkdir(parents=True)
    (old/'cache/index.json').write_text('{}')
    reference = tmp_path/'reference.json'
    reference.write_text('{}')
    config = dict(identity='c'*64, output=str(old), reference=str(reference), early_stopping=policy, microbatch=4)
    model = TinyModel()
    optimizer = torch.optim.AdamW(model.optimizer_groups(), weight_decay=1e-4, foreach=False)
    sampler = EpochSampler(12, 41)
    state = dict(policy=policy, best_loss=None, best_epoch=0, patience_anchor=None,
                 bad_epochs=0, last_validated_epoch=-1, history=[])
    record_validation(state, dict(selection_presence_bce=.2, voice_bce=.2, music_bce=.2), 0)
    best = trainable_state(model)
    sampler.next_batch()
    (model.weight-1).square().sum().backward()
    optimizer.step()
    parent = old/'parent.pt'
    save_checkpoint(parent, model, optimizer, 'cpu', dict(run_identity='a'*64, arm='R1', seed=41,
        step=1, sampler=sampler.state_dict(), diagnostics={}, frozen_sha256=frozen_digest(model),
        elapsed_seconds=1., early_stopping=state, best_model=best))
    request = dict(schema_version=1, output=str(tmp_path/'new'), cache_index_sha256=sha256(old/'cache/index.json'),
        device='cpu', microbatch=4, checkpoint_seconds=300, progress_steps=100,
        deadlines=dict(R1=2., R3=4.), hard_deadline_epoch=6.,
        parent=dict(checkpoint=str(parent), checkpoint_sha256=sha256(parent), run_identity='a'*64),
        fast_step=dict(enabled=False, receipt=None, receipt_sha256=None),
        sources={name: sha256(Path(subject.__file__).with_name(name)) for name in
                 ('panns_submission_train.py', 'panns_fast_step.py')})
    entries = [dict(ID=str(i), role='train', targets=[i % 2, i % 2]) for i in range(12)]
    entries += [dict(ID='selection', role='selection', targets=[1, 0])]
    clock = [0.]
    calls = []

    def step(model, optimizer, selected, *args):
        calls.append([entry['ID'] for entry in selected])
        optimizer.zero_grad(set_to_none=True)
        loss = (model.weight-1).square().sum()
        loss.backward()
        optimizer.step()
        clock[0] += 3.
        return dict(loss=float(loss.detach()), gradient_norms={}), {}

    monkeypatch.setattr(subject, 'validate_config', lambda _: tmp_path)
    monkeypatch.setattr(subject, 'read_cache_index', lambda _: dict(entries=entries))
    monkeypatch.setattr(subject, 'model_for_arm', lambda *_: TinyModel())
    monkeypatch.setattr(subject, 'readout_assets', lambda _: (None, None, None))
    monkeypatch.setattr(subject, 'optimization_step', step)
    monkeypatch.setattr(subject.time, 'time', lambda: clock[0])
    return config, request, sampler, best, calls, clock


def test_time_budget_exports_validated_best_and_preserves_mid_epoch_state(tmp_path, monkeypatch):
    config, request, sampler, best, calls, _ = fixture(tmp_path, monkeypatch)
    expected_batch = [str(i) for i in sampler.next_batch()]
    result = subject.train(config, request, 'R1')
    out = Path(request['output'])/'train_R1_41'
    latest = torch.load(out/'checkpoint.pt', weights_only=True)
    exported = torch.load(out/'export/best_model.pt', weights_only=True)
    assert result['stop_reason'] == 'time_budget' and result['training_converged'] is False
    assert calls == [expected_batch] and latest['step'] == 2
    assert latest['sampler'] == sampler.state_dict()
    assert latest['early_stopping']['last_validated_epoch'] == 0
    assert torch.equal(exported['weight'], best['weight'])
    assert not torch.equal(exported['weight'], latest['model']['weight'])
    assert not (out/'complete.json').exists()
    resumed = TinyModel()
    optimizer = torch.optim.AdamW(resumed.optimizer_groups(), weight_decay=1e-4, foreach=False)
    restored = restore_checkpoint(out/'checkpoint.pt', result['run_identity'], resumed, optimizer)
    assert torch.equal(resumed.weight, latest['model']['weight'])
    assert int(optimizer.state[resumed.weight]['step']) == restored['step'] == 2
    selection = json.loads((out/'export/selection.json').read_text())
    assert selection['checkpoint_sha256'] == sha256(out/'checkpoint.pt')
    subject.train(config, request, 'R1')
    assert calls == [expected_batch]


def test_parent_hash_mismatch_rejects_before_loading_or_creating_output(tmp_path, monkeypatch):
    config, request, _, _, calls, _ = fixture(tmp_path, monkeypatch)
    request['parent']['checkpoint_sha256'] = '0'*64
    with pytest.raises(ValueError, match='parent checkpoint hash'):
        subject.train(config, request, 'R1')
    assert calls == [] and not Path(request['output']).exists()


def test_workflow_reuses_completed_export_and_rejects_changed_weights(tmp_path, monkeypatch):
    import panns_submission_workflow as workflow
    config, request, _, _, calls, _ = fixture(tmp_path, monkeypatch)
    subject.train(config, request, 'R1')
    assert workflow.completed_training(config, request, 'R1') is True
    assert workflow.completed_training(config, request, 'R3') is False
    assert len(calls) == 1
    weights = Path(request['output'])/'train_R1_41/export/best_model.pt'
    with weights.open('ab') as stream:
        stream.write(b'corruption')
    with pytest.raises(ValueError, match='Completed training'):
        workflow.completed_training(config, request, 'R1')


def test_sigterm_continuation_preserves_optimizer_and_next_file_batch(tmp_path, monkeypatch):
    config, request, sampler, _, calls, clock = fixture(tmp_path, monkeypatch)
    original_step = subject.optimization_step
    expected = [[str(i) for i in sampler.next_batch()] for _ in range(2)]

    def interrupt_after_update(*args):
        result = original_step(*args)
        signal.raise_signal(signal.SIGTERM)
        return result

    monkeypatch.setattr(subject, 'optimization_step', interrupt_after_update)
    (tmp_path/'run.json').write_text(json.dumps(config))
    (tmp_path/'request.json').write_text(json.dumps(request))
    monkeypatch.setattr(sys, 'argv', ['panns_submission_train', '--run', str(tmp_path/'run.json'),
                                   '--request', str(tmp_path/'request.json'), '--arm', 'R1'])
    with pytest.raises(SystemExit) as exit_status:
        subject.main()
    assert exit_status.value.code == 2
    out = Path(request['output'])/'train_R1_41'
    interrupted = json.loads((out/'terminal.json').read_text())
    assert interrupted['stop_reason'] == 'interrupted' and not interrupted['export_ready']
    assert not (out/'export/selection.json').exists()
    clock[0] = 0.
    monkeypatch.setattr(subject, 'optimization_step', original_step)
    result = subject.train(config, request, 'R1')
    saved = torch.load(out/'checkpoint.pt', weights_only=True)
    assert calls == expected and result['step'] == 3
    assert saved['sampler'] == sampler.state_dict()
    assert int(next(iter(saved['optimizer']['state'].values()))['step']) == 3
    assert saved['early_stopping']['last_validated_epoch'] == 0
