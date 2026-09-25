"""Trainer lifecycle regression checks with a substituted tiny model, not PANNs integration."""
from copy import deepcopy
import json

import pytest
import torch

import panns_cloud_run as runner
from panns_adaptation_data import sha256, write_json


class TinySuffix(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor([.75], dtype=torch.float64))

    def optimizer_groups(self):
        return [dict(params=[self.weight], lr=.01)]

    def tags_from_block5(self, features):
        return self.weight.expand(len(features), 2) * torch.tensor([1., -1.], dtype=torch.float64)


def loop_setup(tmp_path, monkeypatch, losses, *, patience=2, max_epochs=8):
    policy = dict(max_epochs=max_epochs, patience=patience, min_delta=1e-4,
                  monitor='selection_presence_bce')
    config = dict(identity='early-stop-lifecycle-fixture', output=str(tmp_path),
                  arms=['R3'], seeds=[41], microbatch=4, checkpoint_every=1,
                  early_stopping=policy)
    entries = [dict(ID=f'train-{i}', role='train', targets=[i % 2, i % 2]) for i in range(5)]
    entries.append(dict(ID='selection-0', role='selection', targets=[1, 0]))
    events, controls = [], dict(interrupt_epoch=None, validated_weights={})
    write_json(tmp_path/'cache/index.json', dict(authored_lifecycle_fixture=True))

    def model(*args):
        torch.manual_seed(41)
        return TinySuffix()

    def update(model, optimizer, selected, *args):
        events.append(('update', tuple(row['ID'] for row in selected)))
        optimizer.zero_grad(set_to_none=True)
        target = sum(int(row['ID'].split('-')[1]) for row in selected)/len(selected)
        loss = (model.weight - target - torch.rand((), dtype=torch.float64)).square().mean()
        loss.backward()
        norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), 1.))
        optimizer.step()
        return dict(loss=float(loss.detach()), gradient_norms=dict(suffix=norm)), None

    def metrics(model, selected, heads, labels, groups, device, microbatch, stop_requested):
        assert [row['ID'] for row in selected] == ['selection-0']
        epoch = sum(event[0] == 'update' for event in events)//2
        events.append(('validate', epoch))
        if controls['interrupt_epoch'] == epoch:
            controls['interrupt_epoch'] = None
            return None
        controls['validated_weights'][epoch] = model.weight.detach().clone()
        loss = losses[min(epoch, len(losses)-1)]
        return dict(selection_presence_bce=loss, voice_bce=loss, music_bce=loss, known_counts=[1, 1])

    monkeypatch.setattr(runner, 'validate_config', lambda c: tmp_path)
    monkeypatch.setattr(runner, 'load_budget', lambda *a, **k: (
        dict(training_steps=max_epochs*2, steps_per_pass=2, early_stopping=policy), 'budget'))
    monkeypatch.setattr(runner, 'read_cache_index', lambda c: dict(entries=entries))
    monkeypatch.setattr(runner, 'model_for_arm', model)
    monkeypatch.setattr(runner, 'readout_assets', lambda p: (None, None, None))
    monkeypatch.setattr(runner, 'optimization_step', update)
    monkeypatch.setattr(runner, 'selection_metrics', metrics, raising=False)
    return config, events, controls


@pytest.mark.parametrize('losses,best_epoch,best_loss,anchor', [
    ([.5, .49, .48995, .48994], 3, .48994, .49),
    ([.5, .4, .45, .46], 1, .4, .4),
])
def test_tiny_improvements_save_true_best_without_resetting_significant_patience(
        tmp_path, monkeypatch, losses, best_epoch, best_loss, anchor):
    # Given: two genuine improvements smaller than the meaningful-improvement threshold.
    config, events, controls = loop_setup(tmp_path, monkeypatch, losses)
    # When: the actual trainer executes with a tiny authored model and real AdamW.
    done = runner.train(config, 'R3', 41, 'cpu')
    # Then: epoch 1 resets patience; epochs 2 and 3 consume it despite lower true minima.
    assert done['steps'] == 6
    assert [value for event, value in events if event == 'validate'] == [0, 1, 2, 3]
    saved = torch.load(tmp_path/'train_R3_41/checkpoint.pt', weights_only=True)
    state = saved['early_stopping']
    assert (state['best_epoch'], state['best_loss'], state['patience_anchor'], state['bad_epochs']) == (
        best_epoch, best_loss, anchor, 2)
    assert done['stop_reason'] == 'early_stopping'
    assert done['last_validated_epoch'] == done['epochs_completed'] == 3
    torch.testing.assert_close(saved['best_model']['weight'], controls['validated_weights'][best_epoch], rtol=0, atol=0)
    torch.testing.assert_close(saved['model']['weight'], controls['validated_weights'][3], rtol=0, atol=0)


def test_interrupted_epoch_validation_retries_before_update_and_resumes_exactly(tmp_path, monkeypatch):
    # Given: interruption after a full epoch has checkpointed its optimizer update.
    config, events, controls = loop_setup(tmp_path/'split', monkeypatch, [.5, .4, .45, .46])
    controls['interrupt_epoch'] = 1
    incomplete = runner.train(config, 'R3', 41, 'cpu')
    checkpoint = tmp_path/'split/train_R3_41/checkpoint.pt'
    pending = torch.load(checkpoint, weights_only=True)
    assert incomplete['complete'] is False and pending['step'] == 2
    assert pending['early_stopping']['last_validated_epoch'] == 0
    assert not checkpoint.with_name('complete.json').exists()
    split_event = len(events)
    # When: resuming, the pending validation must run before another optimizer update.
    runner.train(config, 'R3', 41, 'cpu', resume=True)
    resumed = torch.load(checkpoint, weights_only=True)
    assert events[split_event] == ('validate', 1)
    config, continuous_events, _ = loop_setup(tmp_path/'continuous', monkeypatch, [.5, .4, .45, .46])
    runner.train(config, 'R3', 41, 'cpu')
    continuous = torch.load(tmp_path/'continuous/train_R3_41/checkpoint.pt', weights_only=True)
    # Then: real optimizer tensors, sampler, RNG and selected/current parameters agree exactly.
    assert [event for event in events if event[0] == 'update'] == [
        event for event in continuous_events if event[0] == 'update']
    assert resumed['sampler'] == continuous['sampler']
    assert resumed['early_stopping'] == continuous['early_stopping']
    assert resumed['step'] == continuous['step'] == 6
    for key in ('model', 'best_model', 'optimizer', 'torch_rng'):
        torch.testing.assert_close(resumed[key], continuous[key], rtol=0, atol=0)


def test_validation_state_is_saved_even_without_another_optimizer_update(tmp_path, monkeypatch):
    # Given: only the final validation of a two-epoch run remains after interruption.
    config, _, controls = loop_setup(tmp_path, monkeypatch, [.5, .4, .3], max_epochs=2)
    controls['interrupt_epoch'] = 2
    runner.train(config, 'R3', 41, 'cpu')
    path = tmp_path/'train_R3_41/checkpoint.pt'
    before = torch.load(path, weights_only=True)
    assert before['step'] == 4 and before['early_stopping']['last_validated_epoch'] == 1
    # When: resume completes validation at the same optimizer step.
    done = runner.train(config, 'R3', 41, 'cpu', resume=True)
    after = torch.load(path, weights_only=True)
    # Then: completion points to the newly persisted validation state and best weights.
    assert done['stop_reason'] == 'max_epochs'
    assert done['checkpoint_sha256'] == sha256(path)
    assert after['step'] == before['step']
    assert after['early_stopping']['last_validated_epoch'] == after['early_stopping']['best_epoch'] == 2
    assert after['early_stopping']['best_loss'] == .3
    torch.testing.assert_close(after['optimizer'], before['optimizer'], rtol=0, atol=0)
    torch.testing.assert_close(after['best_model']['weight'], after['model']['weight'], rtol=0, atol=0)


@pytest.mark.parametrize('invalid', [float('nan'), float('inf')])
def test_nonfinite_validation_never_completes_or_advances_patience(tmp_path, monkeypatch, invalid):
    # Given: a finite epoch-zero reference followed by an invalid selection measurement.
    config, _, _ = loop_setup(tmp_path, monkeypatch, [.5, invalid])
    # When: the actual training loop receives the invalid measurement.
    with pytest.raises(ValueError, match='(?i)(finite|metric|selection|loss)'):
        runner.train(config, 'R3', 41, 'cpu')
    # Then: no result is certified, and the last durable stopping state remains finite.
    assert not (tmp_path/'train_R3_41/complete.json').exists()
    saved = torch.load(tmp_path/'train_R3_41/checkpoint.pt', weights_only=True)
    state = saved['early_stopping']
    assert state['best_loss'] == .5 and state['bad_epochs'] == 0
    assert state['last_validated_epoch'] == 0


@pytest.mark.parametrize('selection_targets', [None, [None, None]])
def test_missing_or_unsupervised_selection_is_rejected_before_update(tmp_path, monkeypatch, selection_targets):
    config, events, _ = loop_setup(tmp_path, monkeypatch, [.5, .4])
    entries = [dict(ID=f'train-{i}', role='train', targets=[i % 2, i % 2]) for i in range(5)]
    if selection_targets is not None:
        entries.append(dict(ID='selection-0', role='selection', targets=selection_targets))
    monkeypatch.setattr(runner, 'read_cache_index', lambda c: dict(entries=entries))
    with pytest.raises(ValueError, match='(?i)(selection|known|target)'):
        runner.train(config, 'R3', 41, 'cpu')
    assert not any(event == 'update' for event, _ in events)
    assert not (tmp_path/'train_R3_41/complete.json').exists()


@pytest.mark.parametrize('change', [
    {'stop_reason': 'resource_limit'},
    {'steps': 5},
    {'last_validated_epoch': None},
    {'early_stopping': {'max_epochs': 8, 'patience': 3, 'min_delta': 1e-4, 'monitor': 'selection_presence_bce'}},
])
def test_completion_gate_rejects_uncertified_or_different_training(tmp_path, monkeypatch, change):
    config, _, _ = loop_setup(tmp_path, monkeypatch, [.5, .4, .45, .46])
    done = runner.train(config, 'R3', 41, 'cpu')
    from panns_training_state import validate_completion
    budget = runner.load_budget(config, 'cpu')[0]
    validate_completion(done, done['run_identity'], done['checkpoint_sha256'], budget)
    altered = deepcopy(done)
    altered.update(change)
    if altered.get('last_validated_epoch') is None:
        altered.pop('last_validated_epoch')
    with pytest.raises(ValueError):
        validate_completion(altered, done['run_identity'], done['checkpoint_sha256'], budget)


def test_selection_bce_uses_known_file_denominators_and_rejects_other_roles(monkeypatch):
    import panns_adaptation
    from torch.nn import functional as F

    calls = []
    logits = torch.tensor([[1000., -1000.], [1000., 1000.], [.3, -.7]], dtype=torch.float64)
    targets = [[1, 0], [None, 0], [0, 1]]
    segments = [3, 1, 2]
    entries = [dict(ID=str(i), role='selection', targets=target, stems={})
               for i, target in enumerate(targets)]

    class NumericSuffix:
        def tags_from_block5(self, features):
            calls.append('suffix')
            return features

    def cached(entry):
        calls.append('cache')
        i = int(entry['ID'])
        return dict(block5=logits[i].repeat(segments[i], 1).numpy())

    monkeypatch.setattr(runner, 'load_cache', cached)
    monkeypatch.setattr(panns_adaptation, 'presence_logits', lambda tags, *args: tags.mean(0))
    measured = runner.selection_metrics(NumericSuffix(), entries, None, None, None, 'cpu', 1, lambda: False)
    voice = F.binary_cross_entropy_with_logits(logits[[0, 2], 0], torch.tensor([1., 0.], dtype=torch.float64))
    music = F.binary_cross_entropy_with_logits(logits[:, 1], torch.tensor([0., 0., 1.], dtype=torch.float64))
    assert measured['known_counts'] == [2, 3]
    assert measured['voice_bce'] == pytest.approx(float(voice), rel=0, abs=1e-12)
    assert measured['music_bce'] == pytest.approx(float(music), rel=0, abs=1e-12)
    assert measured['selection_presence_bce'] == pytest.approx(float((voice+music)/2), rel=0, abs=1e-12)
    logits[1, 0] = -1000.
    assert runner.selection_metrics(NumericSuffix(), entries, None, None, None, 'cpu', 1, lambda: False) == measured
    previous_calls = list(calls)
    forbidden = [dict(entries[0], role='development'), *entries[1:]]
    with pytest.raises(ValueError, match='selection'):
        runner.selection_metrics(NumericSuffix(), forbidden, None, None, None, 'cpu', 1, lambda: False)
    assert calls == previous_calls


def test_evaluation_uses_selected_best_weights_instead_of_final_training_weights(tmp_path, monkeypatch):
    import panns_adaptation

    config, _, _ = loop_setup(tmp_path, monkeypatch, [.5, .4, .45, .46])
    row = dict(ID='selection-0', role='selection', verified_audio=dict(sha256='authored-audio'),
               FILE_FAKE=0, VOICE_FAKE=0, MUSIC_FAKE=0, VOICE_PRESENT=1, MUSIC_PRESENT=0)
    config['rows'] = str(tmp_path/'rows.jsonl')
    (tmp_path/'rows.jsonl').write_text(json.dumps(row)+'\n')
    fixed = dict(entries=[dict(ID=row['ID'], audio_sha256='authored-audio', FILE_FAKE_PROB=.4,
        VOICE_FAKE_PROB=.8, MUSIC_FAKE_PROB=.3, VOICE_PRESENT_PROB=.5, MUSIC_PRESENT_PROB=.5)])
    config['fixed_predictions'] = str(tmp_path/'fixed.json')
    write_json(tmp_path/'fixed.json', fixed)
    write_json(tmp_path/'model/trained/file_ensemble.json',
               dict(weights=dict(max=1.), tables=dict(max=[0., 1.])))
    cached = runner.read_cache_index(config)
    for entry in cached['entries']:
        entry['stems'] = {}
    monkeypatch.setattr(runner, 'read_cache_index', lambda c: cached)
    monkeypatch.setattr(runner, 'load_fixed_predictions', lambda *args: fixed)
    monkeypatch.setattr(runner, 'load_cache', lambda entry: dict(block5=torch.zeros(2, 2, dtype=torch.float64).numpy()))
    monkeypatch.setattr(panns_adaptation, 'presence_logits', lambda tags, *args: tags.mean(0))
    done = runner.train(config, 'R3', 41, 'cpu')
    saved = torch.load(tmp_path/'train_R3_41/checkpoint.pt', weights_only=True)
    best, current = saved['best_model']['weight'], saved['model']['weight']
    assert saved['early_stopping']['best_epoch'] == 1 and done['epochs_completed'] == 3
    expected_voice, expected_music = float(best.sigmoid()), float((-best).sigmoid())
    assert abs(expected_voice-float(current.sigmoid())) > 1e-6
    result = runner.evaluate(config, 'R3', 41, 'cpu')
    prediction = result['predictions'][0]
    assert prediction['VOICE_PRESENT_PROB'] == pytest.approx(expected_voice, rel=0, abs=1e-12)
    assert prediction['MUSIC_PRESENT_PROB'] == pytest.approx(expected_music, rel=0, abs=1e-12)
    assert prediction['FILE_FAKE_PROB'] == pytest.approx(max(expected_voice*.8, expected_music*.3), rel=0, abs=1e-12)
    assert (prediction['VOICE_FAKE_PROB'], prediction['MUSIC_FAKE_PROB']) == (.8, .3)
    assert result['checkpoint_sha256'] == done['checkpoint_sha256']
