import gc
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import zipfile

import numpy as np
import pytest
import torch

import decision_heads
from panns_adaptation import presence_logits
from panns_adaptation_cache import baseline, load_base, readout_assets
from panns_adaptation_data import load_cache, sha256
from panns_adaptation_run import model_for_arm
from panns_submission import Engine, PannsTagger, SubmissionError, apply_adaptation
from package_panns_submission import package as make_package


@pytest.fixture(scope='module')
def actual_inputs():
    workspace = os.environ.get('PANNS_TEST_WORKSPACE')
    if workspace is None:
        pytest.skip('Set PANNS_TEST_WORKSPACE to the preserved real-audio QA assets')
    work = Path(workspace)
    reference = work/'reference_lock.json'
    package = Path(json.loads(reference.read_text())['package'])
    source = work/'notebook_execution_20260925/presence_fix_20260926/saved_probe/cpu'
    assert sha256(source/'index.json') == (source/'index.sha256').read_text().strip()
    entry = next(e for e in json.loads((source/'index.json').read_text())['entries']
                 if e['ID'].startswith('asvspoof2021') and e['segments'] == 1)
    entry = dict(entry, path=str(source/Path(entry['path']).name))
    config = json.loads((work/'notebook_execution_20260925/cloud_v11/preflight_work/run.json').read_text())
    row = next(json.loads(line) for line in Path(config['rows']).read_text().splitlines()
               if json.loads(line)['ID'] == entry['ID'])
    audio = row['verified_audio']
    with Path(config['archive']).open('rb') as stream:
        stream.seek(audio['tar_offset'])
        payload = stream.read(audio['bytes'])
    assert hashlib.sha256(payload).hexdigest() == audio['sha256']
    return work, reference, package, entry, Path(row['path']).suffix, payload


@pytest.mark.parametrize('arm', ['R1', 'R3'])
def test_trained_suffix_matches_cache_and_raw_audio(actual_inputs, tmp_path, arm):
    torch.set_num_threads(4)
    work, _, package, entry, suffix, payload = actual_inputs
    source = tmp_path/('input'+suffix)
    source.write_bytes(payload)
    base = baseline(package)
    raw = base.load_audio(source)
    peak = float(np.abs(raw).max())
    if peak > .6:
        raw = (raw*(.6/peak)).astype(np.float32)
    arrays = load_cache(entry)
    checkpoint = work/f'pilot_{arm}_cpu_mb1/pilot_checkpoint.pt'
    state = torch.load(checkpoint, map_location='cpu', weights_only=True)['model']
    training = model_for_arm(package, arm, 41, 'cpu')
    with torch.no_grad():
        original_tags = training.tags_from_block5(torch.from_numpy(arrays['block5'])).clone()
        for name, parameter in training.named_parameters():
            if name in state:
                parameter.copy_(state[name])
        expected_tags = training.tags_from_block5(torch.from_numpy(arrays['block5'])).clone()
    assert float((expected_tags-original_tags).abs().max()) > 1e-6
    del training
    gc.collect()
    model = apply_adaptation(load_base(package), state, arm)
    with torch.inference_mode():
        actual_tags = model.tags_from_block5(torch.from_numpy(arrays['block5']))
    torch.testing.assert_close(actual_tags, expected_tags, rtol=0, atol=0)
    waveform_tags, _ = PannsTagger(model).inference(base.make_panns_segments(raw))
    np.testing.assert_allclose(waveform_tags, expected_tags.numpy(), rtol=0, atol=1e-5)
    heads, labels, groups = readout_assets(package)
    expected = presence_logits(expected_tags, entry['stems'], heads, labels, groups).sigmoid().numpy()
    feats = {**entry['stems'], **decision_heads.tagger_features(waveform_tags, labels, groups)}
    observed = decision_heads.presence(heads, feats, (0., 0.))
    np.testing.assert_allclose(observed, expected, rtol=0, atol=1e-5)
    invalid = dict(state, **{'base.bn0.running_mean': torch.zeros(64)})
    with pytest.raises(SubmissionError, match='parameter set'):
        apply_adaptation(model.base, invalid, arm)
    print(json.dumps(dict(arm=arm, ID=entry['ID'], trained_tag_change=float((expected_tags-original_tags).abs().max()),
        raw_cache_tag_max_abs=float(np.abs(waveform_tags-expected_tags.numpy()).max()),
        presence_max_abs=float(np.abs(np.array(observed)-expected).max()))))


def test_real_full_engine_retains_fake_heads_and_uses_selected_suffix(actual_inputs, tmp_path):
    work, reference, package, entry, suffix, payload = actual_inputs
    checkpoint = work/'pilot_R3_cpu_mb1/pilot_checkpoint.pt'
    record = torch.load(checkpoint, map_location='cpu', weights_only=True)
    trained = tmp_path/'trained'
    trained.mkdir()
    for path in (package/'model/trained').iterdir():
        if path.is_file():
            shutil.copyfile(path, trained/path.name)
    weights = trained/'panns_best_model.pt'
    torch.save(record['model'], weights)
    selection = dict(arm='R3', seed=41, checkpoint_sha256=sha256(checkpoint), best_epoch=0,
        stop_reason='time_budget', run_identity=record['run_identity'], config_identity=record['config_identity'],
        weights_sha256=sha256(weights), reference_sha256=sha256(reference))
    (trained/'panns_selection.json').write_text(json.dumps(selection))
    source = tmp_path/('input'+suffix)
    source.write_bytes(payload)
    engine = Engine(package/'model/assets', trained, 'cpu')
    learned = engine.predict(source)
    adapted_model = engine.panns.model
    from panns_adaptation import AdaptedCnn14
    engine.panns = PannsTagger(AdaptedCnn14(load_base(package), train_blocks=()))
    original = engine.predict(source)
    assert learned['VOICE_FAKE_PROB'] == original['VOICE_FAKE_PROB']
    assert learned['MUSIC_FAKE_PROB'] == original['MUSIC_FAKE_PROB']
    assert max(abs(learned[key]-original[key]) for key in ('VOICE_PRESENT_PROB', 'MUSIC_PRESENT_PROB')) > 1e-6
    for name, parameter in adapted_model.named_parameters():
        if name in record['model']:
            torch.testing.assert_close(parameter.cpu(), record['model'][name], rtol=0, atol=0)
    expected_file = decision_heads.file_score(engine.ensemble, dict(Pv=learned['VOICE_PRESENT_PROB'],
        Pm=learned['MUSIC_PRESENT_PROB'], Fv=learned['VOICE_FAKE_PROB'], Fm=learned['MUSIC_FAKE_PROB']))
    assert learned['FILE_FAKE_PROB'] == expected_file
    assert all(np.isfinite(value) and 0 <= value <= 1 for value in learned.values())
    print(json.dumps(dict(ID=entry['ID'], learned=learned, original=original,
        frozen_fake_probabilities_bitwise_equal=True, scope='local pilot integration, not final model quality')))


def test_package_stream_preserves_members_and_rejects_changed_inputs(tmp_path):
    base = tmp_path/'base'
    names = ('script.py', 'requirements.txt', 'model/trained/presence_head.json',
             'model/trained/file_ensemble.json', 'model/runtime/full_bank_submission.py')
    for name in names:
        path = base/name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(('preserved '+name).encode())
    hashes = {name: sha256(base/name) for name in names}
    (base/'model/manifest.json').write_text(json.dumps(dict(choice=dict(candidate='verified-base'), sha256=hashes)))
    hashes['model/manifest.json'] = sha256(base/'model/manifest.json')
    reference = tmp_path/'reference.json'
    reference.write_text(json.dumps(dict(package=str(base), members=hashes)))
    baseline = make_package(reference, None, tmp_path/'baseline.zip')
    with zipfile.ZipFile(tmp_path/'baseline.zip') as archive:
        assert {name: hashlib.sha256(archive.read(name)).hexdigest() for name in archive.namelist()} == hashes
    export = tmp_path/'export'
    export.mkdir()
    (export/'best_model.pt').write_bytes(b'packaging-byte-preservation-fixture')
    selection = dict(arm='R3', seed=41, checkpoint_sha256='1'*64, best_epoch=3, stop_reason='time_budget',
        run_identity='2'*64, config_identity='3'*64, weights_sha256=sha256(export/'best_model.pt'),
        reference_sha256=sha256(reference))
    (export/'selection.json').write_text(json.dumps(selection))
    adapted = make_package(reference, export, tmp_path/'adapted.zip')
    with zipfile.ZipFile(tmp_path/'adapted.zip') as archive:
        for name in names[1:]:
            assert hashlib.sha256(archive.read(name)).hexdigest() == hashes[name]
        assert json.loads(archive.read('model/manifest.json'))['choice']['panns_adaptation'] == selection
        assert archive.read('model/trained/panns_best_model.pt') == (export/'best_model.pt').read_bytes()
    assert baseline['inference_parity_verified'] is adapted['inference_parity_verified'] is False
    deploy = tmp_path/'deploy'
    deploy.mkdir()
    root = Path(__file__).resolve().parent
    for name in ('package_panns_submission.py', 'panns_submission.py', 'panns_submission_selection.py'):
        shutil.copyfile(root/name, deploy/name)
    subprocess.run([sys.executable, str(deploy/'package_panns_submission.py'), '--reference', str(reference),
        '--export', str(export), '--output', str(tmp_path/'deployed.zip')], cwd=deploy,
        env=dict(os.environ, PYTHONPATH=str(root)), check=True, capture_output=True, timeout=60)
    with zipfile.ZipFile(tmp_path/'deployed.zip') as archive:
        assert archive.read('model/runtime/panns_adaptation.py') == (root/'panns_adaptation.py').read_bytes()
    (base/'model/trained/file_ensemble.json').write_bytes(b'changed readout')
    with pytest.raises(SubmissionError, match='input changed'):
        make_package(reference, export, tmp_path/'changed.zip')
    assert not (tmp_path/'changed.zip').exists()


def test_orchestration_cli_imports_with_frozen_bundle_without_submission_engine(tmp_path):
    workspace = os.environ.get('PANNS_TEST_WORKSPACE')
    if workspace is None:
        pytest.skip('Set PANNS_TEST_WORKSPACE to the actual frozen cloud bundle')
    frozen = Path(workspace)/'notebook_execution_20260925/cloud_v12/final_package/bundle/code'
    assert not (frozen/'full_bank_contract.py').exists()
    root = Path(__file__).resolve().parent
    scripts = ('panns_submission_train.py', 'panns_cached_fake.py', 'panns_submission_evaluate.py',
               'package_panns_submission.py', 'panns_submission_verify.py', 'panns_submission_workflow.py')
    for name in (*scripts, 'panns_submission.py', 'panns_submission_selection.py', 'panns_fast_step.py'):
        shutil.copyfile(root/name, tmp_path/name)
    program = ("import runpy,sys; sys.modules['full_bank_submission']=None; "
               "script=sys.argv[1]; sys.argv=[script,'--help']; runpy.run_path(script,run_name='__main__')")
    for name in scripts:
        subprocess.run([sys.executable, '-c', program, str(tmp_path/name)], cwd=tmp_path,
            env=dict(os.environ, PYTHONPATH=os.pathsep.join((str(tmp_path), str(frozen)))),
            check=True, capture_output=True, timeout=60)
