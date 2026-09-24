"""Protect masked supervision, reference numerics, lineage and artifact integrity."""
import hashlib
import io
import json
import tarfile

import numpy as np
import pytest
import soundfile as sf
import torch

from decision_heads import SINGING, VOICE_CORE, logistic, tagger_features
from panns_adaptation import masked_file_loss, presence_logits
from panns_adaptation_run import audit_archive, quarantine_roles


def test_masked_unknown_cannot_change_loss_or_gradient():
    logits = torch.tensor([[.2, -.8], [.7, .1]], dtype=torch.float64, requires_grad=True)
    known = torch.tensor([[True, False], [True, True]])
    fraction = torch.tensor([1., .5], dtype=torch.float64)
    targets = torch.tensor([[1., float('nan')], [0., 1.]], dtype=torch.float64)
    loss = masked_file_loss(logits, targets, known, fraction)
    grad, = torch.autograd.grad(loss, logits)
    targets[0, 1] = -999
    changed = masked_file_loss(logits, targets, known, fraction)
    changed_grad, = torch.autograd.grad(changed, logits)
    expected = (torch.nn.functional.softplus(-logits[0, 0]) +
                torch.nn.functional.softplus(logits[1, 0]) +
                2 * torch.nn.functional.softplus(-logits[1, 1])) / 4
    torch.testing.assert_close(loss, expected, rtol=0, atol=1e-12)
    assert torch.equal(loss, changed) and torch.equal(grad, changed_grad)
    assert grad[0, 1] == 0 and torch.isfinite(grad).all()
    with pytest.raises(ValueError):
        masked_file_loss(logits, targets, known, torch.tensor([1., 0.]))


def test_readout_matches_numpy_and_reaches_selected_tags():
    labels = list(dict.fromkeys([*VOICE_CORE, *SINGING, 'Music', 'Guitar']))
    groups = dict(voice=list(VOICE_CORE) + list(SINGING), music=['Music', 'Guitar'])
    rng = np.random.default_rng(41)
    probs = rng.random((7, len(labels))).astype(np.float32)
    features = tagger_features(probs, labels, groups) | {'stem_acc_frac': .7}
    head = dict(features=list(features), mean=[.1]*16, scale=[1.2]*16,
                coef=rng.normal(size=16).tolist(), intercept=-.8)
    heads = dict(VOICE_PRESENT=head, MUSIC_PRESENT=head)
    tags = torch.tensor(probs, requires_grad=True)
    logits = presence_logits(tags, {'stem_acc_frac': .7}, heads, labels, groups)
    expected = logistic(head, features)
    assert abs(float(logits[0].sigmoid().detach()) - expected) < 1e-7
    logits.sum().backward()
    assert torch.isfinite(tags.grad).all() and tags.grad.abs().sum() > 0
    # Order cannot influence a file's segment statistics or readout.
    reverse = presence_logits(tags.flip(0), {'stem_acc_frac': .7}, heads, labels, groups)
    torch.testing.assert_close(logits, reverse, rtol=0, atol=2e-7)


def test_audit_rejects_corrupt_payload_and_unsafe_members(tmp_path):
    stream = io.BytesIO()
    sf.write(stream, np.linspace(-.2, .2, 800, dtype=np.float32), 16000, format='WAV')
    payload = stream.getvalue()
    row = dict(ID='a', path='audio/a.wav', sha256=hashlib.sha256(payload).hexdigest(), bytes=len(payload))
    archive = tmp_path/'audio.tar'
    def write(name):
        with tarfile.open(archive, 'w') as tar:
            info = tarfile.TarInfo(name); info.size = len(payload)
            tar.addfile(info, io.BytesIO(payload))
    write(row['path'])
    verified, missing = audit_archive(archive, [row])
    assert verified['a']['decoded_samples'] == 800 and not missing
    with pytest.raises(ValueError, match='hash'):
        audit_archive(archive, [row | {'sha256': '0'*64}])
    write('../audio/a.wav')
    with pytest.raises(ValueError, match='Unsafe'):
        audit_archive(archive, [row])


def test_parent_closure_quarantines_train_evaluation_collision():
    rows = [dict(ID='parent', group_keys=['speaker:a'], role='development'),
            dict(ID='child', group_keys=[], parent_ids=['parent'], role='train'),
            dict(ID='clean', group_keys=['speaker:b'], role='train')]
    result = quarantine_roles(rows)
    assert result['parent']['quarantined'] and result['child']['quarantined']
    assert not result['clean']['quarantined']


def test_cache_identity_and_payload_corruption_rejected(tmp_path):
    from panns_adaptation_run import load_cache, sha256
    path = tmp_path/'features.npz'
    np.savez(path, identity='correct', block5=np.ones((1, 1024, 12, 2), np.float32))
    entry = dict(path=str(path), identity='correct', sha256=sha256(path))
    assert np.array_equal(load_cache(entry)['block5'], np.ones((1, 1024, 12, 2), np.float32))
    with pytest.raises(ValueError, match='identity'):
        load_cache(entry | {'identity': 'different-audio-or-prefix'})
    with path.open('ab') as stream:
        stream.write(b'corruption')
    with pytest.raises(ValueError, match='hash'):
        load_cache(entry)


def test_checkpoint_rejects_changed_data_before_loading(tmp_path):
    from panns_adaptation_run import restore_checkpoint
    path = tmp_path/'checkpoint.pt'
    torch.save(dict(run_identity='original', model={}), path)
    with pytest.raises(ValueError, match='identity'):
        restore_checkpoint(path, 'changed-data', None, None)


def test_real_checkpoint_identity_gradient_and_frozen_state():
    """A cached real waveform must drive adapter through the frozen block6."""
    import os
    from pathlib import Path
    from panns_adaptation_run import (frozen_digest, load_cache, model_for_arm,
                                      optimization_step, readout_assets, trainable_state)
    root = os.environ.get('DACON_LOCAL_DATA_ROOT')
    if not root:
        pytest.skip('Real checkpoint/audio not configured; not a readiness pass')
    work = Path(root)/'_workspace/panns_cnn_v1'
    index = json.loads((work/'cache/index.json').read_text())
    entries = index['entries'][:4]
    package = Path(json.loads((work/'reference_lock.json').read_text())['package'])
    heads, labels, groups = readout_assets(package)
    torch.set_num_threads(4)
    model = model_for_arm(package, 'R2', 41, 'cpu')
    original = frozen_digest(model)
    initial = trainable_state(model)
    cached = load_cache(entries[0])
    with torch.no_grad():
        tags = model.tags_from_block5(torch.from_numpy(cached['block5']))
    torch.testing.assert_close(tags, torch.from_numpy(cached['tags']), rtol=0, atol=1e-6)
    model.train(True)
    optimizer = torch.optim.AdamW(model.optimizer_groups(), weight_decay=1e-4, foreach=False)
    for _ in range(5):
        optimization_step(model, optimizer, entries, heads, labels, groups, [1., 1.], 'cpu', 1)
    final = trainable_state(model)
    assert all(torch.linalg.vector_norm(final[k]-v) > 0 for k, v in initial.items())
    assert frozen_digest(model) == original
    with torch.no_grad():
        changed = model.tags_from_block5(torch.from_numpy(cached['block5']))
    assert float((changed-tags).abs().max()) > 1e-6


def test_cache_index_target_or_stem_mutation_is_rejected(tmp_path):
    from panns_adaptation_run import read_cache_index, sha256, cache_provenance
    directory = tmp_path/'cache'; directory.mkdir()
    path = directory/'index.json'
    payload = dict(config_identity='locked', provenance=cache_provenance(), entries=[dict(ID='a', role='train', targets=[1, 0], stems={'mix_rms_db': -23})])
    path.write_text(json.dumps(payload))
    path.with_suffix('.sha256').write_text(sha256(path))
    config = dict(output=str(tmp_path), identity='locked', panel_ids=['a'])
    assert read_cache_index(config)['entries'][0]['targets'] == [1, 0]
    payload['entries'][0]['targets'] = [0, 1]
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match='index hash'):
        read_cache_index(config)


def test_cache_rejects_changed_preprocessing_runtime(tmp_path):
    from panns_adaptation_run import cache_provenance, read_cache_index, sha256
    directory = tmp_path/'cache'; directory.mkdir()
    path = directory/'index.json'
    provenance = cache_provenance()
    provenance['versions']['soundfile'] = 'changed-runtime'
    path.write_text(json.dumps(dict(config_identity='locked', provenance=provenance, entries=[])))
    path.with_suffix('.sha256').write_text(sha256(path))
    with pytest.raises(ValueError, match='provenance'):
        read_cache_index(dict(output=str(tmp_path), identity='locked', panel_ids=[]))


def test_interrupted_namespace_is_preserved_and_budget_counts_replay(tmp_path):
    from panns_adaptation_run import prepare_pilot_output, check_budget, validate_pilot_steps
    sentinel = tmp_path/'progress.json'; sentinel.write_text('interrupted')
    with pytest.raises(FileExistsError, match='incomplete'):
        prepare_pilot_output(tmp_path, resume=False)
    assert sentinel.read_text() == 'interrupted'
    with pytest.raises(ValueError, match='200'):
        validate_pilot_steps(200)
    config = dict(memory_limit_bytes=100, pilot_timeout_seconds=60)
    with pytest.raises(RuntimeError, match='budget'):
        check_budget(config, 10, dict(peak_rss_bytes=101, mps_driver_bytes=0))
    with pytest.raises(RuntimeError, match='budget'):
        check_budget(config, 61, dict(peak_rss_bytes=1, mps_driver_bytes=0))


def test_recovered_noise_cannot_change_role_labels_or_historical_hash(tmp_path):
    from panns_adaptation_data import admit_noise, sha256
    audio = tmp_path/'noise.wav'
    sf.write(audio, np.linspace(-.2, .2, 800, dtype=np.float32), 16000)
    old = dict(ID='noise:a', source='musan_noise', role='development', sha256='original',
               VOICE_PRESENT=0, MUSIC_PRESENT=0, group_keys=['original-group'])
    row = old | dict(path='noise.wav', sha256=sha256(audio), bytes=audio.stat().st_size,
                     original_processed_sha256='original')
    manifest = tmp_path/'manifest.jsonl'
    for mutation in ({'role': 'train'}, {'MUSIC_PRESENT': 1}, {'original_processed_sha256': 'wrong'}):
        manifest.write_text(json.dumps(row | mutation)+'\n')
        with pytest.raises(ValueError, match='historical'):
            admit_noise([old], {}, manifest, sha256(manifest))
    manifest.write_text(json.dumps(row)+'\n')
    verified = {}
    result = admit_noise([old], verified, manifest, sha256(manifest))
    assert result[0]['role'] == 'development' and result[0]['sha256'] == sha256(audio)
    assert verified['noise:a']['sha256'] == sha256(audio)
