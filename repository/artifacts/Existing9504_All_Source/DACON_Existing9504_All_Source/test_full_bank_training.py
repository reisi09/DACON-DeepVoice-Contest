"""Protect full coverage, split isolation, exact MIL, and real voice-head gradients."""
import importlib.util
import tempfile
from pathlib import Path

import numpy as np
import torch

from full_bank_features import window_truth, validate_cache
from mert12_submission import window_layout, WINDOW
from v6_data import sha256
from full_bank_train import Item, epoch_order, forward, loss_weights, step
from localization_models import WindowMIL


def test_full_bank_training_contract() -> None:
    torch.manual_seed(41)
    items = [Item({'ID': str(i), 'split': 'train', 'source': str(i%2), 'generator': ''},
                  np.random.default_rng(i).normal(size=(i+1, 16)).astype(np.float32),
                  np.full(i+1, np.nan, np.float32), float(i%2)) for i in range(9)]
    for epoch in range(1, 4):
        assert sorted(epoch_order(items, 41, epoch).tolist()) == list(range(len(items)))
    val = Item({**items[0].row, 'split': 'val'}, items[0].features, items[0].truth, 0.)
    try:
        epoch_order([*items, val], 41, 1)
    except ValueError:
        pass
    else:
        raise AssertionError('Validation accepted into training schedule')
    model = WindowMIL(torch.zeros(16), torch.ones(16))
    ragged, _, _ = forward(model, items, 'music', 0, 'cpu')
    padded = torch.nn.utils.rnn.pad_sequence([torch.from_numpy(x.features) for x in items], batch_first=True)
    mask = torch.arange(9)[None] < torch.arange(1, 10)[:, None]
    padded[~mask] = float('nan')
    reference = model(padded, mask).file_logits
    torch.testing.assert_close(ragged, reference, atol=2e-7, rtol=1e-6)
    gradient_reference = torch.autograd.grad(reference.sum(), model.head.network[0].weight, retain_graph=True)[0]
    gradient_ragged = torch.autograd.grad(ragged.sum(), model.head.network[0].weight)[0]
    torch.testing.assert_close(gradient_reference, gradient_ragged, atol=1e-6, rtol=1e-5)
    before = model.head.network[0].weight.detach().clone()
    optimizer = torch.optim.AdamW(model.parameters(), lr=.01)
    step(model, optimizer, items, 'music', 0, loss_weights(items), 2, 20, 'cpu')
    assert not torch.equal(before, model.head.network[0].weight)
    before = {k: v.clone() for k, v in model.state_dict().items()}
    try:
        step(model, optimizer, [val], 'music', 0, {'0': 1.}, 1, 20, 'cpu')
    except ValueError:
        pass
    else:
        raise AssertionError('Validation accepted into optimizer')
    assert all(torch.equal(before[k], v) for k, v in model.state_dict().items())

    # Instantiate the actual architecture's classifier; no substitute head layout.
    path = Path('_workspace/redesign_full_20260917/assets/baseline/model/df_arena_1b/conformer.py')
    spec = importlib.util.spec_from_file_location('contract_conformer', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    conformer = module.FinalConformer(emb_size=1280, heads=4, n_encoders=0)
    classifier = conformer.fc5
    voice = [Item(x.row, np.pad(x.features, ((0, 0), (0, 1264))), x.truth, x.label) for x in items]
    logits, _, _ = forward(classifier, voice, 'voice', 0, 'cpu')
    expected = torch.stack([classifier(torch.from_numpy(x.features)).softmax(-1)[:, 0].max() for x in voice])
    torch.testing.assert_close(logits.sigmoid(), expected)
    loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, torch.tensor([x.label for x in voice]))
    loss.backward()
    assert classifier.weight.grad is not None and classifier.weight.grad.abs().sum() > 0
    assert all(p.grad is None for name, p in conformer.named_parameters() if not name.startswith('fc5.'))
    unknown = window_truth({'event_intervals': '', 'MUSIC_FAKE': '1'}, np.array([0]), np.array([64600]))
    assert np.isnan(unknown).all()
    explicit = window_truth({'event_intervals': '[{"component":"music","start":0,"end":1,"label":0},{"component":"music","start":1,"end":2,"label":1}]'},
                            np.array([0, 16000]), np.array([32000, 16000]))
    np.testing.assert_allclose(explicit, [.5, 1.])
    with tempfile.TemporaryDirectory() as folder:
        path = Path(folder) / 'cache.npz'
        samples = 16000 * 31 + 13
        starts = window_layout(samples)
        valid = np.minimum(WINDOW, samples-starts)
        arrays = dict(identity=np.array('bound'), samples=np.array(samples), starts=starts,
                      valid_samples=valid, voice_embeddings=np.zeros((8, 1280), np.float32),
                      mert_windows=np.zeros((len(starts), 1024), np.float32), presence=np.array([.9, .1]),
                      window_truth=np.full(len(starts), np.nan, np.float32))
        np.savez(path, **arrays)
        original_hash = sha256(path)
        validate_cache(path, 'bound', original_hash)
        arrays['starts'] = starts[:-1]
        arrays['valid_samples'] = valid[:-1]
        arrays['mert_windows'] = arrays['mert_windows'][:-1]
        arrays['window_truth'] = arrays['window_truth'][:-1]
        np.savez(path, **arrays)
        try:
            validate_cache(path, 'bound', sha256(path))
        except ValueError:
            pass
        else:
            raise AssertionError('Truncated full-duration cache accepted')
        try:
            validate_cache(path, 'bound', original_hash)
        except ValueError:
            pass
        else:
            raise AssertionError('Changed cache bytes accepted')


def test_monitor_recovers_from_token_failure():
    """A transient credential fetch must not abandon the paid-GPU monitor."""
    import contextlib
    import io
    import runpy
    from types import SimpleNamespace
    from unittest.mock import patch

    class MissingObject(Exception):
        pass

    def missing(**kwargs):
        raise MissingObject()

    remote = SimpleNamespace(exceptions=SimpleNamespace(NoSuchKey=MissingObject), get_object=missing)
    delays = []
    def sleep(seconds):
        delays.append(seconds)
        if seconds == 300:
            raise StopIteration('Recovered and reached the next regular poll')

    with patch('full_bank_upload.client', side_effect=[RuntimeError('token timeout'), (remote, 'bucket', 'prefix/')]) as client, \
         patch('sys.argv', ['full_bank_ops.py', 'watch']), patch('time.sleep', side_effect=sleep), \
         contextlib.redirect_stdout(io.StringIO()):
        try:
            runpy.run_path(str(Path(__file__).with_name('full_bank_ops.py')), run_name='__main__')
        except StopIteration:
            pass
    assert client.call_count == 2 and delays == [60, 300]


def test_startup_detaches_and_resume_rejects_corruption():
    import os
    import signal
    import subprocess
    import tempfile
    from full_bank_ops import STARTUP, LEGACY_STARTUP
    from full_bank_run import audio_valid, digest

    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        (root/'bootstrap.sh').write_text('sleep 5\n')
        for script, expected_blocking in ((LEGACY_STARTUP, True), (STARTUP, False)):
            script = script.decode().replace('/shared/full-bank-mil-8020-20260918', temporary)
            process = subprocess.Popen(['bash', '-c', script], stdout=subprocess.PIPE,
                                       stderr=subprocess.PIPE, start_new_session=True)
            blocked = False
            try:
                process.communicate(timeout=.5)
            except subprocess.TimeoutExpired:
                blocked = True
            finally:
                try:
                    os.killpg(process.pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
                process.wait()
            assert blocked == expected_blocking
        audio = root/'audio.wav'
        audio.write_bytes(b'original verified audio bytes')
        row = {'path': audio.name, 'audio_sha256': digest(audio)}
        assert audio_valid(root, row)
        audio.write_bytes(b'interrupted extraction')
        assert not audio_valid(root, row)
        audio.unlink()
        assert not audio_valid(root, row)


if __name__ == '__main__':
    test_full_bank_training_contract()
    test_monitor_recovers_from_token_failure()
    test_startup_detaches_and_resume_rejects_corruption()
    print('PASS: every-row coverage, no validation optimization, exact MIL/gradients, original fc5 gradients and fake-index mapping, unknown-label masking')
