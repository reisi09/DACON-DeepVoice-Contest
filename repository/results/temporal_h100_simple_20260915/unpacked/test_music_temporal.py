"""Check temporal learning, masked inference and real training/checkpoint integration."""

from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory
import time
import unittest

import numpy as np
import torch

from music_temporal import (TemporalMusic, TemporalRecipe, fit_temporal_head,
                            load_temporal_head, predict_temporal_music)


class TemporalMusicChecks(unittest.TestCase):
    def test_original_branch_preserves_legacy_gradient_accumulation_order(self):
        # A forward-equivalent reordering changed gradients and long training curves.
        from copy import deepcopy
        torch.manual_seed(41)
        model = TemporalMusic()
        reference = deepcopy(model)
        x = torch.randn(8, 20, 768)
        mask = torch.zeros(8, 20, dtype=torch.bool)
        torch.manual_seed(41)
        actual = model(x, mask)
        actual.square().sum().backward()
        torch.manual_seed(41)
        hidden = torch.nn.functional.gelu(reference.projection(reference.input_norm(x).transpose(1, 2)))
        for block in reference.blocks:
            hidden = block(hidden, mask[:, None])
        gates = torch.tanh(reference.gate_tanh(hidden)) * torch.sigmoid(reference.gate_sigmoid(hidden))
        attention = reference.gate_score(gates).squeeze(1).softmax(1)
        attended = (hidden * attention[:, None]).sum(2)
        maxima = hidden.masked_fill(mask[:, None], -torch.inf).amax(2).masked_fill(mask.all(1)[:, None], 0)
        expected = reference.output(torch.cat([attended, maxima], 1)).squeeze(1)
        expected.square().sum().backward()
        torch.testing.assert_close(actual, expected, atol=0, rtol=0)
        for name, parameter in reference.named_parameters():
            torch.testing.assert_close(dict(model.named_parameters())[name].grad,
                                       parameter.grad, atol=0, rtol=0, msg=name)

    def test_simplified_head_executes_two_convolutions_max_only_and_reloads_after_training(self):
        # Protect the requested architecture, gradient connectivity and saved inference.
        torch.manual_seed(41)
        model = TemporalMusic(simplified=True).eval()
        executed = []
        handles = [layer.register_forward_hook(lambda *_: executed.append(1))
                   for block in model.blocks for layer in block.convs]
        x = torch.randn(3, 8, 768)
        mask = torch.zeros(3, 8, dtype=torch.bool)
        mask[0, 5:] = True
        logits = model(x, mask)
        self.assertEqual(len(executed), 2)
        for handle in handles:
            handle.remove()
        self.assertFalse(any('gate' in name for name, _ in model.named_parameters()))
        hidden = torch.nn.functional.gelu(model.projection(
            model.input_norm(x.masked_fill(mask[..., None], 0)).transpose(1, 2)))
        hidden = hidden.masked_fill(mask[:, None], 0)
        hidden = model.blocks[0](hidden, mask[:, None])
        expected = model.output(hidden.masked_fill(mask[:, None], -torch.inf).amax(2)).squeeze(1)
        torch.testing.assert_close(logits, expected)
        torch.nn.functional.binary_cross_entropy_with_logits(logits, torch.tensor([0., 1., 0.])).backward()
        for name, parameter in model.named_parameters():
            self.assertIsNotNone(parameter.grad, name)
            self.assertTrue(torch.isfinite(parameter.grad).all(), name)
            self.assertGreater(float(parameter.grad.abs().sum()), 0., name)
        alone = model(x[:1, :5], mask[:1, :5])
        torch.testing.assert_close(alone[0], logits.detach()[0], atol=1e-6, rtol=1e-5)
        self.assertTrue(torch.isfinite(model(x, torch.ones_like(mask))).all())
        from test_temporal_augmentation import samples
        rows, records = samples()
        recipe = TemporalRecipe(epochs=3, batch_size=4, nsynth_mass=0., crop_probability=.7,
                                simplified=True)
        with TemporaryDirectory() as directory:
            path = Path(directory)
            trained, report = fit_temporal_head(rows[:8], records, (rows[8:], records), path,
                                                time.time()+60, recipe=recipe)
            loaded = load_temporal_head(path/'temporal_music.pt')
            self.assertTrue(loaded.simplified)
            np.testing.assert_allclose(predict_temporal_music(trained, rows[8:], records),
                predict_temporal_music(loaded, rows[8:], records), atol=1e-6, rtol=0)
            probabilities = predict_temporal_music(loaded, rows[8:], records)
            labels = np.array([int(r['MUSIC_FAKE']) for r in rows[8:]])
            loss = -np.log(np.where(labels, probabilities, 1-probabilities)).mean()
            self.assertAlmostEqual(loss, report['best_validation_loss'], places=6)

    def test_padding_other_files_and_empty_music_cannot_change_a_file_prediction(self):
        torch.manual_seed(41)
        model = TemporalMusic().eval()
        original = torch.randn(1, 5, 768)
        alone = model(original, torch.zeros(1, 5, dtype=torch.bool))
        batch = torch.randn(128, 12, 768)
        batch[0, :5] = original[0]
        batch[0, 5:] = 10000
        mask = torch.zeros(128, 12, dtype=torch.bool); mask[0, 5:] = True
        together = model(batch, mask)
        torch.testing.assert_close(alone[0], together[0], atol=1e-6, rtol=1e-5)
        # Temporal order, unlike batch neighbors or padding, must affect the path.
        permuted = model(original.flip(1), torch.zeros(1, 5, dtype=torch.bool))
        self.assertGreater(float((alone - permuted).abs().max()), 1e-6)
        empty = model(batch, torch.ones_like(mask)).sigmoid()
        self.assertTrue(torch.isfinite(empty).all())
        self.assertLess(float(empty.max()), 1e-6)

    def test_every_temporal_convolution_and_attention_gate_receives_training_gradients(self):
        torch.manual_seed(41)
        model = TemporalMusic()
        x = torch.randn(4, 9, 768)
        loss = torch.nn.functional.binary_cross_entropy_with_logits(
            model(x, torch.zeros(4, 9, dtype=torch.bool)), torch.tensor([0., 1., 0., 1.]))
        loss.backward()
        weights = [(name, p) for name, p in model.named_parameters()
                   if name.endswith('weight') and ('blocks' in name or 'gate' in name)]
        for name, parameter in weights:
            self.assertIsNotNone(parameter.grad, name)
            self.assertTrue(torch.isfinite(parameter.grad).all(), name)
            self.assertGreater(float(parameter.grad.abs().sum()), 0., name)

    def test_cached_feature_fit_early_stopping_and_saved_inference_agree(self):
        rows, val, records = [], [], {}
        rng = np.random.default_rng(41)
        for split, count in [('train', 12), ('val', 6)]:
            for i in range(count):
                label = i % 2; identifier = f'{split}-{i}'
                row = dict(ID=identifier, group_id=identifier, pcm_sha256=identifier,
                           MUSIC_FAKE=str(label), source='FMA/FMA' if not label else 'Echoes/a/TTA')
                (rows if split == 'train' else val).append(row)
                x = rng.normal(0, .1, (3 + i % 3, 768)).astype('float32')
                x[:, :64] += (2 * label - 1) * (1 if split == 'train' else -1)
                records[identifier] = dict(music=x, music_audible=np.ones(len(x)),
                                           starts=np.arange(len(x)) * 64600)
        recipe = TemporalRecipe(epochs=12, patience=2, min_delta=1., nsynth_mass=0., dropout=.4)
        torch.manual_seed(recipe.seed)
        initial = TemporalMusic().eval()
        before = predict_temporal_music(initial, rows, records)
        with TemporaryDirectory() as directory:
            output = Path(directory)
            model, report = fit_temporal_head(rows, records, (val, records), output,
                                              time.time() + 60, recipe=recipe)
            self.assertLess(report['epochs_completed'], recipe.epochs)
            self.assertEqual(report['selected_epoch'],
                             int(np.argmin(report['validation_losses'])) + 1)
            after = predict_temporal_music(model, rows, records)
            labels = np.array([int(r['MUSIC_FAKE']) for r in rows])
            before_bce = -np.log(np.where(labels, before, 1 - before)).mean()
            after_bce = -np.log(np.where(labels, after, 1 - after)).mean()
            self.assertLess(after_bce, before_bce - 1e-5)
            val_probability = predict_temporal_music(model, val, records)
            val_labels = np.array([int(r['MUSIC_FAKE']) for r in val])
            val_bce = -np.log(np.where(val_labels, val_probability, 1 - val_probability)).mean()
            self.assertAlmostEqual(val_bce, report['best_validation_loss'], places=6)
            loaded = load_temporal_head(output / 'temporal_music.pt')
            self.assertEqual(loaded.blocks[0].dropout.p, recipe.dropout)
            record = records[val[0]['ID']]
            x = torch.from_numpy(record['music'])[None]
            mask = torch.zeros(1, x.shape[1], dtype=torch.bool)
            torch.testing.assert_close(model(x, mask), loaded(x, mask), atol=0, rtol=0)
            # Validation membership must be rejected before any new optimization.
            with self.assertRaisesRegex(ValueError, 'leakage'):
                fit_temporal_head(rows, records, ([rows[0], rows[1]], records), output,
                                  time.time() + 60, recipe=replace(recipe, epochs=1))


if __name__ == '__main__':
    torch.set_num_threads(1)
    unittest.main()
