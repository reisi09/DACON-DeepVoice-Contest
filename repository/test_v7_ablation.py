"""Protect source balancing and equal-file normalization from dataset-size bias."""

import unittest
import time

import numpy as np

from v7_ablation import Recipe, fit_head, source_weights
from v7_features import crop_starts


class DataAblationChecks(unittest.TestCase):
    def test_early_stopping_restores_best_validation_weights_and_rejects_source_leakage(self):
        # Training improves while a shifted validation distribution gets worse.
        rows, validation, records = [], [], {}
        for split in ('train', 'validation'):
            for i in range(8):
                label = i % 2
                row = dict(ID=f'{split}-{i}', group_id=f'{split}-{i}',
                           pcm_sha256=f'{split}-{i}', MUSIC_FAKE=str(label),
                           source='FMA/FMA' if label == 0 else 'Echoes/a/TTA')
                (rows if split == 'train' else validation).append(row)
                x = np.zeros((1, 768), dtype='float32')
                x[0, 0] = (2 * label - 1) * (1 if split == 'train' else -1)
                if split == 'validation':
                    x = np.concatenate([x, x * .5])
                records[row['ID']] = dict(music=x, music_audible=np.ones(len(x)))
        recipe = Recipe('early-stop-check', 0., True, epochs=30, patience=3)
        stopped, losses, info = fit_head(rows, records, recipe, time.time() + 60,
                                         validation=(validation, records))
        reference, _, _ = fit_head(rows, records,
            Recipe('best-epoch-reference', 0., True, epochs=1), time.time() + 60)
        self.assertEqual(info['selected_epoch'], 1)
        self.assertEqual(info['epochs_completed'], 4)
        self.assertLess(losses[-1], losses[0])
        self.assertGreater(info['validation_losses'][-1], info['validation_losses'][0])
        np.testing.assert_array_equal(stopped.music_weight, reference.music_weight)
        np.testing.assert_array_equal(stopped.mean, reference.mean)
        np.testing.assert_array_equal(stopped.scale, reference.scale)
        probabilities = np.array([stopped.music(records[r['ID']]['music'],
            records[r['ID']]['music_audible']).max() for r in validation])
        targets = np.array([int(r['MUSIC_FAKE']) for r in validation])
        inference_bce = -np.log(np.where(targets, probabilities, 1 - probabilities)).mean()
        self.assertAlmostEqual(info['best_validation_loss'], inference_bce, places=6)
        # A shared source must fail before it can influence stopping decisions.
        leaked = [dict(r, group_id=rows[0]['group_id']) for r in validation]
        with self.assertRaisesRegex(ValueError, 'leakage'):
            fit_head(rows, records, recipe, time.time() + 60, validation=(leaked, records))

    def test_provider_duplication_cannot_increase_its_total_loss_mass(self):
        rows = [dict(ID=str(i), MUSIC_FAKE=str(label), source=source)
                for i, (label, source) in enumerate([(0, 'FMA/FMA'), (0, 'nsynth'),
                    (1, 'Echoes/a/TTA'), (1, 'Echoes/b/TTA')])]
        expanded = rows + [dict(rows[2], ID=f'duplicate-{i}') for i in range(17)]
        for mass in (0., .05):
            def totals(items):
                weights = source_weights(items, mass)
                return {source: sum(w for r, w in zip(items, weights) if r['source'] == source)
                        for source in {r['source'] for r in items}}
            before, after = totals(rows), totals(expanded)
            for source in before:
                self.assertAlmostEqual(before[source], after[source])
            self.assertAlmostEqual(before['nsynth'], mass)

    def test_repeated_windows_do_not_bias_equal_file_statistics_or_optimization(self):
        rows = [dict(ID=str(i), MUSIC_FAKE=str(i % 2), source='FMA/FMA' if i % 2 == 0 else 'Echoes/a/TTA')
                for i in range(8)]
        rng = np.random.default_rng(41)
        records = {}
        for row in rows:
            vector = rng.normal(0, .01, (1, 768)).astype('float32')
            vector[0, 0] = 2 * int(row['MUSIC_FAKE']) - 1
            records[row['ID']] = dict(music=vector, music_audible=np.ones(1))
        expanded = dict(records)
        expanded['0'] = dict(music=np.repeat(records['0']['music'], 31, axis=0), music_audible=np.ones(31))
        recipe = Recipe('check', 0., True, epochs=60)
        a, losses, _ = fit_head(rows, records, recipe, time.time() + 60)
        b, _, _ = fit_head(rows, expanded, recipe, time.time() + 60)
        np.testing.assert_allclose(a.mean, b.mean, atol=1e-7)
        np.testing.assert_allclose(a.scale, b.scale, atol=1e-7)
        pa = np.array([a.music(records[r['ID']]['music'], np.ones(1))[0] for r in rows])
        pb = np.array([b.music(records[r['ID']]['music'], np.ones(1))[0] for r in rows])
        np.testing.assert_allclose(pa, pb, atol=1e-5)
        self.assertLess(losses[-1], losses[0] * .5)
        self.assertGreater(pa[1::2].min(), pa[::2].max())

    def test_four_second_sampling_does_not_repeat_short_audio_or_use_out_of_range_starts(self):
        np.testing.assert_array_equal(crop_starts(64000, 'short'), [0])
        starts = crop_starts(800000, 'long')
        np.testing.assert_array_equal(starts, crop_starts(800000, 'long'))
        self.assertTrue(np.all(starts >= 0) and np.all(starts + 64000 <= 800000))
        self.assertFalse(np.array_equal(starts, crop_starts(800000, 'another-song')))


if __name__ == '__main__':
    unittest.main()
