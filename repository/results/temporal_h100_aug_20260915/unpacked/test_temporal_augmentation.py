"""Protect cached-data integrity, crop labels, and training-only augmentation."""

from dataclasses import replace
from pathlib import Path
import json
from tempfile import TemporaryDirectory
import time
import unittest
from unittest.mock import patch

import numpy as np
import torch

import music_temporal as temporal
from temporal_augmentation_experiment import run_augmentation_search
from temporal_experiment import evaluate_locked_test


def samples():
    rows, records = [], {}
    for i in range(12):
        label = i % 2
        row = dict(ID=str(i), group_id=str(i), pcm_sha256=str(i), MUSIC_PRESENT='1',
                   MUSIC_FAKE=str(label), source='Echoes/a/TTA' if label else 'FMA/FMA',
                   panel='music_seen' if label else 'music_real')
        x = np.random.default_rng(i).normal(size=(20, 768)).astype('float32')
        x[:, :64] += 2 * label - 1
        audible = np.ones(20); audible[[1, 7, 12]] = 0
        rows.append(row)
        records[row['ID']] = dict(music=x, music_audible=audible,
                                  starts=np.arange(20) * 64600)
    return rows, records


class AugmentationChecks(unittest.TestCase):
    def test_nine_run_selection_uses_seed_means_and_locked_evaluation_keeps_all_candidates(self):
        rows, records = samples()
        rows[2]['source'] = 'nsynth/instrument'
        train, tune = rows[:8], rows[8:]
        heldout = [dict(r, ID='test-'+r['ID'], group_id='test-'+r['ID'],
                        pcm_sha256='test-'+r['ID'], FILE_FAKE=r['MUSIC_FAKE'],
                        VOICE_PRESENT='0', VOICE_FAKE='') for r in tune]
        records.update({r['ID']:records[r['ID'].removeprefix('test-')] for r in heldout})
        baseline = [dict(ID=r['ID'], FILE_FAKE_PROB=.3, VOICE_FAKE_PROB=.2,
                        MUSIC_FAKE_PROB=.3, VOICE_PRESENT_PROB=.1, MUSIC_PRESENT_PROB=.9)
                    for r in heldout]
        with TemporaryDirectory() as directory:
            output = Path(directory)
            search = run_augmentation_search(train, tune, records, records, output,
                                             time.time()+60, epochs=1, device='cpu', batch_size=4)
            frozen = (output/'selection.json').read_bytes()
            selection = json.loads(frozen)
            for summary in selection['summaries']:
                values = [r['music_eer'] for r in search if r['setting'] == summary['name']]
                self.assertEqual(len(values), 3)
                self.assertAlmostEqual(summary['music_eer_mean'], float(np.mean(values)))
            reports = evaluate_locked_test(heldout, records, baseline, output, device='cpu')
            self.assertEqual((output/'selection.json').read_bytes(), frozen)
            self.assertEqual({r['name'] for r in reports[2:]}, {r['name'] for r in search})
            for result in reports[1:]:
                predictions = json.loads((output/f'{result["name"]}_test_predictions.json').read_text())
                for row in predictions:
                    self.assertAlmostEqual(row['FILE_FAKE_PROB'], max(.02,.9*row['MUSIC_FAKE_PROB']))

    def test_crops_preserve_time_masks_and_cache_and_pair_across_mask_conditions(self):
        from temporal_augmentation import augment_records
        rows, records = samples()
        originals = {key: {k: v.copy() for k, v in rec.items()} for key, rec in records.items()}
        recipe = replace(temporal.TemporalRecipe(), crop_probability=1.)
        cropped, counts = augment_records(rows, records, recipe,
                                          np.random.default_rng(3), np.random.default_rng(4))
        combined, _ = augment_records(rows, records, replace(recipe, feature_mask_probability=1.),
                                      np.random.default_rng(3), np.random.default_rng(4))
        repeated, _ = augment_records(rows, records, recipe,
                                       np.random.default_rng(3), np.random.default_rng(4))
        self.assertEqual(counts['cropped'], len(rows))
        for row in rows:
            key = row['ID']; value = cropped[key]
            positions = np.searchsorted(records[key]['starts'], value['starts'])
            np.testing.assert_array_equal(np.diff(positions), np.ones(len(positions)-1))
            np.testing.assert_array_equal(value['music'], records[key]['music'][positions])
            np.testing.assert_array_equal(value['music_audible'], records[key]['music_audible'][positions])
            np.testing.assert_array_equal(combined[key]['starts'], value['starts'])
            np.testing.assert_array_equal(repeated[key]['music'], value['music'])
            self.assertTrue(value['music_audible'].any())
            self.assertGreater(np.count_nonzero(value['music'] != combined[key]['music']), 0)
            for field in originals[key]:
                np.testing.assert_array_equal(records[key][field], originals[key][field])

    def test_partial_unknown_labels_and_single_window_are_not_cropped(self):
        from temporal_augmentation import augment_records
        rows, records = samples()
        rows[0]['panel'] = 'splice'
        rows[1]['source'] = 'unknown/provider'
        records['2'] = {k: v[:1].copy() for k, v in records['2'].items()}
        records['3']['music_audible'][:] = 0
        records['3']['music_audible'][19] = 1
        out, _ = augment_records(rows[:4], records,
            replace(temporal.TemporalRecipe(), crop_probability=1.),
            np.random.default_rng(3), np.random.default_rng(4))
        for key in ['0', '1', '2']:
            for field in records[key]:
                np.testing.assert_array_equal(out[key][field], records[key][field])
        self.assertTrue(out['3']['music_audible'].any())

    def test_actual_training_augments_inputs_without_changing_validation(self):
        rows, records = samples()
        train, val = rows[:8], rows[8:]
        recipe = replace(temporal.TemporalRecipe(), epochs=2, batch_size=4,
                         nsynth_mass=0., crop_probability=1., feature_mask_probability=1.)
        training_changed, validation_seen = [], []
        original_batch = temporal._batch

        def observed_batch(batch_rows, batch_records, device):
            for row in batch_rows:
                key = row['ID']
                if row in val:
                    np.testing.assert_array_equal(batch_records[key]['music'], records[key]['music'])
                    validation_seen.append(key)
                else:
                    training_changed.append(not np.array_equal(batch_records[key]['music'], records[key]['music']))
            return original_batch(batch_rows, batch_records, device)

        with TemporaryDirectory() as directory:
            with patch('music_temporal._batch', side_effect=observed_batch):
                model, report = temporal.fit_temporal_head(train, records, (val, records),
                    Path(directory), time.time()+60, recipe=recipe)
            self.assertTrue(all(training_changed))
            self.assertEqual(len(training_changed), len(train)*2)
            self.assertEqual(len(validation_seen), len(val)*2)
            self.assertEqual(sum(r['cropped'] for r in report['augmentation_history']), len(train)*2)
            loaded = temporal.load_temporal_head(Path(directory)/'temporal_music.pt')
            np.testing.assert_allclose(temporal.predict_temporal_music(model, val, records),
                temporal.predict_temporal_music(loaded, val, records), atol=1e-6, rtol=0)


if __name__ == '__main__':
    torch.set_num_threads(1)
    unittest.main()
