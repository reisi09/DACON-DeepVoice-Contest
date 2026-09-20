"""Protect tune-only screening, seed aggregation and frozen checkpoint evaluation."""

from dataclasses import replace
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import time
import unittest

import numpy as np

from music_temporal import TemporalRecipe
from temporal_experiment import evaluate_locked_test
from test_temporal_augmentation import samples


class CompactTuningChecks(unittest.TestCase):
    def test_real_search_selects_seed_mean_and_evaluates_only_frozen_comparisons(self):
        # A seed-41-only winner or test-informed reselection would invalidate this experiment.
        from temporal_tuning import run_compact_search
        rows, records = samples()
        rows[2]['source'] = 'nsynth/instrument'
        base = TemporalRecipe(epochs=1, batch_size=4, simplified=True, compact=True)
        candidates = {'C00': base, 'C01': replace(base, learning_rate=1e-4),
                      'C02': replace(base, dropout=.4)}
        heldout = [dict(r, ID='test-'+r['ID'], group_id='test-'+r['ID'],
                       pcm_sha256='test-'+r['ID'], FILE_FAKE=r['MUSIC_FAKE'],
                       VOICE_PRESENT='0', VOICE_FAKE='') for r in rows[8:]]
        records.update({r['ID']: records[r['ID'].removeprefix('test-')] for r in heldout})
        baseline = [dict(ID=r['ID'], FILE_FAKE_PROB=.3, VOICE_FAKE_PROB=.2,
                        MUSIC_FAKE_PROB=.3, VOICE_PRESENT_PROB=.1, MUSIC_PRESENT_PROB=.9)
                    for r in heldout]
        with TemporaryDirectory() as directory:
            output = Path(directory)
            results = run_compact_search(rows[:8], rows[8:], records, records, output,
                time.time()+60, candidates=candidates, shortlist_size=2, device='cpu')
            selection_bytes = (output/'selection.json').read_bytes()
            selection = json.loads(selection_bytes)
            shortlist = json.loads((output/'shortlist.json').read_text())['candidates']
            for summary in selection['summaries']:
                members = [r for r in results if r['setting'] == summary['name']]
                self.assertEqual({r['seed'] for r in members}, {41, 42, 43})
                self.assertAlmostEqual(summary['music_eer_mean'], np.mean([r['music_eer'] for r in members]))
            eligible = [s for s in selection['summaries'] if s['name'] in shortlist]
            expected = min(eligible, key=lambda s: (s['music_eer_mean'], s['music_eer_std'],
                                                   s['balanced_bce_mean'], s['name']))
            self.assertEqual(selection['winner']['name'], expected['name'])
            reports = evaluate_locked_test(heldout, records, baseline, output, device='cpu')
            self.assertEqual((output/'selection.json').read_bytes(), selection_bytes)
            self.assertEqual({r['name'] for r in reports[2:]}, set(selection['evaluation_candidates']))
            self.assertEqual(set(selection['comparison_groups']), {'S128', 'C00', expected['name']})
            for result in results:
                training = json.loads((output/result['name']/'training.json').read_text())
                self.assertEqual(training['train_ids'], [r['ID'] for r in rows[:8]])
                self.assertEqual(training['validation_ids'], [r['ID'] for r in rows[8:]])


if __name__ == '__main__':
    unittest.main()
