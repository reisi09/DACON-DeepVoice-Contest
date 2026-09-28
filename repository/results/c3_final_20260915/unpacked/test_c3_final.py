"""Protect C3 quality filtering and source-disjoint calibration assignments."""

import unittest

import numpy as np
from sklearn.model_selection import StratifiedGroupKFold

from c3_data import component_ids, usable_windows
from v6_data import assert_disjoint


class FullTrainingChecks(unittest.TestCase):
    def test_sparse_note_is_excluded_but_long_file_keeps_its_active_window(self):
        note = np.zeros(64000, dtype='float32')
        note[:8000] = .1 * np.sin(np.arange(8000) * .1)
        self.assertEqual(usable_windows(note)[1], [])
        long_file = np.zeros(8 * 64600, dtype='float32')
        long_file[:32000] = .1 * np.sin(np.arange(32000) * .1)
        starts, kept = usable_windows(long_file)
        self.assertEqual(kept, [0])
        self.assertEqual(starts[kept[0]], 0)
        self.assertEqual(float(long_file[64600:].max()), 0.)

    def test_transitive_mixture_parents_never_cross_a_calibration_fold(self):
        groups = ['song-a', 'song-a|speaker-b', 'speaker-b|song-c', 'song-c']
        groups += [f'independent-{i}' for i in range(24)]
        rows = [dict(ID=str(i), group_id=g, pcm_sha256=f'pcm-{i}') for i, g in enumerate(groups)]
        components = component_ids(rows)
        self.assertEqual(len(set(components[:4])), 1)
        seen = set()
        splitter = StratifiedGroupKFold(3, shuffle=True, random_state=41)
        for train, held in splitter.split(np.zeros(len(rows)), np.arange(len(rows)) % 2, components):
            assert_disjoint([rows[i] for i in train], [rows[i] for i in held])
            self.assertFalse(seen.intersection(held))
            seen.update(held)
        self.assertEqual(seen, set(range(len(rows))))


if __name__ == '__main__':
    unittest.main()
