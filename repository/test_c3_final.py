"""Protect C3 quality filtering and source-disjoint calibration assignments."""

import unittest
from contextlib import chdir
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import time

import numpy as np
from sklearn.model_selection import StratifiedGroupKFold

from c3_data import component_ids, usable_windows
from v6_data import assert_disjoint
from c3_full_train import train_final
from v6_train import trained_heads
from v7_ablation import Recipe


class FullTrainingChecks(unittest.TestCase):
    def test_oof_held_sources_do_not_select_epochs_and_final_refit_uses_selected_budget(self):
        # Exercise actual nested fitting, serialization and the packaging loader.
        rows, records, quality = [], {}, {}
        for i in range(80):
            label = int(i >= 40)
            source = ('FMA/FMA' if i < 20 else 'nsynth' if i < 40
                      else 'Echoes/a/TTA' if i < 60 else 'Echoes/b/TTA')
            row = dict(ID=str(i), group_id=str(i), pcm_sha256=str(i), source=source,
                       FILE_FAKE=str(label), MUSIC_FAKE=str(label), music_training='1')
            rows.append(row)
            x = np.zeros((1, 768), dtype='float32'); x[0, 0] = 2 * label - 1
            records[str(i)] = dict(music=x, music_audible=np.ones(1), starts=np.array([0]),
                                  presence=np.array([[.2, .9]]), voice=np.array([.1]),
                                  raw=np.array([.2 + .6 * label]))
            quality[str(i)] = dict(starts=[0], usable_indices=[0])
        plan = dict(rows=rows, quality=quality, policy={})
        with TemporaryDirectory() as directory, chdir(directory):
            Path('full_data_plan.json').write_text(json.dumps(plan))
            out = Path('run'); out.mkdir()
            info = train_final(plan, records, (out, 'test-assets', time.time() + 60),
                               recipe=Recipe('nested-check', .05, True, epochs=8,
                                             patience=2, min_delta=1.))
            folds = json.loads((out / 'oof_folds.json').read_text())
            ids = {r['ID'] for r in rows}
            for fold in folds:
                selection = fold['epoch_selection']
                fit_ids, val_ids = set(selection['train_ids']), set(selection['validation_ids'])
                held_ids = set(fold['held_ids'])
                self.assertFalse(fit_ids & val_ids or (fit_ids | val_ids) & held_ids)
                self.assertEqual(fit_ids | val_ids | held_ids, ids)
                self.assertEqual(fold['composition']['epochs_completed'], selection['selected_epoch'])
            selected = [f['epoch_selection']['selected_epoch'] for f in folds]
            self.assertEqual(info['epochs_completed'], int(np.median(selected)))
            self.assertLess(info['epochs_completed'], 8)
            self.assertEqual(info['epochs_requested'], info['epochs_completed'])
            head = trained_heads(out, 'test-assets')
            self.assertGreater(head.music(records['79']['music'], np.ones(1))[0],
                               head.music(records['0']['music'], np.ones(1))[0])

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
