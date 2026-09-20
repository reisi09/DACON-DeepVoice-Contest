"""Protect selection isolation and real cached-feature search/checkpoint integration."""

import json
from pathlib import Path
from tempfile import TemporaryDirectory
import time
import unittest
from unittest.mock import patch
from contextlib import chdir
import hashlib
import zipfile

import numpy as np
import torch

from temporal_experiment import evaluate_locked_test, load_features, run_search, split_experiment
from export_temporal_features import export_features
from music_temporal import _batch


class ExperimentChecks(unittest.TestCase):
    def test_cpu_export_preserves_features_and_rejects_changed_source_or_cache(self):
        with TemporaryDirectory() as directory:
            root=Path(directory); old=root/'old'; old.mkdir(); (old/'cache').mkdir()
            audio=root/'source.wav'; audio.write_bytes(b'verified source fixture')
            audio_hash=hashlib.sha256(audio.read_bytes()).hexdigest()
            row=dict(ID='sample', path=str(audio), audio_sha256=audio_hash)
            identity=audio_hash+':cache-version'
            cache=old/'cache'/(hashlib.sha256(identity.encode()).hexdigest()+'.npz')
            music=np.random.default_rng(41).normal(size=(3,768)).astype('float32')
            np.savez_compressed(cache, identity=np.array(identity), music=music,
                presence=np.ones((3,2)), voice=np.zeros(3), raw=np.zeros(3),
                music_audible=np.ones(3), starts=np.arange(3)*64600)
            (old/'training.json').write_text(json.dumps(dict(cache_id='cache-version',asset_id='asset')))
            (old/'split.json').write_text(json.dumps(dict(expert=[row],fusion=[],dev=[])))
            export_features([row],old,root,root/'features.zip')
            destination=root/'destination';destination.mkdir()
            with zipfile.ZipFile(root/'features.zip') as z:z.extractall(destination)
            with chdir(destination):
                np.testing.assert_array_equal(load_features([row],old,root)['sample']['music'],music)
                copied=destination/'features/sample.npz'
                copied.write_bytes(copied.read_bytes()+b'corrupted')
                with self.assertRaisesRegex(ValueError,'provenance mismatch'):
                    load_features([row],old,root)
            audio.write_bytes(b'changed source')
            with self.assertRaises(AssertionError):
                export_features([row],old,root,root/'invalid.zip')

    def test_real_source_split_excludes_test_parents_and_ignores_test_labels(self):
        rows = json.loads(Path('_workspace/c3_final_upload/full_data_plan.json').read_text())['rows']
        baseline = json.loads(Path('results/v7_a100_20260915/results/baseline_predictions.json').read_text())
        ids = {r['ID'] for r in baseline}
        original = split_experiment(rows, ids)
        changed = [dict(r, FILE_FAKE=str(1-int(r['FILE_FAKE']))) if r['ID'] in ids else r for r in rows]
        perturbed = split_experiment(changed, ids)
        self.assertEqual(original['train'], perturbed['train'])
        self.assertEqual(original['tune'], perturbed['tune'])
        train_groups = {g for r in original['train'] for g in r['group_id'].split('|')}
        parents = {g for r in original['test'] for g in r['group_id'].split('|')}
        self.assertFalse(train_groups & parents)

    def test_search_frozen_selection_and_baseline_hybrid_inference(self):
        rng = np.random.default_rng(41)
        splits, records, baseline = {}, {}, []
        for partition, count in [('train', 24), ('tune', 8), ('test', 8)]:
            rows = []
            for i in range(count):
                identifier, label = f'{partition}-{i}', i % 2
                row = dict(ID=identifier, group_id=identifier, pcm_sha256=identifier,
                    MUSIC_FAKE=str(label), MUSIC_PRESENT='1', VOICE_PRESENT='0', VOICE_FAKE='',
                    FILE_FAKE=str(label), panel='music_seen' if label else 'music_real',
                    source='Echoes/a/TTA' if label else ('nsynth/a' if i % 4 else 'FMA/a'))
                rows.append(row)
                x = rng.normal(0, .1, (3+i%3, 768)).astype('float32')
                x[:, :64] += 2*label-1
                records[identifier] = dict(music=x, music_audible=np.ones(len(x)),
                                           starts=np.arange(len(x))*64600)
                if partition == 'test':
                    baseline.append(dict(ID=identifier, FILE_FAKE_PROB=.3, VOICE_FAKE_PROB=.2,
                        MUSIC_FAKE_PROB=.3, VOICE_PRESENT_PROB=.1, MUSIC_PRESENT_PROB=.9))
            splits[partition] = rows
        with TemporaryDirectory() as directory:
            output = Path(directory)
            with patch('music_temporal._batch', wraps=_batch) as batches:
                search = run_search(splits['train'], splits['tune'], records, records, output,
                                    time.time()+120, epochs=2, device='cpu', batch_size=7)
            sizes = [len(call.args[0]) for call in batches.call_args_list
                     if call.args[0][0]['ID'].startswith('train-')]
            self.assertEqual(set(sizes), {3, 7})
            self.assertEqual(sum(r['optimizer_steps'] for r in search), 8*2*4)
            selection = json.loads((output/'selection.json').read_text())
            expected = min(search, key=lambda r:(r['music_eer'], r['balanced_bce'], r['name']))
            self.assertEqual(selection['winner'], expected)
            frozen = (output/'selection.json').read_bytes()
            reports = evaluate_locked_test(splits['test'], records, baseline, output, device='cpu')
            self.assertEqual((output/'selection.json').read_bytes(), frozen)
            for result in reports[1:]:
                predictions = json.loads((output/f'{result["name"]}_test_predictions.json').read_text())
                for row in predictions:
                    self.assertEqual(row['VOICE_FAKE_PROB'], .2)
                    self.assertEqual(row['VOICE_PRESENT_PROB'], .1)
                    self.assertEqual(row['MUSIC_PRESENT_PROB'], .9)
                    self.assertAlmostEqual(row['FILE_FAKE_PROB'], max(.02,.9*row['MUSIC_FAKE_PROB']))


if __name__ == '__main__':
    torch.set_num_threads(1)
    unittest.main()
