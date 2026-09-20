"""Protect the extra input channels, checkpoint identity and original C08 behavior."""
from pathlib import Path
from dataclasses import replace
from tempfile import TemporaryDirectory
import hashlib
import json
import time
import unittest
import numpy as np
import torch
from music_temporal import TemporalRecipe, _batch, fit_temporal_head, load_temporal_head, predict_temporal_music
from test_temporal_augmentation import samples
from temporal_backbone import REGULARIZED_C08, verify_inputs


class TemporalInputDimensionChecks(unittest.TestCase):
    def test_regularization_changes_training_but_reload_inference_is_deterministic(self):
        # Protect against a configured regularizer being disconnected or active at inference.
        rows, records = samples()
        recipe = replace(REGULARIZED_C08,epochs=1,batch_size=4)
        with TemporaryDirectory() as directory:
            root = Path(directory)
            trained, _ = fit_temporal_head(rows[:8],records,(rows[8:],records),root/'decay',
                                           time.time()+60,recipe=recipe)
            unregularized, _ = fit_temporal_head(rows[:8],records,(rows[8:],records),root/'no_decay',
                time.time()+60,recipe=replace(recipe,weight_decay=0.))
            difference = (trained.projection.weight-unregularized.projection.weight).abs().max()
            self.assertGreater(float(difference.detach()),1e-8)
            loaded = load_temporal_head(root/'decay/temporal_music.pt')
            x, mask = _batch(rows[8:],records,'cpu')
            loaded.train()
            self.assertGreater(float((loaded(x,mask)-loaded(x,mask)).detach().abs().max()),1e-5)
            loaded.eval()
            with torch.inference_mode():
                torch.testing.assert_close(loaded(x,mask),loaded(x,mask),rtol=0,atol=0)
                torch.testing.assert_close(loaded(x,mask),trained(x,mask),rtol=0,atol=1e-6)

    def test_missing_or_modified_control_is_rejected_before_extraction(self):
        # A copied prediction must never silently become a different fixed control.
        with TemporaryDirectory() as directory:
            root = Path(directory)
            names = ['reference95_features_manifest.json','results/linear_control.npz']
            names += [f'results/M95_seed{seed}/{name}' for seed in (41,42,43)
                      for name in ('temporal_music.pt','training.json','tune_predictions.npz')]
            manifest = {}
            for name in names:
                path = root/name
                path.parent.mkdir(parents=True,exist_ok=True)
                path.write_bytes(name.encode())
                manifest[name] = hashlib.sha256(path.read_bytes()).hexdigest()
            (root/'input_manifest.json').write_text(json.dumps(dict(sha256=manifest)))
            verify_inputs(root)
            target = root/'results/M95_seed42/tune_predictions.npz'
            target.write_bytes(b'stale predictions')
            with self.assertRaisesRegex(ValueError,'Input integrity mismatch'):
                verify_inputs(root)
            target.unlink()
            with self.assertRaisesRegex(ValueError,'Input integrity mismatch'):
                verify_inputs(root)

    def test_extra_features_receive_gradients_and_survive_checkpoint_reload(self):
        # Given: only the 256 added feature channels carry class information.
        rows,original=samples()
        records={}
        for i,row in enumerate(rows):
            x=np.zeros((len(original[row['ID']]['music']),1024),dtype='float32')
            rng=np.random.default_rng(i)
            x[:,768:]=rng.normal(.7 if row['MUSIC_FAKE']=='1' else -.7,.15,size=x[:,768:].shape)
            records[row['ID']]=dict(original[row['ID']],music=x)
        with TemporaryDirectory() as directory:
            recipe=TemporalRecipe(epochs=5,learning_rate=1e-3,batch_size=4,nsynth_mass=0.,
                                  simplified=True,compact=True,dropout=.1,input_dim=1024,
                                  softmax_output=True,mlp_output=True)
            # When: fit and reload the wider-input C08 head.
            model,report=fit_temporal_head(rows[:8],records,(rows[8:],records),Path(directory),
                time.time()+60,recipe=recipe)
            loaded=load_temporal_head(Path(directory)/'temporal_music.pt')
            p=predict_temporal_music(loaded,rows[8:],records)
            # Then: additional channels participate, and loading reproduces measured learning.
            np.testing.assert_allclose(p,predict_temporal_music(model,rows[8:],records),atol=1e-6)
            self.assertLess(report['best_validation_loss'],.5)
            gradient=model.projection.weight.grad[:,768:]
            self.assertGreater(float(gradient.abs().sum()),0.)
            self.assertGreater(float(model.output[0].weight.grad.abs().sum()),0.)
            self.assertTrue(torch.all(model.output[-1].weight.grad.abs().sum(1)>0))
            x,mask=_batch(rows[8:],records,'cpu',input_dim=1024)
            with torch.inference_mode():
                logits=loaded(x,mask)
                np.testing.assert_allclose(p,logits.softmax(-1)[:,1].numpy(),atol=1e-6)
                silent=loaded(x,torch.ones_like(mask)).softmax(-1)[:,1]
                self.assertLess(float(silent.max()),1e-6)
            neutral={k:dict(r,music=np.zeros_like(r['music'])) for k,r in records.items()}
            self.assertGreater(float(np.max(np.abs(p-predict_temporal_music(loaded,rows[8:],neutral)))),.1)
            with self.assertRaises(ValueError):
                predict_temporal_music(loaded,rows[8:],original)


if __name__=='__main__':
    torch.set_num_threads(1)
    unittest.main()
