"""Protect temporal detail, overlap ordering, augmentation and two-class training."""

from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory
import time
import unittest

import numpy as np
import torch

from music_temporal import TemporalMusic, TemporalRecipe, _batch, fit_temporal_head, load_temporal_head, predict_temporal_music
from temporal_augmentation import augment_records
from test_temporal_augmentation import samples
from temporal_resolution import pool_frames, load_pair, select_resolution


class TemporalResolutionChecks(unittest.TestCase):
    def test_eer_sum_roundoff_cannot_override_seed_stability_tiebreak(self):
        coarse=dict(name='coarse',music_eer_mean=.2764148651931018,
                    music_eer_std=.011637073215985925,balanced_bce_mean=.5556821661744126)
        fine=dict(name='fine',music_eer_mean=.27641486519310177,
                  music_eer_std=.02147079458608201,balanced_bce_mean=.5337619973881108)
        self.assertEqual(select_resolution([coarse,fine])['name'],'coarse')
        self.assertEqual(select_resolution([fine,dict(coarse,music_eer_mean=.28)])['name'],'fine')

    def test_pooling_preserves_disjoint_frame_values_and_rejects_stale_cache(self):
        hidden = torch.arange(2*19*768,dtype=torch.float32).reshape(2,19,768)/10000
        one,eight,times,counts = pool_frames(hidden,[0,1000],[10,3],[5,2])
        np.testing.assert_allclose(one,hidden.mean(1).numpy())
        for i,indices in enumerate(np.array_split(np.arange(19),8)):
            np.testing.assert_allclose(eight[:,i],hidden[:,indices].mean(1).numpy())
        np.testing.assert_allclose(times[1]-times[0],1000)
        with TemporaryDirectory() as directory:
            path=Path(directory)/'cache.npz'
            np.savez(path,identity='audio:code',music_1=one,music_8=eight,bin_times=times,
                     starts=np.array([0,1000]),music_audible=np.ones(2),frame_counts=counts)
            coarse,fine=load_pair(path,'audio:code')
            np.testing.assert_array_equal(fine['music'],eight)
            with self.assertRaises(ValueError):
                load_pair(path,'wrong:code')
        z=torch.tensor([-.7,.4,1.8])
        y=torch.tensor([0,1,1])
        weights=torch.tensor([.1,.2,.7])
        ce=torch.nn.functional.cross_entropy(torch.stack([torch.zeros_like(z),z],1),y,reduction='none')
        bce=torch.nn.functional.binary_cross_entropy_with_logits(z,y.float(),reduction='none')
        torch.testing.assert_close((ce*weights).sum(),(bce*weights).sum())

    def test_high_resolution_batch_orders_overlapping_tail_and_repeats_validity(self):
        # Flattening windows blindly would move backward in time at the overlapping tail.
        rows = [dict(ID='audio')]
        frames = np.broadcast_to(np.arange(16, dtype='float32').reshape(2,8,1), (2,8,768)).copy()
        records = {'audio': dict(music=frames, music_audible=np.array([1.,0.]),
                                  starts=np.array([0.,16150.]),
                                  bin_times=np.array([0.,16150.])[:,None]+np.arange(8)[None,:]*8075)}
        x, mask = _batch(rows, records, 'cpu', temporal_bins=8)
        order = sorted(range(16), key=lambda i: (i//8)*16150 + (i%8)*8075)
        np.testing.assert_array_equal(x[0,:,0], order)
        np.testing.assert_array_equal(mask[0], np.array(order)>=8)
        with self.assertRaises(ValueError):
            _batch(rows, records, 'cpu', temporal_bins=1)

    def test_crop_and_feature_mask_keep_coarse_window_duration_and_feature_axis(self):
        # Crops must not shrink 8x, nor may feature masking accidentally mask the bin axis.
        rows, coarse = samples()
        fine = {key: dict(record, bin_times=record['starts'][:,None]+np.arange(8)[None,:]*8075, music=record['music'][:,None,:] +
                         (np.arange(8,dtype='float32')-3.5)[None,:,None]*.1)
                for key,record in coarse.items()}
        originals = {key: record['music'].copy() for key,record in fine.items()}
        recipe = TemporalRecipe(crop_probability=1., feature_mask_probability=1.)
        low, low_counts = augment_records(rows, coarse, recipe, np.random.default_rng(3), np.random.default_rng(4))
        high, high_counts = augment_records(rows, fine, recipe, np.random.default_rng(3), np.random.default_rng(4))
        self.assertEqual(low_counts, high_counts)
        for key in fine:
            np.testing.assert_array_equal(low[key]['starts'],high[key]['starts'])
            np.testing.assert_allclose(high[key]['music'].mean(1),low[key]['music'],atol=5e-7)
            self.assertEqual(np.count_nonzero((high[key]['music']==0).all(axis=(0,1))),38)
            np.testing.assert_array_equal(fine[key]['music'], originals[key])

    def test_two_class_training_uses_both_logits_and_reloads_each_time_resolution(self):
        # Catch wrong softmax axis/class, disconnected output rows and wrong input on reload.
        rows, records = samples()
        for bins in (1,8):
            with self.subTest(bins=bins), TemporaryDirectory() as directory:
                source = records if bins==1 else {key:dict(r,bin_times=r['starts'][:,None]+np.arange(8)[None,:]*8075,music=r['music'][:,None,:] +
                    (np.arange(8,dtype='float32')-3.5)[None,:,None]*.1) for key,r in records.items()}
                recipe = TemporalRecipe(epochs=3,batch_size=4,nsynth_mass=0.,simplified=True,
                                        compact=True,temporal_bins=bins,softmax_output=True)
                model, report = fit_temporal_head(rows[:8],source,(rows[8:],source),Path(directory),
                                                   time.time()+60,recipe=recipe)
                loaded = load_temporal_head(Path(directory)/'temporal_music.pt')
                self.assertEqual(loaded.temporal_bins,bins)
                self.assertTrue(loaded.softmax_output)
                p = predict_temporal_music(loaded,rows[8:],source)
                np.testing.assert_allclose(p,predict_temporal_music(model,rows[8:],source),atol=1e-6)
                y = np.array([int(r['MUSIC_FAKE']) for r in rows[8:]])
                self.assertAlmostEqual(float(-np.log(np.where(y,p,1-p)).mean()),report['best_validation_loss'],places=6)
                x,mask = _batch(rows[8:],source,'cpu',temporal_bins=bins)
                logits = loaded(x,mask)
                torch.nn.functional.cross_entropy(logits,torch.tensor(y)).backward()
                self.assertTrue((loaded.output.weight.grad.abs().sum(1)>0).all())
                scalar = TemporalMusic(simplified=True,compact=True,temporal_bins=bins).eval()
                scalar.load_state_dict({k:v for k,v in loaded.state_dict().items() if not k.startswith('output.')},strict=False)
                with torch.no_grad():
                    scalar.output.weight.copy_(loaded.output.weight[1:]-loaded.output.weight[:1])
                    scalar.output.bias.copy_(loaded.output.bias[1:]-loaded.output.bias[:1])
                torch.testing.assert_close(logits.softmax(-1)[:,1],scalar(x,mask).sigmoid(),atol=1e-6,rtol=0)
                empty = loaded(x,torch.ones_like(mask)).softmax(-1)[:,1]
                self.assertTrue(torch.isfinite(empty).all())
                self.assertLess(float(empty.max()),1e-6)


if __name__ == '__main__':
    torch.set_num_threads(1)
    unittest.main()
