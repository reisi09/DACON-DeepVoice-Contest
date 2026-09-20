"""Checks protect temporal evidence, MIL training, leakage and artifact identity."""

import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from v6_core import Heads, fusion_features
from v6_data import assert_disjoint
from v6_train import mil_logits, fit_music, trained_heads
from v6_data import write_json
from v6_runtime import load_cache


class FinalPipelineChecks(unittest.TestCase):
    def test_alignment_does_not_join_different_events(self):
        p = np.array([[.9, .1], [.1, .9]])
        v, m = np.array([.1, .9]), np.array([.8, .2])
        features = fusion_features(p, v, m, np.array([.2, .3]))
        np.testing.assert_allclose(features[:2], [.09, .18])
        self.assertLess(features[0], p[:, 0].max() * v.max())
        with self.assertRaises(ValueError):
            fusion_features(p, v[:1], m, v)

    def test_mil_gradient_uses_file_label_at_strongest_window(self):
        x = torch.tensor([-.8, .7, -.4], requires_grad=True)
        bag = torch.tensor([0, 0, 1])
        loss = torch.nn.functional.binary_cross_entropy_with_logits(
            mil_logits(x, bag, 2), torch.tensor([1., 0.]))
        loss.backward()
        self.assertEqual(x.grad[0].item(), 0.)
        self.assertLess(x.grad[1].item(), 0.)
        self.assertGreater(x.grad[2].item(), 0.)

    def test_head_optimization_and_safe_roundtrip(self):
        rng = np.random.default_rng(41)
        bags = [rng.normal(0, .01, (2, 768)).astype("float32") for _ in range(12)]
        labels = np.array([0, 1] * 6, dtype="float32")
        for x, label in zip(bags, labels):
            x[:, 0] = -1
            if label:
                x[-1, 0] = 1
        head, trace = fit_music(bags, labels, epochs=80, seed=41)
        self.assertLess(trace[-1], trace[0] * .5)
        probabilities = [head.music(x, np.ones(len(x))).max() for x in bags]
        self.assertGreater(min(np.array(probabilities)[labels == 1]),
                           max(np.array(probabilities)[labels == 0]))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "heads.npz"
            head.save(path)
            loaded = Heads.load(path)
            np.testing.assert_array_equal(loaded.music(bags[0], np.ones(2)),
                                          head.music(bags[0], np.ones(2)))

    def test_parent_and_duplicate_pcm_cannot_cross_partitions(self):
        a = [{"ID": "a", "group_id": "song1|speaker1", "pcm_sha256": "x"}]
        for row in ({"ID": "b", "group_id": "speaker1", "pcm_sha256": "y"},
                    {"ID": "c", "group_id": "song2", "pcm_sha256": "x"}):
            with self.assertRaises(ValueError):
                assert_disjoint(a, [row])

    def test_cache_rejects_changed_audio_or_model_identity(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "cache.npz"
            np.savez(path, identity=np.array("audio-a:model-a"), raw=np.array([.5]))
            self.assertIsNone(load_cache(path, "audio-b:model-a"))
            self.assertIsNone(load_cache(path, "audio-a:model-b"))

    def test_smoke_or_incomplete_training_cannot_be_exported(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            for metadata in (dict(status="running"), dict(status="complete", purpose="smoke")):
                write_json(path / "training.json", metadata)
                with self.assertRaises(ValueError):
                    trained_heads(path, "assets")


if __name__ == "__main__":
    unittest.main()
