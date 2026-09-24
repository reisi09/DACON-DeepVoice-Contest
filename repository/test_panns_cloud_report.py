"""Protect the parent-bootstrap calculation against weighted/tied ROC errors."""
import numpy as np
from graph_scoring import _roc_metrics
from panns_cloud_report import curve


def test_weighted_curve_matches_actual_parent_resampling():
    rng=np.random.default_rng(20260925)
    for _ in range(100):
        y=rng.integers(0,2,40)
        scores=rng.integers(0,6,40)/5
        parents=np.repeat(np.arange(10),4)
        counts=np.bincount(rng.integers(0,10,10),minlength=10)
        weights=counts[parents]
        selected=np.repeat(np.arange(len(y)),weights)
        expected=_roc_metrics(y[selected].tolist(),scores[selected].tolist())
        np.testing.assert_allclose(curve(y,scores,weights),expected,rtol=0,atol=1e-12)
