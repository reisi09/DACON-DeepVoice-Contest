"""Tests first: the implementation these assertions require does not yet exist."""
import json
import os
from pathlib import Path
import numpy as np
import pytest

ROOT = Path(os.environ.get('STACKING_INPUT_ROOT', '/mnt/data/stacking_input'))


def test_real_heads_cache_baseline_and_seed_replay():
    from dacon_stacking.features import FixtureAdapter
    adapter = FixtureAdapter(ROOT)
    records = adapter.replay()
    assert len(records) == 10
    for record, entry in zip(records, adapter.index, strict=True):
        assert record['ID'] == entry['ID']
        for key, expected in entry['reference_predictions'].items():
            assert abs(record['diagnostic'][key] - expected) <= 0.002, (record['ID'], key)
        p = record['baseline']
        assert p['FILE_FAKE_PROB'] == max(p['VOICE_PRESENT_PROB']*p['VOICE_FAKE_PROB'], p['MUSIC_PRESENT_PROB']*p['MUSIC_FAKE_PROB'])
        assert len(record['features']) >= 32


def test_different_timebases_and_no_invalid_padding_weight():
    from dacon_stacking.features import time_layout, aligned_summary
    m = time_layout(100000, 'music')
    v = time_layout(100000, 'voice')
    assert m['starts'] == [0, 32300, 35400]
    assert v['starts'] == [0, 35400]
    assert time_layout(10000,'music')['valid_samples'] == [10000]
    result = aligned_summary([0,4],[4,4],[1.,0.], [0,2,4],[4,4,4],[1.,1.,1.], 8, sr=1)
    assert result['mean'] == pytest.approx(.5)
    assert result['coverage'] == pytest.approx(.5)
    assert result['longest_fraction'] == pytest.approx(.5)
    with pytest.raises(ValueError, match='uncovered'):
        aligned_summary([0],[3],[1.], [0],[8],[1.], 8, sr=1)


def test_forbidden_and_missing_features_are_rejected():
    from dacon_stacking.features import strict_frame, FEATURE_SETS
    import pandas as pd
    good = pd.DataFrame([[.2,.3,.4,.5]], columns=FEATURE_SETS['F0'])
    strict_frame(good, FEATURE_SETS['F0'])
    for bad in [good.iloc[:,::-1],good.assign(FILE_FAKE=1),good.assign(Pv=np.nan)]:
        with pytest.raises(ValueError):
            strict_frame(bad, FEATURE_SETS['F0'])
    with pytest.raises(ValueError):
        strict_frame(pd.DataFrame({'window_truth':[1.]}), ['window_truth'])
