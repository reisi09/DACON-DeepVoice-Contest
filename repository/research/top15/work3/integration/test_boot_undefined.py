"""Boot.delta: a single-class slice is 'undefined' (None), never p_gt_0 = 0; a two-class slice keeps a finite CI.
  python research/top15/work3/integration/test_boot_undefined.py"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from fit_ft_fusion import Boot  # noqa: E402


def row(i, file_fake):
    return {'ID': f'r{i}', 'group_id': f'g{i % 5}', 'FILE_FAKE': str(file_fake), 'FILE_FAKE_mask': '1', 'VOICE_PRESENT': '1',
            'VOICE_PRESENT_mask': '1', 'VOICE_FAKE': str(file_fake), 'VOICE_FAKE_mask': '1', 'MUSIC_PRESENT': '0',
            'MUSIC_PRESENT_mask': '1', 'MUSIC_FAKE': '', 'MUSIC_FAKE_mask': '0'}


def preds(rows, s):
    return [{'ID': r['ID'], 'FILE_FAKE_PROB': x, 'VOICE_FAKE_PROB': x, 'MUSIC_FAKE_PROB': 0.} for r, x in zip(rows, s)]


# all-real slice: every EER undefined
one = [row(i, 0) for i in range(20)]
d = Boot(one, reps=50).delta(preds(one, [.1] * 20), preds(one, [.2] * 20))
assert d['delta'] is None and d['p_gt_0'] is None, d
# two classes, no music: music EER undefined -> the whole ADS is undefined as well (official ADS needs all three)
two = [row(i, i % 2) for i in range(40)]
d = Boot(two, reps=50).delta(preds(two, [.5] * 40), preds(two, [(i % 2) * .9 for i in range(40)]))
assert d['delta'] is None, d
# all three components with two classes: finite delta and CI, a better model gives delta > 0
full = [dict(row(i, i % 2), MUSIC_PRESENT='1', MUSIC_FAKE=str(i % 2), MUSIC_FAKE_mask='1') for i in range(40)]
good = [{'ID': r['ID'], 'FILE_FAKE_PROB': (i % 2) * .9, 'VOICE_FAKE_PROB': (i % 2) * .9, 'MUSIC_FAKE_PROB': (i % 2) * .9}
        for i, r in enumerate(full)]
bad = [{'ID': r['ID'], 'FILE_FAKE_PROB': (i % 3) * .3, 'VOICE_FAKE_PROB': (i % 3) * .3, 'MUSIC_FAKE_PROB': (i % 3) * .3}
       for i, r in enumerate(full)]
d = Boot(full, reps=50).delta(bad, good)
assert d['delta'] > 0 and d['ci95'] is not None and d['undefined_resamples'] == 0, d
print('boot undefined checks ok')
