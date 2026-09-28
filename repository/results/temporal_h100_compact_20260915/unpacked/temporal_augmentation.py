"""Training-only views of immutable MERT caches; preserve contiguous time and labels."""

from __future__ import annotations

from typing import TYPE_CHECKING, Final, TypedDict

import numpy as np

if TYPE_CHECKING:
    from music_temporal import Records, Rows, TemporalRecipe

CROP_LENGTHS: Final = (3, 4, 6, 8, 12)
MASK_FRACTION: Final = .05


class AugmentationCounts(TypedDict):
    seen: int
    cropped: int
    masked: int
    crop_eligible: int
    real_cropped: int
    fake_cropped: int
    input_windows: int
    output_windows: int


def empty_counts() -> AugmentationCounts:
    return AugmentationCounts(seen=0, cropped=0, masked=0, crop_eligible=0,
                              real_cropped=0, fake_cropped=0, input_windows=0, output_windows=0)


def augment_records(rows: Rows, records: Records, recipe: TemporalRecipe,
                    crop_rng: np.random.Generator,
                    mask_rng: np.random.Generator) -> tuple[Records, AugmentationCounts]:
    """Whitelist full-track labels; draw crops independently of feature-mask randomness."""
    result: Records = {}
    counts = empty_counts()
    for row in rows:
        record = records[row['ID']]
        music, audible, starts = record['music'], record['music_audible'], record['starts']
        counts['seen'] += 1
        counts['input_windows'] += len(music)
        source = row['source'].lower()
        pure = (row.get('MUSIC_PRESENT') == '1'
                and row.get('panel') in ('music_real', 'music_seen', 'music_ood')
                and ((source.startswith(('fma/', 'nsynth/')) and row['MUSIC_FAKE'] == '0')
                     or (source.startswith('echoes/') and row['MUSIC_FAKE'] == '1')))
        lengths = [n for n in CROP_LENGTHS if n < len(music)]
        eligible = pure and bool(lengths)
        counts['crop_eligible'] += int(eligible)
        if eligible and recipe.crop_probability > 0 and crop_rng.random() < recipe.crop_probability:
            length = int(crop_rng.choice(lengths))
            cumulative = np.r_[0, np.cumsum(audible > 0)]
            positions = np.flatnonzero(cumulative[length:] - cumulative[:-length] > 0)
            if len(positions):
                start = int(crop_rng.choice(positions))
                music = music[start:start+length]
                audible = audible[start:start+length]
                starts = starts[start:start+length]
                counts['cropped'] += 1
                counts['real_cropped' if row['MUSIC_FAKE'] == '0' else 'fake_cropped'] += 1
        if recipe.feature_mask_probability > 0 and mask_rng.random() < recipe.feature_mask_probability:
            music = music.copy()
            channels = mask_rng.choice(music.shape[1], round(music.shape[1]*MASK_FRACTION), replace=False)
            music[:, channels] = 0
            counts['masked'] += 1
        result[row['ID']] = dict(music=music, music_audible=audible, starts=starts)
        counts['output_windows'] += len(music)
    return result, counts
