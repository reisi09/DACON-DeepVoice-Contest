"""Component fusion shared by fitting (data_v4/fusion_fit.py) and the submission runtime (full_bank_submission.py).
VOICE/MUSIC_FAKE = weighted mean of member val CDFs (decision_heads quantile tables): a fixed function of one file's
own member scores, no test-set statistics (DACON rule 2-4). Members: our heads, DF-Arena's own classifier (df0),
AntiDeepfake on the voice stem, SONICS on the capped mix.
"""
from __future__ import annotations

import numpy as np
import torch

from decision_heads import to_cdf


def fuse(spec: dict, values: dict) -> float:
    """Weighted mean of member val CDFs, mapped back onto our head's score scale (inverse of its val CDF) so the FILE
    rules keep their calibrated inputs; monotone, so the component EER is that of the CDF mean. Every member with a
    nonzero weight must be in values: a missing one raises (a runtime member must never drop out silently and change
    VOICE without an error); ours alone passes through unchanged."""
    missing = [n for n, x in spec['weights'].items() if x and n not in values]
    if missing:
        raise KeyError(f'fusion members missing from values: {missing}')
    w = {n: x for n, x in spec['weights'].items() if x}
    if set(w) == {'ours'}:
        return values['ours']
    c = sum(x * to_cdf(spec['tables'][n], values[n]) for n, x in w.items()) / sum(w.values())
    ours = spec['tables'].get('ours')   # absent only for single-member diagnostics
    # clip: quantile_table's strictly-increasing 1e-12 steps put the top of a saturated head's table just above 1
    return c if ours is None else float(np.clip(np.interp(c, np.linspace(0, 1, len(ours)), ours), 0., 1.))


def sonics_prob(model, audio: np.ndarray, device) -> float:
    """SONICS eval preprocessing: zero-pad/crop to max_len (crop keeps the 3/4 offset), divide by std."""
    n = model.config.audio.max_len
    audio = np.pad(audio, (0, n - len(audio))) if len(audio) < n else audio[int((len(audio) - n) / 4 * 3):][:n]
    audio = audio / max(float(np.std(audio)), 1e-6)
    with torch.inference_mode():
        return float(torch.sigmoid(model(torch.from_numpy(audio.astype(np.float32))[None].to(device))).item())


def detector_members(name: str, probs: list[float]) -> dict:
    """AntiDeepfake.predict output [max, top30_mean, *window_probs] -> named members."""
    return {f'{name}_max': probs[0], f'{name}_top30': probs[1], f'{name}_mean': float(np.mean(probs[2:]))}


if __name__ == '__main__':
    t = list(np.linspace(0, 1, 1001))
    spec = {'weights': {'ours': 1, 'b': 3, 'c': 0}, 'tables': {'ours': [x ** 2 for x in t], 'b': t}}
    assert abs(fuse(spec, {'ours': .25, 'b': .6, 'c': 9}) - ((.5 + 3 * .6) / 4) ** 2) < 1e-3, 'CDF mean mapped to ours scale'
    try:
        fuse(spec, {'ours': .25})
        raise AssertionError('a missing weighted member must raise')
    except KeyError:
        pass
    assert fuse({'weights': {'ours': 1, 'c': 0}, 'tables': {'ours': t}}, {'ours': .25}) == .25, 'zero-weight member may be absent'
    assert fuse({'weights': {'ours': 1}, 'tables': {'ours': t}}, {'ours': .37}) == .37
    assert detector_members('x', [.9, .8, .1, .9]) == {'x_max': .9, 'x_top30': .8, 'x_mean': .5}
    sat = {'weights': {'ours': 1, 'b': 1}, 'tables': {'ours': [x + i * 1e-12 for i, x in enumerate([0.] * 5 + [1.] * 6)], 'b': t}}
    assert fuse(sat, {'ours': 1., 'b': 1.}) <= 1., 'saturated heads stay within [0, 1]'
    print('component_fusion self-check ok')
