"""Per-file decision layer shared by feature extraction, evaluation and the submission runtime (numpy only).

presence : small logistic heads over PANNs class-group statistics (+ HTDemucs stem energy) -> VOICE/MUSIC_PRESENT
ensemble : FILE_FAKE = weighted mean of candidate file scores, each mapped through a quantile table fitted on
           validation data. Every mapping is a fixed function of one file's own outputs: no test-set statistics,
           no cross-file ranking (DACON rule 2-4: each evaluation file is predicted independently).
"""
from __future__ import annotations

import numpy as np

VOICE_CORE = ('Speech', 'Male speech, man speaking', 'Female speech, woman speaking', 'Conversation', 'Narration, monologue')
SINGING = ('Singing', 'Choir', 'Male singing', 'Female singing', 'Child singing', 'Synthetic singing', 'Rapping', 'Humming')


# ---------------------------------------------------------------- presence features
def _stats(prefix: str, x: np.ndarray) -> dict:
    return {f'{prefix}_max': float(x.max()), f'{prefix}_mean': float(x.mean()), f'{prefix}_q90': float(np.quantile(x, .9))}


def tagger_features(probs: np.ndarray, labels, groups: dict) -> dict:
    """probs: [segments, 527] PANNs clipwise scores (same segments as the legacy presence loop)."""
    idx = {l: i for i, l in enumerate(labels)}
    col = lambda names: probs[:, [idx[n] for n in names if n in idx]].max(1)
    out = {}
    for name, cols in (('voice', groups['voice']), ('music', groups['music']), ('speech', VOICE_CORE), ('singing', SINGING)):
        out.update(_stats(f'tag_{name}', col(cols)))
    out.update(_stats('tag_music_only', probs[:, idx['Music']]))
    return out


def stem_features(mix: np.ndarray, vocals: np.ndarray, sr: int, win_s: float = 3.0) -> dict:
    """Separated-vocal / accompaniment energy relative to the mixture per 3 s window (mono arrays, same sr)."""
    w = max(1, int(win_s * sr))
    n = max(1, len(mix) // w)
    acc = mix[:len(vocals)] - vocals[:len(mix)]
    e = lambda x, i: float(np.mean(np.square(x[i * w:(i + 1) * w], dtype=np.float64))) + 1e-10
    vr = np.array([np.sqrt(e(vocals, i) / e(mix, i)) for i in range(n)])
    ar = np.array([np.sqrt(e(acc, i) / e(mix, i)) for i in range(n)])
    out = {**_stats('stem_vocal_ratio', vr), **_stats('stem_acc_ratio', ar)}
    out['stem_vocal_frac'] = float((vr >= .35).mean())
    out['stem_acc_frac'] = float((ar >= .35).mean())
    out['mix_rms_db'] = float(10 * np.log10(np.mean(np.square(mix, dtype=np.float64)) + 1e-10))
    return out


def presence_features(probs, labels, groups, raw16, voice16) -> dict:
    return {**tagger_features(probs, labels, groups), **stem_features(raw16, voice16, 16000)}


def legacy_presence(probs, labels, groups) -> tuple[float, float]:
    """Exactly the baseline value: max over segments and group classes."""
    idx = {l: i for i, l in enumerate(labels)}
    return (float(probs[:, [idx[n] for n in groups['voice']]].max()), float(probs[:, [idx[n] for n in groups['music']]].max()))


def logistic(head: dict, feats: dict) -> float:
    x = np.array([feats[n] for n in head['features']], float)
    z = float(((x - np.array(head['mean'])) / np.array(head['scale'])) @ np.array(head['coef']) + head['intercept'])
    return 1 / (1 + np.exp(-z))


def presence(heads: dict | None, feats: dict | None, fallback: tuple[float, float]) -> tuple[float, float]:
    if not heads or feats is None:
        return fallback
    return logistic(heads['VOICE_PRESENT'], feats), logistic(heads['MUSIC_PRESENT'], feats)


# ---------------------------------------------------------------- final FILE ensemble
CANDIDATES = {
    'max': lambda d: max(d['Pv'] * d['Fv'], d['Pm'] * d['Fm']),                    # current rule
    'noisy_or': lambda d: 1 - (1 - d['Pv'] * d['Fv']) * (1 - d['Pm'] * d['Fm']),   # evidence from both components
    'max_nopresence': lambda d: max(d['Fv'], d['Fm']),                             # robust to presence errors
    'prev': lambda d: d.get('prev_file'),                                          # previous best package, if present
}


def quantile_table(values: np.ndarray, n: int = 1001) -> list[float]:
    """Monotone table of val quantiles; interp(x, table, levels) is the val CDF of a single score."""
    t = np.quantile(np.asarray(values, float), np.linspace(0, 1, n))
    return (np.maximum.accumulate(t) + np.arange(n) * 1e-12).tolist()   # strictly increasing for np.interp


def to_cdf(table: list[float], x: float) -> float:
    return float(np.interp(x, table, np.linspace(0, 1, len(table))))


def file_score(ens: dict | None, d: dict) -> float:
    """Legacy max rule when no ensemble is fitted; otherwise weighted mean of per-candidate val CDFs."""
    if not ens:
        return CANDIDATES['max'](d)
    num = den = 0.
    for name, w in ens['weights'].items():
        v = CANDIDATES[name](d)
        if w and v is not None:
            num += w * to_cdf(ens['tables'][name], v)
            den += w
    return num / den if den else CANDIDATES['max'](d)


def self_check() -> None:
    rng = np.random.default_rng(0)
    labels = list({*VOICE_CORE, *SINGING, 'Music', 'Guitar'}) + [f'c{i}' for i in range(500)]
    groups = {'voice': list(VOICE_CORE) + list(SINGING), 'music': ['Music', 'Guitar']}
    p = rng.random((4, len(labels)))
    f = tagger_features(p, labels, groups)
    vi = [labels.index(n) for n in groups['voice']]
    assert f['tag_voice_max'] == p[:, vi].max() == legacy_presence(p, labels, groups)[0]
    mix = rng.standard_normal(16000 * 7).astype(np.float32)
    s = stem_features(mix, mix * .5, 16000)
    assert abs(s['stem_vocal_ratio_mean'] - .5) < 1e-3 and s['stem_vocal_frac'] == 1.
    head = {'features': ['a', 'b'], 'mean': [0, 0], 'scale': [1, 1], 'coef': [1, -1], 'intercept': 0.}
    assert abs(logistic(head, {'a': 1, 'b': 1}) - .5) < 1e-12
    assert presence(None, None, (.3, .4)) == (.3, .4)
    d = dict(Pv=.9, Fv=.8, Pm=.1, Fm=.2)
    assert file_score(None, d) == max(.9 * .8, .1 * .2)
    vals = {k: [CANDIDATES[k](dict(Pv=a, Fv=b, Pm=c, Fm=e)) for a, b, c, e in rng.random((200, 4))] for k in ('max', 'noisy_or')}
    ens = {'weights': {'max': 1, 'noisy_or': 1}, 'tables': {k: quantile_table(v) for k, v in vals.items()}}
    lo, hi = file_score(ens, dict(Pv=.1, Fv=.1, Pm=.1, Fm=.1)), file_score(ens, dict(Pv=.9, Fv=.9, Pm=.9, Fm=.9))
    assert 0 <= lo < hi <= 1, 'monotone in the evidence'
    assert file_score(ens, d) == file_score(ens, dict(d)), 'deterministic per file, no batch state'
    t = quantile_table([0.5] * 50)
    assert all(b > a for a, b in zip(t, t[1:])), 'ties made strictly increasing'
    print('decision_heads self-check ok')


if __name__ == '__main__':
    self_check()
