"""Trained presence heads (VOICE_PRESENT / MUSIC_PRESENT) on frozen-model features.

Why: swapping PANNs for AST/CED gave the same voice-presence AUC (~0.81-0.84) on real clips;
singing inside songs is the miss. A small head over AudioSet group scores + HTDemucs stem-energy
features learns that from our presence labels. Both feature sources already run in the submission.

  extract : python data_v4/presence.py extract --manifest OUT/manifest.jsonl --root OUT --panns DIR --out OUT/presence_feats.jsonl
  train   : python data_v4/presence.py train --feats ... --manifest ... --splits OUT/splits.csv --out OUT/presence_head.json
  runtime : head = json.load(...); predict(head, features(...))    (numpy only, no pickle)
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

SR, WIN = 16000, 64600                     # baseline presence windows (4.04 s)
VOICE_CORE = ('Speech', 'Male speech, man speaking', 'Female speech, woman speaking', 'Conversation', 'Narration, monologue')
SINGING = ('Singing', 'Choir', 'Male singing', 'Female singing', 'Child singing', 'Synthetic singing', 'Rapping', 'Humming')
TARGETS = ('VOICE_PRESENT', 'MUSIC_PRESENT')


def _windows(y: np.ndarray, n: int = WIN) -> list[np.ndarray]:
    if len(y) <= n:
        return [np.pad(y, (0, n - len(y)))]
    starts = list(range(0, len(y) - n + 1, n))
    if starts[-1] != len(y) - n:
        starts.append(len(y) - n)
    return [y[s:s + n] for s in starts]


def _stats(prefix: str, x: np.ndarray) -> dict:
    return {f'{prefix}_max': float(x.max()), f'{prefix}_mean': float(x.mean()), f'{prefix}_q90': float(np.quantile(x, .9))}


def tagger_features(probs: np.ndarray, labels: list[str], groups: dict) -> dict:
    """probs: [windows, 527] AudioSet sigmoid scores."""
    idx = {l: i for i, l in enumerate(labels)}
    col = lambda names: probs[:, [idx[n] for n in names if n in idx]].max(1)
    out = {}
    out.update(_stats('tag_voice', col(groups['voice'])))
    out.update(_stats('tag_music', col(groups['music'])))
    out.update(_stats('tag_speech', col(VOICE_CORE)))
    out.update(_stats('tag_singing', col(SINGING)))
    out.update(_stats('tag_music_only', probs[:, idx['Music']]))
    return out


def stem_features(mix: np.ndarray, vocals: np.ndarray, sr: int, win_s: float = 3.0) -> dict:
    """Energy of separated vocals / accompaniment relative to the mixture, per 3 s window (mono arrays)."""
    w = max(1, int(win_s * sr))
    n = max(1, len(mix) // w)
    acc = mix[:len(vocals)] - vocals[:len(mix)]
    e = lambda x, i: float(np.mean(x[i * w:(i + 1) * w] ** 2)) + 1e-10
    vr = np.array([np.sqrt(e(vocals, i) / e(mix, i)) for i in range(n)])
    ar = np.array([np.sqrt(e(acc, i) / e(mix, i)) for i in range(n)])
    out = {**_stats('stem_vocal_ratio', vr), **_stats('stem_acc_ratio', ar)}
    out['stem_vocal_frac'] = float((vr >= .35).mean())
    out['stem_acc_frac'] = float((ar >= .35).mean())
    out['mix_rms_db'] = float(10 * np.log10(np.mean(mix ** 2) + 1e-10))
    return out


class Extractor:
    """PANNs CNN14 (same checkpoint/class groups as the submission) + HTDemucs."""

    def __init__(self, panns_dir: Path, device: str = 'cuda'):
        import torch
        from panns_inference import AudioTagging, labels
        from demucs.pretrained import get_model
        self.torch, self.device = torch, device
        self.tagger = AudioTagging(checkpoint_path=str(panns_dir / 'Cnn14_mAP=0.431.pth'), device=device)
        self.labels = list(labels)
        self.groups = json.loads((panns_dir / 'component_labels.json').read_text())
        self.sep = get_model('htdemucs').to(device).eval()

    def __call__(self, y16: np.ndarray) -> dict:
        import librosa
        from demucs.apply import apply_model
        seg = np.stack([librosa.resample(w, orig_sr=SR, target_sr=32000) for w in _windows(y16)])
        probs, _ = self.tagger.inference(seg)
        x = librosa.resample(y16, orig_sr=SR, target_sr=self.sep.samplerate)
        t = self.torch.from_numpy(x).float()[None].repeat(2, 1)[None].to(self.device)
        with self.torch.inference_mode():
            stems = apply_model(self.sep, t, split=True, overlap=.1, progress=False)[0]
        voc = stems[self.sep.sources.index('vocals')].mean(0).cpu().numpy()
        return {**tagger_features(probs, self.labels, self.groups), **stem_features(x, voc, self.sep.samplerate)}


# ---------------------------------------------------------------- head: standardized logistic regression, JSON
def fit(X: np.ndarray, y: np.ndarray, names: list[str], C: float = 0.05) -> dict:
    from sklearn.linear_model import LogisticRegression
    mu, sd = X.mean(0), X.std(0) + 1e-9
    m = LogisticRegression(C=C, max_iter=5000).fit((X - mu) / sd, y)
    return dict(features=names, mean=mu.tolist(), scale=sd.tolist(), coef=m.coef_[0].tolist(), intercept=float(m.intercept_[0]), C=C)


def predict(head: dict, feats: dict | list[dict]) -> np.ndarray:
    rows = [feats] if isinstance(feats, dict) else feats
    X = np.array([[r[n] for n in head['features']] for r in rows], float)
    z = ((X - np.array(head['mean'])) / np.array(head['scale'])) @ np.array(head['coef']) + head['intercept']
    return 1 / (1 + np.exp(-z))


def grouped_cv(X, y, groups, names, base='tag_voice_max', folds=5, C=0.05) -> dict:
    """Held-out AUC of the head vs the raw baseline score (max over tagger class group), same folds."""
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import GroupKFold
    out = {'head': [], 'baseline_raw_group_max': []}
    for tr, ev in GroupKFold(min(folds, len(set(groups)))).split(X, y, groups):
        if len(set(y[tr])) < 2 or len(set(y[ev])) < 2:
            continue
        h = fit(X[tr], y[tr], names, C)
        out['head'].append(roc_auc_score(y[ev], predict(h, [dict(zip(names, r)) for r in X[ev]])))
        out['baseline_raw_group_max'].append(roc_auc_score(y[ev], X[ev, names.index(base)]))
    return {k: [round(v, 4) for v in vs] for k, vs in out.items()}


def train(feats: list[dict], labels: dict, splits: dict | None, target: str) -> tuple[dict, dict]:
    names = sorted(k for k in feats[0] if k != 'ID')
    rows = [(f, labels[f['ID']]) for f in feats if labels.get(f['ID']) is not None]
    tr = [(f, y) for f, y in rows if splits is None or splits.get(f['ID']) == 'train']
    X = np.array([[f[n] for n in names] for f, _ in tr])
    y = np.array([v for _, v in tr])
    base = 'tag_voice_max' if target == 'VOICE_PRESENT' else 'tag_music_max'
    head = fit(X, y, names)
    report = {'target': target, 'train_n': len(tr), 'train_pos': int(y.sum())}
    from sklearn.metrics import roc_auc_score
    for s in ('val', 'test'):
        ev = [(f, v) for f, v in rows if splits and splits.get(f['ID']) == s]
        ye = np.array([v for _, v in ev])
        if len(set(ye)) == 2:
            report[s] = {'n': len(ev), 'head_auc': round(roc_auc_score(ye, predict(head, [f for f, _ in ev])), 4),
                         'baseline_auc': round(roc_auc_score(ye, [f[base] for f, _ in ev]), 4)}
    return head, report


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('cmd', choices=['extract', 'train'])
    ap.add_argument('--manifest', type=Path, required=True)
    ap.add_argument('--root', type=Path)
    ap.add_argument('--panns', type=Path)
    ap.add_argument('--feats', type=Path)
    ap.add_argument('--splits', type=Path)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--per-cell', type=int, default=400, help='rows per (source, presence pattern) to extract')
    a = ap.parse_args()
    rows = [json.loads(l) for l in a.manifest.open()]
    if a.cmd == 'extract':
        import librosa
        # deterministic cap per cell keeps GPU time bounded; all presence patterns/sources stay represented
        cells: dict = {}
        for r in sorted(rows, key=lambda r: hashlib.sha256(r['ID'].encode()).hexdigest()):
            if r['VOICE_PRESENT'] is None and r['MUSIC_PRESENT'] is None:
                continue
            key = (r['source'], r['VOICE_PRESENT'], r['MUSIC_PRESENT'])
            if len(cells.setdefault(key, [])) < a.per_cell:
                cells[key].append(r)
        todo = [r for rs in cells.values() for r in rs]
        done = {json.loads(l)['ID'] for l in a.out.open()} if a.out.exists() else set()
        ex = Extractor(a.panns)
        with a.out.open('a') as f:
            for i, r in enumerate(todo):
                if r['ID'] in done:
                    continue
                y = librosa.load(a.root / r['path'], sr=SR, mono=True)[0]
                f.write(json.dumps({'ID': r['ID'], **ex(y)}) + '\n')
                if i % 500 == 0:
                    f.flush()
                    print(f'presence feats {i}/{len(todo)}', flush=True)
    else:
        import csv
        feats = [json.loads(l) for l in a.feats.open()]
        splits = {r['ID']: r['split'] for r in csv.DictReader(a.splits.open())} if a.splits else None
        heads, reports = {}, {}
        for t in TARGETS:
            heads[t], reports[t] = train(feats, {r['ID']: r[t] for r in rows}, splits, t)
        a.out.write_text(json.dumps(heads, indent=1) + '\n')
        a.out.with_suffix('.report.json').write_text(json.dumps(reports, indent=1) + '\n')
        print(json.dumps(reports, indent=1))


def self_check() -> None:
    rng = np.random.default_rng(0)
    labels = ['Speech', 'Singing', 'Music', 'Guitar'] + [f'c{i}' for i in range(523)]
    groups = {'voice': ['Speech', 'Singing'], 'music': ['Music', 'Guitar']}
    p = rng.random((3, 527))
    f = tagger_features(p, labels, groups)
    assert f['tag_voice_max'] == p[:, :2].max() and f['tag_singing_max'] == p[:, 1].max()
    mix = rng.standard_normal(SR * 9).astype(np.float32)
    s = stem_features(mix, mix * .5, SR)
    assert abs(s['stem_vocal_ratio_mean'] - .5) < 1e-3 and s['stem_vocal_frac'] == 1.0
    assert len(_windows(np.zeros(WIN * 2 + 10))) == 3 and len(_windows(np.zeros(100))) == 1
    # head must recover a signal the raw baseline score misses (vocals hidden in music)
    n = 400
    vp = rng.integers(0, 2, n)
    X = np.c_[rng.random(n) * .3 + .1 * vp, .2 + .5 * vp + rng.normal(0, .1, n)]    # tag weak, stem strong
    names = ['tag_voice_max', 'stem_vocal_ratio_max']
    h = fit(X, vp, names)
    assert json.loads(json.dumps(h)) == h, 'head must be JSON-serialisable'
    cv = grouped_cv(X, vp, np.arange(n) // 4, names)
    assert np.mean(cv['head']) > np.mean(cv['baseline_raw_group_max']) + .1, cv
    print('presence self-check ok')


if __name__ == '__main__':
    self_check() if sys.argv[1:] == ['--self-check'] else main()
