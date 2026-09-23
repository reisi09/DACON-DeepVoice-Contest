"""Train presence heads (VOICE_PRESENT / MUSIC_PRESENT) for decision_heads.presence.

Why: swapping PANNs for AST/CED gave the same voice-presence AUC (~0.81-0.84) on real clips; singing inside
songs is the miss (DESIGN.md §6.5). A regularised logistic head over PANNs class-group statistics + HTDemucs
stem energy learns it from our presence labels.

Features come from decision_heads.presence_features computed with the *baseline module* (same load_audio, gain,
separate_voice_and_music and make_panns_segments as full_bank_features.py and the submission runtime), so
training and inference see identical features. Rows need no fake label: noise/none rows train presence too.

  extract : python data_v4/presence.py extract --manifest OUT/manifest.jsonl --root OUT --assets ASSETS --out OUT/presence_feats.jsonl
  train   : python data_v4/presence.py train --manifest ... --feats ... --splits OUT/splits.csv --out OUT/presence_head.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import decision_heads  # noqa: E402

TARGETS = ('VOICE_PRESENT', 'MUSIC_PRESENT')
BASE = {'VOICE_PRESENT': 'tag_voice_max', 'MUSIC_PRESENT': 'tag_music_max'}   # legacy score = raw PANNs group max


class BaselineExtractor:
    """Runtime-identical presence features (see module docstring)."""

    def __init__(self, assets: Path, device: str = 'cuda'):
        import torch
        from panns_inference import labels
        from v6_runtime import baseline_module
        from mert12_submission import pin_torchaudio_loader
        self.base = baseline_module(assets)
        pin_torchaudio_loader(self.base)
        self.device = torch.device(device)
        self.torch = torch
        self.demucs = self.base.load_htdemucs_model().to(self.device).eval().requires_grad_(False)
        self.panns, _, _ = self.base.load_panns_model(self.device)
        self.labels = list(labels)
        self.groups = json.loads((self.base.PANNS_DIR / 'component_labels.json').read_text())

    def __call__(self, path: Path) -> dict:
        import tempfile
        import soundfile as sf
        raw = self.base.load_audio(path)
        peak = float(np.abs(raw).max())
        if peak > .6:
            raw = (raw * (.6 / peak)).astype(np.float32)
        with tempfile.TemporaryDirectory(prefix='presence_') as tmp:
            wav = Path(tmp) / 'normalized.wav'
            sf.write(wav, raw, 16000, subtype='FLOAT')
            with self.torch.inference_mode():
                voice, _ = self.base.separate_voice_and_music(wav, self.demucs, self.device)
        voice = np.pad(voice[:len(raw)], (0, max(0, len(raw) - len(voice))))
        probs = self.panns.inference(self.base.make_panns_segments(raw))[0]
        return decision_heads.presence_features(probs, self.labels, self.groups, raw, voice)


def fit(X: np.ndarray, y: np.ndarray, names: list[str], C: float = 0.05) -> dict:
    """Standardised L2 logistic regression -> JSON consumed by decision_heads.logistic (numpy only)."""
    from sklearn.linear_model import LogisticRegression
    mu, sd = X.mean(0), X.std(0) + 1e-9
    m = LogisticRegression(C=C, max_iter=5000).fit((X - mu) / sd, y)
    return dict(features=names, mean=mu.tolist(), scale=sd.tolist(), coef=m.coef_[0].tolist(), intercept=float(m.intercept_[0]), C=C)


def predict(head: dict, rows: list[dict]) -> np.ndarray:
    return np.array([decision_heads.logistic(head, r) for r in rows])


def grouped_cv(X, y, groups, names, base='tag_voice_max', folds=5, C=0.05) -> dict:
    """Held-out AUC of the head vs the raw baseline score on the same grouped folds."""
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


def train(feats: list[dict], labels: dict, splits: dict | None, target: str, C: float = 0.05) -> tuple[dict, dict]:
    from sklearn.metrics import roc_auc_score
    names = sorted(k for k in feats[0] if k != 'ID')
    rows = [(f, labels[f['ID']]) for f in feats if labels.get(f['ID']) is not None]
    tr = [(f, y) for f, y in rows if splits is None or splits.get(f['ID']) == 'train']
    X, y = np.array([[f[n] for n in names] for f, _ in tr]), np.array([v for _, v in tr])
    head = fit(X, y, names, C)
    report = {'target': target, 'train_n': len(tr), 'train_pos': int(y.sum()), 'C': C}
    for s in ('val', 'test'):
        ev = [(f, v) for f, v in rows if splits and splits.get(f['ID']) == s]
        ye = np.array([v for _, v in ev])
        if len(set(ye)) == 2:
            report[s] = {'n': len(ev), 'head_auc': round(roc_auc_score(ye, predict(head, [f for f, _ in ev])), 4),
                         'baseline_auc': round(roc_auc_score(ye, [f[BASE[target]] for f, _ in ev]), 4)}
    return head, report


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('cmd', choices=['extract', 'train'])
    ap.add_argument('--manifest', type=Path, required=True)
    ap.add_argument('--root', type=Path)
    ap.add_argument('--assets', type=Path)
    ap.add_argument('--feats', type=Path)
    ap.add_argument('--splits', type=Path)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--per-cell', type=int, default=400, help='rows per (source, presence pattern) to extract')
    ap.add_argument('--C', type=float, default=0.05)
    a = ap.parse_args()
    rows = [json.loads(l) for l in a.manifest.open()]
    if a.cmd == 'extract':
        cells: dict = {}   # deterministic cap per cell keeps GPU time bounded; every source/pattern stays represented
        for r in sorted(rows, key=lambda r: hashlib.sha256(r['ID'].encode()).hexdigest()):
            if r['VOICE_PRESENT'] is None and r['MUSIC_PRESENT'] is None:
                continue
            key = (r['source'], r['VOICE_PRESENT'], r['MUSIC_PRESENT'])
            if len(cells.setdefault(key, [])) < a.per_cell:
                cells[key].append(r)
        todo = [r for rs in cells.values() for r in rs]
        done = {json.loads(l)['ID'] for l in a.out.open()} if a.out.exists() else set()
        ex = BaselineExtractor(a.assets)
        with a.out.open('a') as f:
            for i, r in enumerate(todo):
                if r['ID'] not in done:
                    f.write(json.dumps({'ID': r['ID'], **ex(a.root / r['path'])}) + '\n')
                if i % 500 == 0:
                    f.flush()
                    print(f'presence feats {i}/{len(todo)}', flush=True)
    else:
        import csv
        feats = [json.loads(l) for l in a.feats.open()]
        splits = {r['ID']: r['split'] for r in csv.DictReader(a.splits.open())} if a.splits else None
        heads, reports = {}, {}
        for t in TARGETS:
            heads[t], reports[t] = train(feats, {r['ID']: r[t] for r in rows}, splits, t, a.C)
        a.out.write_text(json.dumps(heads, indent=1) + '\n')
        a.out.with_suffix('.report.json').write_text(json.dumps(reports, indent=1) + '\n')
        print(json.dumps(reports, indent=1))


def self_check() -> None:
    rng = np.random.default_rng(0)
    n = 400
    vp = rng.integers(0, 2, n)
    X = np.c_[rng.random(n) * .3 + .1 * vp, .2 + .5 * vp + rng.normal(0, .1, n)]    # tag weak, stem strong
    names = ['tag_voice_max', 'stem_vocal_ratio_max']
    h = fit(X, vp, names)
    assert json.loads(json.dumps(h)) == h, 'head must be JSON-serialisable'
    feat_rows = [dict(zip(names, r)) for r in X[:5]]
    np.testing.assert_allclose(predict(h, feat_rows), [decision_heads.logistic(h, r) for r in feat_rows])
    cv = grouped_cv(X, vp, np.arange(n) // 4, names)
    assert np.mean(cv['head']) > np.mean(cv['baseline_raw_group_max']) + .1, cv
    feats = [dict(ID=str(i), **dict(zip(names, r))) for i, r in enumerate(X)]
    heads = {t: train(feats, {str(i): int(v) for i, v in enumerate(vp)}, None, t)[0] for t in TARGETS}
    p = decision_heads.presence(heads, feats[0], (0., 0.))
    assert all(0 < v < 1 for v in p), 'trained heads plug into the runtime decision layer'
    print('presence self-check ok')


if __name__ == '__main__':
    self_check() if sys.argv[1:] == ['--self-check'] else main()
