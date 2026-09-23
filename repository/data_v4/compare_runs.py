"""Score the held-out test with several trained head sets and audit the file-length shortcut.

  python data_v4/compare_runs.py --test-bank OUT/bank/bank_test.csv --cache CACHE_TEST \
      --run prev_existing9504=DIR --run new=TRAIN [--run new_balanced=TRAIN_BAL] --out REPORT.json
Every run is scored with the legacy decision (PANNs presence, max rule) from the same cached features, so the
runs differ only in their heads. A run dir that also holds test_predictions_decision.csv (file_ensemble.py)
is audited for that variant too. Per target (official masks): EER, the AUC that log-duration alone reaches
(shortcut available in the data), the EER after reweighting both classes to one duration histogram (score
without length help) and the within-class rank correlation of score with duration.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from full_bank_batches import duration_balance  # noqa: E402

TARGETS = (('file', 'FILE_FAKE', 'FILE_FAKE_PROB', None), ('voice', 'VOICE_FAKE', 'VOICE_FAKE_PROB', 'VOICE_PRESENT'),
           ('music', 'MUSIC_FAKE', 'MUSIC_FAKE_PROB', 'MUSIC_PRESENT'))


def eer(y, s, w=None) -> float:
    """DACON formula (roc_curve drop_intermediate=False, argmin |FPR-FNR|), optionally sample-weighted."""
    from sklearn.metrics import roc_curve
    fpr, tpr, _ = roc_curve(y, s, sample_weight=w, drop_intermediate=False)
    i = int(np.argmin(np.abs(fpr - (1 - tpr))))
    return float((fpr[i] + 1 - tpr[i]) / 2)


def rank_corr(a, b) -> float:
    if len(a) < 3:
        return float('nan')
    ra, rb = np.argsort(np.argsort(a)), np.argsort(np.argsort(b))
    return float(np.corrcoef(ra, rb)[0, 1])


def audit(y, s, samples) -> dict:
    from sklearn.metrics import roc_auc_score
    y, s, d = np.asarray(y, float), np.asarray(s, float), np.asarray(samples, float)
    w = duration_balance(y, d)
    return {'n': int(len(y)), 'fake': int(y.sum()), 'eer': round(eer(y, s), 5),
            'duration_only_auc': round(float(roc_auc_score(y, np.log(d))), 4),
            'duration_matched_eer': round(eer(y, s, w), 5),
            'rho_score_duration_real': round(rank_corr(s[y == 0], d[y == 0]), 4),
            'rho_score_duration_fake': round(rank_corr(s[y == 1], d[y == 1]), 4)}


def audit_predictions(rows: list[dict], preds: dict[str, dict], samples: dict[str, int]) -> dict:
    out = {}
    for name, target, column, present in TARGETS:
        keep = [r for r in rows if r[target] != '' and (present is None or r[present] == '1')]
        out[name] = audit([int(r[target]) for r in keep], [float(preds[r['ID']][column]) for r in keep],
                          [samples[r['ID']] for r in keep])
    return out


def ads(metrics: dict) -> float | None:
    m = metrics
    if None in (m.get('file_eer'), m.get('voice_eer'), m.get('music_eer')):
        return None
    return round(.5 * (1 - m['file_eer']) + .2 * (1 - m['voice_eer']) + .3 * (1 - m['music_eer']), 5)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--test-bank', type=Path, required=True)
    ap.add_argument('--cache', type=Path, required=True, help='feature cache holding the test rows')
    ap.add_argument('--run', action='append', required=True, help='name=trained_dir (music_4x.pt, voice_4x.pt)')
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--device', default='cuda')
    a = ap.parse_args()
    from full_bank_evaluate import score_rows, five_columns
    from graph_scoring import official_metrics, SUBMISSION_COLUMNS
    rows = list(csv.DictReader(a.test_bank.open()))
    samples = {k: v['samples'] for k, v in json.loads((a.cache / 'index.json').read_text())['items'].items()}
    report = {'test_rows': len(rows)}
    for spec in a.run:
        name, directory = spec.split('=', 1)
        directory = Path(directory)
        preds = [five_columns(s) for s in score_rows(rows, a.cache, directory, a.device)]
        with (a.out.parent / f'test_predictions_{name}_legacy.csv').open('w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=SUBMISSION_COLUMNS); w.writeheader(); w.writerows(preds)
        official = official_metrics(rows, preds)
        entry = {'dir': str(directory), 'legacy': {'official': official, 'ads_known_subset': ads(official['metrics']),
                                                    'audit': audit_predictions(rows, {p['ID']: p for p in preds}, samples)}}
        decision = directory / 'test_predictions_decision.csv'
        if decision.exists():
            dp = {r['ID']: r for r in csv.DictReader(decision.open())}
            drows = [r for r in rows if r['ID'] in dp]
            official = official_metrics(drows, [dp[r['ID']] for r in drows])
            entry['decision'] = {'rows': len(drows), 'official': official, 'ads_known_subset': ads(official['metrics']),
                                 'audit': audit_predictions(drows, dp, samples)}
        report[name] = entry
        print(name, json.dumps({k: v.get('ads_known_subset') for k, v in entry.items() if isinstance(v, dict)}), flush=True)
    a.out.write_text(json.dumps(report, indent=1, default=str) + '\n')


def self_check() -> None:
    rng = np.random.default_rng(0)
    n = 4000
    y = rng.integers(0, 2, n).astype(float)
    d = np.exp(rng.normal(np.where(y == 1, 11.2, 11.6), .4))          # fakes shorter, like the voice rows
    cheat = np.log(d) * -1 + rng.normal(0, .05, n)                    # score = "short means fake"
    honest = y + rng.normal(0, .8, n)
    a_cheat, a_honest = audit(y, cheat, d), audit(y, honest, d)
    assert a_cheat['duration_only_auc'] < .3 and a_cheat['eer'] < .35, a_cheat
    assert a_cheat['duration_matched_eer'] > a_cheat['eer'] + .1, 'matching must expose a length-only score'
    assert abs(a_honest['duration_matched_eer'] - a_honest['eer']) < .03, 'length-free score is unaffected'
    w = duration_balance(y, d)
    b = np.searchsorted(np.unique(np.quantile(np.log(d), np.linspace(0, 1, 11)[1:-1])), np.log(d), side='right')
    h = [np.bincount(b[y == c], w[y == c], 10) / w[y == c].sum() for c in (0., 1.)]
    assert np.allclose(h[0], h[1], atol=1e-9) and np.isclose(w[y == 1].sum(), (y == 1).sum()), 'one histogram, class totals kept'
    assert eer(np.array([0, 0, 1, 1]), [.1, .2, .8, .9]) == 0.
    print('compare_runs self-check ok')


if __name__ == '__main__':
    self_check() if sys.argv[1:] == ['--self-check'] else main()
