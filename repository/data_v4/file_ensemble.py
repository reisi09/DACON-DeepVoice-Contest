"""Fit the final FILE_FAKE ensemble (decision_heads.file_score) on val; report val and one-shot test.

  python data_v4/file_ensemble.py --bank OUT/bank/bank_trainval.csv --test-bank OUT/bank/bank_test.csv \
      --cache CACHE --train-dir TRAIN [--presence-head OUT/presence_head.json] [--prev-val P.csv --prev-test Q.csv]
writes TRAIN/file_ensemble.json (+ .report.json). Candidates: max (current rule), noisy_or, max_nopresence,
prev (previous package's FILE_FAKE_PROB per ID, if given). Weights: 0.1-step simplex grid on val File EER
(official formula); equal weights are kept unless the grid wins by >= min_gain (guards val overfit).
"""
from __future__ import annotations

import argparse
import csv
import itertools
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import decision_heads as dh  # noqa: E402
from full_bank_contract import role


def eer(y, s) -> float:
    """DACON formula: roc_curve(drop_intermediate=False), argmin |FPR-FNR|, mean of the two."""
    from sklearn.metrics import roc_curve
    fpr, tpr, _ = roc_curve(y, s, pos_label=1, drop_intermediate=False)
    fnr = 1 - tpr
    i = int(np.argmin(np.abs(fpr - fnr)))
    return float((fpr[i] + fnr[i]) / 2)


def candidate_matrix(files: list[dict], names: list[str]) -> dict[str, np.ndarray]:
    return {n: np.array([dh.CANDIDATES[n](f) for f in files], float) for n in names}


def fit(files: list[dict], y: np.ndarray, step: float = .1, min_gain: float = .002) -> tuple[dict, dict]:
    names = [n for n in dh.CANDIDATES if all(dh.CANDIDATES[n](f) is not None for f in files)]
    cand = candidate_matrix(files, names)
    tables = {n: dh.quantile_table(cand[n]) for n in names}
    cdf = {n: np.array([dh.to_cdf(tables[n], v) for v in cand[n]]) for n in names}
    score = lambda w: eer(y, sum(w[n] * cdf[n] for n in names) / sum(w.values()))
    k = round(1 / step)
    grid = [dict(zip(names, c)) for c in itertools.product(range(k + 1), repeat=len(names)) if sum(c) == k]
    results = sorted(((score(w), json.dumps(w, sort_keys=True)) for w in grid if sum(w.values())), key=lambda t: t)
    equal = {n: 1 for n in names}
    e_equal, (e_best, w_best) = score(equal), results[0]
    chosen = json.loads(w_best) if e_equal - e_best >= min_gain else equal
    report = {'candidates_alone': {n: round(eer(y, cand[n]), 5) for n in names}, 'equal_weights_eer': round(e_equal, 5),
              'grid_best_eer': round(e_best, 5), 'grid_best_weights': json.loads(w_best), 'chosen_weights': chosen,
              'chosen_rule': 'grid' if chosen != equal else f'equal (grid gain {e_equal - e_best:.4f} < {min_gain})', 'val_n': len(y)}
    return {'weights': chosen, 'tables': tables, 'fitted_on': 'val', 'candidates': names}, report


def select_rows(rows, split, intended_role):
    """An unknown FILE label masks that metric only, never the other four outputs."""
    return [r for r in rows if r['split'] == split and
            (role(r) == intended_role if r.get('role') else True)]


def known_file_panel(rows, files):
    if [r['ID'] for r in rows] != [f['ID'] for f in files]:
        raise ValueError('Prediction IDs do not match labels')
    known = [(r, f) for r, f in zip(rows, files, strict=True)
             if r.get('FILE_FAKE_mask', '1') == '1' and r.get('FILE_FAKE') in ('0', '1')]
    return [f for r, f in known], np.array([int(r['FILE_FAKE']) for r, f in known])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--bank', type=Path, required=True, help='bank_trainval.csv (val rows are used)')
    ap.add_argument('--test-bank', type=Path, required=True)
    ap.add_argument('--cache', type=Path, required=True)
    ap.add_argument('--test-cache', type=Path, help='feature cache of the test rows (default: --cache)')
    ap.add_argument('--train-dir', type=Path, required=True)
    ap.add_argument('--presence-head', type=Path)
    ap.add_argument('--prev-val', type=Path)
    ap.add_argument('--prev-test', type=Path)
    ap.add_argument('--device', default='cuda')
    a = ap.parse_args()
    from full_bank_evaluate import score_rows
    heads = json.loads(a.presence_head.read_text()) if a.presence_head else None
    prev = lambda p: {r['ID']: float(r['FILE_FAKE_PROB']) for r in csv.DictReader(p.open())} if p else {}

    def files_for(bank: Path, split: str, prev_scores: dict, cache: Path):
        rows = select_rows(list(csv.DictReader(bank.open())), split, 'calibration' if split == 'val' else 'development')
        out = []
        for s in score_rows(rows, cache, a.train_dir, a.device):
            vp, mp = dh.presence(heads, s['presence_features'], (s['Pv'], s['Pm']))
            out.append({**s, 'Pv_raw': s['Pv'], 'Pm_raw': s['Pm'], 'Pv': vp, 'Pm': mp, 'prev_file': prev_scores.get(s['ID'])})
        return out, rows

    val, val_rows = files_for(a.bank, 'val', prev(a.prev_val), a.cache)
    fit_files, yv = known_file_panel(val_rows, val)
    if set(yv.tolist()) != {0, 1}:
        raise ValueError('Calibration needs both FILE classes')
    ens, report = fit(fit_files, yv)
    ens['fitted_on'] = 'calibration' if all(r.get('role') == 'calibration' for r in val_rows) else 'val'
    report['fit_role'] = ens['fitted_on']
    test, test_rows = files_for(a.test_bank, 'test', prev(a.prev_test), a.test_cache or a.cache)
    if any(f['prev_file'] is None for f in test) and ens['weights'].get('prev', 0):
        raise SystemExit('prev candidate fitted on val but missing for test; pass --prev-test')
    test_known, yt = known_file_panel(test_rows, test)
    if set(yt.tolist()) != {0, 1}:
        raise ValueError('Development FILE metric needs both classes')
    if {r['component_id'] for r in val_rows} & {r['component_id'] for r in test_rows}:
        raise ValueError('Calibration and development components overlap')
    report['development'] = {'n': len(yt), 'ensemble_eer': round(eer(yt, [dh.file_score(ens, f) for f in test_known]), 5),
                               'max_rule_eer': round(eer(yt, [dh.CANDIDATES['max'](f) for f in test_known]), 5)}
    # all five columns + official metrics, before (legacy PANNs max, max rule) and after (presence head + ensemble)
    from graph_scoring import official_metrics, SUBMISSION_COLUMNS
    from full_bank_evaluate import five_columns
    for split, fs, rows in (('val', val, val_rows), ('test', test, test_rows)):
        for tag, ph, en in (('legacy', None, None), ('decision', heads, ens)):
            preds = [five_columns({**f, 'Pv': f['Pv_raw'], 'Pm': f['Pm_raw']}, ph, en) for f in fs]
            with (a.train_dir / f'{split}_predictions_{tag}.csv').open('w', newline='') as out:
                w = csv.DictWriter(out, fieldnames=SUBMISSION_COLUMNS); w.writeheader(); w.writerows(preds)
            report[f'{split}_official_{tag}'] = official_metrics(rows, preds)
    (a.train_dir / 'file_ensemble.json').write_text(json.dumps(ens) + '\n')
    (a.train_dir / 'file_ensemble.report.json').write_text(json.dumps(report, indent=1, default=str) + '\n')
    print(json.dumps(report, indent=1, default=str))


def self_check() -> None:
    rng = np.random.default_rng(3)
    n = 600
    y = rng.integers(0, 2, n)
    # voice evidence informative, presence noisy: max_nopresence should carry signal the max rule loses
    files = [dict(Pv=float(np.clip(rng.normal(.5, .3), 0, 1)), Pm=float(rng.random()),
                  Fv=float(np.clip(.35 * t + rng.normal(.3, .15), 0, 1)), Fm=float(rng.random())) for t in y]
    ens, rep = fit(files, y)
    assert set(ens['weights']) == {'max', 'noisy_or', 'max_nopresence'}, 'prev absent -> not a candidate'
    e_ens = eer(y, [dh.file_score(ens, f) for f in files])
    assert e_ens <= rep['candidates_alone']['max'] + 1e-9, rep
    assert abs(eer(np.array([0, 0, 1, 1]), [.1, .2, .8, .9])) < 1e-12 and abs(eer(np.array([0, 1]), [.9, .1]) - 1) < 1e-12
    same = [dh.file_score(ens, f) for f in files[:10]]
    assert same == [dh.file_score(ens, f) for f in reversed(files[:10])][::-1], 'order-independent per-file scores'
    assert json.loads(json.dumps(ens)) == ens
    print('file_ensemble self-check ok')


if __name__ == '__main__':
    self_check() if sys.argv[1:] == ['--self-check'] else main()
