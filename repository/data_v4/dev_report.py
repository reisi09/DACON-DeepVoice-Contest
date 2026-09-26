"""Score head sets on the S2 dev rows with the official metric (graph_scoring.official_metrics, official masks).

  python data_v4/dev_report.py --out REPORT.json name=HEADS_DIR [name=HEADS_DIR ...]
Dev = never-trained rows: held-out AI-Hub calls (probe), KtelSpeech, unseen AI-Hub sessions/pieces, Jamendo artists,
the Sonauto family and two MLAAD-ko systems through held-out call channels. Each head set uses its own
presence_head.json / file_ensemble.json when present (else legacy PANNs max + max rule), like the runtime.
Also reports per-source median FILE score and the share above the model's own dev File-EER threshold.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_curve

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from full_bank_evaluate import score_rows, five_columns  # noqa: E402
from graph_scoring import official_metrics  # noqa: E402

W = ROOT / '_workspace'
DEV = [(W / 'aihub_probe/heldout.csv', W / 'aihub_probe/cache_heldout'),
       (W / 's2_extra/bank_A_dev.csv', W / 's2_extra/cache_A_dev'),
       (W / 's2_extra/bank_B_dev.csv', W / 's2_extra/cache_B_dev')]
PRESENCE = ROOT / 'results/data_v4_20260923/data_v4_results_20260923T1639Z/train/presence_head.json'


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('runs', nargs='+', help='name=heads_dir')
    a = ap.parse_args()
    parts = [(list(csv.DictReader(m.open())), c) for m, c in DEV if m.exists() and (c / 'index.json').exists()]
    report = {'dev_rows': sum(len(r) for r, _ in parts)}
    for spec in a.runs:
        name, heads = spec.split('=', 1)
        heads = Path(heads)
        ensemble = json.loads((heads / 'file_ensemble.json').read_text()) if (heads / 'file_ensemble.json').exists() else None
        presence = json.loads(PRESENCE.read_text()) if ensemble is not None else None
        rows, preds = [], []
        for part, cache in parts:
            rows += part
            preds += [five_columns(s, presence, ensemble) for s in score_rows(part, cache, heads, 'cpu')]
        m = official_metrics(rows, preds)['metrics']
        ads = .5 * (1 - m['file_eer']) + .2 * (1 - m['voice_eer']) + .3 * (1 - m['music_eer'])
        y = np.array([int(r['FILE_FAKE']) for r in rows]); s = np.array([float(p['FILE_FAKE_PROB']) for p in preds])
        fpr, tpr, th = roc_curve(y, s, drop_intermediate=False)
        t = th[int(np.argmin(np.abs(fpr - (1 - tpr))))]
        by = {}
        for r, p in zip(rows, preds):
            by.setdefault(f"{'FAKE' if r['FILE_FAKE'] == '1' else 'REAL'} {r['source']}", []).append(float(p['FILE_FAKE_PROB']))
        report[name] = {'dev_ads': round(ads, 5), 'metrics': {k: None if v is None else round(v, 5) for k, v in m.items()},
                        'per_source': {k: {'n': len(v), 'median_file': round(float(np.median(v)), 4),
                                           'error_at_dev_thr': round(float(np.mean(np.array(v) >= t) if k.startswith('REAL')
                                                                         else np.mean(np.array(v) < t)), 4)}
                                       for k, v in sorted(by.items())}}
        print(name, report[name]['dev_ads'], report[name]['metrics'], flush=True)
    a.out.write_text(json.dumps(report, indent=1, ensure_ascii=False) + '\n')


if __name__ == '__main__':
    main()
