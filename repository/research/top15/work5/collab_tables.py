"""Summary tables for docs/collab (counts only; no audio, features or row-level lists):
  bank_v4_sources.csv   data v4 bank: source x split x class, route, licence (the rows every package trained on)
  new_data_summary.csv  new rows (research/top15/work5/newdata/rows/new_rows.csv): dataset/source x split x class, licence
  train_weight_shares.csv  per head/class mass shares of the P1 finals (the 'weights ...' lines of the final logs)
  python research/top15/work5/collab_tables.py"""
import csv
import json
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
BANK = ROOT / 'results/data_v4_20260923/data_v4_results_20260923T1639Z/data/bank'
OUT = ROOT / 'docs/collab/tables'
NEW = ROOT / 'research/top15/work5/newdata/rows/new_rows.csv'
CLS = lambda r: {'0': 'real', '1': 'fake'}.get(r['FILE_FAKE'], 'unknown')


def table(rows, key, path, extra=()):
    c = defaultdict(lambda: defaultdict(int))
    meta = {}
    for r in rows:
        k = key(r)
        c[k][f"{r['split']}_{CLS(r)}"] += 1
        meta[k] = {e: r.get(e, '') for e in extra}
    cols = sorted({x for v in c.values() for x in v})
    with path.open('w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['source', *extra, *cols, 'total'])
        for k in sorted(c, key=lambda k: -sum(c[k].values())):
            w.writerow([k, *[meta[k][e] for e in extra], *[c[k].get(x, 0) for x in cols], sum(c[k].values())])


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    bank = list(csv.DictReader((BANK / 'bank_trainval.csv').open())) + list(csv.DictReader((BANK / 'bank_test.csv').open()))
    table(bank, lambda r: r['source'], OUT / 'bank_v4_sources.csv', ('route', 'license'))
    if NEW.exists():
        table(list(csv.DictReader(NEW.open())), lambda r: r['source'], OUT / 'new_data_summary.csv', ('dataset', 'route', 'license'))
    with (OUT / 'train_weight_shares.csv').open('w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['final', 'head', 'class', 'rows', 'kish_rows', 'kish_groups', 'max_group_share', 'stratum_share', 'top_sources'])
        for log in sorted((ROOT / '_workspace').glob('final_*_20260928.log')):
            seen = set()
            for line in log.read_text(errors='ignore').splitlines():
                if line.startswith('weights '):
                    _, head, scheme, js = line.split(' ', 3)
                    if (head, scheme) in seen:
                        continue
                    seen.add((head, scheme))
                    for cls, d in json.loads(js).items():
                        w.writerow([log.stem, head, cls, d['rows'], d['kish_rows'], d['kish_groups'], d['max_group_share'],
                                    json.dumps(d['stratum_share']), json.dumps(d['top_sources'])])
    print('written', sorted(p.name for p in OUT.iterdir()))


if __name__ == '__main__':
    main()
