"""Analysis-driven train/val/test split (no random/seed assignment).

  python data_v4/split.py analyze --manifest OUT/manifest.jsonl --report OUT/split_analysis
  (read the report, then write data_v4/split_rules.json by hand — DESIGN.md §7)
  python data_v4/split.py apply   --manifest ... --rules data_v4/split_rules.json --out OUT/splits.csv
  python data_v4/split.py verify  --manifest ... --splits OUT/splits.csv
Unit of assignment = connected component of shared speaker/track/prompt/text/parent keys.
"""
from __future__ import annotations

import argparse
import collections
import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from build_manifest import LABELS, components  # noqa: E402


def load(path: Path) -> list[dict]:
    return [json.loads(l) for l in path.open()]


def rtype(r: dict) -> str:
    v, m = r['VOICE_PRESENT'], r['MUSIC_PRESENT']
    if v is None or m is None:
        return 'unknown_presence'
    return {(1, 0): 'voice', (0, 1): 'music', (1, 1): 'mixed', (0, 0): 'none'}[(v, m)]


def cell(r: dict, keys: list[str]) -> str:
    get = {'type': rtype(r), 'FILE_FAKE': str(r['FILE_FAKE']), 'source': r['source'],
           'generator': str(r.get('generator')), 'language': str(r.get('language')), 'codec': r.get('codec_chain', '')}
    return '|'.join(get[k] for k in keys)


# ---------------------------------------------------------------- analyze
def analyze(rows: list[dict]) -> dict:
    comp = components(rows)
    sizes = collections.Counter(comp.values())
    by_comp = collections.defaultdict(list)
    for r in rows:
        by_comp[comp[r['ID']]].append(r)
    big = sorted(by_comp.items(), key=lambda kv: -len(kv[1]))[:15]
    cells = collections.defaultdict(lambda: [0, set()])
    for r in rows:
        c = cells[cell(r, ['type', 'FILE_FAKE', 'source', 'generator'])]
        c[0] += 1
        c[1].add(comp[r['ID']])
    lengths = collections.defaultdict(list)
    codecs = collections.defaultdict(collections.Counter)
    for r in rows:
        k = f"{rtype(r)}|{r['FILE_FAKE']}"
        lengths[k].append(r.get('crop_len_s', 0))
        codecs[k][r.get('codec_chain')] += 1
    q = lambda xs: [round(sorted(xs)[int(p * (len(xs) - 1))], 1) for p in (.05, .25, .5, .75, .95)] if xs else []
    return dict(
        rows=len(rows), components=len(sizes),
        component_size_quantiles=q(list(sizes.values())),
        largest_components=[dict(size=len(rs), sources=collections.Counter(r['source'] for r in rs).most_common(5),
                                 labels=collections.Counter(str(r['FILE_FAKE']) for r in rs)) for _, rs in big],
        type_x_label=collections.Counter(f"{rtype(r)}|{r['FILE_FAKE']}" for r in rows),
        masks={n: sum(r[n] is None for r in rows) for n in LABELS},
        cells={k: dict(files=v[0], components=len(v[1])) for k, v in sorted(cells.items())},
        length_quantiles_s={k: q(v) for k, v in sorted(lengths.items())},
        codec_by_type_label={k: dict(v) for k, v in sorted(codecs.items())},
        metadata_probe=metadata_probe(rows, comp),
    )


def metadata_probe(rows: list[dict], comp: dict) -> dict:
    """Grouped-CV AUC of FILE_FAKE from non-content metadata only. High AUC = a shortcut to inspect
    (evidence, not proof the model uses it — review P1). Reported with fold spread, no pass/fail redraw."""
    try:
        import numpy as np
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import roc_auc_score
        from sklearn.model_selection import GroupKFold
    except ImportError:
        return {'status': 'NOT_RUN', 'reason': 'sklearn missing'}
    out = {}
    codecs = sorted({r.get('codec_chain', '') for r in rows})
    for t in ('voice', 'music', 'mixed'):
        rs = [r for r in rows if rtype(r) == t and r['FILE_FAKE'] is not None]
        y = np.array([r['FILE_FAKE'] for r in rs])
        if len(rs) < 50 or len(set(y)) < 2:
            out[t] = {'status': 'NOT_EVALUATED', 'n': len(rs)}
            continue
        x = np.array([[r.get('crop_len_s', 0), r.get('orig_duration_s', 0), r.get('channels', 1), r.get('gain_db', 0)]
                      + [r.get('codec_chain') == c for c in codecs] for r in rs], float)
        g = [comp[r['ID']] for r in rs]
        aucs = []
        for tr, ev in GroupKFold(5).split(x, y, g):
            if len(set(y[ev])) < 2 or len(set(y[tr])) < 2:
                continue
            m = LogisticRegression(max_iter=2000).fit((x[tr] - x[tr].mean(0)) / (x[tr].std(0) + 1e-9), y[tr])
            aucs.append(roc_auc_score(y[ev], m.predict_proba((x[ev] - x[tr].mean(0)) / (x[tr].std(0) + 1e-9))[:, 1]))
        out[t] = {'n': len(rs), 'auc_folds': [round(a, 3) for a in aucs]}
    return out


# ---------------------------------------------------------------- apply
def matches(r: dict, rule: dict) -> bool:
    return all(str(r.get(k)) in map(str, v) for k, v in rule.items())


def apply(rows: list[dict], rules: dict) -> tuple[dict[str, str], dict]:
    """1) components touching any test rule -> test (domain/generator holdout, stated in rules);
    2) val: greedy deterministic fill of per-cell targets over remaining components
       (rarest cell first, smallest component first, ties by component id);
    3) rest -> train."""
    comp = components(rows)
    by_comp = collections.defaultdict(list)
    for r in rows:
        by_comp[comp[r['ID']]].append(r)
    split = {}
    for c, rs in by_comp.items():
        if any(matches(r, rule) for r in rs for rule in rules.get('test_holdout', [])):
            split[c] = 'test'
    keys = rules['cell_keys']
    frac, min_c = rules['val_fraction'], rules.get('min_val_components_per_cell', 2)
    pool = {c: rs for c, rs in by_comp.items() if c not in split}
    cell_files = collections.Counter(cell(r, keys) for rs in pool.values() for r in rs)
    cell_comps = collections.defaultdict(set)
    for c, rs in pool.items():
        for r in rs:
            cell_comps[cell(r, keys)].add(c)
    # a cell with fewer than 2*min_c components cannot be split without emptying train -> train only
    target = {k: (max(1, round(n * frac)) if len(cell_comps[k]) >= 2 * min_c else 0) for k, n in cell_files.items()}
    got = collections.Counter()
    for k in sorted(target, key=lambda k: (cell_files[k], k)):
        for c in sorted(cell_comps[k], key=lambda c: (len(pool[c]), c)):
            if got[k] >= target[k]:
                break
            if c in split:
                continue
            contrib = collections.Counter(cell(r, keys) for r in pool[c])
            # skip a component that would overshoot any cell by more than half its target
            if any(got[j] + n > target[j] * 1.5 + 1 for j, n in contrib.items()):
                continue
            split[c] = 'val'
            got.update(contrib)
    for c in pool:
        split.setdefault(c, 'train')
    report = dict(val_target_vs_got={k: [target[k], got[k]] for k in sorted(target) if target[k]},
                  train_only_cells=sorted(k for k, t in target.items() if t == 0))
    return {r['ID']: split[comp[r['ID']]] for r in rows}, report


# ---------------------------------------------------------------- verify
def verify(rows: list[dict], split: dict[str, str]) -> dict:
    comp = components(rows)
    per_comp = collections.defaultdict(set)
    for r in rows:
        per_comp[comp[r['ID']]].add(split[r['ID']])
    leaks = [c for c, s in per_comp.items() if len(s) > 1]
    counts = collections.Counter(f"{split[r['ID']]}|{rtype(r)}|{r['FILE_FAKE']}" for r in rows)
    one_class = [f'{s}|{t}' for s in ('train', 'val', 'test') for t in ('voice', 'music', 'mixed')
                 if (counts[f'{s}|{t}|0'] == 0) != (counts[f'{s}|{t}|1'] == 0)]
    return dict(ok=not leaks, component_leaks=len(leaks), counts=dict(sorted(counts.items())),
                single_class_type_splits=one_class)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('cmd', choices=['analyze', 'apply', 'verify'])
    ap.add_argument('--manifest', type=Path, required=True)
    ap.add_argument('--report', type=Path)
    ap.add_argument('--rules', type=Path)
    ap.add_argument('--out', type=Path)
    ap.add_argument('--splits', type=Path)
    a = ap.parse_args()
    rows = load(a.manifest)
    if a.cmd == 'analyze':
        rep = analyze(rows)
        a.report.with_suffix('.json').write_text(json.dumps(rep, indent=1, ensure_ascii=False, default=str) + '\n')
        print(json.dumps({k: rep[k] for k in ('rows', 'components', 'component_size_quantiles', 'type_x_label', 'metadata_probe')},
                         indent=1, default=str))
    elif a.cmd == 'apply':
        split, rep = apply(rows, json.loads(a.rules.read_text()))
        with a.out.open('w', newline='') as f:
            w = csv.writer(f)
            w.writerow(['ID', 'split'])
            w.writerows(sorted(split.items()))
        a.out.with_suffix('.report.json').write_text(json.dumps({**rep, **verify(rows, split)}, indent=1) + '\n')
        print(json.dumps(verify(rows, split), indent=1))
    else:
        split = {r['ID']: r['split'] for r in csv.DictReader(a.splits.open())}
        res = verify(rows, split)
        print(json.dumps(res, indent=1))
        sys.exit(0 if res['ok'] else 1)


def self_check() -> None:
    from build_manifest import lab
    rows = []
    for spk in range(20):   # voice: 20 speakers x 3 utts, gen 'a' fakes for even speakers, gen 'h' holdout on spk 19
        for i in range(3):
            fake = spk % 2 == 0
            rows.append(dict(ID=f'v{spk}_{i}', source='s', generator='h' if spk == 19 else ('a' if fake else 'bona'),
                             group_keys=[f'spk:{spk}'], **lab(1, 0, int(fake or spk == 19), None)))
    rows.append(dict(ID='mix0', source='mix', generator='x', group_keys=['spk:3', 'trk:0'], **lab(1, 1, 0, 1)))
    rows.append(dict(ID='m0', source='m', generator='g', group_keys=['trk:0'], **lab(0, 1, None, 1)))
    rules = dict(test_holdout=[{'generator': ['h']}], cell_keys=['type', 'FILE_FAKE', 'source', 'generator'],
                 val_fraction=0.2, min_val_components_per_cell=2)
    split, rep = apply(rows, rules)
    assert apply(rows, rules)[0] == split, 'assignment must be deterministic'
    assert all(split[f'v19_{i}'] == 'test' for i in range(3)), 'holdout generator -> test'
    assert split['mix0'] == split['m0'] == split['v3_0'], 'shared parent keys keep one component together'
    assert verify(rows, split)['ok']
    assert {s for s in split.values()} >= {'train', 'val', 'test'}
    split['v3_1'] = 'val' if split['v3_0'] != 'val' else 'train'
    assert not verify(rows, split)['ok'], 'leak must be detected'
    print('split self-check ok')


if __name__ == '__main__':
    self_check() if sys.argv[1:] == ['--self-check'] else main()
