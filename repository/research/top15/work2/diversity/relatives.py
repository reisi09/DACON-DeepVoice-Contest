"""Nearest train relatives of the six largest held-out error families -> relatives.json (design step: uses held-out
FEATURES and LABELS to define families, never held-out predictions or errors).

R(F) = lineage cells (same dataset / construction recipe, fixed by hand below) UNION nearest-neighbour cells: every
family file on a route (voice: DF-Arena fc5-input file mean; music: MERT 6+12 file mean; train-standardised, cosine)
is assigned to its nearest TRAIN file of the same route and label; train cells are ranked by assigned files and added
until they cover >= 50% of that (route, label) part of the family (parts with >= 20 files). Only the family's
ERROR label (pre-declared from the brief's held-out error concentration: HAIM/MLAAD/SpeechFake fakes are missed, real
music/FLEURS-ko/enhanced reals are false positives) gets nearest-neighbour cells, so e.g. HAIM's few real files do not
pull the whole real-music pool into its relatives.
    python relatives.py
"""
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import subsets as S  # noqa: E402

ROOT = S.ROOT
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / 'data_v4'))
import full_bank_train as T  # noqa: E402
import train_backbone_heads as H  # noqa: E402

FAMILIES = {   # test-side membership
    'haim': lambda r: r['source'] == 'haim',
    'real_music': lambda r: r['source'] in ('musan_music', 'constructed_song', 'constructed_mix') and r['MUSIC_FAKE'] in ('0', '0.0'),
    'fleurs_ko': lambda r: r['source'] == 'fleurs_ko',
    'constructed_enhanced': lambda r: r['source'] == 'constructed_enhanced',
    'mlaad': lambda r: r['source'] == 'mlaad',
    'speechfake': lambda r: r['source'] == 'speechfake',
}


ERROR_LABEL = {'haim': 1., 'real_music': 0., 'fleurs_ko': 0., 'constructed_enhanced': 0., 'mlaad': 1., 'speechfake': 1.}


def lineage(fam, train, test):
    """same benchmark / same construction recipe / same generator identity"""
    if fam == 'haim':
        return {S.cell(r) for r in train if r['source'] in ('haim', 'existing:HAIM')}
    if fam == 'real_music':
        return {S.cell(r) for r in train if r['source'] in ('constructed_mix', 'existing:constructed_music') and r['FILE_FAKE'] == '0'}
    if fam == 'constructed_enhanced':
        return {S.cell(r) for r in train if r['source'] == 'constructed_enhanced'}
    if fam == 'mlaad':
        gens = {r['generator'] for r in test if r['source'] == 'mlaad'}
        return {S.cell(r) for r in train if r['source'] == 'mlaad' and r['generator'] in gens}
    if fam == 'speechfake':
        return {S.cell(r) for r in train if r['source'] in ('speechfake', 'constructed_splice')}
    return set()


def means(rows):
    """file-mean features per route with the head's labels (same item rules as the heads)"""
    out = {'voice': {}, 'music': {}}
    for r in rows:
        labels = T.head_labels(r)
        if 'voice' in labels:
            with np.load(H.OLD[r['split']] / f"{r['ID']}.npz") as o:
                silent = bool(o['voice_silent'])
                if not (r['route'] != 'voice' and silent):
                    out['voice'][r['ID']] = (o['voice_embeddings'].astype(np.float32).mean(0), labels['voice'])
        if 'music' in labels:
            with np.load(S.CACHE / f"{r['ID']}.npz") as d:
                L = list(d['layers'])
                f = d['mert_layers'][:, [L.index(6), L.index(12)]].astype(np.float32).reshape(len(d['mert_layers']), -1)
            out['music'][r['ID']] = (f.mean(0), labels['music'])
    return out


def main():
    rows = S.load_rows()
    train, test = rows['train'], rows['test']
    cell = {r['ID']: S.cell(r) for r in train}
    M = {s: means(rows[s]) for s in ('train', 'test')}
    rel, report = {}, {}
    for fam, member in FAMILIES.items():
        fam_ids = {r['ID'] for r in test if member(r)}
        nn_cells, parts = set(), {}
        for route in ('voice', 'music'):
            tr = M['train'][route]
            ids = list(tr)
            X = np.stack([tr[i][0] for i in ids])
            mu, sd = X.mean(0), X.std(0) + 1e-6
            Z = (X - mu) / sd
            Z /= np.linalg.norm(Z, axis=1, keepdims=True)
            ylab = np.array([tr[i][1] for i in ids])
            for y in (ERROR_LABEL[fam],):
                q = [i for i in fam_ids if i in M['test'][route] and M['test'][route][i][1] == y]
                if len(q) < 20:
                    continue
                Q = (np.stack([M['test'][route][i][0] for i in q]) - mu) / sd
                Q /= np.linalg.norm(Q, axis=1, keepdims=True)
                pool = np.flatnonzero(ylab == y)
                nn = pool[(Q @ Z[pool].T).argmax(1)]
                tally = Counter(cell[ids[j]] for j in nn)
                chosen, cov = [], 0
                for c, n in tally.most_common():
                    if cov >= .5 * len(q):
                        break
                    chosen.append(c)
                    cov += n
                nn_cells |= set(chosen)
                parts[f'{route}|{int(y)}'] = {'n_family': len(q), 'nn_cells': chosen, 'covered': round(cov / len(q), 3),
                                              'top_tally': [[c, n] for c, n in tally.most_common(6)]}
        lin = lineage(fam, train, test)
        cells = sorted(lin | nn_cells)
        removed = Counter(S.stratum(r) for r in train if S.cell(r) in cells)
        total = Counter(S.stratum(r) for r in train)
        rel[fam] = {'cells': cells}
        report[fam] = {'n_family_test_rows': len(fam_ids), 'lineage_cells': sorted(lin), 'nn': parts,
                       'rows_removed': sum(removed.values()),
                       'removed_share_by_stratum': {s: [n, round(n / total[s], 3)] for s, n in sorted(removed.items())}}
        print(fam, json.dumps(report[fam], indent=1), flush=True)
    (HERE / 'relatives.json').write_text(json.dumps(rel, indent=1))
    (HERE / 'relatives_report.json').write_text(json.dumps(report, indent=1))


if __name__ == '__main__':
    main()
